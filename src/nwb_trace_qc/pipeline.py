"""End-to-end pipeline orchestrator: discover → cache → compute → threshold → override → report.

Each stage is timed and its counts/errors collected into a `run_report.json` written
next to the QC report. The metric-compute stage streams results via `pool.imap_unordered`
and flushes to the cache every `flush_every` results so a Ctrl-C mid-run preserves work.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import PIPELINE_VERSION, __version__
from .cache import append_rows, cached_hashes, filter_for_version, load_cache
from .config import ProjectConfig, default_families
from .families import METRIC_TO_FAMILY
from .manifest import build_manifest, load_acquisition_index, save_manifest, unique_nwbs
from .metrics import compute_metrics
from .overrides import apply_overrides, init_overrides_file, load_overrides
from .report import write_report
from .stimuli import StimulusFamilyMap
from .thresholds import evaluate, load_thresholds
from . import vision as _vision

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def _compute_one(args):
    nwb_path, nwb_sha256, families, use_efel, trim_bad_ending = args
    fm = StimulusFamilyMap(families)
    metrics = compute_metrics(nwb_path, fm,
                                use_efel=use_efel, trim_bad_ending=trim_bad_ending)
    metrics.update({
        "nwb_sha256": nwb_sha256,
        "nwb_path": str(nwb_path),
        "pipeline_version": PIPELINE_VERSION,
    })
    return metrics


def _peak_rss_mb() -> float:
    """Return peak RSS for this process (and its children) in megabytes.

    `ru_maxrss` is bytes on macOS, kilobytes on Linux/BSD.
    """
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        kids_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except Exception:
        kids_rss = 0
    raw = max(self_rss, kids_rss)
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _stratified_picks(candidates: list, max_n: int = 3) -> list:
    """From an ordered list of candidates, pick at most max_n samples that are
    spread across the list (first / middle / last for max_n=3) rather than just
    the head. Surfaces within-family drift in the static report.
    """
    if len(candidates) <= max_n:
        return list(candidates)
    if max_n <= 1:
        return [candidates[0]]
    # Even-spaced indices including endpoints
    last = len(candidates) - 1
    idxs = [round(i * last / (max_n - 1)) for i in range(max_n)]
    # Dedup while preserving order (defensive: rounding can collide on tiny lists)
    seen, picks = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i); picks.append(candidates[i])
    return picks


def _provenance_sweep_names(triggered: list[dict], metric_row: dict | None) -> dict[str, list[str]]:
    """For each failing metric, look up the provenance fields in `metric_row` and
    return the sweep names that drove the value. Maps family → ordered list of
    sweep names to render. For drift metrics (`*_drift_pct`, `*_drift_mv`,
    `vrest_session_drift_mv` …) this is [first_sweep, last_sweep] — the two
    snapshots whose comparison produced the failure. For median-based metrics,
    the provenance carries `first` and `last` of the contributing set; we use
    those as the bookends so the user can spot drift even within a family.
    """
    if not metric_row:
        return {}
    by_family: dict[str, list[str]] = {}
    for t in triggered:
        metric = t.get("metric")
        if metric not in METRIC_TO_FAMILY:
            continue
        family = METRIC_TO_FAMILY[metric]
        # Most provenance lives under `<base_metric>_provenance`:
        #   vrest_mv → vrest_mv_provenance, rs_drift_pct → rs_mohm_provenance
        prov_key_candidates = [
            f"{metric}_provenance",
            "rs_mohm_provenance" if metric.startswith("rs_") else None,
            "vrest_mv_provenance" if metric.startswith("vrest_") else None,
            "ap_amp_overshoot_mv_provenance" if metric.startswith("ap_") else None,
        ]
        prov_raw = None
        for key in prov_key_candidates:
            if key and key in metric_row and metric_row[key] not in (None, ""):
                prov_raw = metric_row[key]; break
        if not prov_raw:
            continue
        try:
            prov = json.loads(prov_raw) if isinstance(prov_raw, str) else dict(prov_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        names = []
        if prov.get("first"): names.append(prov["first"])
        if prov.get("last") and prov.get("last") != prov.get("first"):
            names.append(prov["last"])
        if names:
            by_family.setdefault(family, [])
            for n in names:
                if n not in by_family[family]:
                    by_family[family].append(n)
    return by_family


def _make_thumbnail(nwb_path: Path, out_path: Path, *,
                    families: dict[str, list[str]],
                    triggered_metrics: list[dict],
                    metric_row: dict | None = None) -> tuple[Path | None, str]:
    """Render up to 3 representative sweeps for an NWB and return (path, status).

    Status is one of: 'rendered', 'no_voltage_sweeps' (the NWB has zero voltage
    acquisitions — pathological), 'render_error' (matplotlib / I/O blew up).

    Sweep selection priority:
      1. If a failing metric has provenance (e.g. rs_drift_pct knows which
         first/last test_pulse sweeps drove the value), render *those exact
         sweeps* — that's the most diagnostically useful picture.
      2. Otherwise fall back to family-matched stratified picks (first/middle/last
         from the families implicated by the failing metrics).
      3. Last resort: any voltage sweep (with a warning logged).

    Each subplot title includes the failing metric(s) tied to its family, so a
    triager can read it as "this is what to look at for the vrest_mv FLAG".
    """
    from .nwb_io import open_nwb
    try:
        reasons = [t.get("metric") for t in triggered_metrics if isinstance(t, dict)]
        with open_nwb(nwb_path) as f:
            wanted_families = {METRIC_TO_FAMILY[r] for r in reasons
                               if r in METRIC_TO_FAMILY}
            if not wanted_families:
                wanted_families = {"spontaneous_hold", "ap_waveform"}
            # Group failing metric names by family so we can annotate titles
            metrics_by_family: dict[str, list[str]] = {}
            for t in triggered_metrics:
                m = t.get("metric") if isinstance(t, dict) else None
                fam = METRIC_TO_FAMILY.get(m) if m else None
                if fam:
                    metrics_by_family.setdefault(fam, []).append(m)
            fm = StimulusFamilyMap(families)

            voltage_acquisitions = []
            name_to_obj: dict[str, tuple] = {}
            for name, obj in f.acquisition.items():
                unit = (getattr(obj, "unit", "") or "").lower()
                if unit not in {"volts", "v", ""}:
                    continue
                stim = name.split("__")[1] if "__" in name and name.count("__") >= 2 else name
                fam = fm.family_of(stim)
                voltage_acquisitions.append((fam, name, obj))
                name_to_obj[name] = (fam, name, obj)
            if not voltage_acquisitions:
                return None, "no_voltage_sweeps"

            # Priority 1: provenance-driven picks (the actual sweeps that drove
            # the failing metrics). Cap at 3 total to keep the PNG readable.
            picks: list[tuple] = []
            picked_names: set[str] = set()
            provenance_picks = _provenance_sweep_names(triggered_metrics, metric_row)
            for fam, names in provenance_picks.items():
                for name in names:
                    if name in name_to_obj and name not in picked_names:
                        picks.append(name_to_obj[name]); picked_names.add(name)
                    if len(picks) >= 3:
                        break
                if len(picks) >= 3:
                    break

            # Priority 2: fill any remaining slots from family-matched stratified picks.
            if len(picks) < 3:
                family_matches = [t for t in voltage_acquisitions
                                  if t[0] in wanted_families and t[1] not in picked_names]
                strat = _stratified_picks(family_matches, max_n=3 - len(picks))
                for t in strat:
                    if t[1] not in picked_names:
                        picks.append(t); picked_names.add(t[1])

            # Priority 3 (fallback): any voltage sweep.
            if not picks:
                log.warning(
                    "thumbnail: %s — no sweeps matched %s (families found: %s); "
                    "falling back to first 3 voltage sweeps. Map your lab's protocols "
                    "to stimulus_protocols in the project YAML for targeted plots.",
                    nwb_path.name, sorted(wanted_families),
                    sorted({f or "?unmapped?" for f, _, _ in voltage_acquisitions}),
                )
                picks = _stratified_picks(voltage_acquisitions, max_n=3)

            fig, axes = plt.subplots(len(picks), 1, figsize=(6, 1.8 * len(picks)), sharex=False)
            if len(picks) == 1:
                axes = [axes]
            for ax, (fam, name, obj) in zip(axes, picks):
                data = np.asarray(obj.data[:]).reshape(-1)
                rate = float(getattr(obj, "rate", 0) or 0)
                t = np.arange(len(data)) / rate if rate > 0 else np.arange(len(data))
                ax.plot(t, data * 1000.0, lw=0.6, color="#222")
                fam_label = fam or "unmapped"
                # Annotate with failing metrics tied to this family so the user
                # can read "spontaneous_hold · vrest_mv: ic__SponHold30__001" and
                # know exactly which failed metric this sweep is illustrating.
                metric_tag = ""
                if fam in metrics_by_family:
                    ms = metrics_by_family[fam]
                    metric_tag = f" · {', '.join(ms[:2])}{'…' if len(ms) > 2 else ''}"
                ax.set_title(f"{fam_label}{metric_tag}: {name[:50]}", fontsize=8)
                ax.set_ylabel("mV", fontsize=7)
                ax.tick_params(labelsize=6)
            axes[-1].set_xlabel("time (s)", fontsize=7)
            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=80)
            plt.close(fig)
            return out_path, "rendered"
    except Exception as e:  # noqa: BLE001
        log.warning("thumbnail: %s — render failed: %s", nwb_path.name, e)
        return None, "render_error"


def run(
    cfg: ProjectConfig,
    *,
    filter_dataset: str | None = None,
    report_only: bool = False,
    progress_callback: ProgressCallback | None = None,
    flush_every: int = 100,
    max_cost_usd: float | None = None,
) -> dict:
    """Execute the full pipeline.

    Returns a summary dict with timing + counts. Writes:
      - manifest parquet, cache parquet
      - report HTML + CSV
      - thumbnails
      - run_report.json (per-stage timing, counts, vision cost, memory peak)

    `progress_callback(stage, done, total)` is called periodically during compute
    (every result) and at stage boundaries (done=total=0 marks a stage start; done=total
    of >0 marks completion).
    """
    t_run_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    init_overrides_file(cfg.overrides_path)

    stages: dict[str, dict[str, Any]] = {}
    compute_errors: list[dict[str, Any]] = []

    def _stage_start(name: str, total: int = 0) -> float:
        log.info("stage: %s starting (total=%d)", name, total)
        # Only ping the progress bar for stages with real work — instant stages
        # (overrides, report, disabled vision) should rely on the INFO log alone
        # and avoid colliding with stderr log lines on the same TTY row.
        if progress_callback and total > 0:
            progress_callback(name, 0, total)
        return time.time()

    def _stage_end(name: str, t0: float, extra: dict[str, Any]) -> None:
        elapsed = round(time.time() - t0, 3)
        stages[name] = {"elapsed_s": elapsed, **extra}
        log.info("stage: %s done in %.2fs (%s)", name, elapsed,
                 ", ".join(f"{k}={v}" for k, v in extra.items() if not isinstance(v, list)))
        n_total = int(extra.get("n_total", 0) or 0)
        if progress_callback and n_total > 0:
            progress_callback(name, n_total, n_total)

    # ─── Stage 1: manifest ───────────────────────────────────────
    t0 = _stage_start("manifest_build")
    manifest = build_manifest(cfg)
    if filter_dataset:
        manifest = manifest[manifest["dataset"] == filter_dataset].reset_index(drop=True)
    save_manifest(manifest, cfg.manifest_path)
    manifest_stats = list(manifest.attrs.get("manifest_stats", []))
    n_sha_reused = sum(s.get("n_sha256_reused", 0) for s in manifest_stats)
    n_sha_recomp = sum(s.get("n_sha256_recomputed", 0) for s in manifest_stats)
    _stage_end("manifest_build", t0, {
        "n_files": int(len(manifest)),
        "n_unique_hashes": int(unique_nwbs(manifest).shape[0]) if not manifest.empty else 0,
        "n_sha256_reused": int(n_sha_reused),
        "n_sha256_recomputed": int(n_sha_recomp),
        "n_total": int(len(manifest)),
    })

    if manifest.empty:
        _write_run_report(cfg, started_at, t_run_start, stages, compute_errors,
                          manifest_stats, vision_stats={"enabled": False},
                          n_workers=cfg.n_workers)
        return {"n_cells": 0, "elapsed_s": round(time.time() - t_run_start, 2),
                "note": "no NWBs found",
                "run_report": str(cfg.output_dir / "run_report.json")}

    # ─── Stage 2: compute metrics for unique NWBs not in cache ───
    cache_df = load_cache(cfg.cache_path)
    have = cached_hashes(cache_df)
    uniq = unique_nwbs(manifest)
    todo = uniq[~uniq["nwb_sha256"].isin(have)]
    n_cache_hits = int(uniq.shape[0] - todo.shape[0])
    n_compute = 0

    t0 = _stage_start("metric_compute", total=int(todo.shape[0]))
    n_rs_fallback_cells = 0   # NWBs whose Rs computation fell back to the 50 pA hack
    if not report_only and not todo.empty:
        args_list = [(Path(row.nwb_path), row.nwb_sha256, cfg.stimulus_protocols,
                       cfg.use_efel, cfg.trim_bad_ending)
                     for row in todo.itertuples(index=False)]
        batch: list[dict] = []
        total = len(args_list)

        def _accumulate(metrics: dict) -> None:
            nonlocal n_rs_fallback_cells
            batch.append(metrics)
            if metrics.get("compute_error"):
                compute_errors.append({
                    "nwb_path": metrics.get("nwb_path"),
                    "nwb_sha256": metrics.get("nwb_sha256"),
                    "error": metrics["compute_error"],
                })
            if int(metrics.get("n_rs_fallback_sweeps", 0) or 0) > 0:
                n_rs_fallback_cells += 1

        if cfg.n_workers and cfg.n_workers > 1:
            with mp.Pool(cfg.n_workers) as pool:
                for i, metrics in enumerate(pool.imap_unordered(_compute_one, args_list), start=1):
                    _accumulate(metrics)
                    n_compute += 1
                    if progress_callback:
                        progress_callback("metric_compute", i, total)
                    if len(batch) >= flush_every:
                        append_rows(cfg.cache_path, batch)
                        batch.clear()
        else:
            for i, a in enumerate(args_list, start=1):
                metrics = _compute_one(a)
                _accumulate(metrics)
                n_compute += 1
                if progress_callback:
                    progress_callback("metric_compute", i, total)
                if len(batch) >= flush_every:
                    append_rows(cfg.cache_path, batch)
                    batch.clear()
        if batch:
            append_rows(cfg.cache_path, batch)
            batch.clear()
    _stage_end("metric_compute", t0, {
        "n_new": int(n_compute),
        "n_cache_hits": int(n_cache_hits),
        "n_workers": int(cfg.n_workers),
        "n_errors": len(compute_errors),
        "n_rs_fallback": int(n_rs_fallback_cells),
        "n_total": int(todo.shape[0]),
    })
    if n_rs_fallback_cells > 0:
        log.warning(
            "%d NWB(s) had no paired CurrentClampStimulusSeries — Rs values for "
            "those test-pulse sweeps used the legacy 50 pA assumption. Verdicts on "
            "rs_mohm_* and rs_drift_pct may be off by a multiplicative factor for "
            "labs that use a different test-pulse amplitude.",
            n_rs_fallback_cells,
        )

    # ─── Stage 3: apply thresholds → per-cell verdicts ───────────
    t0 = _stage_start("thresholds", total=int(manifest.shape[0]))
    cache_df = filter_for_version(load_cache(cfg.cache_path))
    if cfg.thresholds_file is None or not cfg.thresholds_file.exists():
        raise FileNotFoundError(f"thresholds_file not found: {cfg.thresholds_file}")
    thresholds = load_thresholds(cfg.thresholds_file)
    # v0.6.0: critical-metric whitelist. Empty config list ⇒ bundled defaults.
    from .families import DEFAULT_CRITICAL_METRICS
    critical_metrics = (set(cfg.critical_metrics) if cfg.critical_metrics
                         else DEFAULT_CRITICAL_METRICS)
    rows = []
    total_th = int(manifest.shape[0])
    for i, r in enumerate(manifest.itertuples(index=False), start=1):
        if progress_callback and (i % 50 == 0 or i == total_th):
            progress_callback("thresholds", i, total_th)
        metric_row = cache_df[cache_df["nwb_sha256"] == r.nwb_sha256]
        if metric_row.empty:
            rows.append({
                "cell_id": r.cell_id, "dataset": r.dataset,
                "nwb_path": r.nwb_path, "nwb_sha256": r.nwb_sha256,
                "computed_verdict": "flag",
                "triggered_metrics": [{"metric": "_no_cache", "value": None,
                                        "verdict": "flag", "reason": "no metrics computed",
                                        "critical": True}],
            })
            continue
        m = metric_row.iloc[0].to_dict()
        verdict, triggered = evaluate(m, thresholds, critical_metrics=critical_metrics)
        # Spread every metric column from the cache row through to the report
        # row — the previous hardcoded whitelist was last touched in v0.1.x and
        # silently dropped every v0.4.0+ addition (n_sweeps_trimmed,
        # bad_ending_at_sweep, ap_amplitude_mv, rac_variability_pct,
        # rs_compensation_pct, held_vm_mv, …), which is why the viewer's
        # trim banner and chips were never firing — the data was there,
        # just not propagated. Exclude only the routing/internal fields we
        # don't want duplicated.
        _NON_METRIC_COLUMNS = {
            "cell_id", "dataset", "nwb_path", "nwb_sha256",
            "pipeline_version", "computed_verdict", "triggered_metrics",
        }
        row_payload = {k: v for k, v in m.items() if k not in _NON_METRIC_COLUMNS}
        rows.append({
            "cell_id": r.cell_id, "dataset": r.dataset,
            "nwb_path": r.nwb_path, "nwb_sha256": r.nwb_sha256,
            "computed_verdict": verdict, "triggered_metrics": triggered,
            **row_payload,
        })
    verdicts = pd.DataFrame(rows)
    rule_counts = verdicts["computed_verdict"].value_counts().to_dict() if not verdicts.empty else {}
    _stage_end("thresholds", t0, {
        "n_pass": int(rule_counts.get("pass", 0)),
        "n_flag": int(rule_counts.get("flag", 0)),
        "n_fail": int(rule_counts.get("fail", 0)),
        "n_total": int(verdicts.shape[0]),
    })

    # ─── Stage 3.5: thumbnails for non-pass cells ────────────────
    total_thumbs = int((verdicts["computed_verdict"] != "pass").sum())
    t0 = _stage_start("thumbnails", total=total_thumbs)
    thumbs: dict[str, list[Path]] = {}
    cfg.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    seen_for_sha: dict[str, list[Path]] = {}
    n_generated = 0
    n_skipped = 0
    n_no_voltage = 0
    n_render_error = 0
    done_thumbs = 0
    for r in verdicts.itertuples(index=False):
        if r.computed_verdict == "pass":
            continue
        done_thumbs += 1
        sha8 = r.nwb_sha256[:8]
        if r.nwb_sha256 in seen_for_sha:
            thumbs[r.cell_id] = seen_for_sha[r.nwb_sha256]
        else:
            triggered = [t for t in (r.triggered_metrics or []) if isinstance(t, dict)]
            # Look up this cell's full metric row in the cache so the picker can
            # use provenance fields (e.g. rs_mohm_provenance.first/last) to
            # render the specific sweeps that drove the failed metric values.
            metric_row = None
            metric_match = cache_df[cache_df["nwb_sha256"] == r.nwb_sha256]
            if not metric_match.empty:
                metric_row = metric_match.iloc[0].to_dict()
            out = cfg.thumbnails_dir / f"{sha8}__{Path(r.nwb_path).stem}.png"
            if out.exists():
                n_skipped += 1
            else:
                _, status = _make_thumbnail(Path(r.nwb_path), out,
                                             families=cfg.stimulus_protocols,
                                             triggered_metrics=triggered,
                                             metric_row=metric_row)
                if status == "rendered":
                    n_generated += 1
                elif status == "no_voltage_sweeps":
                    n_no_voltage += 1
                elif status == "render_error":
                    n_render_error += 1
            if out.exists():
                seen_for_sha[r.nwb_sha256] = [out]
                thumbs[r.cell_id] = [out]
        if progress_callback:
            progress_callback("thumbnails", done_thumbs, total_thumbs)
    _stage_end("thumbnails", t0, {
        "n_generated": int(n_generated),
        "n_skipped_existing": int(n_skipped),
        "n_no_voltage_sweeps": int(n_no_voltage),
        "n_render_errors": int(n_render_error),
        "n_total": int(total_thumbs),
    })

    # ─── Stage 3.7: optional vision judge ────────────────────────
    vision_stats: dict[str, Any] = {"enabled": False}
    t0 = _stage_start("vision")
    if cfg.vision_judge and cfg.vision_judge.enabled:
        # CLI override for soft cap
        if max_cost_usd is not None:
            cfg.vision_judge.max_cost_usd = float(max_cost_usd)
        metrics_by_sha = {r.nwb_sha256: cache_df[cache_df["nwb_sha256"] == r.nwb_sha256].iloc[0].to_dict()
                          for r in verdicts.itertuples(index=False)
                          if r.nwb_sha256 in set(cache_df["nwb_sha256"])}
        vverdicts, vision_stats = _vision.run_vision_pass(
            verdicts_df=verdicts,
            metrics_by_sha=metrics_by_sha,
            thumbnails=thumbs,
            cfg=cfg.vision_judge,
            cached_responses=None,
        )
        if vverdicts:
            verdicts = _vision.apply_vision_verdicts(verdicts, vverdicts)
    _stage_end("vision", t0, {**vision_stats, "n_total": int(vision_stats.get("n_called", 0))})

    # ─── Stage 4: apply human overrides ──────────────────────────
    t0 = _stage_start("overrides")
    overrides = load_overrides(cfg.overrides_path)
    final = apply_overrides(verdicts, overrides)
    _stage_end("overrides", t0, {"n_overrides_loaded": int(len(overrides)) if overrides is not None else 0})

    # ─── Stage 5: render report ──────────────────────────────────
    t0 = _stage_start("report")
    cohort_stats_path = cfg.output_dir / "cohort_stats.json"
    cohort_stats: dict | None = None
    if cohort_stats_path.exists():
        try:
            cohort_stats = json.loads(cohort_stats_path.read_text())
        except (json.JSONDecodeError, OSError):
            cohort_stats = None
    write_report(final, thumbs,
                 html_path=cfg.report_html,
                 csv_path=cfg.report_csv,
                 project_name=cfg.project_name,
                 pipeline_version=PIPELINE_VERSION,
                 thresholds_fp=str(cfg.thresholds_file),
                 viewer_url=cfg.viewer_url,
                 thresholds=thresholds,
                 cohort_stats=cohort_stats,
                 config_path=str(cfg.config_path) if cfg.config_path else None)
    viewer_src = Path(__file__).parent / "templates" / "viewer.html"
    if viewer_src.exists():
        viewer_dst = cfg.report_html.parent / "qc_viewer.html"
        viewer_dst.write_text(viewer_src.read_text())
    _stage_end("report", t0, {})

    # ─── Write run_report.json ───────────────────────────────────
    _write_run_report(cfg, started_at, t_run_start, stages, compute_errors,
                      manifest_stats, vision_stats=vision_stats,
                      n_workers=cfg.n_workers, final_df=final)

    return {
        "n_cells": int(len(final)),
        "n_unique_nwbs": int(uniq.shape[0]),
        "n_computed_this_run": n_compute,
        "n_nwb_opens": n_compute,
        "n_pass": int((final["final_verdict"] == "pass").sum()),
        "n_flag": int((final["final_verdict"] == "flag").sum()),
        "n_fail": int((final["final_verdict"] == "fail").sum()),
        "elapsed_s": round(time.time() - t_run_start, 2),
        "report": str(cfg.report_html),
        "viewer": str(cfg.report_html.parent / "qc_viewer.html"),
        "run_report": str(cfg.output_dir / "run_report.json"),
        "vision": vision_stats,
        "manifest_stats": manifest_stats,
    }


def _write_run_report(
    cfg: ProjectConfig,
    started_at: str,
    t_run_start: float,
    stages: dict[str, dict[str, Any]],
    compute_errors: list[dict[str, Any]],
    manifest_stats: list[dict[str, Any]],
    *,
    vision_stats: dict[str, Any],
    n_workers: int,
    final_df: pd.DataFrame | None = None,
) -> Path:
    elapsed_s = round(time.time() - t_run_start, 2)
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    verdicts_counts: dict[str, int] = {"pass": 0, "flag": 0, "fail": 0, "total": 0}
    if final_df is not None and "final_verdict" in final_df.columns:
        for v in ("pass", "flag", "fail"):
            verdicts_counts[v] = int((final_df["final_verdict"] == v).sum())
        verdicts_counts["total"] = int(len(final_df))
    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": elapsed_s,
        "pipeline_version": PIPELINE_VERSION,
        "package_version": __version__,
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "project_name": cfg.project_name,
        "stages": stages,
        "verdicts": verdicts_counts,
        "memory": {"peak_rss_mb": round(_peak_rss_mb(), 1), "n_workers": int(n_workers)},
        "system": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "nwb_trace_qc": __version__,
        },
        "manifest_stats": manifest_stats,
        "compute_errors": compute_errors,
    }
    out_path = cfg.output_dir / "run_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote run report: %s", out_path)
    return out_path
