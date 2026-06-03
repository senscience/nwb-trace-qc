"""End-to-end pipeline orchestrator: discover → cache → compute → threshold → override → report."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import PIPELINE_VERSION
from .cache import append_rows, cached_hashes, filter_for_version, load_cache
from .config import ProjectConfig, default_families
from .manifest import build_manifest, load_acquisition_index, save_manifest, unique_nwbs
from .metrics import compute_metrics
from .overrides import apply_overrides, init_overrides_file, load_overrides
from .report import write_report
from .stimuli import StimulusFamilyMap
from .thresholds import evaluate, load_thresholds
from . import vision as _vision


def _compute_one(args):
    nwb_path, nwb_sha256, families = args
    fm = StimulusFamilyMap(families)
    metrics = compute_metrics(nwb_path, fm)
    metrics.update({
        "nwb_sha256": nwb_sha256,
        "nwb_path": str(nwb_path),
        "pipeline_version": PIPELINE_VERSION,
    })
    return metrics


def _make_thumbnail(nwb_path: Path, out_path: Path, *, families: dict[str, list[str]], reasons: list[str]) -> Path | None:
    """Render up to 3 representative sweeps (one per offending family if possible)."""
    import pynwb
    try:
        with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
            f = io.read()
            picks = []
            wanted_families = set()
            # Choose sweeps relevant to the failing reason
            for r in reasons:
                if r in ("vrest_mv", "vrest_drift_mv"):
                    wanted_families.add("spontaneous_hold")
                if r in ("rs_mohm_final", "rs_drift_pct", "rs_mohm_initial"):
                    wanted_families.add("test_pulse")
                if r in ("ap_amp_overshoot_mv", "ap_threshold_drift_mv"):
                    wanted_families.add("ap_waveform")
                if r == "baseline_rms_mv":
                    wanted_families.add("spontaneous_hold")
            if not wanted_families:
                wanted_families = {"spontaneous_hold", "ap_waveform"}
            fm = StimulusFamilyMap(families)
            for name, obj in f.acquisition.items():
                unit = (getattr(obj, "unit", "") or "").lower()
                if unit not in {"volts", "v", ""}: continue
                stim = name.split("__")[1] if "__" in name and name.count("__") >= 2 else name
                fam = fm.family_of(stim)
                if fam in wanted_families:
                    picks.append((fam, name, obj))
                    if len(picks) >= 3: break
            if not picks: return None
            fig, axes = plt.subplots(len(picks), 1, figsize=(6, 1.8 * len(picks)), sharex=False)
            if len(picks) == 1: axes = [axes]
            for ax, (fam, name, obj) in zip(axes, picks):
                data = np.asarray(obj.data[:]).reshape(-1)
                rate = float(getattr(obj, "rate", 0) or 0)
                t = np.arange(len(data)) / rate if rate > 0 else np.arange(len(data))
                ax.plot(t, data * 1000.0, lw=0.6, color="#222")
                ax.set_title(f"{fam}: {name[:50]}", fontsize=8)
                ax.set_ylabel("mV", fontsize=7)
                ax.tick_params(labelsize=6)
            axes[-1].set_xlabel("time (s)", fontsize=7)
            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=80)
            plt.close(fig)
            return out_path
    except Exception:
        return None


def run(cfg: ProjectConfig, *, filter_dataset: str | None = None, report_only: bool = False) -> dict:
    """Execute the full pipeline.

    Returns a summary dict with timing + counts. Side effects:
    writes manifest, cache, report HTML+CSV, thumbnails.
    """
    t0 = time.time()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    init_overrides_file(cfg.overrides_path)

    # Stage 1: manifest
    manifest = build_manifest(cfg)
    if filter_dataset:
        manifest = manifest[manifest["dataset"] == filter_dataset].reset_index(drop=True)
    save_manifest(manifest, cfg.manifest_path)
    if manifest.empty:
        return {"n_cells": 0, "elapsed_s": time.time() - t0, "note": "no NWBs found"}

    # Stage 2: compute metrics for unique NWBs not in cache
    cache_df = load_cache(cfg.cache_path)
    have = cached_hashes(cache_df)
    uniq = unique_nwbs(manifest)
    todo = uniq[~uniq["nwb_sha256"].isin(have)]
    n_compute = 0
    n_opens = 0
    if not report_only and not todo.empty:
        args = [(Path(row.nwb_path), row.nwb_sha256, cfg.stimulus_protocols)
                for row in todo.itertuples(index=False)]
        if cfg.n_workers and cfg.n_workers > 1:
            with mp.Pool(cfg.n_workers) as pool:
                computed = pool.map(_compute_one, args)
        else:
            computed = [_compute_one(a) for a in args]
        append_rows(cfg.cache_path, computed)
        n_compute = len(computed)
        n_opens = len(computed)

    # Stage 3: apply thresholds → per-cell verdicts
    cache_df = filter_for_version(load_cache(cfg.cache_path))
    if cfg.thresholds_file is None or not cfg.thresholds_file.exists():
        raise FileNotFoundError(f"thresholds_file not found: {cfg.thresholds_file}")
    thresholds = load_thresholds(cfg.thresholds_file)
    rows = []
    for r in manifest.itertuples(index=False):
        metric_row = cache_df[cache_df["nwb_sha256"] == r.nwb_sha256]
        if metric_row.empty:
            rows.append({
                "cell_id": r.cell_id, "dataset": r.dataset,
                "nwb_path": r.nwb_path, "nwb_sha256": r.nwb_sha256,
                "computed_verdict": "flag", "triggered_metrics": [{"metric": "_no_cache", "value": None, "verdict": "flag", "reason": "no metrics computed"}],
            })
            continue
        m = metric_row.iloc[0].to_dict()
        verdict, triggered = evaluate(m, thresholds)
        rows.append({
            "cell_id": r.cell_id, "dataset": r.dataset,
            "nwb_path": r.nwb_path, "nwb_sha256": r.nwb_sha256,
            "computed_verdict": verdict, "triggered_metrics": triggered,
            **{k: m.get(k) for k in [
                "vrest_mv","vrest_drift_mv","rs_mohm_initial","rs_mohm_final","rs_drift_pct",
                "rin_mohm","ap_amp_overshoot_mv","ap_threshold_drift_mv","baseline_rms_mv",
                "n_sweeps_total","n_sweeps_clipped","n_sweeps_nan","qc_protocol_coverage",
                "rac_decay_residual_rel","vm_drift_within_sweep_mv_per_s",
                "ap_failure_fraction","ap_amp_cv","late_instability_index",
                "compute_error",
            ]},
        })
    verdicts = pd.DataFrame(rows)

    # Stage 3.5: thumbnails for non-pass cells (cached per-sha256 to avoid rework).
    # We render now so the optional vision pass can reuse them.
    thumbs: dict[str, list[Path]] = {}
    cfg.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    seen_for_sha: dict[str, list[Path]] = {}
    for r in verdicts.itertuples(index=False):
        if r.computed_verdict == "pass":
            continue
        sha8 = r.nwb_sha256[:8]
        if r.nwb_sha256 in seen_for_sha:
            thumbs[r.cell_id] = seen_for_sha[r.nwb_sha256]
            continue
        reasons = [t["metric"] for t in (r.triggered_metrics or []) if isinstance(t, dict)]
        out = cfg.thumbnails_dir / f"{sha8}__{Path(r.nwb_path).stem}.png"
        if not out.exists():
            _make_thumbnail(Path(r.nwb_path), out, families=cfg.stimulus_protocols, reasons=reasons)
        if out.exists():
            seen_for_sha[r.nwb_sha256] = [out]
            thumbs[r.cell_id] = [out]

    # Stage 3.7: optional vision judge for borderline (flag) cells.
    vision_stats: dict = {"enabled": False}
    if cfg.vision_judge and cfg.vision_judge.enabled:
        metrics_by_sha = {r.nwb_sha256: cache_df[cache_df["nwb_sha256"] == r.nwb_sha256].iloc[0].to_dict()
                          for r in verdicts.itertuples(index=False)
                          if r.nwb_sha256 in set(cache_df["nwb_sha256"])}
        cached_responses = None  # not yet wired into the cache parquet; the vision call cache lives inside vision.py per-process for now
        vverdicts, vision_stats = _vision.run_vision_pass(
            verdicts_df=verdicts,
            metrics_by_sha=metrics_by_sha,
            thumbnails=thumbs,
            cfg=cfg.vision_judge,
            cached_responses=cached_responses,
        )
        if vverdicts:
            verdicts = _vision.apply_vision_verdicts(verdicts, vverdicts)

    # Stage 4: apply human overrides (always last; human wins)
    overrides = load_overrides(cfg.overrides_path)
    final = apply_overrides(verdicts, overrides)

    # Stage 6: render report (static HTML + CSV)
    write_report(final, thumbs,
                 html_path=cfg.report_html,
                 csv_path=cfg.report_csv,
                 project_name=cfg.project_name,
                 pipeline_version=PIPELINE_VERSION,
                 thresholds_fp=str(cfg.thresholds_file))

    # Stage 7: write the viewer.html template into the output dir so `nwb-qc serve` works
    viewer_src = Path(__file__).parent / "templates" / "viewer.html"
    if viewer_src.exists():
        viewer_dst = cfg.report_html.parent / "qc_viewer.html"
        viewer_dst.write_text(viewer_src.read_text())

    return {
        "n_cells": int(len(final)),
        "n_unique_nwbs": int(uniq.shape[0]),
        "n_computed_this_run": n_compute,
        "n_nwb_opens": n_opens,
        "n_pass": int((final["final_verdict"] == "pass").sum()),
        "n_flag": int((final["final_verdict"] == "flag").sum()),
        "n_fail": int((final["final_verdict"] == "fail").sum()),
        "elapsed_s": round(time.time() - t0, 2),
        "report": str(cfg.report_html),
        "viewer": str(cfg.report_html.parent / "qc_viewer.html"),
        "vision": vision_stats,
        "manifest_stats": list(manifest.attrs.get("manifest_stats", [])),
    }
