"""Self-contained HTML report + CSV emission.

Single file, no external resources, no JS deps. Sort/filter via vanilla JS.
"""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .families import (
    METRIC_DESCRIPTIONS,
    METRIC_TO_FAMILY,
    PSEUDO_METRIC_LABELS,
    implicated_families,
)


HEALTH_CARD_METRICS: list[tuple[str, str]] = [
    ("vrest_mv",                       "Vrest"),
    ("rs_mohm_final",                  "Rs (final)"),
    ("rin_mohm",                       "Rin"),
    ("ap_amp_overshoot_mv",            "AP overshoot"),
    ("holding_current_pa",             "Holding I"),
    ("vrest_session_drift_mv",         "Session drift"),
]


def _format_health_value(metric: str, value) -> str:
    """Compact display for the health card (units inline)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if metric.endswith("_mv"):           return f"{float(value):.1f} mV"
    if metric.endswith("_mohm") or metric.endswith("_mohm_final") or metric.endswith("_mohm_initial"):
        return f"{float(value):.0f} MΩ"
    if metric.endswith("_pa"):           return f"{float(value):.0f} pA"
    if metric.endswith("_pct"):          return f"{float(value):.0f}%"
    if isinstance(value, float):         return f"{value:.3g}"
    return html.escape(str(value))


def _evaluate_card_metric(metric: str, value, thresholds) -> str:
    """Run the same rule logic against one metric → return verdict ('pass'/'flag'/'fail'/'na')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "na"
    if not thresholds or metric not in thresholds:
        return "na"
    from .thresholds import evaluate_metric
    verdict, _ = evaluate_metric(value, thresholds[metric])
    return verdict


def _render_health_card(row, thresholds) -> str:
    """A 2×3 mini-heatmap of canonical whole-cell QC signals at a glance, plus
    an optional yellow strip when the recording was trimmed due to a bad ending."""
    cells = []
    for metric, label in HEALTH_CARD_METRICS:
        if metric not in row.index:
            cells.append(f"<div class='hc-cell hc-na'><div class='hc-lbl'>{label}</div>"
                         f"<div class='hc-val'>—</div></div>")
            continue
        val = row[metric]
        verdict = _evaluate_card_metric(metric, val, thresholds)
        cls = {"pass": "hc-pass", "flag": "hc-flag", "fail": "hc-fail"}.get(verdict, "hc-na")
        cells.append(
            f"<div class='hc-cell {cls}'>"
            f"<div class='hc-lbl'>{html.escape(label)}</div>"
            f"<div class='hc-val'>{_format_health_value(metric, val)}</div>"
            f"<div class='hc-vd'>{verdict.upper()}</div>"
            f"</div>"
        )
    health = f"<div class='health-card'>{''.join(cells)}</div>"

    # Recording-trim strip (v0.4.0): when bad-ending detection fired, surface
    # the cutoff sweep + reason above the health card so the triager knows the
    # metric scalars exclude the degraded tail.
    n_trimmed = row.get("n_sweeps_trimmed")
    cutoff = row.get("bad_ending_at_sweep")
    reason = row.get("bad_ending_reason")
    if n_trimmed is not None and not _is_nan_scalar(n_trimmed) and int(n_trimmed) > 0:
        n_total = row.get("n_sweeps_total")
        cutoff_str = ""
        if cutoff is not None and not _is_nan_scalar(cutoff):
            cutoff_str = f" at sweep {int(cutoff)}"
            if n_total is not None and not _is_nan_scalar(n_total):
                cutoff_str += f"/{int(n_total)}"
        reason_str = f" ({html.escape(str(reason))})" if reason and str(reason) != "nan" else ""
        strip = (
            f"<div class='trim-strip'>"
            f"<b>Recording trimmed{cutoff_str}{reason_str}.</b> "
            f"{int(n_trimmed)} tail sweep(s) excluded from the metric scalars. "
            f"The metrics shown reflect the pre-degradation period only."
            f"</div>"
        )
        return strip + health
    return health


def _is_nan_scalar(v) -> bool:
    """Tiny helper to skip the NaN check guards (pd.isna chokes on some types)."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _parse_provenance(raw) -> dict | None:
    """Some provenance fields arrive as JSON strings from the cache parquet."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _explain_trigger(trigger: dict, row, cohort_stats: dict | None) -> str:
    """Build a plain-English explanation for a triggered metric, drawing on
    provenance (which sweeps drove it) and cohort percentiles when available.
    """
    metric = trigger.get("metric", "?")
    value = trigger.get("value")
    reason = (trigger.get("reason") or "").strip()
    desc = METRIC_DESCRIPTIONS.get(metric, {})
    parts: list[str] = []
    if desc.get("what"):
        parts.append(desc["what"])

    # Provenance — most metrics now carry a *_provenance JSON column
    prov = _parse_provenance(row.get(f"{metric}_provenance"))
    if prov:
        first, last, n = prov.get("first"), prov.get("last"), prov.get("n")
        if first and last and first != last:
            parts.append(f"Computed from {n} sweep{'s' if (n or 0) != 1 else ''}, "
                          f"first <code>{html.escape(first)}</code> → last <code>{html.escape(last)}</code>.")
        elif first:
            parts.append(f"Computed from sweep <code>{html.escape(first)}</code>.")
    # Hand-tuned bridges for the metrics that don't carry provenance
    elif metric == "rs_drift_pct":
        init = row.get("rs_mohm_initial"); fin = row.get("rs_mohm_final")
        rs_prov = _parse_provenance(row.get("rs_mohm_provenance"))
        if init is not None and fin is not None and not (pd.isna(init) or pd.isna(fin)):
            sweeps_str = ""
            if rs_prov and rs_prov.get("first") and rs_prov.get("last"):
                sweeps_str = (f" (<code>{html.escape(rs_prov['first'])}</code> → "
                              f"<code>{html.escape(rs_prov['last'])}</code>)")
            parts.append(f"Rs drifted from {float(init):.1f} MΩ to {float(fin):.1f} MΩ{sweeps_str}.")
    elif metric == "qc_protocol_coverage" and value is False:
        parts.append("NWB is missing one of the essential families "
                     "(spontaneous_hold, test_pulse, ap_waveform). Most often this means "
                     "your <code>stimulus_protocols:</code> mapping doesn't include "
                     "this lab's protocol names — check the YAML's UNMAPPED block.")

    # Cohort comparison
    if cohort_stats and metric in cohort_stats and value is not None and not (
            isinstance(value, float) and pd.isna(value)):
        ps = cohort_stats[metric]
        if all(k in ps for k in ("p10", "p50", "p90")):
            try:
                v = float(value)
                pct = _approx_percentile(v, ps)
                parts.append(f"Cohort percentile ≈ P{pct}; cohort range "
                             f"P10 {ps['p10']:.2g} · P50 {ps['p50']:.2g} · P90 {ps['p90']:.2g}.")
            except (TypeError, ValueError):
                pass

    if desc.get("healthy"):
        parts.append(f"Healthy: {html.escape(desc['healthy'])}")

    rule_text = f"Triggered rule: <code>{html.escape(reason)}</code>." if reason else ""
    if rule_text:
        parts.append(rule_text)

    return " ".join(parts) or "No additional context available."


def _approx_percentile(v: float, stats: dict) -> int:
    """Bucket a value into the nearest cohort percentile from p10/p50/p90/p99."""
    breaks = [(10, stats.get("p10")), (50, stats.get("p50")),
              (90, stats.get("p90")), (99, stats.get("p99"))]
    breaks = [(p, b) for p, b in breaks if b is not None]
    breaks.sort(key=lambda x: x[1])
    if not breaks:
        return -1
    if v <= breaks[0][1]:
        return breaks[0][0]
    if v >= breaks[-1][1]:
        return breaks[-1][0]
    for (p1, b1), (p2, b2) in zip(breaks, breaks[1:]):
        if b1 <= v <= b2:
            # Linear interp
            frac = (v - b1) / (b2 - b1) if b2 != b1 else 0
            return int(round(p1 + frac * (p2 - p1)))
    return -1


VERDICT_COLORS = {
    "pass": "#d4edda",
    "flag": "#fff3cd",
    "fail": "#f8d7da",
}
VERDICT_TEXT = {
    "pass": "#155724",
    "flag": "#856404",
    "fail": "#721c24",
}


def _png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _format_value(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return html.escape(str(v))


def render_html(report_df: pd.DataFrame, thumbnails: dict[str, list[Path]], *,
                project_name: str, pipeline_version: str, thresholds_fp: str = "",
                metric_columns: list[str] | None = None,
                viewer_url: str = "http://127.0.0.1:8765",
                thresholds: dict[str, dict] | None = None,
                cohort_stats: dict[str, dict] | None = None) -> str:
    """Render the full self-contained HTML report.

    `report_df` must have columns: cell_id, dataset, computed_verdict, final_verdict,
    triggered_metrics (JSON list-of-dicts as string or list), <metric columns>...,
    optional override_note/override_reviewer/override_date.
    `thumbnails` is a dict {cell_id: [paths-to-PNGs]}.
    `viewer_url` is the base URL where `nwb-qc serve` is reachable — used for the
    per-cell "Inspect all sweeps →" deep link.
    """
    metric_columns = metric_columns or [
        "vrest_mv", "rs_mohm_final", "rs_drift_pct", "ap_amp_overshoot_mv",
        "rac_decay_residual_rel", "vm_drift_within_sweep_mv_per_s",
        "ap_failure_fraction", "ap_amp_cv", "late_instability_index",
        "baseline_rms_mv", "n_sweeps_total", "n_sweeps_clipped", "n_sweeps_nan",
        "qc_protocol_coverage",
    ]
    metric_columns = [c for c in metric_columns if c in report_df.columns]

    n_total = len(report_df)
    counts = report_df["final_verdict"].value_counts().to_dict()
    n_pass = int(counts.get("pass", 0))
    n_flag = int(counts.get("flag", 0))
    n_fail = int(counts.get("fail", 0))
    by_dataset = (
        report_df.groupby(["dataset", "final_verdict"]).size().unstack(fill_value=0)
        .reindex(columns=["pass", "flag", "fail"], fill_value=0)
    )

    def _parse_triggers(raw):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        return raw if isinstance(raw, list) else (json.loads(raw) if raw else [])

    def _trigger_chips(items, row):
        """Render colored chips for each triggered metric, tagged with its implicated
        stimulus family so triagers know which sweeps to inspect. Each chip is
        clickable: the plain-English explanation under it expands on click.
        """
        chips = []
        for idx, it in enumerate(items):
            cls = it.get("verdict", "pass")
            metric = it.get("metric", "?")
            label = f"{metric} {it.get('reason', '') or ''}".strip()
            fam = METRIC_TO_FAMILY.get(metric) or PSEUDO_METRIC_LABELS.get(metric)
            fam_tag = (f"<span class='fam-tag'>· {html.escape(fam)}</span>"
                       if fam else "")
            explanation = _explain_trigger(it, row, cohort_stats)
            tip = html.escape((METRIC_DESCRIPTIONS.get(metric) or {}).get("what", ""))
            chips.append(
                f"<details class='chip-details'><summary class='chip chip-{cls}' title='{tip}'>"
                f"{html.escape(label)}{fam_tag}</summary>"
                f"<div class='chip-explain'>{explanation}</div>"
                f"</details>"
            )
        return "".join(chips)

    rows_html = []
    datasets_seen = sorted(report_df["dataset"].dropna().astype(str).unique())
    for i, row in report_df.iterrows():
        cell_id = str(row["cell_id"])
        dataset = str(row.get("dataset", ""))
        verdict = str(row.get("final_verdict", "pass"))
        computed = str(row.get("computed_verdict", verdict))
        triggers_list = _parse_triggers(row.get("triggered_metrics"))
        triggered = _trigger_chips(triggers_list, row)
        health_card = _render_health_card(row, thresholds)
        override_note = str(row.get("override_note", "") or "")
        # Build metric cells
        metric_cells = "".join(
            f"<td>{_format_value(row.get(c))}</td>" for c in metric_columns
        )
        # Build expansion panel (full metric values + thumbnails)
        full_metrics = "".join(
            f"<tr><th>{html.escape(c)}</th><td>{_format_value(row.get(c))}</td></tr>"
            for c in report_df.columns
            if c not in {"cell_id", "dataset", "final_verdict", "computed_verdict",
                          "triggered_metrics", "override_note", "override_reviewer",
                          "override_date", "nwb_sha256", "nwb_path"}
        )
        thumbs = thumbnails.get(cell_id, [])
        thumb_html = "".join(
            f"<img src='data:image/png;base64,{_png_b64(t)}' alt='{html.escape(t.name)}' />"
            for t in thumbs
        )
        # "Inspect these families" hint — derived from triggered metrics, so
        # the triager knows which sweep types to look at instead of scrolling
        # through every voltage acquisition in the NWB.
        impl = sorted(implicated_families(triggers_list))
        if impl:
            impl_chips = "".join(f"<span class='fam-chip'>{html.escape(f)}</span>" for f in impl)
            implicated_block = (
                f"<div class='implicated-line'><b>Inspect these families:</b> {impl_chips}"
                f"<a class='viewer-link' href='{html.escape(viewer_url)}/?cell={html.escape(cell_id)}' "
                f"target='_blank' rel='noopener' "
                f"title='Requires `nwb-qc serve` to be running'>Inspect all sweeps in viewer →</a></div>"
            )
        else:
            implicated_block = (
                f"<div class='implicated-line'>"
                f"<a class='viewer-link' href='{html.escape(viewer_url)}/?cell={html.escape(cell_id)}' "
                f"target='_blank' rel='noopener' "
                f"title='Requires `nwb-qc serve` to be running'>Inspect all sweeps in viewer →</a></div>"
            )
        override_block = ""
        if override_note or row.get("override_reviewer") or computed != verdict:
            override_block = (
                f"<div class='override'>"
                f"<b>Override active</b> — computed verdict was "
                f"<span class='chip chip-{computed}'>{computed}</span>; "
                f"reviewer: {html.escape(str(row.get('override_reviewer','')))}; "
                f"note: {html.escape(override_note)}</div>"
            )
        vision_block = ""
        vv = row.get("vision_verdict")
        if vv is not None and not (isinstance(vv, float) and pd.isna(vv)) and str(vv) != "":
            try:
                conf = float(row.get("vision_confidence") or 0)
            except (TypeError, ValueError):
                conf = 0.0
            vision_block = (
                f"<div class='override' style='background:#e7f1f9; border-left-color:#5a9ad6;'>"
                f"<b>Vision judge</b> — <span class='chip chip-{vv}'>{vv}</span> "
                f"(confidence {conf:.2f}); "
                f"reason: {html.escape(str(row.get('vision_reason','')))}; "
                f"notes: {html.escape(str(row.get('vision_notes','')))}</div>"
            )
        row_metrics_tags = " ".join(sorted({str(t.get("metric", "")) for t in triggers_list
                                              if t.get("verdict") in {"flag", "fail"}}))
        rows_html.append(f"""
<tr class='row row-{verdict}' data-verdict='{verdict}' data-dataset='{html.escape(dataset)}' data-cellid='{html.escape(cell_id)}' data-trigmetrics='{html.escape(row_metrics_tags)}'>
  <td class='expander' onclick='toggle(this)'><span class='caret'>▸</span></td>
  <td class='cellid'>{html.escape(cell_id)}</td>
  <td>{html.escape(dataset)}</td>
  <td><span class='verdict v-{verdict}'>{verdict.upper()}</span></td>
  <td class='triggered'>{triggered}</td>
  {metric_cells}
</tr>
<tr class='detail' data-cellid='{html.escape(cell_id)}'>
  <td colspan='{5 + len(metric_columns)}'>
    {override_block}
    {vision_block}
    {health_card}
    {implicated_block}
    <div class='detail-grid'>
      <div class='detail-metrics'><table class='kv'>{full_metrics}</table></div>
      <div class='detail-thumbs'>{thumb_html}</div>
    </div>
    <div class='override-template'>
      To override, append to <code>qc_overrides.csv</code>:
      <pre>{html.escape(cell_id)},pass,&lt;your reason&gt;,&lt;your name&gt;,{datetime.now(timezone.utc).date()}</pre>
    </div>
  </td>
</tr>""")

    dataset_buttons = "".join(
        f"<button data-filter-dataset='{html.escape(d)}'>{html.escape(d)}</button>"
        for d in datasets_seen
    )
    metric_headers = "".join(f"<th>{c}</th>" for c in metric_columns)
    per_ds_rows = "".join(
        f"<tr><td>{html.escape(d)}</td>"
        f"<td>{int(by_dataset.loc[d, 'pass']) if d in by_dataset.index else 0}</td>"
        f"<td>{int(by_dataset.loc[d, 'flag']) if d in by_dataset.index else 0}</td>"
        f"<td>{int(by_dataset.loc[d, 'fail']) if d in by_dataset.index else 0}</td></tr>"
        for d in datasets_seen
    )

    # Failure-recipe: count how often each metric triggered across the cohort.
    from collections import Counter as _Counter
    failure_counter: _Counter = _Counter()
    for _, row in report_df.iterrows():
        for it in _parse_triggers(row.get("triggered_metrics")):
            if it.get("verdict") in {"flag", "fail"}:
                failure_counter[it.get("metric", "?")] += 1
    failure_recipe_html = ""
    if failure_counter:
        items_html = "".join(
            f"<span class='recipe-item' data-recipe-metric='{html.escape(m)}'>"
            f"{html.escape(m)}<span class='count'>{n}</span></span>"
            for m, n in failure_counter.most_common()
        )
        failure_recipe_html = (
            f"<div class='failure-recipe'><h3>Failures by metric "
            f"(click to filter to cells affected)</h3>{items_html}"
            f"<span class='recipe-item' data-recipe-metric='__clear__' "
            f"style='background:#e7f1f9'>clear filter</span></div>"
        )

    return f"""<!doctype html><html><head><meta charset='utf-8'/>
<title>{html.escape(project_name)} — QC report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 1rem 2rem; color: #222; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1rem; }}
  .summary {{ display: flex; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap; }}
  .summary > div {{ background: #f4f6f8; padding: 0.75rem 1rem; border-radius: 6px; }}
  .summary table {{ border-collapse: collapse; }}
  .summary th, .summary td {{ padding: 0.15rem 0.5rem; text-align: right; }}
  .summary th {{ background: transparent; }}
  .controls {{ display: flex; gap: 1rem; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; }}
  .controls button {{ padding: 0.3rem 0.7rem; border: 1px solid #ccc; background: #fff; cursor: pointer; border-radius: 4px; }}
  .controls button.active {{ background: #333; color: #fff; }}
  .controls input {{ padding: 0.3rem 0.5rem; border: 1px solid #ccc; border-radius: 4px; min-width: 200px; }}
  table.cells {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  table.cells th, table.cells td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }}
  table.cells th {{ background: #fafafa; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  .verdict {{ padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.78rem; font-weight: 600; }}
  .v-pass {{ background: {VERDICT_COLORS['pass']}; color: {VERDICT_TEXT['pass']}; }}
  .v-flag {{ background: {VERDICT_COLORS['flag']}; color: {VERDICT_TEXT['flag']}; }}
  .v-fail {{ background: {VERDICT_COLORS['fail']}; color: {VERDICT_TEXT['fail']}; }}
  .chip {{ display: inline-block; padding: 0.1rem 0.4rem; margin: 0.05rem; border-radius: 3px; font-size: 0.72rem; cursor: pointer; }}
  .chip-flag {{ background: {VERDICT_COLORS['flag']}; color: {VERDICT_TEXT['flag']}; }}
  .chip-fail {{ background: {VERDICT_COLORS['fail']}; color: {VERDICT_TEXT['fail']}; }}
  .chip-pass {{ background: {VERDICT_COLORS['pass']}; color: {VERDICT_TEXT['pass']}; }}
  .chip .fam-tag {{ margin-left: 0.3rem; opacity: 0.75; font-style: italic; font-size: 0.68rem; }}
  details.chip-details {{ display: inline-block; }}
  details.chip-details > summary {{ list-style: none; }}
  details.chip-details > summary::-webkit-details-marker {{ display: none; }}
  details.chip-details[open] .chip-explain {{ display: block; }}
  .chip-explain {{ display: none; background: #fbfbfd; border-left: 3px solid #bbb; padding: 0.4rem 0.6rem; margin: 0.25rem 0 0.5rem 0; font-size: 0.78rem; line-height: 1.45; color: #333; max-width: 720px; }}
  .chip-explain code {{ background: #efefef; padding: 0 0.2rem; border-radius: 2px; font-size: 0.92em; }}
  /* Health-summary card */
  .health-card {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.35rem; margin: 0.5rem; max-width: 720px; }}
  .hc-cell {{ background: #f7f8fa; border: 1px solid #e1e4e8; border-radius: 4px; padding: 0.4rem 0.55rem; }}
  .hc-cell .hc-lbl {{ font-size: 0.7rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; }}
  .hc-cell .hc-val {{ font-size: 1rem; font-weight: 600; margin-top: 0.1rem; color: #222; }}
  .hc-cell .hc-vd {{ font-size: 0.65rem; font-weight: 600; margin-top: 0.15rem; letter-spacing: 0.05em; }}
  .hc-pass {{ background: {VERDICT_COLORS['pass']}; border-color: {VERDICT_TEXT['pass']}; }}
  .hc-pass .hc-vd {{ color: {VERDICT_TEXT['pass']}; }}
  .hc-flag {{ background: {VERDICT_COLORS['flag']}; border-color: {VERDICT_TEXT['flag']}; }}
  .hc-flag .hc-vd {{ color: {VERDICT_TEXT['flag']}; }}
  .hc-fail {{ background: {VERDICT_COLORS['fail']}; border-color: {VERDICT_TEXT['fail']}; }}
  .hc-fail .hc-vd {{ color: {VERDICT_TEXT['fail']}; }}
  .hc-na .hc-vd {{ color: #888; }}
  /* Recording-trim warning */
  .trim-strip {{ background: #fff3cd; color: {VERDICT_TEXT['flag']}; padding: 0.4rem 0.75rem; border-left: 3px solid #e0b54a; margin: 0.5rem; font-size: 0.85rem; }}
  /* Failure recipe sidebar */
  .failure-recipe {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 4px; padding: 0.5rem 0.75rem; margin: 0.5rem 0; font-size: 0.85rem; }}
  .failure-recipe h3 {{ margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #444; }}
  .failure-recipe .recipe-item {{ display: inline-block; margin: 0.15rem 0.3rem 0.15rem 0; padding: 0.2rem 0.5rem; background: #f4f6f8; border-radius: 3px; cursor: pointer; font-size: 0.78rem; }}
  .failure-recipe .recipe-item:hover {{ background: #e7ecf2; }}
  .failure-recipe .recipe-item .count {{ display: inline-block; background: #fff; border-radius: 2px; padding: 0 0.3rem; margin-left: 0.3rem; font-weight: 600; color: #c00; }}
  .implicated-line {{ background: #fff8db; padding: 0.4rem 0.75rem; border-left: 3px solid #e0b54a; margin: 0.5rem; font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }}
  .implicated-line .fam-chip {{ display: inline-block; background: #fff3cd; color: {VERDICT_TEXT['flag']}; padding: 0.1rem 0.5rem; border-radius: 3px; font-size: 0.75rem; font-weight: 600; }}
  .implicated-line .viewer-link {{ margin-left: auto; color: #1c4f8b; text-decoration: none; font-weight: 600; font-size: 0.82rem; }}
  .implicated-line .viewer-link:hover {{ text-decoration: underline; }}
  .expander {{ cursor: pointer; width: 1.2rem; }}
  .detail {{ display: none; background: #fcfcfd; }}
  .detail.open {{ display: table-row; }}
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 0.5rem; }}
  .detail-thumbs img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 0.5rem; }}
  table.kv {{ border-collapse: collapse; font-size: 0.8rem; }}
  table.kv th, table.kv td {{ padding: 0.2rem 0.6rem; border-bottom: 1px solid #f0f0f0; text-align: left; }}
  .override {{ background: #fff3cd; padding: 0.5rem 0.75rem; border-left: 3px solid #f0ad4e; margin: 0.5rem; }}
  .override-template {{ font-size: 0.8rem; color: #666; padding: 0.5rem; }}
  pre {{ background: #f4f6f8; padding: 0.4rem; border-radius: 3px; white-space: pre-wrap; }}
  .row.hidden, .detail.hidden {{ display: none !important; }}
</style>
</head><body>
<h1>{html.escape(project_name)} — QC report</h1>
<div class='meta'>
  Pipeline v{pipeline_version} · Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · Thresholds: <code>{html.escape(thresholds_fp)}</code>
  · <a href="qc_viewer.html" title="Available when served via `nwb-qc serve`">Open interactive viewer →</a>
</div>

<div class='summary'>
  <div>
    <b>Overall</b><br>
    {n_total} cells &nbsp; · &nbsp;
    <span class='verdict v-pass'>PASS {n_pass}</span>
    <span class='verdict v-flag'>FLAG {n_flag}</span>
    <span class='verdict v-fail'>FAIL {n_fail}</span>
  </div>
  <div>
    <b>By dataset</b>
    <table><tr><th>dataset</th><th>pass</th><th>flag</th><th>fail</th></tr>{per_ds_rows}</table>
  </div>
</div>

{failure_recipe_html}

<div class='controls'>
  <span>Verdict:</span>
  <button id='v-all'>All</button>
  <button id='v-attention' class='active'>Fail+Flag only</button>
  <button id='v-fail'>Fail only</button>
  <button id='v-pass'>Pass only</button>
  <span>Dataset:</span>
  <button data-filter-dataset='__all__' class='active'>All</button>
  {dataset_buttons}
  <input type='search' id='search' placeholder='search by cell_id…' />
</div>

<table class='cells'>
<thead><tr>
  <th></th>
  <th data-sort='cellid'>cell_id</th>
  <th data-sort='dataset'>dataset</th>
  <th data-sort='verdict'>verdict</th>
  <th>triggered</th>
  {metric_headers}
</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>

<script>
function toggle(td) {{
  const tr = td.parentElement;
  const detail = tr.nextElementSibling;
  if (detail && detail.classList.contains('detail')) detail.classList.toggle('open');
  const caret = td.querySelector('.caret');
  if (caret) caret.textContent = caret.textContent === '▸' ? '▾' : '▸';
}}
let _recipeFilter = null;   // metric name string, or null = no recipe filter
function applyFilters() {{
  const verdictMode = document.querySelector('.controls .active[id^="v-"]').id;
  const dsBtn = document.querySelector('.controls .active[data-filter-dataset]');
  const ds = dsBtn ? dsBtn.dataset.filterDataset : '__all__';
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('tr.row').forEach(tr => {{
    const v = tr.dataset.verdict, d = tr.dataset.dataset, c = tr.dataset.cellid.toLowerCase();
    const tm = (tr.dataset.trigmetrics || '').split(' ');
    let visible = true;
    if (verdictMode === 'v-attention') visible = (v === 'flag' || v === 'fail');
    else if (verdictMode === 'v-fail') visible = (v === 'fail');
    else if (verdictMode === 'v-pass') visible = (v === 'pass');
    if (visible && ds !== '__all__') visible = (d === ds);
    if (visible && q) visible = c.includes(q);
    if (visible && _recipeFilter) visible = tm.includes(_recipeFilter);
    tr.classList.toggle('hidden', !visible);
    const det = tr.nextElementSibling;
    if (det && det.classList.contains('detail')) det.classList.toggle('hidden', !visible);
  }});
}}
document.querySelectorAll('.failure-recipe .recipe-item').forEach(el => {{
  el.addEventListener('click', () => {{
    const m = el.dataset.recipeMetric;
    _recipeFilter = (m === '__clear__') ? null : m;
    document.querySelectorAll('.failure-recipe .recipe-item').forEach(x =>
      x.style.outline = (x === el && _recipeFilter) ? '2px solid #5a9ad6' : 'none');
    applyFilters();
  }});
}});
document.querySelectorAll('.controls button').forEach(b => b.addEventListener('click', e => {{
  if (b.id && b.id.startsWith('v-')) {{
    document.querySelectorAll('.controls button[id^="v-"]').forEach(x => x.classList.remove('active'));
  }} else if (b.dataset.filterDataset) {{
    document.querySelectorAll('.controls button[data-filter-dataset]').forEach(x => x.classList.remove('active'));
  }}
  b.classList.add('active');
  applyFilters();
}}));
document.getElementById('search').addEventListener('input', applyFilters);
applyFilters();
// Naive column sort
document.querySelectorAll('th[data-sort]').forEach(th => th.addEventListener('click', () => {{
  const tbody = th.closest('table').tbody = th.closest('table').querySelector('tbody');
  const key = th.dataset.sort;
  const rows = Array.from(tbody.querySelectorAll('tr.row'));
  const dir = th.dataset.dir === 'asc' ? -1 : 1;
  th.dataset.dir = dir === 1 ? 'asc' : 'desc';
  rows.sort((a,b) => {{
    const av = a.dataset[key] || '', bv = b.dataset[key] || '';
    return av.localeCompare(bv) * dir;
  }});
  rows.forEach(r => {{
    const d = r.nextElementSibling;
    tbody.appendChild(r); if (d && d.classList.contains('detail')) tbody.appendChild(d);
  }});
}}));
</script>
</body></html>"""


def write_report(report_df: pd.DataFrame, thumbnails: dict[str, list[Path]], *,
                 html_path: Path, csv_path: Path, project_name: str,
                 pipeline_version: str, thresholds_fp: str = "",
                 viewer_url: str = "http://127.0.0.1:8765",
                 thresholds: dict[str, dict] | None = None,
                 cohort_stats: dict[str, dict] | None = None) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_str = render_html(
        report_df, thumbnails,
        project_name=project_name,
        pipeline_version=pipeline_version,
        thresholds_fp=thresholds_fp,
        viewer_url=viewer_url,
        thresholds=thresholds,
        cohort_stats=cohort_stats,
    )
    html_path.write_text(html_str, encoding="utf-8")
    # CSV: flatten triggered_metrics to a string
    csv_df = report_df.copy()
    if "triggered_metrics" in csv_df.columns:
        csv_df["triggered_metrics"] = csv_df["triggered_metrics"].apply(
            lambda v: json.dumps(v) if not isinstance(v, str) else v
        )
    csv_df.to_csv(csv_path, index=False)
