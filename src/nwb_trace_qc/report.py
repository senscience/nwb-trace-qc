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
                metric_columns: list[str] | None = None) -> str:
    """Render the full self-contained HTML report.

    `report_df` must have columns: cell_id, dataset, computed_verdict, final_verdict,
    triggered_metrics (JSON list-of-dicts as string or list), <metric columns>...,
    optional override_note/override_reviewer/override_date.
    `thumbnails` is a dict {cell_id: [paths-to-PNGs]}.
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

    def _trigger_chips(raw):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return ""
        items = raw if isinstance(raw, list) else json.loads(raw) if raw else []
        chips = []
        for it in items:
            cls = it.get("verdict", "pass")
            label = f"{it.get('metric', '?')} {it.get('reason', '') or ''}".strip()
            chips.append(f"<span class='chip chip-{cls}'>{html.escape(label)}</span>")
        return "".join(chips)

    rows_html = []
    datasets_seen = sorted(report_df["dataset"].dropna().astype(str).unique())
    for i, row in report_df.iterrows():
        cell_id = str(row["cell_id"])
        dataset = str(row.get("dataset", ""))
        verdict = str(row.get("final_verdict", "pass"))
        computed = str(row.get("computed_verdict", verdict))
        triggered = _trigger_chips(row.get("triggered_metrics"))
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
        rows_html.append(f"""
<tr class='row row-{verdict}' data-verdict='{verdict}' data-dataset='{html.escape(dataset)}' data-cellid='{html.escape(cell_id)}'>
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
  .chip {{ display: inline-block; padding: 0.1rem 0.4rem; margin: 0.05rem; border-radius: 3px; font-size: 0.72rem; }}
  .chip-flag {{ background: {VERDICT_COLORS['flag']}; color: {VERDICT_TEXT['flag']}; }}
  .chip-fail {{ background: {VERDICT_COLORS['fail']}; color: {VERDICT_TEXT['fail']}; }}
  .chip-pass {{ background: {VERDICT_COLORS['pass']}; color: {VERDICT_TEXT['pass']}; }}
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
function applyFilters() {{
  const verdictMode = document.querySelector('.controls .active[id^="v-"]').id;
  const dsBtn = document.querySelector('.controls .active[data-filter-dataset]');
  const ds = dsBtn ? dsBtn.dataset.filterDataset : '__all__';
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('tr.row').forEach(tr => {{
    const v = tr.dataset.verdict, d = tr.dataset.dataset, c = tr.dataset.cellid.toLowerCase();
    let visible = true;
    if (verdictMode === 'v-attention') visible = (v === 'flag' || v === 'fail');
    else if (verdictMode === 'v-fail') visible = (v === 'fail');
    else if (verdictMode === 'v-pass') visible = (v === 'pass');
    if (visible && ds !== '__all__') visible = (d === ds);
    if (visible && q) visible = c.includes(q);
    tr.classList.toggle('hidden', !visible);
    const det = tr.nextElementSibling;
    if (det && det.classList.contains('detail')) det.classList.toggle('hidden', !visible);
  }});
}}
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
                 pipeline_version: str, thresholds_fp: str = "") -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_str = render_html(
        report_df, thumbnails,
        project_name=project_name,
        pipeline_version=pipeline_version,
        thresholds_fp=thresholds_fp,
    )
    html_path.write_text(html_str, encoding="utf-8")
    # CSV: flatten triggered_metrics to a string
    csv_df = report_df.copy()
    if "triggered_metrics" in csv_df.columns:
        csv_df["triggered_metrics"] = csv_df["triggered_metrics"].apply(
            lambda v: json.dumps(v) if not isinstance(v, str) else v
        )
    csv_df.to_csv(csv_path, index=False)
