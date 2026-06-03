# Usage — running `nwb-trace-qc` on a dataset

A linear, step-by-step walkthrough from a folder of NWBs to a triaged, human-reviewed verdict for every cell. Each step shows the exact command, the output you should see, and what to check before moving on.

The fast path is **Steps 1 → 4** and you have a report. **Step 5** opens the interactive viewer for visual verification. **Step 6** is the iteration loop (thresholds, overrides, optional LLM second opinion).

---

## Step 0 — Install

One-time. Picks up `pynwb`, `efel`, `pandas`, `matplotlib`, `pyarrow`, `pyyaml`, `click`, `pydantic`, and both `anthropic` and `openai` SDKs (the vision judge is ready to use once you set an API key — no second install).

```bash
pip install -e /path/to/nwb-trace-qc        # editable install of the repo
# or, once published:
# pip install nwb-trace-qc
nwb-qc --version
```

**Check before moving on:** `nwb-qc --help` lists the five subcommands `init-config`, `list-cells`, `run`, `report`, `thresholds`, and `serve`.

---

## Step 1 — Auto-discover and write a project config

Point `init-config` at the folder that contains your NWB files (subfolders are fine). It walks the tree, groups NWBs by top-level subfolder into named datasets, detects any wrangler parquet tables, and writes a ready-to-run YAML.

```bash
cd /where/you/want/the/config
nwb-qc init-config /path/to/your/data
```

**Output:**

```
Wrote ./mydata_project.yaml
      (3 sources, 2 acquisition tables, thresholds at mydata_thresholds.yaml)
Next: nwb-qc list-cells --config ./mydata_project.yaml
```

Two files appear in the current directory:

- `mydata_project.yaml` — project config (paths, datasets, stimulus-protocol families, output paths, worker count, vision config, …).
- `mydata_thresholds.yaml` — a per-project copy of the bundled default thresholds so you can edit without touching the global defaults.

**Flags you may want:**

| Flag | Effect |
|---|---|
| `--name myproj` | Override the project name (default = folder basename). |
| `--output path.yaml` | Override the YAML filename/location. A directory path also works. |
| `--no-guess-tables` | Skip the parquet scan if you don't want auto-registered acquisition tables. |

**Check before moving on:** open `mydata_project.yaml` and verify (a) the `nwb_sources:` paths and globs look right, and (b) the `stimulus_protocols:` mapping matches your lab's naming (defaults are LNMC/BBP; e.g. swap `BL_hold` into the `spontaneous_hold:` list if that's what your lab uses).

---

## Step 2 — Dry-run: confirm what will be processed

No NWBs are opened in this step (other than reading file sizes); it just enumerates what the config picked up.

```bash
nwb-qc list-cells --config mydata_project.yaml
```

**Output:**

```
Project: mydata
Sources: 3
  - cohort_a: 412 NWBs at /path/to/your/data/cohort_a
  - cohort_b: 18  NWBs at /path/to/your/data/cohort_b
  - cohort_c: 7   NWBs at /path/to/your/data/cohort_c
Total NWB rows: 437
Unique by sha256: 437 (dedup saves 0 compute steps)
Acquisition tables registered: 2
Cell table: /path/to/your/data/cells.csv
```

**Check before moving on:**

- Counts per dataset match what you expect.
- Total ≈ what `find . -name '*.nwb' | wc -l` returns under your root.
- If a number is off, edit `nwb_sources[*].glob:` in the YAML (most often the fix is `"**/data/*/*.nwb"` for nested archive layouts) and rerun this step.

---

## Step 3 — First real run

Computes every metric on every cell, applies thresholds, writes the report. Wall-clock is dominated by NWB I/O divided by `n_workers`; first run on 30 cells ≈ 90 s, on 2,300 cells ≈ 2 h on 6 workers. Every subsequent run hits the per-NWB cache and skips Stage 2 entirely.

```bash
nwb-qc run --config mydata_project.yaml
```

**What happens, in order:**

1. **Manifest** — sha256-hash every NWB.
2. **Cache lookup** — skip NWBs already in `_qc_cache.parquet`.
3. **Metric compute** (parallel) — open new NWBs and extract Vrest, Rs, AP overshoot, the five visual-defect metrics (test-pulse decay shape, within-sweep Vm drift, failed-spike fraction, AP-amplitude CV, late-recording instability), and signal-hygiene counters.
4. **Apply thresholds** → `pass` / `flag` / `fail` per cell.
5. **Render thumbnails** for non-pass cells (cached on disk).
6. **(Optional) Vision judge** on `flag` cells — only if enabled in YAML or via `--with-vision` (see Step 6c).
7. **Apply overrides** from `qc_output_*/qc_overrides.csv`.
8. **Write report** (static HTML + CSV + emit `qc_viewer.html` for Step 5).

**Output is a single JSON line:**

```json
{
  "n_cells": 437,
  "n_unique_nwbs": 437,
  "n_computed_this_run": 437,
  "n_nwb_opens": 437,
  "n_pass": 102,
  "n_flag": 88,
  "n_fail": 247,
  "elapsed_s": 1842.3,
  "report": "/path/to/qc_output_mydata/qc_report.html",
  "viewer": "/path/to/qc_output_mydata/qc_viewer.html",
  "vision": {"enabled": false}
}
```

**Tip — smoke-test first.** If your cohort is large, run on a single dataset first to validate thresholds before committing to the full run:

```bash
nwb-qc run --config mydata_project.yaml --filter dataset=cohort_a
```

**Check before moving on:** `n_computed_this_run` equals `n_unique_nwbs`. On a second consecutive run it should be **0** (cache hits everything) — that's how you verify caching works.

---

## Step 4 — Open the static report

```bash
open /path/to/qc_output_mydata/qc_report.html
```

**What you see:**

- Top: total pass/flag/fail counts overall and per-dataset.
- Filter strip: toggles for verdict (defaults to Fail+Flag only — passes hidden) and dataset, plus a `cell_id` search.
- Table: one row per cell with verdict badge, triggered metrics as colored chips, and key metric values.
- Click any row to expand → full metric table, inline trace thumbnails of the offending sweeps, a copyable override template, and (when present) vision-judge / human-override banners.

The HTML has zero external resources. You can copy it to a colleague's laptop and everything still renders.

**Check before moving on:** the cells that show up in fail/flag actually look bad in their thumbnails. If many cells fail on a single metric that *shouldn't* be cohort-wide (e.g. `qc_protocol_coverage`), that points to a stimulus-name or threshold mismatch — handle in Step 6.

---

## Step 5 — Interactive verification (any sweep on demand)

The static report only renders the offending sweep thumbnails. To inspect any sweep of any cell, start the local viewer:

```bash
nwb-qc serve --config mydata_project.yaml
```

This starts a localhost-only HTTP server (default port 8765) and opens your browser. The viewer reads the same `qc_report.csv` for the cell list and lazy-loads sweep data from the underlying NWB files only when you click — only the bytes you ask for are decoded.

**Layout:**

- **Left** — cell list with verdict chips, filterable by `cell_id`, sorted fail/flag first.
- **Centre-left** — sweeps for the selected cell, grouped by stimulus family.
- **Right** — 2 × 2 grid of plot panels. Click a sweep to drop it into the next free slot; click "×" on a panel to clear it.

Plots render with vanilla Canvas (no JS libraries), include a dashed 0 mV reference line, and decimate via LTTB to ~2,500 points regardless of source sampling rate. Stop the server with `Ctrl-C`.

You can also hit the API directly for scripting:

```bash
curl http://127.0.0.1:8765/api/sweeps/<cell_id>
curl 'http://127.0.0.1:8765/api/trace/<cell_id>/<sweep_idx>?max_points=1000'
```

**Flags:** `--port N` to change the port, `--no-browser` to skip auto-opening.

**Check before moving on:** flagged cells' sweeps visually agree with the rules' verdict. Where they disagree, that's input to Step 6.

---

## Step 6 — Iterate

Three independent dials, none requires code changes.

### 6a — Tune thresholds

Edit `mydata_thresholds.yaml` and rerun:

```bash
nwb-qc run --config mydata_project.yaml
```

The cache hits all NWBs (no recompute); only manifest hashing and report rendering happen. For thousands of cells the wall-clock is single-digit minutes.

Rule grammar:

```yaml
metric_name:
  fail_above: 30        # value > 30 → fail
  fail_below: 0         # value < 0  → fail
  flag_above: 25
  flag_below: 10
  fail_if_false: true   # boolean: must be truthy or fail
```

Verdict precedence: any `fail` wins; else any `flag`; else `pass`. NaN values produce a soft flag (insufficient data), not a fail.

### 6b — Stick a verdict by hand

Append to `qc_output_mydata/qc_overrides.csv` (the expand-row panel in the static report shows a copyable template per cell):

```csv
cell_id,override_verdict,note,reviewer,date
sample_42,pass,manually inspected — overshoot loss is end-of-recording artefact,you,2026-06-03
```

Overrides survive re-runs and threshold edits. They're applied last, so a human verdict trumps everything else.

### 6c — Get a second opinion from an LLM vision judge (optional)

Off by default. When enabled, only **`flag`-verdict cells** are sent to a vision model — cells that already clearly `pass` or `fail` by the rules don't trigger API calls, so cost is bounded (≤ `max_borderline_cells`, default 100).

**Set an API key** (one or both — `anthropic` and `openai` SDKs are pre-installed):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

**Enable per-project** in your YAML:

```yaml
vision_judge:
  enabled: true
  provider: anthropic              # 'anthropic' | 'openai' | 'mock'
  model: claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY
  max_borderline_cells: 100
  prompt_template: null            # null = bundled default
  cache_responses: true
```

**Or one-shot via flag:**

```bash
nwb-qc run --config mydata_project.yaml --with-vision
```

**Verdict precedence (each later step wins over the previous):**

1. Rule-based verdict.
2. Vision judge:
   - rules `flag` + vision `fail` → final `fail` (reason: `vision_escalated`).
   - rules `flag` + vision `pass` → final stays `flag` (reason: `vision_suggests_pass`) — vision can't auto-pass a borderline cell, only flag it for human review.
   - rules `fail` is never downgraded; rules `pass` is never escalated.
3. Human override — always wins.

The HTML report's per-cell expand panel shows a blue "Vision judge" banner with confidence + notes whenever the judge weighed in. Use `provider: mock` to exercise the integration deterministically without API calls.

---

## The shortest possible loop

Once configured:

```bash
nwb-qc run --config mydata_project.yaml && open qc_output_mydata/qc_report.html
```

Or, for visual spot-checks instead of opening the static report:

```bash
nwb-qc run --config mydata_project.yaml && nwb-qc serve --config mydata_project.yaml
```

---

## Reference — CLI subcommands

| Command | What it does |
|---|---|
| `nwb-qc init-config <root>` | Auto-discover NWBs + parquets and write a starter project YAML and per-project thresholds YAML. |
| `nwb-qc list-cells --config <file>` | Dry-run: print discovered NWBs, dedup info, and which cells map to which dataset. |
| `nwb-qc run --config <file> [--filter dataset=X] [--with-vision/--no-vision] [--report-only]` | Full pipeline. |
| `nwb-qc report --config <file>` | Re-render the HTML/CSV from the existing cache without NWB I/O. |
| `nwb-qc thresholds --config <file> --dry-run` | Show how the current thresholds would classify cached cells (counts only). |
| `nwb-qc serve --config <file> [--port N] [--no-browser]` | Interactive trace viewer. |
| `nwb-qc --version` | Print the package version. |

---

## Reference — what each metric checks

**Scalar (v0.1.0):** `vrest_mv`, `vrest_drift_mv`, `rs_mohm_initial` / `rs_mohm_final` / `rs_drift_pct`, `ap_amp_overshoot_mv`, `ap_threshold_drift_mv`, `baseline_rms_mv`, `n_sweeps_total` / `n_sweeps_clipped` / `n_sweeps_nan`, `qc_protocol_coverage`.

**Visual-defect (v0.2.0):**

| Metric | What it catches |
|---|---|
| `rac_decay_residual_rel` | Test-pulse / step-response recovery that's glitchy or rings instead of a clean exponential. |
| `vm_drift_within_sweep_mv_per_s` | Drifting seal — within-sweep Vm slope (e.g. −70 mV → −20 mV over one long sweep). |
| `ap_failure_fraction` | Spikes that initiate (dV/dt threshold crossing) but never reach overshoot. |
| `ap_amp_cv` | AP peak amplitudes that vary inconsistently within one train. |
| `late_instability_index` | Orderly firing degrading to runaway oscillation in the latter portion of long sweeps. |

All metrics, scalar and visual, share the same rule grammar in `default_thresholds.yaml`.

---

## Troubleshooting

- **"thresholds_file not found"** — `init-config` couldn't find the bundled defaults to copy. Either run from a directory where the repo's `configs/default_thresholds.yaml` is reachable, or point `thresholds_file:` in your project YAML at an absolute path.
- **All cells flagged on `qc_protocol_coverage`** — your `stimulus_protocols` mapping doesn't match your data's protocol names. Open one NWB with `pynwb`, list `acquisition.keys()`, and add the relevant tokens to the right family in your project YAML.
- **Absolute Rs values look unrealistic** — the Rs estimator currently assumes a nominal 50 pA test pulse (it reads only the voltage trace, not the paired stimulus current). Trust `rs_drift_pct` over `rs_mohm_final` until proper Rs from the paired stimulus is implemented; relax or remove the `fail_above` rule on `rs_mohm_final`.
- **First run feels slow** — it's I/O-bound when NWBs aren't in the OS page cache. Re-runs hit the cache and skip metric compute entirely; only manifest hashing remains.
- **Vision judge runs but no cells get queried** — only `flag`-verdict cells are sent. If your cohort is all `pass` or all `fail`, the vision judge has nothing to do; widen the borderline by loosening `flag_above` / `fail_above`.
- **`nwb-qc serve` says "manifest not found"** — you need to have run `nwb-qc run` at least once for the project. The viewer reads `_qc_manifest.parquet` and `qc_report.csv` produced by `run`.

---

See [`jy_quickstart.md`](jy_quickstart.md) for a worked example with real cohort numbers (2,302 NWBs across VPL / Red Nucleus / OBI Thalamus).
