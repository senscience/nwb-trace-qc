# Usage — running `nwb-trace-qc` on a dataset

A linear, step-by-step walkthrough from a folder of NWBs to a triaged, human-reviewed verdict for every cell. Each step shows the exact command, the output you should see, and what to check before moving on.

The fast path is **Steps 1 → 4** and you have a report. **Step 5** opens the interactive viewer for visual verification. **Step 6** is the iteration loop (thresholds, overrides, optional LLM second opinion).

---

## What changed in v0.6.0

`PIPELINE_VERSION` bumped to `0.6.0`; v0.5.x cache invalidates automatically. Two report-UX changes addressing the "all cells fail / report too complicated" feedback:

- **Critical-metric whitelist for cell verdicts.** With ~25 metrics in v0.5, almost every cell tripped at least one (often a missing-data NaN or a peripheral metric like `rs_compensation_pct`), so 100% of cells failed. Now only fails on a **critical** metric promote to a cell-level fail. The bundled critical set (`families.DEFAULT_CRITICAL_METRICS`) is: `rs_drift_pct`, `vrest_mv`, `ap_amp_overshoot_mv`, `ap_amplitude_mv`, `holding_current_drift_pa`, `qc_protocol_coverage`, `n_sweeps_clipped`, `n_sweeps_nan`. Everything else is advisory — its trigger chip is still surfaced (dimmer, behind a "+N advisory" toggle) but the cell verdict is capped at `flag`. Override per-project via `critical_metrics:` in the project YAML.
- **Tiered report layout per cell.** The expand panel now leads with the health card + **critical-only** triggered chips. Advisory chips collapse behind a `+N advisory` toggle so the at-a-glance view stays focused on what actually drove the verdict. The full metric table (every column the cache emits) also collapses behind a "show all metric values" toggle. Thumbnails and viewer deep-link stay where they were. 90% of triage decisions should now be possible without expanding either fold.

## What changed in v0.5.0

LNMC experimenter-guidance additions (see "Parameters to consider while evaluating quality of whole cell, current clamp recordings" — the eCode protocol guidance document). `PIPELINE_VERSION` bumped to `0.5.0` — the v0.4.x cache is invalidated automatically.

- **Spontaneous-hold family split.** The legacy `spontaneous_hold` family is split into two semantically distinct families:
  - `spontaneous_no_hold` (true Vrest — no holding current): `SponNonHold30`, `SponNoHold30`, `StartNoHold` (default). Drives `vrest_mv`.
  - `spontaneous_held` (held at a target Vm under Ihld): `SponHold3`, `SponHold30`, `StartHold` (default). Drives the new `held_vm_mv` metric and is the canonical source of `holding_current_pa`.
  - Legacy `spontaneous_hold` family still works as a mixed-semantics fallback; the pipeline logs once per run noting the cohort should be migrated to the split.
- **New `ap_amplitude_mv` metric.** Peak − threshold (canonical AP amplitude per LNMC's definition; eFEL's `AP_amplitude` feature). Distinct from `ap_amp_overshoot_mv` (peak above 0 mV) — both are reported. Default rule: `flag_below: 60 mV, fail_below: 40 mV`.
- **New `rs_compensation_pct` metric.** Read from `IntracellularElectrode.resistance_comp_correction` in the NWB icephys metadata (or a `lab_meta_data` field whose name contains "rs"/"resistance"/"compensation"). Catches cohorts where the experimenter forgot to enable Rs compensation. Default rule: `flag_below: 50, fail_below: 0`.
- **New `rac_variability_pct` metric.** Coefficient of variation (std/median × 100) of per-Rac Rs estimates. Catches non-monotonic Rs instability that the existing first-vs-last `rs_drift_pct` misses. Default rule: `flag_above: 20, fail_above: 40`.
- **Reframed `test_pulse_edge_overshoot_mv`.** Per the LNMC PDF: a *sharp* edge transient is the *good* signature of active Rs compensation; a smooth exponential decay indicates no compensation. The default auto rule has been removed — the metric is now informational and the interpretation depends on `rs_compensation_pct`. `rac_variability_pct` covers the cohort-stability check.

## What changed in v0.4.0

`PIPELINE_VERSION` bumped to `0.4.0` — the v0.3.x metric cache is invalidated automatically (first run will recompute). Three substantive additions:

- **Bad-ending detection + auto-trim.** When a recording degrades near the end (cell dying, seal collapsing) the pipeline now finds the cutoff sweep and excludes the post-degradation period from the metric scalars. New metrics: `bad_ending_at_sweep`, `n_sweeps_trimmed`, `bad_ending_reason` (`vrest_depolarisation` / `rs_explosion` / `ap_collapse`). The default threshold flags any cell with `n_sweeps_trimmed > 0`. Disable with `trim_bad_ending: false` in the project YAML.
- **eFEL-sourced AP / Vrest features.** Canonical features (`voltage_base`, `AP_amplitude_from_voltagebase`, `AP_begin_voltage`, …) now come from the BBP/LNMC-standard eFEL library where it has a direct equivalent. Our custom helpers stay as fallbacks for malformed sweeps. Disable with `use_efel: false`.
- **Metric provenance.** New `nwb-qc inventory-metrics --config <yaml>` subcommand walks a few NWBs and reports whether any pre-computed analysis modules exist that already carry canonical metric values. Also a permanent `docs/metrics_reference.md` listing what every metric is and where its value comes from.

## What changed in v0.3.0

If you're upgrading from v0.2.x: the pipeline now reads paired `CurrentClampStimulusSeries` from each NWB so Rs comes from the actual injected current (not the 50 pA assumption); Rin from IV protocol; holding current per sweep; and session-level degradation deltas (first half vs second half of the recording, median-based). Plus three sketch-aligned defect metrics: `test_pulse_edge_overshoot_mv`, `ap_amp_overshoot_min_mv`, `ap_amp_attenuation_frac`. `PIPELINE_VERSION` bumped to `0.3.0`, which invalidates the v0.2.x metric cache automatically — your first run on this version will recompute every NWB. The new `nwb-qc calibrate` subcommand suggests project-specific thresholds from cohort statistics.

---

## Step 0 — Install

One-time. Picks up `pynwb`, `efel`, `pandas`, `matplotlib`, `pyarrow`, `pyyaml`, `click`, `pydantic`, and both `anthropic` and `openai` SDKs (the vision judge is ready to use once you set an API key — no second install).

```bash
pip install -e /path/to/nwb-trace-qc        # editable install of the repo
# or, once published:
# pip install nwb-trace-qc
nwb-qc --version
```

**Check before moving on:** `nwb-qc --help` lists the subcommands `inspect`, `init-config`, `list-cells`, `run`, `report`, `thresholds`, `serve`, and `start`.

---

## Step 0.1 — Quickest path: the guided wizard

If you'd rather walk through the whole flow once with prompts between each step instead of running four commands by hand, use:

```bash
nwb-qc start /path/to/your/data
```

This walks you through five stages, pausing for confirmation between each:

1. **Inspect** — prints the inventory (same as `nwb-qc inspect`). `[Enter]` to continue.
2. **Propose config** — generates the project YAML and shows it on screen. The YAML header includes a discovery block listing every stimulus protocol found in your NWBs, and a yellow `⚠ UNMAPPED tokens` callout when some don't match the default family map (this catches the "all cells fail qc_protocol_coverage" failure mode). `[a]ccept`, `[m]ap-unmapped` (only shown when unmapped tokens exist — opens an interactive walk; at the top you pick `[w]` walk-each / `[a]` accept-all-heuristic-suggestions / `[c]` cancel; the per-token walk lets you slot any token into a family or skip, while accept-all takes every heuristic-derived guess in one keystroke and leaves tokens without a guess in the UNMAPPED block), `[e]dit` in `$EDITOR`, or `[q]uit`.
3. **Review thresholds** — shows the active `thresholds_file` YAML so you can sanity-check the rules before paying the metric-compute cost. `[a]ccept` (Enter, default — proceed to dry-run with the current rules) / `[e]dit` (opens the YAML in `$EDITOR`; re-display + re-prompt after save) / `[q]uit`. No cohort data exists yet at this point — this is editorial review, not statistical calibration. After the first run you can come back via `[t]une-thresholds` (cohort-aware) in the outcome menu or via `nwb-qc tune` standalone.
4. **Dry-run** — shows which NWBs were discovered and how many will be processed. `[r]un`, `[b]ack` to re-edit the config, or `[q]uit`.
5. **Run** — executes the pipeline with a live `[stage 2/6 metric-compute] 142/2302 cells …  ETA 28m` progress line. After the pipeline completes, the wizard also auto-runs `calibrate` against the freshly computed cache and writes two side files: `cohort_stats.json` (consumed by future reports to add cohort-percentile context to triggered-metric chips) and `<thresholds_stem>_thresholds_suggested.yaml` (a thresholds YAML derived from cohort percentiles, ready to opt into).
6. **Outcome** — prints the report paths, the cohort-stats and suggested-thresholds paths, and offers `[o]pen` report / `[s]erve` viewer / `[t]une-thresholds` / `[c]alibrate-and-re-run` / `[d]one`. The default is `[d]one` so a first-time run can complete with bundled defaults by just pressing Enter. Choosing `[t]` walks you through every threshold rule interactively (see Step 6f below). Choosing `[c]` rewrites the project YAML's `thresholds_file:` to point at the auto-generated suggested file and re-runs the pipeline immediately — the re-run is cache-fast (only the threshold layer re-evaluates), so you can compare bundled-defaults vs cohort-calibrated verdicts in seconds.

**Re-entering the wizard**: running `nwb-qc start <root>` a second time, with an existing project YAML and warm cache, the wizard detects the prior state and offers to skip straight to the outcome stage (so you can tune / serve / open the existing report without re-walking inspect → propose → dry-run → run). Pick `[r]estart` to do the full flow instead, or `[q]uit` to exit.

Every step is reversible; nothing is committed to disk until you accept the config in step 2. After the run, every project artefact is exactly the same as if you'd run `init-config` / `list-cells` / `run` separately, so you can keep iterating with those commands afterward.

Pass `--max-cost-usd 0.50` (or any number) to cap the vision-judge spend for this run, and `-v` / `--verbose` on the top-level group to see DEBUG-level logs on stderr.

---

## Step 0.5 — (Optional) Inspect the source tree before configuring

Before committing to a config, you can get a structured inventory of what's in a folder of wrangler outputs — NWB counts, parquet schemas, fair2.json / README / run_state summaries, and a per-parquet "QC-eligible?" check that tells you which tables `init-config` will register and what column-mapping you'd need for the ones it can't.

```bash
nwb-qc inspect /path/to/your/data
```

**Output** (excerpt — real JY example):

```
[4] output_20260601_195219/  ·  2.8 MB
    └─ jy_vpl_intracellular_electrophysiology_jane_yi_epfl_lnmc/
       ├─ README.md               7.4 KB  "JY VPL Intracellular Electrophysiology — Jane Yi, EPFL LNMC"
       ├─ fair2.json             68.3 KB  Croissant 1.x, JY VPL …
       ├─ data_dictionary.csv     8.6 KB  31 variable definitions
       ├─ run_state.json          6.4 KB  v1.0
       ├─ parquet/
       │  ├─ acquisitions.parquet         293,499 rows · 11 cols  (map columns.stimulus_type → 'protocol')
       │  ├─ session_metadata.parquet       2,198 rows · 14 cols  (missing: stimulus_type)
       └─ scripts/                         extract_nwb_sessions_and_acquisitions.py

Summary
───────
  6 datasets · 2,374 NWB files · 33.40 GB total
  1 acquisition-table parquet would be registered by `init-config`
```

**Flags:**

| Flag | Effect |
|---|---|
| `--output PATH / -o PATH` | Where to write the full Markdown inventory (default: `./<root>_inventory.md`). |
| `--json` | Emit JSON instead of Markdown (good for piping). |
| `--no-write` | Skip writing the inventory file; only print to stdout. |

Read-only — never opens NWB files, just counts and sizes them. Useful when you want to know what's in a folder you didn't write yourself, or to confirm that a wrangler output is complete before kicking off a QC run on it.

### When the wrangler didn't copy source NWBs into the package

Some wrangler runs leave `source_material/` as just a `source_manifest.json` (Croissant schema v5) — a JSON list of every source NWB by **absolute `original_location`**, with size + sha256 + mtime per file. `inspect` surfaces this manifest and `init-config` will use it as the discovery source, so you can QC files **in place** without the wrangler copying them.

```
nwb-qc inspect /path/to/wrangler-output:
…
├─ source_material/source_manifest.json   schema v5 · 33 files · 1.06 GB · preservation: not copied · sample 5/5 present on disk
…
```

`init-config` against such a tree auto-emits a `manifest:`-form source instead of a `path:` + `glob:` source:

```yaml
nwb_sources:
  - dataset: jy_red_nucleus_intracellular_ephys_epfl_lnmc
    manifest: /…/source_material/source_manifest.json
    only_processed: true   # skip files the wrangler marked was_processed=false
```

The pipeline reads the manifest's `original_location` to find every NWB, and **reuses the manifest's pre-computed sha256** when the on-disk file's size and mtime still match (±1 s). For a cohort whose NWBs total ~1 GB this saves the entire hashing step on every subsequent run. `list-cells` reports the diagnostics:

```
Manifest-source diagnostics:
  - jy_red_nucleus_…: 33 in manifest · 32 eligible · 32 present · 0 missing ·
                       sha256 reused 32 / recomputed 0
```

If a source file has been touched (mtime change) or moved (missing on disk), the manifest stats show it and the pipeline either recomputes the hash or skips the missing file with a logged warning — no silent staleness.

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

`init-config` also opens up to 3 NWBs per source to discover which stimulus protocols are actually present in your data, and writes the result as comments in the YAML header — both the protocols that already map to a family and any **unmapped** tokens you'll need to slot in:

```yaml
# Stimulus protocols discovered by sampling your NWBs:
#   ap_waveform: APWaveform (72)
#   iv_subthreshold: IV (48)
#   rest_firing: IDRest (90)
#
# ⚠ UNMAPPED tokens (21 unique, 462 sweeps in sampled NWBs):
#   C1step_ag  (130 sweeps)
#   sAHP  (78 sweeps)
#   Test_eCode  (31 sweeps)
#   ...
# Heads-up: no protocols mapped to ['spontaneous_hold', 'test_pulse'] —
# qc_protocol_coverage will be False for every cell until you assign
# at least one unmapped token to each essential family.
```

When you see an UNMAPPED block, edit `stimulus_protocols:` below in the same file to add each lab-specific name to the right family (e.g. add `IRrest` and `Spontaneous` to `spontaneous_hold`, `Test_eCode` and `Delta` to `test_pulse`). Without this step, the QC pipeline still runs, but `qc_protocol_coverage` will be `False` for every cell and downstream metrics that need missing families (`vrest_mv`, `rs_mohm_*`, `baseline_rms_mv`) will be `NaN`.

**Flags you may want:**

| Flag | Effect |
|---|---|
| `--name myproj` | Override the project name (default = folder basename). |
| `--output path.yaml` | Override the YAML filename/location. A directory path also works. |
| `--no-guess-tables` | Skip the parquet scan if you don't want auto-registered acquisition tables. |

**Check before moving on:** open `mydata_project.yaml` and verify (a) the `nwb_sources:` paths and globs look right, and (b) the discovery header at the top — if there's an UNMAPPED block, assign those tokens to families before running. The wizard (`nwb-qc start`) prints a yellow callout above the YAML when unmapped tokens are present so you don't miss them.

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
  "run_report": "/path/to/qc_output_mydata/run_report.json",
  "vision": {"enabled": false}
}
```

**`run_report.json`** is the structured per-run record. Per-stage timing, LLM token spend, peak memory, manifest-source diagnostics, and the catalogue of any `compute_error`s land in this file every run, so you can diff successive runs and track regressions. Schema (top-level keys): `started_at` / `finished_at` / `elapsed_s`, `stages.{manifest_build, metric_compute, thresholds, thumbnails, vision, overrides, report}`, `verdicts`, `memory.peak_rss_mb`, `system`, `manifest_stats`, `compute_errors`.

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

Each triggered-metric chip now carries the implicated stimulus **family** as a small italic tag, e.g. `vrest_mv nan · spontaneous_hold`. **Clicking any chip** expands a plain-English explanation underneath: what the metric measures, which sweeps drove the value (first/last from per-metric provenance fields like `vrest_mv_provenance`), the cohort percentile (when `cohort_stats.json` is present from `nwb-qc calibrate` — see Step 6e), and what a healthy range looks like.

At the top of each expanded cell, a **health-summary card** shows the six canonical signals at a glance — Vrest, Rs (final), Rin, AP overshoot, Holding current, session drift — each coloured by its individual rule outcome. Below it, the **"Inspect these families:"** strip lists the unique families implicated by that cell's triggers plus an **"Inspect all sweeps in viewer →"** deep link to Step 5's interactive viewer (pre-selects the cell when `nwb-qc serve` is running).

A new **"Failures by metric"** strip at the top of the report (visible whenever any cell has flag/fail triggers) lists every triggering metric with the cohort-wide count. Click any item to filter the table to just the cells affected — useful for telling apart "one bad metric is dragging the whole cohort down" (likely a threshold mis-calibration) from "several cells fail on multiple metrics" (real cell-quality issues).

The static thumbnail PNG itself still stacks at most 3 representative sweeps. Within a matched family, picks are **stratified** (first / middle / last) rather than just the first 3, so a cell with 30 APWaveform sweeps shows sweeps #1, #15, #30 — easier to spot within-family drift at a glance.

**Sharing the report.** `qc_report.html` is fully self-contained — every thumbnail is base64-inlined, every CSS/JS is embedded. Email or Slack the single file to a collaborator and they can open it in any browser with **no `nwb-qc` install required**. The only feature that needs the tool installed is the "Inspect all sweeps in viewer →" links — those require `nwb-qc serve` running locally (plus the collaborator's own copy of the project YAML + the NWBs). When the viewer isn't reachable, clicking the link unfolds a banner with the exact start command instead of showing a browser error.

**Check before moving on:** the cells that show up in fail/flag actually look bad in their thumbnails. If many cells fail on a single metric that *shouldn't* be cohort-wide (e.g. `qc_protocol_coverage`), that points to a stimulus-name or threshold mismatch — handle in Step 6.

---

## Step 5 — Interactive verification (any sweep on demand)

The static report only renders the offending sweep thumbnails. To inspect any sweep of any cell, start the local viewer:

```bash
nwb-qc serve --config mydata_project.yaml
```

This starts a localhost-only HTTP server (default port 8765) and opens your browser. The viewer reads the same `qc_report.csv` for the cell list and lazy-loads sweep data from the underlying NWB files only when you click — only the bytes you ask for are decoded.

**Layout:**

- **Left** — `flag` cell list (pass/fail rows aren't surfaced here — the canonical record is `qc_report.csv`), filterable by `cell_id`.
- **Centre-left** — **sweep grid**: every voltage acquisition in the selected cell's NWB rendered as a small lazy-loaded thumbnail (~220×70 px), grouped by stimulus family. Implicated-family sweeps get a ⚠ badge and a yellow border; non-implicated sweeps are dimmed. A toggle above the grid switches between **"implicated only"** (default — matches what the triggered metrics actually point to) and **"all sweeps"** (the full ~100–200 sweeps per NWB).
- **Right** — 2 × 2 grid of plot panels. Click a sweep tile to drop the full-resolution trace into the next free slot; click "×" on a panel to clear it.

Thumbnails are rendered **on demand server-side** when a tile scrolls into view (IntersectionObserver). The pipeline doesn't pre-generate them — first hit per `(cell, sweep)` opens the NWB, decimates via LTTB, and caches the PNG both in memory (LRU of 256) and on disk under `qc_output_*/traces/viewer/`. The second visit to the same cell is instant. Full-resolution Canvas plots use the existing `/api/trace` endpoint and the LRU-cached NWB handles, so the four default panel fetches after a cell click open the NWB once.

**Deep-linking:** the static report's per-cell "Inspect all sweeps →" link uses `?cell=<cell_id>` — the viewer auto-selects that cell on load. Use this to keep the cohort triage flow tight: static report for the at-a-glance verdict + the 3-stacked thumbnail, viewer for any-sweep depth.

**Unmapped-protocols warning:** if a selected cell's implicated families have *zero* matching sweeps in the NWB (typically: your `stimulus_protocols:` mapping doesn't cover this lab's protocol names), the sweep grid shows a yellow callout pointing at the YAML's `# ⚠ UNMAPPED tokens` block and a "show all sweeps" shortcut. Fix the mapping (Step 1), rerun, and the warning disappears.

You can also hit the API directly for scripting:

```bash
curl http://127.0.0.1:8765/api/sweeps/<cell_id>
curl 'http://127.0.0.1:8765/api/trace/<cell_id>/<sweep_idx>?max_points=1000'
curl -o sweep_005.png 'http://127.0.0.1:8765/api/thumb/<cell_id>/5?w=220&h=100'
curl http://127.0.0.1:8765/api/families
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

### 6g — Inventory pre-computed metrics in your NWBs (`nwb-qc inventory-metrics`)

```bash
nwb-qc inventory-metrics --config configs/mydata_project.yaml [--n-samples 5]
```

Walks a sample of NWBs from the configured sources and inspects their `processing` / `lab_meta_data` / `scratch` / `intervals` containers. For every canonical QC metric, the output reports whether any sampled file already carries a pre-computed value (`source=nwb_processing/<module>/<name>`) or whether `nwb-trace-qc` computes it (`source=nwb-trace-qc (computed)`).

Writes a markdown report to `<output_dir>/metric_inventory.md`. Useful when onboarding a new cohort — for raw NWBs (Maria's / JY's case) every metric is computed; for cohorts that ran their own analysis pre-export, you'll see which features are already there. See also `docs/metrics_reference.md` for the full per-metric algorithm reference.

### 6f — Tune thresholds interactively (`nwb-qc tune`)

```bash
nwb-qc tune --config configs/mydata_project.yaml
```

Walks every threshold rule with cohort context. For each metric you see how many cells the rule currently affects, the cohort's P10/P50/P90/P99, and the calibrate-suggested value. At the top of the walk you can choose:

- `[w]` walk through each metric one-by-one (per-rule prompts with `[Enter]=accept suggested` / type a number to override / type `s` to skip the metric)
- `[a]` accept all suggested values in one keystroke
- `[c]` cancel without writing

After the walk, you see a preview of the new verdict counts and confirm before saving. Optionally re-runs the pipeline immediately — cache-fast (only thresholds + report re-evaluate).

Flags: `--no-rerun` to save without prompting to re-run; `--only-failing` to walk only metrics with at least one cell currently affected (faster for iterating on a noisy cohort).

The wizard's `[t]une-thresholds` option in the outcome stage calls into the same flow.

### 6e — Calibrate thresholds from the cohort itself

The bundled `default_thresholds.yaml` is calibrated for cortical/hippocampal pyramidal neurons and assumes paired stimulus traces are available (so Rs is accurate). For other cell types, or for cohorts you're trying to understand fresh, you can derive thresholds from the cohort's own metric distributions:

```bash
nwb-qc calibrate --config mydata_project.yaml
```

This reads the cache parquet (must have been written by a prior `nwb-qc run`), computes per-metric percentiles (P10 / P50 / P90 / P99), and writes a suggested-thresholds YAML next to your current one:

```
Wrote suggested thresholds: configs/mydata_thresholds_suggested.yaml
      cohort stats        : qc_output_mydata/cohort_stats.json
Next: review the suggested YAML, then point thresholds_file:
      in mydata_project.yaml at mydata_thresholds_suggested.yaml and re-run.
```

The suggester is conservative: it only **tightens** `flag_above` rules (when the cohort's P90 is below the bundled default) and only **loosens** `flag_below` rules. `fail_*` rules are never auto-suggested — those should stay laboratory-judgment calls. Every suggestion shows both the bundled default and the cohort percentiles in YAML comments so you can sanity-check before adopting.

`cohort_stats.json` is also a side-effect output: the next `nwb-qc run` finds it and decorates each triggered-metric chip in the report with cohort-percentile context (e.g. *"Cohort percentile ≈ P95; cohort range P10 -68 · P50 -65 · P90 -55"*). Run `nwb-qc calibrate` once after each substantive cohort change.

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
  model: claude-haiku-4-5          # default; opt up to sonnet/opus per-project
  api_key_env: ANTHROPIC_API_KEY
  max_borderline_cells: 100
  max_cost_usd: 1.0                # soft cap; stops vision pass when reached
  prompt_template: null            # null = bundled default
  cache_responses: true
```

**Or one-shot via flag:**

```bash
nwb-qc run --config mydata_project.yaml --with-vision --max-cost-usd 0.50
```

The default model is the cheaper Haiku tier; you can opt up by editing your YAML. `max_cost_usd` is a **soft cap**: the pipeline stops calling the vision provider as soon as the running estimated cost reaches the cap, then continues to render the report with whatever vision verdicts were already collected. Un-judged borderline cells keep their rule-based `flag` verdict (no errors). Per-run spend, token totals, and whether the cap was hit are recorded under `stages.vision` in `run_report.json`.

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
| `nwb-qc start <root>` | Guided wizard: inspect → propose config → dry-run → run → outcome, with a confirm prompt at each step. See Step 0.1. |
| `nwb-qc inspect <root>` | Read-only inventory of a wrangler-output tree (NWB counts, parquet schemas, fair2.json / README / run_state summaries, per-parquet QC-eligibility check). See Step 0.5. |
| `nwb-qc init-config <root>` | Auto-discover NWBs + parquets and write a starter project YAML and per-project thresholds YAML. |
| `nwb-qc list-cells --config <file>` | Dry-run: print discovered NWBs, dedup info, and which cells map to which dataset. |
| `nwb-qc run --config <file> [--filter dataset=X] [--with-vision/--no-vision] [--max-cost-usd N] [--report-only]` | Full pipeline. |
| `nwb-qc report --config <file>` | Re-render the HTML/CSV from the existing cache without NWB I/O. |
| `nwb-qc thresholds --config <file> --dry-run` | Show how the current thresholds would classify cached cells (counts only). |
| `nwb-qc calibrate --config <file>` | Suggest cohort-specific thresholds from cached metric distributions. Writes a `*_thresholds_suggested.yaml` you can opt into + a `cohort_stats.json` consumed by the next `run` to add percentile context to triggered chips. |
| `nwb-qc tune --config <file> [--no-rerun] [--only-failing]` | Interactive threshold-tuning walk. Top-of-walk options: `[w]` per-rule, `[a]` accept-all-suggested, `[c]` cancel. After the walk previews new verdict counts and (optionally) re-runs the pipeline — cache-fast since metrics don't recompute. |
| `nwb-qc inventory-metrics --config <file> [--n-samples N]` | Walk N (default 5) NWBs and report which canonical QC metrics they pre-compute internally vs. which `nwb-trace-qc` will compute. Writes `<output_dir>/metric_inventory.md`. See also `docs/metrics_reference.md` for the per-metric algorithm reference. |
| `nwb-qc serve --config <file> [--port N] [--no-browser]` | Interactive trace viewer — restricted to `flag` cells only; pass/fail rows stay in `qc_report.csv`. |
| `nwb-qc -v <subcmd>` | DEBUG-level logging on stderr. |
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
- **All cells flagged on `qc_protocol_coverage`** — your `stimulus_protocols` mapping doesn't match your data's protocol names. The fastest fix is to re-run `nwb-qc init-config` (or `nwb-qc start`) and read the `# ⚠ UNMAPPED tokens` block in the generated YAML header — it lists every stimulus token discovered in your NWBs that no family claims, with per-token sweep counts. Slot them into the right family under `stimulus_protocols:` and rerun. If you'd rather inspect manually: open one NWB with `pynwb`, list `acquisition.keys()`, and add the relevant tokens.
- **Absolute Rs values look unrealistic** — the Rs estimator currently assumes a nominal 50 pA test pulse (it reads only the voltage trace, not the paired stimulus current). Trust `rs_drift_pct` over `rs_mohm_final` until proper Rs from the paired stimulus is implemented; relax or remove the `fail_above` rule on `rs_mohm_final`.
- **First run feels slow** — it's I/O-bound when NWBs aren't in the OS page cache. Re-runs hit the cache and skip metric compute entirely; only manifest hashing remains.
- **Vision judge runs but no cells get queried** — only `flag`-verdict cells are sent. If your cohort is all `pass` or all `fail`, the vision judge has nothing to do; widen the borderline by loosening `flag_above` / `fail_above`.
- **`nwb-qc serve` says "manifest not found"** — you need to have run `nwb-qc run` at least once for the project. The viewer reads `_qc_manifest.parquet` and `qc_report.csv` produced by `run`.

---

See [`jy_quickstart.md`](jy_quickstart.md) for a worked example with real cohort numbers (2,302 NWBs across VPL / Red Nucleus / OBI Thalamus).
