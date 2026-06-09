# nwb-trace-qc

Cohort-scale, configurable quality control for patch-clamp electrophysiology
traces stored in NWB (icephys). Designed for large cohorts where opening every
cell by hand is impractical: compute standardized health metrics once per NWB,
classify each cell as **pass / flag / fail** against editable thresholds, and
surface only the cells that need human attention in a self-contained HTML report
plus an interactive sweep viewer.

Current pipeline version: **0.8.0** — viewer-driven curation.

## What it computes

For every NWB file the pipeline derives the metrics below. Each metric documents
its stimulus family, its eFEL/in-house provenance, and its default healthy
range in [`docs/metrics_reference.md`](docs/metrics_reference.md).

| Family | Metrics |
|---|---|
| Resting state (`spontaneous_no_hold`) | `vrest_mv`, `vrest_drift_mv`, `vrest_session_drift_mv`, `baseline_rms_mv` |
| Held membrane (`spontaneous_held`) | `held_vm_mv`, `holding_current_pa`, `holding_current_drift_pa` |
| Access (`test_pulse`) | `rs_mohm_initial`, `rs_mohm_final`, `rs_drift_pct`, `rs_session_drift_pct`, `rs_compensation_pct`, `rac_variability_pct`, `rac_decay_residual_rel`, `test_pulse_edge_overshoot_mv` |
| Input (`iv_subthreshold`) | `rin_mohm`, `rin_r2` |
| Action potential (`ap_waveform`, `rest_firing`) | `ap_amplitude_mv` (LNMC peak−threshold), `ap_amp_overshoot_mv`, `ap_amp_overshoot_min_mv`, `ap_amp_attenuation_frac`, `ap_threshold_drift_mv`, `ap_overshoot_session_drift_mv`, `ap_amp_cv`, `ap_failure_fraction`, `n_spikes_total`, `late_instability_index`, `vm_drift_within_sweep_mv_per_s` |
| Recording integrity | `bad_ending_at_sweep`, `bad_ending_reason`, `n_sweeps_trimmed`, `n_sweeps_total`, `n_sweeps_clipped`, `n_sweeps_nan` |
| Coverage | `qc_protocol_coverage` |

Stimulus *families* are the configurable layer — your lab can use whatever
protocol names you want; map them in your project YAML and the pipeline routes
sweeps accordingly. **eFEL** is used as the canonical source for the AP and
Vrest features where a direct equivalent exists; in-house helpers stay as
fallbacks for malformed sweeps (the per-cell `n_efel_fallback_sweeps` counter
flags how often that happened).

## Install

```bash
pip install -e .
```

Optional but recommended: a dedicated venv. The defaults install `anthropic` and
`openai` clients so the optional vision-judge stage works out of the box once
you set the corresponding API-key env var.

## Quick start — guided

```bash
nwb-qc start /path/to/your/data
```

The wizard walks you through six stages: **Inspect → Propose config → Review
thresholds → Dry-run → Run → Outcome**. It detects unmapped stimulus tokens
during scan, lets you accept all heuristic mappings in one keystroke, and
surfaces the report + interactive viewer at the end. A typical session looks
like this:

```
$ nwb-qc start ~/Pilot-Tests/datasets/Henrys/JY/single_cell_dataset

══════════════════════════════════════════════════════════════════
STAGE 1/6 · Inspect  →  ~/Pilot-Tests/datasets/Henrys/JY/single_cell_dataset
──────────────────────────────────────────────────────────────────
Found 3 NWB files (1.2 GB total) · scanning stimulus tokens…
  ✓ 14 known tokens mapped to families
  ⚠ 6 unmapped tokens:  SetAmpl  RPip  RSealOpen  RSealClose
                         SineSpec  PosCheops
  hint: SetAmpl looks like test_pulse (heuristic)
        RPip/RSealOpen/RSealClose look like seal-test (skip)

unmapped tokens — pick an action:
  [w] walk through each (one prompt per token)
  [a] accept all heuristic-suggested mappings in one keystroke
  [c] cancel (leave YAML unchanged)
action [w]: a

  ✓ updated single_cell_dataset_project.yaml: assigned 4 token(s); 2 still unmapped

══════════════════════════════════════════════════════════════════
STAGE 2/6 · Propose config  →  ./single_cell_dataset_project.yaml
──────────────────────────────────────────────────────────────────
[YAML preview]
review ([a]ccept/[e]dit/[q]uit) [a]: a

══════════════════════════════════════════════════════════════════
STAGE 3/6 · Review thresholds  →  ./single_cell_dataset_thresholds.yaml
──────────────────────────────────────────────────────────────────
[bundled defaults preview — edit now or tune from cohort data after the run]
review ([a]ccept/[e]dit/[q]uit) [a]: a

══════════════════════════════════════════════════════════════════
STAGE 4/6 · Dry-run  →  3 cells in 1 dataset
──────────────────────────────────────────────────────────────────
proceed to full run? [Y/n]: y

══════════════════════════════════════════════════════════════════
STAGE 5/6 · Run
──────────────────────────────────────────────────────────────────
[manifest_build]    ████████████████████████  3/3   elapsed 0.4s
[metric_compute]    ████████████████████████  3/3   elapsed 22.1s
[thresholds]        ████████████████████████  3/3   elapsed 0.1s
[thumbnails]        ████████████████████████  3/3   elapsed 3.8s
[report]            ████████████████████████        elapsed 0.5s
  → qc_output_single_cell_dataset/qc_report.html  (412 KB · curation log)

══════════════════════════════════════════════════════════════════
STAGE 6/6 · Outcome  ·  0 pass · 2 flag · 1 fail
──────────────────────────────────────────────────────────────────
  [s] serve the interactive viewer (curate cells here — primary path)
  [o] open the static curation log (qc_report.html — queue + decisions)
  [t] tune thresholds from cohort percentiles + re-render
  [d] done
next [s]: s
  starting viewer at http://127.0.0.1:8765/ …
```

You can re-enter the wizard at any time with `nwb-qc start <root>` — it's
idempotent (won't clobber existing config/thresholds without your
`[e]dit`-then-save) and the wrangler cache means re-runs are seconds, not
hours.

## Quick start — manual

```bash
# 1. Auto-discover NWBs (and any wrangler parquets) under a root path
#    and pre-fill a project YAML you can review.
nwb-qc init-config /path/to/your/data
#   -> writes ./<root>_project.yaml and ./<root>_thresholds.yaml

# 2. Dry-run to see what will be processed (no NWB reads beyond a hash).
nwb-qc list-cells --config <root>_project.yaml

# 3. First real run. Computes everything, caches per-NWB-sha256.
nwb-qc run --config <root>_project.yaml

# 4. Open the report.
open qc_output/qc_report.html

# 5. Drill into the flagged sweeps interactively.
nwb-qc serve --config <root>_project.yaml
```

See [`docs/usage.md`](docs/usage.md) for the full step-by-step walkthrough,
threshold tuning, cohort calibration, the override loop, and network-sharing
the viewer.

## Design principles

- **General.** Every dataset-specific assumption (paths, filenames, stimulus
  protocol names, key columns, thresholds) lives in YAML. The same binary runs
  on any cohort.
- **No recomputation.** The cache is keyed by NWB content sha256 + pipeline
  version. A second run over an unchanged corpus reads only the cache (seconds,
  not hours).
- **Reuse what exists.** If you already have wrangler-style acquisition
  parquets (`stimulus_type`, `rate_hz`, `n_samples`, …), point the config at
  them and the pipeline reads them instead of re-deriving from NWB.
- **Critical vs advisory metrics.** Only the seven critical metrics (Rs drift,
  Vrest, AP overshoot, AP amplitude, holding-current drift, protocol coverage,
  signal hygiene) cascade pass/flag/fail to the cell verdict. Failures on
  advisory metrics are demoted to *flag* so a single soft signal doesn't sink a
  cell. The whitelist is editable in your project YAML.
- **Bad-ending trim.** Sessions where Vrest depolarises, Rs explodes, or AP
  overshoot collapses get the bad tail automatically trimmed from the metric
  scalars. The report and viewer both make the trim point visible (banner,
  per-sweep ✂ marker, cell-list chip).
- **Viewer-driven curation (v0.8.0).** The auto-pipeline produces a triage
  *queue*; the human decision lives in the viewer. A Decision block next to
  each cell lets you tag PASS / FLAG / FAIL with a reason — your name and
  the date stamp automatically (curator name comes from `curator:` in your
  project YAML, falling back to a one-shot prompt). Decisions persist to
  `qc_overrides.csv` (same file the pipeline has always used) so they
  survive re-runs and threshold edits.
- **Curation log as the report.** `qc_report.html` is now a curation log,
  not a thumbnail dashboard: two sections — *Awaiting review* (cells the
  curator hasn't touched yet, sorted fail → flag) and *Curated* (cells
  with a saved decision, showing the curator, date, and reason). Sweep
  exploration happens in the viewer where the overlay plot + family
  toggles + zoom already work well, so the report dropped its per-cell
  thumbnail grid; file size goes from ~10–50 MB down to under ~500 KB,
  making it actually shareable.
- **Editable thresholds + trim, live in the viewer (v0.7.0).** Every headline
  metric in the viewer has an inline pencil icon to edit its pass/flag/fail
  bounds; saving writes a `threshold_overrides.yaml` overlay that the next
  `nwb-qc run` picks up automatically. A trim slider above the overlay plot
  lets you adjust where a recording is cut; dragging triggers a live
  recompute so the headline metrics update in place, and "save trim" persists
  the cutoff to `qc_trim_overrides.csv`.

## Configuration

See [`configs/example_project.yaml`](configs/example_project.yaml) for a fully
commented template and
[`configs/default_thresholds.yaml`](configs/default_thresholds.yaml) for the
default pass/flag/fail ranges.

## CLI

| Command | What it does |
|---|---|
| `nwb-qc start <root>` | Guided 6-stage wizard: inspect → propose config → review thresholds → dry-run → run → outcome menu |
| `nwb-qc init-config <root>` | Non-interactive: auto-discover NWBs + parquets, emit a starter project YAML and per-project thresholds YAML |
| `nwb-qc list-cells --config <file>` | Dry-run: print discovered NWBs, dedup info, and which cells map to which dataset |
| `nwb-qc inventory --config <file>` | Stimulus-token inventory and family-coverage map for the cohort |
| `nwb-qc run --config <file>` | Full pipeline: hash → cache lookup → metric compute (only for new NWBs) → verdicts → report |
| `nwb-qc run --config <file> --filter dataset=NAME` | Restrict to one logical dataset (handy for smoke tests) |
| `nwb-qc run --config <file> --with-vision` | Also run the optional LLM vision judge on flagged sweeps (needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) |
| `nwb-qc report --config <file>` | Re-render the HTML/CSV from the existing cache (no NWB I/O) |
| `nwb-qc tune --config <file>` | Cohort-percentile-aware interactive threshold walk |
| `nwb-qc calibrate --config <file>` | Suggest threshold ranges from cohort-level percentiles |
| `nwb-qc thresholds --config <file> --dry-run` | Show how a threshold edit would change verdict counts without writing anything |
| `nwb-qc serve --config <file> [--host H] [--port N]` | Interactive sweep viewer for non-pass cells. `--host 0.0.0.0` for network sharing |

## Project layout

```
src/nwb_trace_qc/   library + CLI
configs/            example + default YAMLs (plus per-project configs as they accumulate)
tests/              pytest suite + tiny synthetic NWB fixtures (170 tests)
docs/               long-form usage + metric reference
scripts/            developer utilities (e.g. build_metrics_reference.py)
```

## Documentation

- [`docs/usage.md`](docs/usage.md) — full walkthrough: wizard, manual runs,
  thresholds, calibration, vision judge, viewer, network sharing, overrides.
- [`docs/metrics_reference.md`](docs/metrics_reference.md) — per-metric: what
  it measures, healthy range, stimulus family, provenance (eFEL vs in-house).
  Regenerate with `python scripts/build_metrics_reference.py`.
- [`docs/jy_quickstart.md`](docs/jy_quickstart.md) — worked example on the
  JY cohort (VPL + Red Nucleus + OBI Thalamus) with real cell counts and
  cohort-specific tuning notes.

## License

MIT — see [`LICENSE`](LICENSE).
