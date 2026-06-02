# nwb-trace-qc

Cohort-scale, configurable quality control for patch-clamp electrophysiology traces stored in NWB (icephys). Designed for large cohorts where opening every cell by hand is impractical: compute standardized health metrics once per NWB, classify each cell as **pass / flag / fail** against editable thresholds, and surface only the cells that need human attention in a single self-contained HTML report.

## What it computes

For every NWB file the pipeline derives:

| Family | Metric | Stimulus family used |
|---|---|---|
| Resting state | `vrest_mv`, `vrest_drift_mv` | `spontaneous_hold` |
| Access | `rs_mohm_initial`, `rs_mohm_final`, `rs_drift_pct` | `test_pulse` |
| Input | `rin_mohm` | `iv_subthreshold` |
| Action potential | `ap_amp_overshoot_mv`, `ap_threshold_drift_mv` | `ap_waveform`, `rest_firing` |
| Signal hygiene | `baseline_rms_mv`, `n_sweeps_clipped`, `n_sweeps_nan` | any |
| Coverage | `qc_protocol_coverage` (bool) | meta |

Stimulus *families* are the configurable layer — your lab can use whatever protocol names you want; map them in your project YAML.

## Install

```bash
pip install -e .
```

Optional but recommended: a dedicated venv.

## Quick start

```bash
# 1. Auto-discover NWBs and any wrangler parquets under a root path,
#    and pre-fill a project YAML you can review.
nwb-qc init-config /path/to/your/data
#   -> writes ./<root_basename>_project.yaml and ./<root_basename>_thresholds.yaml

# 2. Dry-run to see what will be processed (no NWB reads beyond a hash).
nwb-qc list-cells --config <root>_project.yaml

# 3. First real run. Computes everything, caches per-NWB-sha256.
nwb-qc run --config <root>_project.yaml

# 4. Open the report.
open qc_output/qc_report.html
```

## Design principles

- **General**: every dataset-specific assumption (paths, filenames, stimulus protocol names, key columns, thresholds) lives in YAML. The same binary runs on any cohort.
- **No recomputation**: the cache is keyed by NWB content sha256. A second run over an unchanged corpus reads only the cache (seconds, not hours).
- **Reuse what exists**: if you already have wrangler-style acquisition parquets (`stimulus_type`, `rate_hz`, `n_samples`, …), point the config at them and the pipeline reads them instead of re-deriving from NWB.
- **Human-first triage**: the HTML report defaults to flag+fail only, so 90% of cells stay invisible to the reviewer. Each flagged cell shows the specific metrics that triggered and inline trace thumbnails of the offending sweeps.
- **Sticky overrides**: `qc_overrides.csv` is the only file humans edit; verdict overrides survive re-runs, threshold edits, and pipeline upgrades.

## Configuration

See [`configs/example_project.yaml`](configs/example_project.yaml) for a fully commented template and [`configs/default_thresholds.yaml`](configs/default_thresholds.yaml) for the default pass/flag/fail ranges.

## CLI

| Command | What it does |
|---|---|
| `nwb-qc init-config <root>` | Auto-discover NWBs + parquets, emit a starter project YAML and per-project thresholds YAML |
| `nwb-qc list-cells --config <file>` | Dry-run: print discovered NWBs, dedup info, and which cells map to which dataset |
| `nwb-qc run --config <file>` | Full pipeline: hash → cache lookup → metric compute (only for new NWBs) → verdicts → report |
| `nwb-qc run --config <file> --filter dataset=NAME` | Restrict to one logical dataset (handy for smoke tests) |
| `nwb-qc report --config <file>` | Re-render the HTML/CSV from the existing cache (no NWB I/O) |
| `nwb-qc thresholds --config <file> --dry-run` | Show how a threshold edit would change verdict counts without writing anything |

## Project layout

```
src/nwb_trace_qc/   library + CLI
configs/            example + default YAMLs (plus per-project configs as they accumulate)
tests/              pytest suite + tiny synthetic NWB fixtures
docs/               longer-form usage docs
```

## Status

Initial scaffold. See [`docs/usage.md`](docs/usage.md) for the longer-form walkthrough.
