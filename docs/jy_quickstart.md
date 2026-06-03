# JY cohort — quickstart

The JY datasets (VPL + Red Nucleus + OBI Thalamus) are the first consumer of `nwb-trace-qc`. This page documents how to run them and what the inaugural results looked like, so the next run is a one-liner.

## Run

From the repo root:

```bash
nwb-qc run --config configs/jy_project.yaml
```

The config points at the absolute paths under `Henrys/JY/{VPL,RN,OBI}/` and reuses the three wrangler `*.parquet` acquisition tables. Outputs land under `qc_output_jy/` (gitignored).

For a fast smoke-test, restrict to one dataset:
```bash
nwb-qc run --config configs/jy_project.yaml --filter dataset=rn
```

After the first full run, edits to `configs/default_thresholds.yaml` or `qc_output_jy/qc_overrides.csv` re-render the report from the cache in ~6 minutes (manifest hashing) without re-opening any NWB.

## Inaugural results — pipeline v0.1.0

| | cells | pass | flag | fail | wall-clock |
|---|---:|---:|---:|---:|---:|
| VPL | 2,198 | 49 | 97 | 2,052 |  |
| Red Nucleus | 32 | 3 | 6 | 23 |  |
| OBI Thalamus | 72 | 5 | 16 | 51 |  |
| **Total** | **2,302** | **57** | **119** | **2,126** | first run ~2 h 11 m on 6 workers; cache-only re-run ~6 m 37 s |

- 0 compute errors across 2,302 NWBs.
- 35 MB self-contained HTML report at `qc_output_jy/qc_report.html`, 0 external resources.
- 1,182 trace thumbnails generated for non-pass cells (25 MB).

## Known cohort-specific tuning opportunities

The inaugural defaults flag many cells for two diagnosable cohort-specific reasons; these are notes for future threshold tweaks, not pipeline bugs.

1. **Coverage failure dominates (1,873 cells)** — most VPL recordings don't include any `SponHold*` sweeps, so `qc_protocol_coverage` legitimately fails (we lack the data for Vrest). Suggested tweak in `configs/default_thresholds.yaml`:
   ```yaml
   qc_protocol_coverage:
     flag_if_false: true   # demote from fail → flag for cells lacking the QC protocol
   ```
2. **Absolute Rs is inflated by an unknown factor.** The current Rs estimator assumes a nominal 50 pA test pulse because only the voltage trace is read. For this cohort the real test-pulse current is larger, so Rs values run hot (cohort p50 = 124 MΩ). The relative `rs_drift_pct` is unaffected and trustworthy. Suggested tweak:
   ```yaml
   rs_mohm_final:
     flag_above: 200       # cohort-relative outlier flag rather than absolute fail
   rs_drift_pct:
     fail_above: 30
     flag_above: 20
   ```
   The long-term fix is to read the paired `ics__...` current series and compute Rs properly.

## Working through the report

```bash
open qc_output_jy/qc_report.html
```

Defaults to **Fail + Flag** only — pass cells are hidden until you toggle them on. Filter strip at the top supports dataset, verdict, brain region, and cell-id search. Click any row to expand the full metric table and the inline thumbnails of the offending sweeps.

To stick a verdict from human review, append a row to `qc_output_jy/qc_overrides.csv`:

```csv
cell_id,override_verdict,note,reviewer,date
JY160222_A_3,pass,visually inspected — overshoot loss is an end-of-recording artefact,cg,2026-06-02
```

Overrides survive re-runs and threshold changes. The expand-row panel in the HTML shows a one-click copyable template per row.
