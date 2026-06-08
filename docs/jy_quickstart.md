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

## Pipeline v0.2.0 (visual-defect metrics + vision judge + interactive viewer)

The 0.2.0 bump invalidates the per-NWB cache and recomputes everything; the next full JY run will pick up the five new visual-defect metrics:

- `rac_decay_residual_rel` — glitchy / ringing test-pulse recovery
- `vm_drift_within_sweep_mv_per_s` — within-sweep Vm slope (drifting seal)
- `ap_failure_fraction` — initiated spikes that don't reach overshoot
- `ap_amp_cv` — AP-amplitude inconsistency within a train
- `late_instability_index` — orderly firing degrading to oscillation late in a long sweep

RN smoke run on 0.2.0 (32 cells, 45 s wall-clock): the new metrics had non-trivial coverage (`rac_decay_residual_rel` on 22 of 32 cells, `vm_drift_within_sweep_mv_per_s` on 21, `ap_failure_fraction` on 19, `ap_amp_cv` on 14, `late_instability_index` on 5). The reduced cohort moved from 3-6-23 (pass-flag-fail) on 0.1.0 to 0-5-27 on 0.2.0 — the new metrics flagged a few cells previously passing.

Vision judge is off by default; enable with `--with-vision` once you've set `ANTHROPIC_API_KEY`. Interactive viewer is one command: `nwb-qc serve --config configs/jy_project.yaml`.

## Pipeline v0.3.0 – v0.6.0 changes affecting this cohort

Each version bumps `PIPELINE_VERSION` and invalidates the per-NWB cache, so the
next full run picks up everything below. Threshold rules and the metric
reference are documented in [`metrics_reference.md`](metrics_reference.md).

- **v0.3.0** — whole-cell-patch-clamp methodology fixes:
  - `rs_*` now read the paired `ics__...` stimulus current series (no longer
    assumes 50 pA), so absolute Rs is meaningful. The cohort-specific Rs
    inflation noted above is resolved.
  - `rin_mohm` is computed from the actual IV slope, not from a derived ratio.
  - `holding_current_pa` and `holding_current_drift_pa` are real (sourced from
    `spontaneous_held` family sweeps, e.g. SponHold3 / SponHold30).
  - Session-drift metrics: `vrest_session_drift_mv`, `rs_session_drift_pct`,
    `ap_overshoot_session_drift_mv`.
- **v0.4.0** — bad-ending detection and trim:
  - Changepoint on Vrest depolarisation / Rs explosion / AP overshoot collapse;
    a clean tail is excluded from metric scalars before thresholds run.
    `bad_ending_at_sweep`, `bad_ending_reason`, `n_sweeps_trimmed` are reported.
  - eFEL parity for AP-overshoot, AP-threshold, Vrest features.
- **v0.5.0** — LNMC experimenter-guidance metrics:
  - `vrest_mv` now strictly comes from `spontaneous_no_hold` family;
    `held_vm_mv` from `spontaneous_held`. The legacy `spontaneous_hold` family
    name still works but is split into the two above.
  - `ap_amplitude_mv` (peak − threshold; LNMC canonical).
  - `rs_compensation_pct` (read from NWB `IntracellularElectrode` metadata).
  - `rac_variability_pct` (CV of per-Rac Rs across reps — catches
    non-monotonic instability that `rs_drift_pct` misses).
- **v0.6.0** — tiered report:
  - Seven *critical* metrics drive the verdict; failures on *advisory* metrics
    are demoted to flag, so a single soft signal can't sink a cell. Whitelist
    is editable per project (`critical_metrics:` in the YAML).
  - Report layout: critical chips first, advisory folded under "+N advisory",
    full metric values behind an expand. Trim is surfaced via banner +
    per-sweep ✂ markers + cell-list chip.
  - Viewer sorts flag-first, accepts `--host 0.0.0.0` for one-serves-many
    network sharing.

The expected effect on the JY cohort: the v0.1.0 coverage-failure mass should
move from `fail` → `flag` (advisory demotion), absolute Rs is now meaningful
(v0.3.0 fix), and bad-ending trim eliminates spurious fails caused by the last
few sweeps of a degrading session. Run with the v0.6.0 binary and the existing
`configs/jy_project.yaml` to see current numbers.

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
