# `nwb-trace-qc` metric reference (v0.4.0)

For every metric `nwb-trace-qc` emits, this doc says (a) what the metric
measures, (b) the healthy range, (c) the implicated stimulus family,
(d) where the value comes from — an eFEL feature or our own helper —
and (e) the default threshold rule (see `configs/default_thresholds.yaml`).

**eFEL is used as the canonical source** for the AP and Vrest features
where it has a direct equivalent. Our custom helpers stay as fallbacks
(when eFEL refuses on a malformed sweep, the cell's `n_efel_fallback_sweeps`
counter increments). Rs, Rin, holding current, the visual-defect metrics,
and the recording-trim diagnostics have no eFEL equivalent and are
always sourced from our compute.

Regenerate this doc with `python scripts/build_metrics_reference.py`.

## `ap_amp_attenuation_frac`

- **What it measures.** Fraction of all detected APs with overshoot < 15 mV.
- **Healthy range.** < 10% — sustained attenuation indicates cell decline.
- **Implicated family.** `ap_waveform`
- **Computed by.** `nwb-trace-qc` — Fraction of detected APs whose individual overshoot < 15 mV.

## `ap_amp_cv`

- **What it measures.** Coefficient of variation of AP peak amplitudes within a sweep.
- **Healthy range.** < 0.10 ideal, < 0.20 acceptable.
- **Implicated family.** `ap_waveform`
- **Computed by.** eFEL feature `AP_amplitude` (derived (std/|mean| over per-spike amplitudes)). Falls back to our custom helper on malformed sweeps.

## `ap_amp_overshoot_min_mv`

- **What it measures.** Worst-case AP overshoot across sweeps — catches sporadic attenuation.
- **Healthy range.** > 10 mV; < 0 mV is failure.
- **Implicated family.** `ap_waveform`
- **Computed by.** eFEL feature `AP_amplitude_from_voltagebase` (derived (min across sweeps)). Falls back to our custom helper on malformed sweeps.

## `ap_amp_overshoot_mv`

- **What it measures.** Median AP peak above 0 mV across all ap_waveform / rest_firing sweeps.
- **Healthy range.** +20 to +40 mV (cortical pyramidal); ≥10 mV minimally acceptable.
- **Implicated family.** `ap_waveform`
- **Computed by.** eFEL feature `AP_amplitude_from_voltagebase` (derived (vbase + median(AP_amp_from_vbase))). Falls back to our custom helper on malformed sweeps.

## `ap_failure_fraction`

- **What it measures.** Fraction of dV/dt spike initiations that fail to reach overshoot.
- **Healthy range.** ≈ 0 in healthy cells.
- **Implicated family.** `ap_waveform`
- **Computed by.** eFEL feature `Spikecount` (derived (dV/dt initiations vs. Spikecount)). Falls back to our custom helper on malformed sweeps.

## `ap_overshoot_session_drift_mv`

- **What it measures.** AP overshoot delta between 2nd half and 1st half (medians).
- **Healthy range.** > −10 mV (i.e. < 10 mV drop). Sharp drop = cell dying.
- **Implicated family.** `ap_waveform`
- **Computed by.** `nwb-trace-qc` — Median AP overshoot (2nd half) − median (1st half).

## `ap_threshold_drift_mv`

- **What it measures.** Voltage delta between first and last detected AP threshold.
- **Healthy range.** |Δ| < 5 mV.
- **Implicated family.** `ap_waveform`
- **Computed by.** eFEL feature `AP_begin_voltage` (used directly). Falls back to our custom helper on malformed sweeps.

## `bad_ending_at_sweep`

- **What it measures.** First sweep index where the recording started to degrade (Vrest depolarised, Rs exploded, or AP overshoot collapsed). Everything ≥ this index is excluded from the metric scalars.
- **Healthy range.** NaN — recording ended cleanly.
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — First chronologically-ordered sweep index where the recording degraded (Vrest depolarised >10 mV vs running median, OR Rs > 1.75× running median, OR AP overshoot < 10 mV after a healthy run). Guarded: ignored if it falls in the first 30% or last 5% of the session.

## `bad_ending_reason`

- **What it measures.** Why the recording was trimmed: vrest_depolarisation / rs_explosion / ap_collapse.
- **Healthy range.** None (recording ended cleanly).
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — vrest_depolarisation / rs_explosion / ap_collapse — the channel that fired first.

## `baseline_rms_mv`

- **What it measures.** RMS noise of the spontaneous_hold baseline voltage.
- **Healthy range.** < 1 mV (mainly electrode/amplifier coupling).
- **Implicated family.** `spontaneous_hold`
- **Computed by.** `nwb-trace-qc` — RMS noise of the centred voltage in spontaneous_hold sweeps; median across sweeps.

## `holding_current_drift_pa`

- **What it measures.** Change in holding current from first to last sweep.
- **Healthy range.** |Δ| < 50 pA — creeping demand signals seal leak.
- **Implicated family.** `spontaneous_hold`
- **Computed by.** `nwb-trace-qc` — Last-sweep baseline − first-sweep baseline (pA).

## `holding_current_pa`

- **What it measures.** Baseline (pre-step) holding current; reflects seal quality.
- **Healthy range.** |Ihld| < 100 pA at Vrest.
- **Implicated family.** `spontaneous_hold`
- **Computed by.** `nwb-trace-qc` — Median across sweeps of the mean baseline current (pre-step, 5 ms) in pA.

## `late_instability_index`

- **What it measures.** Max ratio of late-quartile-to-early-quartile activity within a sweep, minus 1.
- **Healthy range.** ≈ 0; > 1 = late-sweep runaway oscillation / firing.
- **Implicated family.** `rest_firing`
- **Computed by.** `nwb-trace-qc` — Max ratio of late-quartile vs early-quartile rate/variance within a sweep, minus 1.

## `n_sweeps_clipped`

- **What it measures.** Sweeps that hit the voltage rails (±150 / +80 mV) for ≥1 ms.
- **Healthy range.** 0 — any clipped sweep is suspect.
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — Sweeps that touched the voltage rails (±150 / +80 mV) for ≥1 ms.

## `n_sweeps_nan`

- **What it measures.** Sweeps containing NaN samples.
- **Healthy range.** 0.
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — Sweeps containing NaN samples.

## `n_sweeps_total`

- **What it measures.** Number of voltage acquisitions in the NWB.
- **Healthy range.** > 10 for a useful recording.
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — Count of voltage acquisitions iterated by the metric pass.

## `n_sweeps_trimmed`

- **What it measures.** How many tail sweeps were excluded from metric scalars due to bad-ending detection.
- **Healthy range.** 0.
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — Tail sweeps excluded from metric reductions due to bad-ending detection.

## `qc_protocol_coverage`

- **What it measures.** Boolean: NWB carries at least one sweep from each essential family (spontaneous_hold, test_pulse, ap_waveform).
- **Healthy range.** True. False ⇒ stimulus_protocols mapping incomplete or recording cut short.
- **Implicated family.** `—`
- **Computed by.** `nwb-trace-qc` — Boolean — does the NWB carry ≥1 sweep in each essential family.

## `rac_decay_residual_rel`

- **What it measures.** Relative residual of an exponential fit to the test-pulse recovery.
- **Healthy range.** < 0.05; > 0.15 indicates ringing/glitches.
- **Implicated family.** `test_pulse`
- **Computed by.** `nwb-trace-qc` — Relative residual of an exponential fit to the test-pulse recovery.

## `rin_mohm`

- **What it measures.** Input resistance from subthreshold IV slope (V = Rin·I + offset).
- **Healthy range.** 50–150 MΩ for cortical pyramidal.
- **Implicated family.** `iv_subthreshold`
- **Computed by.** `nwb-trace-qc` — Linear regression of (V, I) pairs from iv_subthreshold sweeps (1000× mV/pA → MΩ).

## `rin_r2`

- **What it measures.** R² of the linear fit used to derive Rin.
- **Healthy range.** > 0.9 (clean linear region).
- **Implicated family.** `iv_subthreshold`
- **Computed by.** `nwb-trace-qc` — R² of the same Rin linear fit.

## `rs_drift_pct`

- **What it measures.** Rs drift between first and last test-pulse sweep, as % of initial.
- **Healthy range.** < 20% over a typical session.
- **Implicated family.** `test_pulse`
- **Computed by.** `nwb-trace-qc` — (Rs_final − Rs_initial) / Rs_initial × 100%.

## `rs_mohm_final`

- **What it measures.** Access resistance at the last test-pulse sweep.
- **Healthy range.** 10–25 MΩ; ≤30 MΩ acceptable.
- **Implicated family.** `test_pulse`
- **Computed by.** `nwb-trace-qc` — Access resistance from last test_pulse sweep.

## `rs_mohm_initial`

- **What it measures.** Access resistance at the first test-pulse sweep.
- **Healthy range.** 10–25 MΩ for somatic patch.
- **Implicated family.** `test_pulse`
- **Computed by.** `nwb-trace-qc` — Access resistance from first test_pulse sweep (ΔV/ΔI in MΩ).

## `rs_session_drift_pct`

- **What it measures.** Rs delta between 2nd half and 1st half (medians), as % of 1st-half.
- **Healthy range.** < 25% — seal stability check.
- **Implicated family.** `test_pulse`
- **Computed by.** `nwb-trace-qc` — Median Rs in 2nd half vs 1st half, expressed as % of 1st half.

## `test_pulse_edge_overshoot_mv`

- **What it measures.** Max peak deviation in 5–10 ms after a test-pulse edge vs the 20–50 ms plateau.
- **Healthy range.** < 5 mV (smooth settling); > 20 mV = capacitive ringing / bad compensation.
- **Implicated family.** `test_pulse`
- **Computed by.** `nwb-trace-qc` — Peak |dV| in 0–10 ms after a test-pulse edge minus the 20–50 ms plateau.

## `vm_drift_within_sweep_mv_per_s`

- **What it measures.** Max within-sweep Vm slope from spontaneous_hold (linear regression).
- **Healthy range.** < 0.5 mV/s; > 2 mV/s indicates active seal drift.
- **Implicated family.** `rest_firing`
- **Computed by.** `nwb-trace-qc` — Max within-sweep linear regression slope (spontaneous_hold).

## `vrest_drift_mv`

- **What it measures.** Vrest delta from first to last spontaneous_hold sweep.
- **Healthy range.** |Δ| < 5 mV across a healthy recording.
- **Implicated family.** `spontaneous_hold`
- **Computed by.** `nwb-trace-qc` — first - last in chronological order across spontaneous_hold sweeps; emitted in mV.

## `vrest_mv`

- **What it measures.** Resting membrane potential.
- **Healthy range.** −65 to −80 mV (cortical pyramidal); −55 to −90 mV more broadly.
- **Implicated family.** `spontaneous_hold`
- **Computed by.** eFEL feature `voltage_base` (used directly). Falls back to our custom helper on malformed sweeps.

## `vrest_session_drift_mv`

- **What it measures.** Vrest delta between 2nd half and 1st half of the session (medians).
- **Healthy range.** |Δ| < 5 mV — early-warning seal degradation otherwise.
- **Implicated family.** `spontaneous_hold`
- **Computed by.** `nwb-trace-qc` — median(Vrest in second half) − median(Vrest in first half) of the session.
