#!/usr/bin/env python3
"""Generate docs/metrics_reference.md from the canonical metric table.

Run once after editing `families.METRIC_DESCRIPTIONS` or the metric→eFEL
mapping. The output doc lives at `docs/metrics_reference.md`.

  python scripts/build_metrics_reference.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from nwb_trace_qc import PIPELINE_VERSION
from nwb_trace_qc.families import METRIC_DESCRIPTIONS, METRIC_TO_FAMILY


# Each entry: (eFEL feature name, "used directly" or "derived")
# Mirrors the table in metrics.py's _efel_or_fallback_* helpers.
METRIC_TO_EFEL: dict[str, tuple[str, str]] = {
    "vrest_mv":                ("voltage_base", "used directly (on spontaneous_no_hold sweeps)"),
    "held_vm_mv":              ("voltage_base", "used directly (on spontaneous_held sweeps)"),
    "ap_amp_overshoot_mv":     ("AP_amplitude_from_voltagebase", "derived (vbase + median(AP_amp_from_vbase))"),
    "ap_amp_overshoot_min_mv": ("AP_amplitude_from_voltagebase", "derived (min across sweeps)"),
    "ap_amplitude_mv":         ("AP_amplitude", "used directly (canonical: peak − threshold)"),
    "ap_threshold_drift_mv":   ("AP_begin_voltage", "used directly"),
    "ap_amp_cv":               ("AP_amplitude", "derived (std/|mean| over per-spike amplitudes)"),
    "ap_failure_fraction":     ("Spikecount", "derived (dV/dt initiations vs. Spikecount)"),
}

CUSTOM_ALGORITHMS: dict[str, str] = {
    "vrest_drift_mv": "first - last in chronological order across spontaneous_hold sweeps; emitted in mV.",
    "vrest_session_drift_mv": "median(Vrest in second half) − median(Vrest in first half) of the session.",
    "baseline_rms_mv": "RMS noise of the centred voltage in spontaneous_hold sweeps; median across sweeps.",
    "rs_mohm_initial": "Access resistance from first test_pulse sweep (ΔV/ΔI in MΩ).",
    "rs_mohm_final":   "Access resistance from last test_pulse sweep.",
    "rs_drift_pct":    "(Rs_final − Rs_initial) / Rs_initial × 100%.",
    "rs_session_drift_pct": "Median Rs in 2nd half vs 1st half, expressed as % of 1st half.",
    "rin_mohm": "Linear regression of (V, I) pairs from iv_subthreshold sweeps (1000× mV/pA → MΩ).",
    "rin_r2": "R² of the same Rin linear fit.",
    "holding_current_pa": "Median across sweeps of the mean baseline current (pre-step, 5 ms) in pA.",
    "holding_current_drift_pa": "Last-sweep baseline − first-sweep baseline (pA).",
    "rac_decay_residual_rel": "Relative residual of an exponential fit to the test-pulse recovery.",
    "vm_drift_within_sweep_mv_per_s": "Max within-sweep linear regression slope (spontaneous_hold).",
    "late_instability_index": "Max ratio of late-quartile vs early-quartile rate/variance within a sweep, minus 1.",
    "test_pulse_edge_overshoot_mv": "Peak |dV| in 0–10 ms after a test-pulse edge minus the 20–50 ms plateau.",
    "ap_overshoot_session_drift_mv": "Median AP overshoot (2nd half) − median (1st half).",
    "ap_amp_attenuation_frac": "Fraction of detected APs whose individual overshoot < 15 mV.",
    "rs_compensation_pct": "Read from the NWB's IntracellularElectrode.resistance_comp_correction "
                            "(or a lab_meta_data 'Rs' field). 0..1 fractions are normalised to 0..100%.",
    "rac_variability_pct": "CV (std/median × 100) of per-Rac Rs estimates across the test_pulse sweeps; "
                            "needs ≥3 Rac sweeps. Catches non-monotonic Rs instability that rs_drift_pct misses.",
    "qc_protocol_coverage": "Boolean — does the NWB carry ≥1 sweep in each essential family (spontaneous + test_pulse + ap_waveform).",
    "n_sweeps_total": "Count of voltage acquisitions iterated by the metric pass.",
    "n_spikes_total": "Sum across ap_waveform + rest_firing sweeps of dV/dt-detected initiations "
                       "that reach ≥ 0 mV (successful APs). Excludes trimmed sweeps.",
    "n_sweeps_clipped": "Sweeps that touched the voltage rails (±150 / +80 mV) for ≥1 ms.",
    "n_sweeps_nan": "Sweeps containing NaN samples.",
    "bad_ending_at_sweep": "First chronologically-ordered sweep index where the recording degraded "
                            "(Vrest depolarised >10 mV vs running median, OR Rs > 1.75× running median, "
                            "OR AP overshoot < 10 mV after a healthy run). Guarded: ignored if it falls "
                            "in the first 30% or last 5% of the session.",
    "n_sweeps_trimmed": "Tail sweeps excluded from metric reductions due to bad-ending detection.",
    "bad_ending_reason": "vrest_depolarisation / rs_explosion / ap_collapse — the channel that fired first.",
}


def render() -> str:
    lines: list[str] = []
    lines.append(f"# `nwb-trace-qc` metric reference (v{PIPELINE_VERSION})")
    lines.append("")
    lines.append(dedent("""
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
    """).strip())
    lines.append("")

    for metric in sorted(METRIC_DESCRIPTIONS.keys()):
        desc = METRIC_DESCRIPTIONS[metric]
        family = METRIC_TO_FAMILY.get(metric, "—")
        lines.append(f"## `{metric}`")
        lines.append("")
        if desc.get("what"):
            lines.append(f"- **What it measures.** {desc['what']}")
        if desc.get("healthy"):
            lines.append(f"- **Healthy range.** {desc['healthy']}")
        lines.append(f"- **Implicated family.** `{family}`")

        if metric in METRIC_TO_EFEL:
            feature, kind = METRIC_TO_EFEL[metric]
            lines.append(f"- **Computed by.** eFEL feature `{feature}` ({kind}). "
                         f"Falls back to our custom helper on malformed sweeps.")
        elif metric in CUSTOM_ALGORITHMS:
            lines.append(f"- **Computed by.** `nwb-trace-qc` — {CUSTOM_ALGORITHMS[metric]}")
        else:
            lines.append("- **Computed by.** `nwb-trace-qc` (custom).")

        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "docs" / "metrics_reference.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    out_path.write_text(text)
    print(f"Wrote {out_path}  ({len(text.splitlines())} lines)")
