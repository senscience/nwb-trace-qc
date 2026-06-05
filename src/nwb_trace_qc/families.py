"""Map QC metrics → stimulus families they were derived from.

Triagers need to know "vrest_mv is a spontaneous_hold metric, so I should look at
spontaneous_hold sweeps to verify" without memorizing the table. This module
exposes that mapping in one place, used by:

- pipeline._make_thumbnail (which sweeps to render for a flagged cell)
- report.py (per-cell expand panel: triggered-metric chips + "Inspect these families:")
- server.py (/api/families endpoint; the viewer filters its sweep grid by implicated set)

Keep the table flat — a metric maps to exactly one family (the *primary* one).
If a metric is genuinely multi-family in the future, switch to set values.
"""
from __future__ import annotations

from typing import Iterable


# v0.6.0: which metrics' fails should cascade to a cell-level fail. Everything
# outside this set is advisory — its fail verdict is demoted to flag at the
# cell level. Calibrated for LNMC eCode-protocol cohorts; override per-project
# via the `critical_metrics:` list in the project YAML.
DEFAULT_CRITICAL_METRICS: frozenset[str] = frozenset({
    "rs_drift_pct",             # access-resistance stability over the session
    "vrest_mv",                 # true resting membrane potential
    "ap_amp_overshoot_mv",      # AP peak above 0 mV
    "ap_amplitude_mv",          # canonical AP amplitude (peak − threshold)
    "holding_current_drift_pa", # seal stability (orthogonal to Rs)
    "qc_protocol_coverage",     # missing essential families → can't QC at all
    "n_sweeps_clipped",         # voltage rails hit → trace corrupted
    "n_sweeps_nan",             # NaN samples → trace corrupted
})


METRIC_TO_FAMILY: dict[str, str] = {
    # Spontaneous-derived. v0.5.0 splits the family into:
    #   spontaneous_no_hold — true resting membrane potential (no current injected)
    #   spontaneous_held    — held under holding current (different semantic)
    "vrest_mv":                       "spontaneous_no_hold",
    "vrest_drift_mv":                 "spontaneous_no_hold",
    "vrest_session_drift_mv":         "spontaneous_no_hold",
    "held_vm_mv":                     "spontaneous_held",
    "baseline_rms_mv":                "spontaneous_no_hold",
    "holding_current_pa":             "spontaneous_held",
    "holding_current_drift_pa":       "spontaneous_held",
    # Test-pulse-derived (Rs, decay shape, edge artifact)
    "rs_mohm_initial":                "test_pulse",
    "rs_mohm_final":                  "test_pulse",
    "rs_drift_pct":                   "test_pulse",
    "rs_session_drift_pct":           "test_pulse",
    "rs_compensation_pct":            "test_pulse",
    "rac_variability_pct":            "test_pulse",
    "rac_decay_residual_rel":         "test_pulse",
    "test_pulse_edge_overshoot_mv":   "test_pulse",
    # IV-derived
    "rin_mohm":                       "iv_subthreshold",
    "rin_r2":                         "iv_subthreshold",
    # AP-waveform-derived
    "ap_amp_overshoot_mv":            "ap_waveform",
    "ap_amp_overshoot_min_mv":        "ap_waveform",
    "ap_amp_attenuation_frac":        "ap_waveform",
    "ap_overshoot_session_drift_mv":  "ap_waveform",
    "ap_threshold_drift_mv":          "ap_waveform",
    "ap_amplitude_mv":                "ap_waveform",
    "ap_amp_cv":                      "ap_waveform",
    "ap_failure_fraction":            "ap_waveform",
    # Long-sweep / firing-train metrics
    "vm_drift_within_sweep_mv_per_s": "rest_firing",
    "late_instability_index":         "rest_firing",
}


# Short human-readable descriptions surfaced as report tooltips + chip
# explanations. Lives here so report.py and viewer can stay in sync.
METRIC_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "vrest_mv": {
        "what": "Resting membrane potential — true Vrest (no holding current injected). "
                "Sourced from `spontaneous_no_hold` family sweeps (e.g. SponNonHold30).",
        "healthy": "−65 to −80 mV (cortical pyramidal); −55 to −90 mV more broadly.",
    },
    "held_vm_mv": {
        "what": "Held membrane potential under holding current. Sourced from "
                "`spontaneous_held` family sweeps (e.g. SponHold3 / SponHold30). "
                "Distinct from vrest_mv — this is the *target* potential held by Ihld.",
        "healthy": "Typically −60 to −80 mV. Should match the lab's protocol holding voltage.",
    },
    "vrest_drift_mv": {
        "what": "Vrest delta from first to last spontaneous_hold sweep.",
        "healthy": "|Δ| < 5 mV across a healthy recording.",
    },
    "vrest_session_drift_mv": {
        "what": "Vrest delta between 2nd half and 1st half of the session (medians).",
        "healthy": "|Δ| < 5 mV — early-warning seal degradation otherwise.",
    },
    "baseline_rms_mv": {
        "what": "RMS noise of the spontaneous_hold baseline voltage.",
        "healthy": "< 1 mV (mainly electrode/amplifier coupling).",
    },
    "rs_mohm_initial": {
        "what": "Access resistance at the first test-pulse sweep.",
        "healthy": "10–25 MΩ for somatic patch.",
    },
    "rs_mohm_final": {
        "what": "Access resistance at the last test-pulse sweep.",
        "healthy": "10–25 MΩ; ≤30 MΩ acceptable.",
    },
    "rs_drift_pct": {
        "what": "Rs drift between first and last test-pulse sweep, as % of initial.",
        "healthy": "< 20% over a typical session.",
    },
    "rs_session_drift_pct": {
        "what": "Rs delta between 2nd half and 1st half (medians), as % of 1st-half.",
        "healthy": "< 25% — seal stability check.",
    },
    "rs_compensation_pct": {
        "what": "Series-resistance compensation percentage read from the NWB's "
                "IntracellularElectrode metadata (resistance_comp_correction). "
                "Indicates how much of Rs the experimenter compensated for at acquisition time.",
        "healthy": "Typically 60–80% (lab-dependent). 0 or NaN ⇒ no compensation recorded.",
    },
    "rac_variability_pct": {
        "what": "Coefficient of variation (std/median × 100) of per-Rac Rs estimates "
                "across the test_pulse sweeps. Catches non-monotonic instability "
                "across repetitions that rs_drift_pct (first vs last) misses.",
        "healthy": "< 20%; > 40% indicates dropping recording performance.",
    },
    "rin_mohm": {
        "what": "Input resistance from subthreshold IV slope (V = Rin·I + offset).",
        "healthy": "50–150 MΩ for cortical pyramidal.",
    },
    "rin_r2": {
        "what": "R² of the linear fit used to derive Rin.",
        "healthy": "> 0.9 (clean linear region).",
    },
    "holding_current_pa": {
        "what": "Baseline (pre-step) holding current; reflects seal quality.",
        "healthy": "|Ihld| < 100 pA at Vrest.",
    },
    "holding_current_drift_pa": {
        "what": "Change in holding current from first to last sweep.",
        "healthy": "|Δ| < 50 pA — creeping demand signals seal leak.",
    },
    "ap_amp_overshoot_mv": {
        "what": "Median AP peak above 0 mV across all ap_waveform / rest_firing sweeps. "
                "Distinct from `ap_amplitude_mv` (peak − threshold).",
        "healthy": "+20 to +40 mV (cortical pyramidal); ≥10 mV minimally acceptable.",
    },
    "ap_amplitude_mv": {
        "what": "Canonical AP amplitude (LNMC definition): peak voltage minus threshold "
                "voltage (the dV/dt-triggered onset). Median across all detected spikes "
                "in ap_waveform / rest_firing sweeps. Independent of resting-Vm baseline.",
        "healthy": "60–100 mV for healthy cortical pyramidal cells; < 40 mV is degraded.",
    },
    "ap_amp_overshoot_min_mv": {
        "what": "Worst-case AP overshoot across sweeps — catches sporadic attenuation.",
        "healthy": "> 10 mV; < 0 mV is failure.",
    },
    "ap_amp_attenuation_frac": {
        "what": "Fraction of all detected APs with overshoot < 15 mV.",
        "healthy": "< 10% — sustained attenuation indicates cell decline.",
    },
    "ap_overshoot_session_drift_mv": {
        "what": "AP overshoot delta between 2nd half and 1st half (medians).",
        "healthy": "> −10 mV (i.e. < 10 mV drop). Sharp drop = cell dying.",
    },
    "ap_threshold_drift_mv": {
        "what": "Voltage delta between first and last detected AP threshold.",
        "healthy": "|Δ| < 5 mV.",
    },
    "ap_amp_cv": {
        "what": "Coefficient of variation of AP peak amplitudes within a sweep.",
        "healthy": "< 0.10 ideal, < 0.20 acceptable.",
    },
    "ap_failure_fraction": {
        "what": "Fraction of dV/dt spike initiations that fail to reach overshoot.",
        "healthy": "≈ 0 in healthy cells.",
    },
    "rac_decay_residual_rel": {
        "what": "Relative residual of an exponential fit to the test-pulse recovery.",
        "healthy": "< 0.05; > 0.15 indicates ringing/glitches.",
    },
    "vm_drift_within_sweep_mv_per_s": {
        "what": "Max within-sweep Vm slope from spontaneous_hold (linear regression).",
        "healthy": "< 0.5 mV/s; > 2 mV/s indicates active seal drift.",
    },
    "late_instability_index": {
        "what": "Max ratio of late-quartile-to-early-quartile activity within a sweep, minus 1.",
        "healthy": "≈ 0; > 1 = late-sweep runaway oscillation / firing.",
    },
    "test_pulse_edge_overshoot_mv": {
        "what": "Magnitude of the step-edge voltage transient (5–10 ms post-edge vs 20–50 ms plateau). "
                "Note: per LNMC experimenter guidance, a SHARP transient is the GOOD signature of "
                "active Rs compensation; a smooth exponential decay indicates NO compensation. "
                "So this metric is informational — interpretation depends on whether you expect "
                "compensation in this protocol. Cross-reference rs_compensation_pct from metadata.",
        "healthy": "Cohort-dependent. Stable across reps is the real signal; rac_variability_pct quantifies that.",
    },
    "qc_protocol_coverage": {
        "what": "Boolean: NWB carries at least one sweep from each essential family "
                "(spontaneous_hold, test_pulse, ap_waveform).",
        "healthy": "True. False ⇒ stimulus_protocols mapping incomplete or recording cut short.",
    },
    "n_sweeps_total":   {"what": "Number of voltage acquisitions in the NWB.",     "healthy": "> 10 for a useful recording."},
    "n_sweeps_clipped": {"what": "Sweeps that hit the voltage rails (±150 / +80 mV) for ≥1 ms.",
                          "healthy": "0 — any clipped sweep is suspect."},
    "n_sweeps_nan":     {"what": "Sweeps containing NaN samples.", "healthy": "0."},
    "bad_ending_at_sweep": {
        "what": "First sweep index where the recording started to degrade "
                "(Vrest depolarised, Rs exploded, or AP overshoot collapsed). "
                "Everything ≥ this index is excluded from the metric scalars.",
        "healthy": "NaN — recording ended cleanly.",
    },
    "n_sweeps_trimmed": {
        "what": "How many tail sweeps were excluded from metric scalars due to bad-ending detection.",
        "healthy": "0.",
    },
    "bad_ending_reason": {
        "what": "Why the recording was trimmed: vrest_depolarisation / rs_explosion / ap_collapse.",
        "healthy": "None (recording ended cleanly).",
    },
}

# Synthetic non-metric triggers that surface in `triggered_metrics` lists but
# aren't tied to a specific stimulus family. Mapped to a short label for display.
PSEUDO_METRIC_LABELS: dict[str, str] = {
    "qc_protocol_coverage": "coverage",
    "_no_cache":            "cache",
    "n_sweeps_clipped":     "signal-hygiene",
    "n_sweeps_nan":         "signal-hygiene",
    "n_sweeps_total":       "signal-hygiene",
}


def family_for_metric(metric: str | None) -> str | None:
    """Return the implicated stimulus family for a metric, or None if unmapped."""
    if not metric:
        return None
    return METRIC_TO_FAMILY.get(metric)


def implicated_families(triggered_metrics: Iterable[dict | str] | None) -> set[str]:
    """Set of stimulus families implicated by a cell's triggered metrics.

    Accepts the list-of-dicts shape used in the pipeline (each dict has a
    `metric` key) or a list of bare metric names. Unknown / pseudo metrics
    contribute nothing — they're not a stimulus family.
    """
    if not triggered_metrics:
        return set()
    out: set[str] = set()
    for t in triggered_metrics:
        if isinstance(t, dict):
            name = t.get("metric")
        else:
            name = t
        fam = METRIC_TO_FAMILY.get(name or "")
        if fam:
            out.add(fam)
    return out
