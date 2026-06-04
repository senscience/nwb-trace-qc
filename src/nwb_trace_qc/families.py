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


METRIC_TO_FAMILY: dict[str, str] = {
    # Spontaneous-hold-derived
    "vrest_mv":                       "spontaneous_hold",
    "vrest_drift_mv":                 "spontaneous_hold",
    "baseline_rms_mv":                "spontaneous_hold",
    # Test-pulse-derived (Rs, decay shape)
    "rs_mohm_initial":                "test_pulse",
    "rs_mohm_final":                  "test_pulse",
    "rs_drift_pct":                   "test_pulse",
    "rac_decay_residual_rel":         "test_pulse",
    # AP-waveform-derived
    "ap_amp_overshoot_mv":            "ap_waveform",
    "ap_threshold_drift_mv":          "ap_waveform",
    "ap_amp_cv":                      "ap_waveform",
    "ap_failure_fraction":            "ap_waveform",
    # Long-sweep / firing-train metrics
    "vm_drift_within_sweep_mv_per_s": "rest_firing",
    "late_instability_index":         "rest_firing",
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
