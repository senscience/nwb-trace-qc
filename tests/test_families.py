"""METRIC_TO_FAMILY mapping + implicated_families derivation."""
from __future__ import annotations

import pytest

from nwb_trace_qc.families import (
    METRIC_TO_FAMILY,
    PSEUDO_METRIC_LABELS,
    family_for_metric,
    implicated_families,
)


def test_essential_metric_mappings():
    """A handful of metrics that drive QC verdicts must map to a family."""
    assert family_for_metric("vrest_mv") == "spontaneous_hold"
    assert family_for_metric("rs_drift_pct") == "test_pulse"
    assert family_for_metric("ap_amp_overshoot_mv") == "ap_waveform"
    assert family_for_metric("late_instability_index") == "rest_firing"


def test_unknown_metric_returns_none():
    assert family_for_metric(None) is None
    assert family_for_metric("") is None
    assert family_for_metric("not_a_real_metric") is None


def test_implicated_families_from_dict_list():
    triggered = [
        {"metric": "vrest_mv", "verdict": "flag"},
        {"metric": "rs_drift_pct", "verdict": "fail"},
        {"metric": "qc_protocol_coverage", "verdict": "flag"},   # pseudo — no family
        {"metric": "_no_cache", "verdict": "flag"},               # pseudo — no family
        {"metric": "vrest_drift_mv", "verdict": "flag"},          # dup family with vrest_mv
    ]
    fams = implicated_families(triggered)
    assert fams == {"spontaneous_hold", "test_pulse"}


def test_implicated_families_from_bare_strings():
    """Also accept a list of bare metric names (used by some client code paths)."""
    assert implicated_families(["ap_amp_overshoot_mv", "vrest_mv"]) == \
        {"ap_waveform", "spontaneous_hold"}


def test_implicated_families_empty_input():
    assert implicated_families(None) == set()
    assert implicated_families([]) == set()
    assert implicated_families([{}]) == set()


def test_pseudo_labels_have_human_friendly_strings():
    """Pseudo metrics surface a short tag for display, but no family."""
    assert "qc_protocol_coverage" in PSEUDO_METRIC_LABELS
    assert PSEUDO_METRIC_LABELS["qc_protocol_coverage"] == "coverage"
    # And they don't accidentally appear in METRIC_TO_FAMILY
    for k in PSEUDO_METRIC_LABELS:
        assert k not in METRIC_TO_FAMILY
