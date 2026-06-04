"""Bad-ending detection + auto-trim (v0.4.0)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.config import default_families
from nwb_trace_qc.metrics import _detect_bad_ending, compute_metrics
from nwb_trace_qc.stimuli import StimulusFamilyMap


def test_detect_vrest_depolarisation_late_in_session():
    """Vrest stable around -65 mV for sweeps 0..11, then collapses to -40 from sweep 12 onward.
    Detector should return cutoff=12, reason='vrest_depolarisation'."""
    # Volts (so depolarisation = +25 mV ≈ +0.025 V)
    vrest_seq = [(-0.065, i) for i in range(12)] + [(-0.040, i) for i in range(12, 20)]
    cutoff, reason = _detect_bad_ending(vrest_seq, [], [], n_total_sweeps=20)
    assert cutoff == 12
    assert reason == "vrest_depolarisation"


def test_detect_rs_explosion():
    """Rs steady around 18 MΩ for sweeps 0..9, then jumps to 50 MΩ at sweep 10."""
    rs_seq = [(18.0, i) for i in range(10)] + [(50.0, i) for i in range(10, 20)]
    cutoff, reason = _detect_bad_ending([], rs_seq, [], n_total_sweeps=20)
    assert cutoff == 10
    assert reason == "rs_explosion"


def test_detect_ap_collapse():
    """AP overshoot stable around +25 mV for sweeps 0..8, then drops below +10 mV from sweep 9."""
    overshoots = [(25.0, i) for i in range(9)] + [(5.0, i) for i in range(9, 20)]
    cutoff, reason = _detect_bad_ending([], [], overshoots, n_total_sweeps=20)
    assert cutoff == 9
    assert reason == "ap_collapse"


def test_clean_session_returns_none():
    """No degradation across 20 sweeps → detector returns (None, None)."""
    vrest_seq = [(-0.065 + 0.001 * (i % 3), i) for i in range(20)]   # tiny noise, no jump
    rs_seq = [(18.0 + 0.5 * (i % 3), i) for i in range(20)]
    overshoots = [(25.0 + 0.5 * (i % 3), i) for i in range(20)]
    cutoff, reason = _detect_bad_ending(vrest_seq, rs_seq, overshoots, n_total_sweeps=20)
    assert cutoff is None
    assert reason is None


def test_early_degradation_ignored_by_30pct_guardrail():
    """Vrest collapse at sweep 2 of 20 (10% in) is ignored — that's not a bad ending,
    it's a cell that wasn't healthy from the start."""
    vrest_seq = [(-0.065, 0), (-0.065, 1)] + [(-0.040, i) for i in range(2, 20)]
    cutoff, reason = _detect_bad_ending(vrest_seq, [], [], n_total_sweeps=20)
    assert cutoff is None
    assert reason is None


def test_last_sweep_glitch_ignored_by_95pct_guardrail():
    """A single bad sweep in the last 5% is not a degradation pattern."""
    vrest_seq = [(-0.065, i) for i in range(19)] + [(-0.040, 19)]
    cutoff, reason = _detect_bad_ending(vrest_seq, [], [], n_total_sweeps=20)
    assert cutoff is None or cutoff >= 19    # last-5% guardrail in effect


def test_short_session_returns_none():
    """Too few sweeps to make a confident judgement → bail out."""
    vrest_seq = [(-0.065, 0), (-0.040, 1)]
    cutoff, reason = _detect_bad_ending(vrest_seq, [], [], n_total_sweeps=2)
    assert cutoff is None


def _make_session_nwb(path: Path, vrest_v_sequence: list[float], rate=10000.0) -> None:
    """Build an NWB with one spontaneous_hold sweep per Vrest level."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)
    for i, v in enumerate(vrest_v_sequence):
        trace = np.full(int(rate * 1.0), v, dtype=np.float64)
        acq = pynwb.icephys.CurrentClampSeries(
            name=f"ic__SponHold30__{i:03d}",
            data=trace, electrode=elec,
            gain=1.0, starting_time=i * 1.0, rate=rate, unit="volts",
        )
        nwbfile.add_acquisition(acq)
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_compute_metrics_trims_bad_ending_from_vrest_median(tmp_path: Path):
    """20 sweeps: first 8 at -65 mV, last 12 at -40 mV.
    With trim: cutoff=8, n_sweeps_trimmed=12, Vrest median = first 8 → -65 mV.
    Without trim: median of 20 sweeps where 12 are -40 → -40 mV.
    """
    path = tmp_path / "drift.nwb"
    vrest_v_sequence = [-0.065] * 8 + [-0.040] * 12
    _make_session_nwb(path, vrest_v_sequence)

    families = StimulusFamilyMap(default_families())
    out = compute_metrics(path, families, use_efel=False, trim_bad_ending=True)

    assert int(out["n_sweeps_trimmed"]) == 12
    assert int(out["bad_ending_at_sweep"]) == 8
    assert out["bad_ending_reason"] == "vrest_depolarisation"
    # Trimmed: median over first 8 sweeps ≈ -65 mV
    assert -66 <= out["vrest_mv"] <= -64

    # Without trim: median over all 20 sweeps where 12 are at -40 → ≈ -40 mV
    out_untrimmed = compute_metrics(path, families, use_efel=False, trim_bad_ending=False)
    assert -41 <= out_untrimmed["vrest_mv"] <= -39
    assert int(out_untrimmed["n_sweeps_trimmed"]) == 0
