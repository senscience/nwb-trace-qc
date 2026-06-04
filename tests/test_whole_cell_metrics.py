"""Whole-cell QC metric correctness — Rs/Rin/Ihld/session-drift/edge artifact (Parts 2-5b)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.config import default_families
from nwb_trace_qc.metrics import (
    _holding_current_pa,
    _iv_subthreshold_pair,
    _rin_mohm_from_iv_pairs,
    _rs_from_test_pulse_mohm,
    _test_pulse_edge_overshoot_mv,
    compute_metrics,
)
from nwb_trace_qc.stimuli import StimulusFamilyMap


# ---------- Rs ----------------------------------------------------------------

def _step_voltage(rate, n=2000, baseline=-0.065, plateau=-0.060):
    """Voltage trace with a clean step from baseline → plateau at t=5 ms; recovers at 50 ms."""
    v = np.full(n, baseline, dtype=np.float64)
    s = int(0.005 * rate); e = int(0.050 * rate)
    v[s:e] = plateau
    return v


def _step_current(rate, n=2000, baseline_a=0.0, step_a=50e-12):
    i = np.full(n, baseline_a, dtype=np.float64)
    s = int(0.005 * rate); e = int(0.050 * rate)
    i[s:e] = step_a
    return i


def test_rs_with_proper_stim_50pA():
    """ΔV=5 mV, ΔI=50 pA → Rs = 100 MΩ (not 100 MΩ from the 50 pA fallback hack —
    really computed from the stimulus)."""
    rate = 10000.0
    v = _step_voltage(rate, plateau=-0.060)        # 5 mV down
    i = _step_current(rate, step_a=50e-12)         # 50 pA
    rs, used_fallback = _rs_from_test_pulse_mohm(v, rate, i)
    assert used_fallback is False
    assert 95 <= rs <= 105


def test_rs_with_proper_stim_100pA_halves_value():
    """Same ΔV with double the stimulus current → Rs halves. Proves the constant is gone."""
    rate = 10000.0
    v = _step_voltage(rate, plateau=-0.060)        # 5 mV down
    i = _step_current(rate, step_a=100e-12)        # 100 pA
    rs, used_fallback = _rs_from_test_pulse_mohm(v, rate, i)
    assert used_fallback is False
    assert 45 <= rs <= 55


def test_rs_fallback_when_no_stim():
    """Without paired stim, falls back to 50 pA assumption and flags as fallback."""
    rate = 10000.0
    v = _step_voltage(rate, plateau=-0.060)
    rs, used_fallback = _rs_from_test_pulse_mohm(v, rate, None)
    assert used_fallback is True
    assert 95 <= rs <= 105


# ---------- Rin ---------------------------------------------------------------

def test_rin_from_iv_pairs_returns_slope():
    """IV pairs (-100 pA → -10 mV), (-50 pA → -5 mV), (0 pA → 0 mV) → 100 MΩ slope."""
    pairs = [(-100.0, -10.0), (-50.0, -5.0), (0.0, 0.0)]
    rin, r2 = _rin_mohm_from_iv_pairs(pairs)
    # slope = ΔV(mV) / ΔI(pA) = 0.1 mV/pA → 100 MΩ
    assert 99 <= rin <= 101
    assert r2 > 0.99


def test_rin_too_few_points_returns_nan():
    pairs = [(-100.0, -10.0), (-50.0, -5.0)]
    rin, r2 = _rin_mohm_from_iv_pairs(pairs)
    assert math.isnan(rin) and math.isnan(r2)


def test_iv_subthreshold_pair_rejects_suprathreshold():
    """A sweep whose steady-state is above AP threshold (~-40 mV) is rejected."""
    rate = 10000.0
    v = np.full(2000, 0.020, dtype=np.float64)   # +20 mV — clearly suprathreshold
    i = _step_current(rate, step_a=100e-12)
    assert _iv_subthreshold_pair(i, v, rate) is None


# ---------- Holding current --------------------------------------------------

def test_holding_current_pa_from_stim_baseline():
    rate = 10000.0
    i = np.full(2000, -40e-12, dtype=np.float64)   # constant -40 pA
    assert _holding_current_pa(i, rate) == pytest.approx(-40, rel=1e-6)


def test_holding_current_pa_nan_when_no_stim():
    assert math.isnan(_holding_current_pa(None, 10000.0))


# ---------- Test-pulse edge overshoot (Part 5b sketch metric) ----------------

def test_edge_overshoot_clean_pulse_low_value():
    """A square step with no transient — edge overshoot ~0."""
    rate = 10000.0
    v = _step_voltage(rate, plateau=-0.060)
    i = _step_current(rate, step_a=50e-12)
    e = _test_pulse_edge_overshoot_mv(v, rate, i)
    assert 0 <= e <= 1


def test_edge_overshoot_with_capacitive_ringing_high_value():
    """The BAD Rac/Delta sketch: sharp 20 mV transient at the leading edge."""
    rate = 10000.0
    v = _step_voltage(rate, plateau=-0.060)
    # Inject a transient ring 1 ms into the step
    transient_idx = int(0.006 * rate)
    v[transient_idx:transient_idx + 10] = -0.040   # 20 mV up from plateau
    i = _step_current(rate, step_a=50e-12)
    e = _test_pulse_edge_overshoot_mv(v, rate, i)
    assert e >= 18


# ---------- Session-level drift via compute_metrics --------------------------

def _make_session_nwb(path: Path, vrest_sequence: list[float], rate=10000.0) -> None:
    """Build an NWB with one spontaneous_hold sweep per Vrest level."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)
    for i, v in enumerate(vrest_sequence):
        trace = np.full(int(rate * 1.0), v, dtype=np.float64)
        acq = pynwb.icephys.CurrentClampSeries(
            name=f"ic__SponHold30__{i:03d}",
            data=trace, electrode=elec,
            gain=1.0, starting_time=i * 1.0, rate=rate, unit="volts",
        )
        nwbfile.add_acquisition(acq)
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_vrest_session_drift_fires_on_half_vs_half_delta(tmp_path: Path):
    """First-half Vrest ≈ -65 mV, second-half ≈ -55 mV → +10 mV session drift."""
    path = tmp_path / "drift.nwb"
    _make_session_nwb(path, vrest_sequence=[-0.065, -0.065, -0.055, -0.055])
    families = StimulusFamilyMap(default_families())
    out = compute_metrics(path, families)
    # Median(2nd half) - Median(1st half) = -55 - -65 = +10 mV
    assert out["vrest_session_drift_mv"] == pytest.approx(10.0, abs=0.5)
    # Sanity: scalar Vrest is the median across all sweeps
    assert -65 < out["vrest_mv"] < -55
