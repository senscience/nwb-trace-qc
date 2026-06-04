"""eFEL feature wrapper + per-cell fallback bookkeeping (v0.4.0)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.config import default_families
from nwb_trace_qc.efel_features import (
    EFEL_AP_AMPLITUDE_FROM_VBASE,
    EFEL_VOLTAGE_BASE,
    efel_features_for_sweep,
    feature_scalar,
)
from nwb_trace_qc.metrics import compute_metrics
from nwb_trace_qc.stimuli import StimulusFamilyMap


def test_voltage_base_matches_baseline_on_flat_trace():
    """A flat trace at -65 mV (SI: -0.065 V) — eFEL voltage_base should equal -65 mV."""
    rate = 10000.0
    voltage_v = np.full(int(rate * 0.5), -0.065, dtype=np.float64)
    feats = efel_features_for_sweep(voltage_v, None, rate, features=[EFEL_VOLTAGE_BASE])
    assert feats is not None
    base_mv = feature_scalar(feats.get(EFEL_VOLTAGE_BASE))
    # eFEL returns mV
    assert -66 <= base_mv <= -64


def test_feature_scalar_handles_empty_and_nan():
    assert math.isnan(feature_scalar(None))
    assert math.isnan(feature_scalar([]))
    assert math.isnan(feature_scalar([float("nan"), float("nan")]))
    assert feature_scalar([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_efel_features_returns_none_on_empty_trace():
    feats = efel_features_for_sweep(np.array([]), None, 10000.0, features=[EFEL_VOLTAGE_BASE])
    assert feats is None


def test_efel_features_returns_none_on_zero_rate():
    voltage_v = np.full(1000, -0.065, dtype=np.float64)
    feats = efel_features_for_sweep(voltage_v, None, 0.0, features=[EFEL_VOLTAGE_BASE])
    assert feats is None


def _make_spike_trace(rate: float, peak_v: float, n_samples: int = 5000) -> np.ndarray:
    """Synthetic baseline at -65 mV with one AP-like peak (Gaussian) in the middle."""
    t = np.arange(n_samples) / rate
    baseline = -0.065
    peak_center = 0.05    # 50 ms
    peak_width = 0.001    # 1 ms half-width
    spike = (peak_v - baseline) * np.exp(-((t - peak_center) ** 2) / (2 * peak_width ** 2))
    return baseline + spike


def _make_apwaveform_nwb(path: Path, peak_v: float = 0.025) -> None:
    """One ap_waveform sweep with a synthetic AP peaking at peak_v (V)."""
    rate = 10000.0
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)
    trace = _make_spike_trace(rate, peak_v)
    acq = pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001",
        data=trace, electrode=elec,
        gain=1.0, starting_time=0.0, rate=rate, unit="volts",
    )
    nwbfile.add_acquisition(acq)
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_compute_metrics_records_efel_fallback_count(tmp_path: Path):
    """n_efel_fallback_sweeps populated when eFEL refuses or use_efel=False."""
    path = tmp_path / "ap.nwb"
    _make_apwaveform_nwb(path, peak_v=0.025)
    families = StimulusFamilyMap(default_families())

    # use_efel=False forces every sweep to use our custom helpers (no eFEL calls)
    out_no_efel = compute_metrics(path, families, use_efel=False, trim_bad_ending=False)
    # With eFEL disabled we don't increment the fallback counter (it's only
    # incremented when eFEL was tried and refused).
    assert int(out_no_efel["n_efel_fallback_sweeps"]) == 0

    out_with_efel = compute_metrics(path, families, use_efel=True, trim_bad_ending=False)
    # Either eFEL succeeded (fallback==0) or it raised and we fell back. Both
    # are valid outcomes; the metric should be populated in either case.
    assert out_with_efel["ap_amp_overshoot_mv"] is not None
    assert not math.isnan(out_with_efel["ap_amp_overshoot_mv"])
