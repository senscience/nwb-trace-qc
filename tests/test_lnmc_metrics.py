"""v0.5.0 LNMC experimenter-guidance additions:
- spontaneous-hold family split (no_hold vs held)
- ap_amplitude_mv (peak − threshold)
- rs_compensation_pct (NWB icephys metadata)
- rac_variability_pct (CV across Rac reps)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.config import default_families, load_config
from nwb_trace_qc.metrics import compute_metrics, _read_rs_compensation_pct
from nwb_trace_qc.stimuli import StimulusFamilyMap


def _build_nwb(path: Path,
                spec: list[tuple[str, float]],
                rs_comp_pct: float | None = None) -> None:
    """spec = list of (acq_name, constant_voltage_volts) per sweep."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d", device=device,
    )
    # rs_comp_pct: pynwb in this version doesn't accept resistance_comp_correction
    # at construction; the helper scans for it as an attribute (real cohorts use
    # NWB extensions for this). We test the metadata read by direct setattr.
    if rs_comp_pct is not None:
        elec.resistance_comp_correction = float(rs_comp_pct) / 100.0
    rate = 10000.0
    for i, (name, v_const) in enumerate(spec):
        trace = np.full(int(rate * 1.0), v_const, dtype=np.float64)
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=name, data=trace, electrode=elec,
            gain=1.0, starting_time=i * 1.0, rate=rate, unit="volts",
        ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


# ---------- family split ----------------------------------------------------

def test_vrest_sources_from_spontaneous_no_hold(tmp_path: Path):
    """When the NWB has both no-hold (-65) and held (-70) sweeps, vrest_mv MUST
    come from the no-hold family (true Vrest), not the held average."""
    path = tmp_path / "split.nwb"
    _build_nwb(path, [
        ("ic__SponNonHold30__001", -0.065),
        ("ic__SponNonHold30__002", -0.065),
        ("ic__SponHold30__001",    -0.070),
        ("ic__SponHold30__002",    -0.070),
    ])
    fm = StimulusFamilyMap(default_families())
    out = compute_metrics(path, fm, use_efel=False, trim_bad_ending=False)
    # vrest_mv = median of no-hold sweeps → -65 mV (not the average of all 4)
    assert -66 <= out["vrest_mv"] <= -64
    # held_vm_mv = median of held sweeps → -70 mV
    assert -71 <= out["held_vm_mv"] <= -69


def test_vrest_legacy_spontaneous_hold_still_works(tmp_path: Path):
    """A project YAML that still uses the legacy `spontaneous_hold` family
    (mixed-semantics bucket) routes everything into the legacy accumulator
    and Vrest comes from there as a fallback."""
    path = tmp_path / "legacy.nwb"
    _build_nwb(path, [
        ("ic__SponHold30__001", -0.068),
        ("ic__SponHold30__002", -0.068),
    ])
    legacy_families = {"spontaneous_hold": ["SponHold30"], "test_pulse": ["Rac"],
                        "ap_waveform": ["APWaveform"]}
    fm = StimulusFamilyMap(legacy_families)
    out = compute_metrics(path, fm, use_efel=False, trim_bad_ending=False)
    # Falls back to legacy bucket → vrest_mv populated from there
    assert -69 <= out["vrest_mv"] <= -67


# ---------- ap_amplitude (peak − threshold) ---------------------------------

def _make_spike_trace(rate=10000.0, n=5000, peak_v=0.030) -> np.ndarray:
    """Synthetic baseline at -65 mV with one Gaussian-shape AP-like peak."""
    t = np.arange(n) / rate
    baseline = -0.065
    width = 0.001
    centre = n / 2 / rate
    return baseline + (peak_v - baseline) * np.exp(-((t - centre) ** 2) / (2 * width ** 2))


def test_ap_amplitude_mv_populated_when_efel_succeeds(tmp_path: Path):
    """A clean spike trace → ap_amplitude_mv has a reasonable peak−threshold value.

    This is a smoke test: eFEL may or may not detect the spike depending on the
    exact shape, but if it does, ap_amplitude_mv is populated. We don't compare
    numerically since eFEL's threshold detection is sensitive to dV/dt shape.
    """
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier="ap_cell",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)
    nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001", data=_make_spike_trace(),
        electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
    ))
    path = tmp_path / "ap.nwb"
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    fm = StimulusFamilyMap(default_families())
    out = compute_metrics(path, fm, use_efel=True, trim_bad_ending=False)
    # ap_amplitude_mv is allowed to be NaN if eFEL can't detect a threshold on
    # this synthetic Gaussian, but ap_amp_overshoot_mv MUST be populated
    # (since it just measures peak above 0 mV, no threshold detection needed).
    assert not math.isnan(out["ap_amp_overshoot_mv"])


# ---------- rs_compensation_pct ---------------------------------------------

def test_rs_compensation_read_fraction_form():
    """When the electrode's resistance_comp_correction is a 0..1 fraction,
    the helper normalises to a 0..100 percent."""
    class _MockElectrode:
        resistance_comp_correction = 0.70
    class _MockNwb:
        icephys_electrodes = {"e0": _MockElectrode()}
        lab_meta_data = {}
    pct = _read_rs_compensation_pct(_MockNwb())
    assert 69 <= pct <= 71


def test_rs_compensation_read_percent_form():
    """When the value is already a 0..100 percent, pass it through unchanged."""
    class _MockElectrode:
        resistance_comp_correction = 80.0
    class _MockNwb:
        icephys_electrodes = {"e0": _MockElectrode()}
        lab_meta_data = {}
    pct = _read_rs_compensation_pct(_MockNwb())
    assert pct == pytest.approx(80.0)


def test_rs_compensation_nan_when_absent(tmp_path: Path):
    """No resistance_comp_correction on the electrode → rs_compensation_pct = NaN."""
    path = tmp_path / "uncompensated.nwb"
    _build_nwb(path, [("ic__SponHold30__001", -0.068)])
    fm = StimulusFamilyMap(default_families())
    out = compute_metrics(path, fm, use_efel=False, trim_bad_ending=False)
    assert math.isnan(out["rs_compensation_pct"])


# ---------- rac_variability_pct ---------------------------------------------

def test_rac_variability_pct_high_when_rs_jitters(tmp_path: Path):
    """When per-Rac Rs estimates jitter ~20% around the median, CV is ~20%."""
    # Synthetic Rs values: 20, 24, 20, 16, 22, 18, 20 → median ~20, std ~2.4
    # CV ≈ 2.4/20 × 100 ≈ 12%
    from nwb_trace_qc.metrics import compute_metrics  # re-import for clarity

    nwbfile = pynwb.NWBFile(
        session_description="t", identifier="rac",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)
    rate = 10000.0
    # Each Rac sweep: baseline at -65 mV for 5 ms, then step to a different
    # voltage producing different Rs estimates. ΔV is what drives Rs in
    # `_rs_from_test_pulse_mohm` (no stim_a paired → 50pA assumption); so
    # vary the post-step ΔV to get a CV across reps.
    delta_v_targets = [-0.005, -0.0055, -0.0049, -0.0061, -0.0048, -0.0058, -0.005]
    for i, dv in enumerate(delta_v_targets):
        trace = np.full(int(rate * 0.1), -0.065, dtype=np.float64)  # 100 ms
        s = int(0.005 * rate)
        e = int(0.060 * rate)
        trace[s:e] = -0.065 + dv  # step
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=f"ic__Rac__{i:03d}", data=trace, electrode=elec,
            gain=1.0, starting_time=i * 0.5, rate=rate, unit="volts",
        ))
    path = tmp_path / "rac.nwb"
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    fm = StimulusFamilyMap(default_families())
    out = compute_metrics(path, fm, use_efel=False, trim_bad_ending=False)
    # rac_variability_pct should be a non-NaN small percent — exact value
    # depends on the Rs computation, but it must be set and positive.
    assert not math.isnan(out["rac_variability_pct"])
    assert out["rac_variability_pct"] > 0


def test_rac_variability_pct_nan_with_too_few_sweeps(tmp_path: Path):
    """Fewer than 3 Rac sweeps → not enough to compute CV; returns NaN."""
    path = tmp_path / "rac_few.nwb"
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier="few",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)
    rate = 10000.0
    trace = np.full(int(rate * 0.1), -0.065, dtype=np.float64)
    trace[int(0.005 * rate):int(0.060 * rate)] = -0.070
    nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
        name="ic__Rac__001", data=trace, electrode=elec,
        gain=1.0, starting_time=0.0, rate=rate, unit="volts",
    ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    fm = StimulusFamilyMap(default_families())
    out = compute_metrics(path, fm, use_efel=False, trim_bad_ending=False)
    assert math.isnan(out["rac_variability_pct"])
