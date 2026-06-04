"""Unit-normalisation helpers + paired stimulus discovery (Parts 0 + 1)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.nwb_io import (
    current_si,
    find_paired_stimulus,
    open_nwb,
    voltage_si,
)


# ---------- voltage_si / current_si ----------------------------------------

class _FakeSeries:
    """Minimal stand-in for a TimeSeries with data/conversion/offset/unit attrs."""
    def __init__(self, data, unit, conversion=1.0, offset=0.0):
        self.data = np.array(data, dtype=np.float64)
        self.unit = unit
        self.conversion = conversion
        self.offset = offset


def test_voltage_si_already_volts():
    """unit='volts', conversion=1.0 — straight passthrough."""
    obj = _FakeSeries([-0.065, -0.070], unit="volts")
    result = voltage_si(obj)
    np.testing.assert_allclose(result, [-0.065, -0.070])


def test_voltage_si_millivolt_string_form():
    """unit='millivolts' or 'mV' — data is in mV magnitudes, must be downscaled to V."""
    for unit in ("millivolts", "mV"):
        obj = _FakeSeries([-65.0, -70.0], unit=unit)
        result = voltage_si(obj)
        np.testing.assert_allclose(result, [-0.065, -0.070])


def test_voltage_si_conversion_scales_to_si():
    """unit='volts' but conversion=0.001 — raw values are mV magnitudes; converted to V."""
    obj = _FakeSeries([-65.0, -70.0], unit="volts", conversion=0.001)
    result = voltage_si(obj)
    np.testing.assert_allclose(result, [-0.065, -0.070])


def test_voltage_si_offset_applied():
    """offset != 0 must be applied per NWB spec."""
    obj = _FakeSeries([0.0, 10.0], unit="volts", conversion=1.0, offset=-0.05)
    result = voltage_si(obj)
    np.testing.assert_allclose(result, [-0.05, 9.95])


def test_current_si_pA_unit():
    """unit='pA' — values are pA magnitudes; converted to A."""
    obj = _FakeSeries([50.0, 100.0], unit="pA")
    result = current_si(obj)
    np.testing.assert_allclose(result, [50e-12, 100e-12])


def test_current_si_already_amps():
    obj = _FakeSeries([50e-12, 100e-12], unit="amperes")
    result = current_si(obj)
    np.testing.assert_allclose(result, [50e-12, 100e-12])


def test_current_si_picoamps_long_form():
    obj = _FakeSeries([50.0, 100.0], unit="picoamps")
    result = current_si(obj)
    np.testing.assert_allclose(result, [50e-12, 100e-12])


# ---------- find_paired_stimulus -------------------------------------------

def _make_nwb_with_paired_stim(path: Path, *, with_stim: bool = True,
                                acq_name: str = "ic__Rac__001") -> str:
    """Build a tiny NWB with one CurrentClampSeries acquisition and (optionally)
    a paired CurrentClampStimulusSeries with the same name."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="amp", description="d")
    elec = nwbfile.create_icephys_electrode(name="elec0", description="d", device=device)

    # Voltage acquisition
    vtrace = np.full(1000, -0.065, dtype=np.float64)
    vtrace[200:500] = -0.060   # tiny step
    acq = pynwb.icephys.CurrentClampSeries(
        name=acq_name, data=vtrace, electrode=elec,
        gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
    )
    nwbfile.add_acquisition(acq)

    if with_stim:
        # Paired stimulus current at the same name
        itrace = np.zeros(1000, dtype=np.float64)
        itrace[200:500] = 50e-12   # 50 pA step
        stim = pynwb.icephys.CurrentClampStimulusSeries(
            name=acq_name, data=itrace, electrode=elec,
            gain=1.0, starting_time=0.0, rate=10000.0, unit="amperes",
        )
        nwbfile.add_stimulus(stim)

    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    return acq_name


def test_find_paired_stimulus_exact_name(tmp_path: Path):
    path = tmp_path / "cell.nwb"
    acq_name = _make_nwb_with_paired_stim(path, with_stim=True)
    with open_nwb(path) as nwbfile:
        acq = nwbfile.acquisition[acq_name]
        stim = find_paired_stimulus(nwbfile, acq_name, acq)
        assert stim is not None
        assert stim.name == acq_name
        assert getattr(stim, "unit", "").lower() in ("amperes", "a", "")


def test_find_paired_stimulus_returns_none_when_missing(tmp_path: Path):
    path = tmp_path / "cell.nwb"
    acq_name = _make_nwb_with_paired_stim(path, with_stim=False)
    with open_nwb(path) as nwbfile:
        acq = nwbfile.acquisition[acq_name]
        assert find_paired_stimulus(nwbfile, acq_name, acq) is None
