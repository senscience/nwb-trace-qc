"""_make_thumbnail fallbacks: targeted picks, last-resort voltage sweeps, error paths."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.config import default_families
from nwb_trace_qc.pipeline import _make_thumbnail, _stratified_picks


def _make_nwb_with(sweep_names: list[str], path: Path) -> None:
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    for sn in sweep_names:
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=sn, data=np.linspace(-0.07, 0.03, 200),
            electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
        ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_targeted_pick_when_family_matches(tmp_path: Path):
    """When the NWB has sweeps in the wanted family, picks come from that family."""
    nwb = tmp_path / "cell.nwb"
    _make_nwb_with(["ic__APWaveform__001", "ic__APWaveform__002", "ic__IDRest__001"], nwb)
    out = tmp_path / "thumb.png"
    result, status = _make_thumbnail(
        nwb, out, families=default_families(),
        reasons=["ap_amp_overshoot_mv"],   # → wanted = {ap_waveform}
    )
    assert result is not None and out.exists()
    assert status == "rendered"


def test_fallback_when_no_family_match(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """When the NWB has no sweeps in the wanted families (lab-specific protocol names
    not mapped), fall back to the first 3 voltage sweeps and warn."""
    nwb = tmp_path / "cell.nwb"
    # All these tokens are unmapped under the LNMC defaults
    _make_nwb_with([
        "ic__Test_eCode__001", "ic__sAHP__001", "ic__C1step_ag__001", "ic__Spontaneous__001",
    ], nwb)
    out = tmp_path / "thumb.png"
    with caplog.at_level("WARNING"):
        result, status = _make_thumbnail(
            nwb, out, families=default_families(),
            reasons=["vrest_mv", "rs_drift_pct"],  # → wanted = {spontaneous_hold, test_pulse}
        )
    assert result is not None and out.exists()
    assert status == "rendered"
    # The warning explicitly tells the user their mapping is incomplete
    assert any("falling back to first 3 voltage sweeps" in r.message for r in caplog.records)


def test_no_voltage_returns_status(tmp_path: Path):
    """An NWB with zero voltage acquisitions returns a clean (None, no_voltage_sweeps)."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier="empty",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    # No add_acquisition calls → empty acquisitions
    nwb = tmp_path / "empty.nwb"
    with pynwb.NWBHDF5IO(str(nwb), mode="w") as io:
        io.write(nwbfile)
    out = tmp_path / "thumb.png"
    result, status = _make_thumbnail(
        nwb, out, families=default_families(), reasons=["vrest_mv"],
    )
    assert result is None
    assert status == "no_voltage_sweeps"
    assert not out.exists()


def test_render_error_returns_status(tmp_path: Path):
    """A bad NWB path returns (None, render_error) — never silently."""
    out = tmp_path / "thumb.png"
    result, status = _make_thumbnail(
        Path("/nonexistent/path.nwb"), out,
        families=default_families(), reasons=["vrest_mv"],
    )
    assert result is None
    assert status == "render_error"


def test_stratified_picks_takes_endpoints_and_middle():
    """With 10 candidates and max_n=3, picks should be #0, #5 (or #4), #9 —
    not #0/#1/#2. Surfaces within-family drift."""
    candidates = list(range(10))
    picks = _stratified_picks(candidates, max_n=3)
    assert picks[0] == 0
    assert picks[-1] == 9
    # Middle pick is within ±1 of the true midpoint (rounding tolerance)
    assert abs(picks[1] - 5) <= 1


def test_stratified_picks_returns_all_when_short():
    assert _stratified_picks([7, 8, 9], max_n=3) == [7, 8, 9]
    assert _stratified_picks([7, 8], max_n=3) == [7, 8]
    assert _stratified_picks([42], max_n=3) == [42]
    assert _stratified_picks([], max_n=3) == []


def test_stratified_picks_dedups_on_tiny_lists():
    """max_n=3 over a list of 2 rounds to indices (0, 1, 1) — dedup to (0, 1)."""
    picks = _stratified_picks([10, 20], max_n=3)
    assert picks == [10, 20]


def test_make_thumbnail_uses_stratified_picks(tmp_path: Path):
    """An NWB with 5 APWaveform sweeps should have its 1st / 3rd-or-middle / 5th
    rendered in the static thumbnail (not 1/2/3)."""
    nwb = tmp_path / "cell.nwb"
    _make_nwb_with([f"ic__APWaveform__{i:03d}" for i in range(1, 6)], nwb)
    out = tmp_path / "thumb.png"
    result, status = _make_thumbnail(
        nwb, out, families=default_families(),
        reasons=["ap_amp_overshoot_mv"],  # → wanted = {ap_waveform}
    )
    assert status == "rendered"
    assert result is not None
    # We can't inspect the PNG content easily, but the rendering worked end-to-end.
    # The stratified-picks unit test above pins down the selection itself.
