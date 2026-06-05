"""Server endpoint smoke tests — LTTB downsampler + handler routing + thumb cache."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc import server as srv_mod
from nwb_trace_qc.server import (
    _build_cells_for_viewer,
    _build_flag_cells,
    _is_nan,
    _lttb,
    _sanitise_for_json,
)


def test_lttb_passes_through_when_short():
    x = np.linspace(0, 1, 50)
    y = np.sin(x)
    xo, yo = _lttb(x, y, n_out=100)
    assert len(xo) == 50 and len(yo) == 50


def test_lttb_downsamples_to_n_out():
    x = np.linspace(0, 10, 10_000)
    y = np.sin(x) + np.random.RandomState(0).normal(0, 0.05, size=x.size)
    xo, yo = _lttb(x, y, n_out=500)
    assert len(xo) == 500
    assert xo[0] == x[0] and xo[-1] == x[-1]


def test_lttb_preserves_extrema_approximately():
    # Spike in the middle of a flat trace; LTTB should retain the peak (or close to it)
    x = np.linspace(0, 1, 10_000)
    y = np.zeros_like(x)
    y[5000] = 10.0
    xo, yo = _lttb(x, y, n_out=200)
    # The retained samples should include something near the spike
    near_spike = (xo > 0.45) & (xo < 0.55)
    assert yo[near_spike].max() >= 1.0  # spike substantially preserved


def test_build_cells_for_viewer_includes_flag_and_fail_strips_nans(tmp_path: Path):
    """The viewer cell list shows BOTH flag and fail (any non-pass); pass cells
    are excluded; per-cell NaN columns are stripped; fail rows come before flag."""
    csv = tmp_path / "qc_report.csv"
    csv.write_text(
        "cell_id,dataset,final_verdict,vrest_mv,rs_drift_pct,ap_amp_overshoot_mv\n"
        "c1,ds,pass,-65.0,2.0,80.0\n"          # pass → excluded
        "c2,ds,flag,-60.0,,75.0\n"             # rs_drift_pct stripped (NaN)
        "c3,ds,flag,-62.0,5.0,\n"              # ap_amp_overshoot_mv stripped
        "c4,ds,fail,-50.0,12.0,40.0\n"         # fail → INCLUDED in v0.5.x
    )
    cells, total = _build_cells_for_viewer(csv)
    assert total == 4
    # Pass excluded; flag + fail kept
    assert {c["cell_id"] for c in cells} == {"c2", "c3", "c4"}
    # Server-side ordering: FLAG first (borderline → triage), then FAIL.
    # Ties broken alphabetically by cell_id.
    assert cells[0]["final_verdict"] == "flag"
    assert cells[-1]["cell_id"] == "c4"
    assert cells[-1]["final_verdict"] == "fail"
    # Per-cell NaN stripping still in effect
    c2 = next(c for c in cells if c["cell_id"] == "c2")
    c3 = next(c for c in cells if c["cell_id"] == "c3")
    assert "rs_drift_pct" not in c2
    assert "ap_amp_overshoot_mv" in c2
    assert "ap_amp_overshoot_mv" not in c3
    assert "rs_drift_pct" in c3
    # Legacy alias still importable + returns the same shape
    assert _build_flag_cells(csv) == _build_cells_for_viewer(csv)


def test_sanitise_for_json_replaces_nan_with_none():
    """The JSON sanitiser drops NaN → None so the browser's JSON.parse doesn't
    choke on the invalid `NaN` token that json.dumps would otherwise emit."""
    import math, json
    payload = {
        "a": float("nan"),
        "b": [1.0, float("nan"), {"c": float("nan")}],
        "d": "fine",
        "e": 42,
    }
    clean = _sanitise_for_json(payload)
    assert clean["a"] is None
    assert clean["b"][0] == 1.0
    assert clean["b"][1] is None
    assert clean["b"][2]["c"] is None
    assert clean["d"] == "fine"
    assert clean["e"] == 42
    # And the result is valid JSON that the browser would accept
    json.loads(json.dumps(clean))    # would raise if NaN slipped through


def test_is_nan_handles_python_floats_and_pandas_na():
    import pandas as pd
    assert _is_nan(None) is True
    assert _is_nan(float("nan")) is True
    assert _is_nan(pd.NA) is True
    assert _is_nan(0.0) is False
    assert _is_nan("") is False  # empty string is a value, not NaN
    assert _is_nan(-65.3) is False


@pytest.fixture
def _isolated_server_state(tmp_path: Path, monkeypatch):
    """Reset module-level server caches so tests don't bleed state."""
    monkeypatch.setattr(srv_mod, "_THUMB_LRU", srv_mod.OrderedDict())
    monkeypatch.setattr(srv_mod, "_HANDLE_LRU", srv_mod.OrderedDict())
    monkeypatch.setattr(srv_mod, "_thumbnails_dir", tmp_path)
    monkeypatch.setattr(srv_mod, "_thumb_disk_cache_enabled", True)
    yield


def _write_min_nwb(path: Path, n_sweeps: int = 3) -> str:
    """Write a minimal HDF5 NWB with `n_sweeps` voltage acquisitions."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    for i in range(n_sweeps):
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=f"ic__APWaveform__{i:03d}",
            data=np.linspace(-0.07, 0.03, 800),
            electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
        ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    return "sha-fake-" + path.stem


def test_render_sweep_thumb_returns_png_and_caches(_isolated_server_state, tmp_path: Path):
    """First call renders + writes disk + populates LRU; second call is a memory hit."""
    nwb = tmp_path / "cell.nwb"
    sha = _write_min_nwb(nwb, n_sweeps=2)

    png1 = srv_mod._render_sweep_thumb(nwb, sha, 0, 220, 100)
    assert png1.startswith(b"\x89PNG")
    assert (sha, 0, 220, 100) in srv_mod._THUMB_LRU
    # Disk cache too
    disk = srv_mod._thumb_disk_path(sha, 0, 220, 100)
    assert disk is not None and disk.is_file()

    # Second call: same bytes (cache hit)
    png2 = srv_mod._render_sweep_thumb(nwb, sha, 0, 220, 100)
    assert png1 == png2


def test_render_sweep_thumb_bad_index_raises(_isolated_server_state, tmp_path: Path):
    nwb = tmp_path / "cell.nwb"
    sha = _write_min_nwb(nwb, n_sweeps=2)
    with pytest.raises(IndexError):
        srv_mod._render_sweep_thumb(nwb, sha, 99, 220, 100)


def test_render_sweep_thumb_loads_from_disk_on_lru_miss(_isolated_server_state,
                                                          tmp_path: Path, monkeypatch):
    """If the LRU is cleared but the disk file exists, the second call uses disk
    (which is faster than re-opening the NWB)."""
    nwb = tmp_path / "cell.nwb"
    sha = _write_min_nwb(nwb, n_sweeps=2)
    png1 = srv_mod._render_sweep_thumb(nwb, sha, 0, 220, 100)
    srv_mod._THUMB_LRU.clear()  # simulate eviction
    # Spy on _get_handle to confirm we did NOT touch the NWB the second time
    called = {"n": 0}
    real_get_handle = srv_mod._get_handle
    def spy(p):
        called["n"] += 1
        return real_get_handle(p)
    monkeypatch.setattr(srv_mod, "_get_handle", spy)
    png2 = srv_mod._render_sweep_thumb(nwb, sha, 0, 220, 100)
    assert png2 == png1
    assert called["n"] == 0  # served from disk, never opened the NWB
