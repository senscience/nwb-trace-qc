"""Tests for HDF5/Zarr NWB I/O dispatch + fingerprint hashing.

Writes a minimal NWB to HDF5 and to Zarr via the real pynwb/hdmf-zarr writers,
then verifies open_nwb dispatches correctly and the fingerprint hash is stable
and changes when content changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest
from hdmf_zarr import NWBZarrIO

from nwb_trace_qc.nwb_io import (
    is_hdf5,
    is_nwb,
    is_zarr,
    nwb_mtime,
    nwb_sha256,
    nwb_size,
    open_nwb,
)


def _make_minimal_nwbfile() -> pynwb.NWBFile:
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-cell-001",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0",
        description="test electrode",
        device=nwbfile.create_device(name="amp1", description="test amplifier"),
    )
    sweep = pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001",
        data=np.linspace(-0.07, 0.03, 1000),
        electrode=elec,
        gain=1.0,
        starting_time=0.0,
        rate=10000.0,
        unit="volts",
    )
    nwbfile.add_acquisition(sweep)
    return nwbfile


@pytest.fixture
def hdf5_nwb(tmp_path: Path) -> Path:
    p = tmp_path / "cell_a.nwb"
    with pynwb.NWBHDF5IO(str(p), mode="w") as io:
        io.write(_make_minimal_nwbfile())
    return p


@pytest.fixture
def zarr_nwb(tmp_path: Path) -> Path:
    p = tmp_path / "cell_b.nwb.zarr"
    with NWBZarrIO(str(p), mode="w") as io:
        io.write(_make_minimal_nwbfile())
    return p


def test_is_nwb_detects_both(hdf5_nwb: Path, zarr_nwb: Path):
    assert is_hdf5(hdf5_nwb) and not is_zarr(hdf5_nwb)
    assert is_zarr(zarr_nwb) and not is_hdf5(zarr_nwb)
    assert is_nwb(hdf5_nwb) and is_nwb(zarr_nwb)


def test_open_nwb_hdf5_reads_acquisition(hdf5_nwb: Path):
    with open_nwb(hdf5_nwb) as nwbfile:
        assert "ic__APWaveform__001" in nwbfile.acquisition


def test_open_nwb_zarr_reads_acquisition(zarr_nwb: Path):
    with open_nwb(zarr_nwb) as nwbfile:
        assert "ic__APWaveform__001" in nwbfile.acquisition


def test_zarr_size_is_dir_total(zarr_nwb: Path):
    size = nwb_size(zarr_nwb)
    # Many small chunk + .zattrs/.zgroup files; total should be modest but >0
    assert size > 0
    # And matches manual recursive sum
    manual = sum(p.stat().st_size for p in zarr_nwb.rglob("*") if p.is_file())
    assert size == manual


def test_zarr_fingerprint_is_stable_and_changes_on_modification(zarr_nwb: Path):
    h1 = nwb_sha256(zarr_nwb)
    h2 = nwb_sha256(zarr_nwb)
    assert h1 == h2  # stable across calls
    # Modify a chunk file's size → fingerprint must change
    chunk_files = [p for p in zarr_nwb.rglob("*") if p.is_file() and p.name not in {".zgroup", ".zattrs"}]
    assert chunk_files
    chunk_files[0].write_bytes(chunk_files[0].read_bytes() + b"extra")
    h3 = nwb_sha256(zarr_nwb)
    assert h3 != h1


def test_zarr_mtime_uses_marker_files(zarr_nwb: Path):
    m = nwb_mtime(zarr_nwb)
    assert m > 0


def test_hdf5_sha256_is_streaming_content_hash(hdf5_nwb: Path):
    import hashlib
    expected = hashlib.sha256(hdf5_nwb.read_bytes()).hexdigest()
    assert nwb_sha256(hdf5_nwb) == expected


def test_open_nwb_rejects_non_nwb(tmp_path: Path):
    p = tmp_path / "not_an_nwb.txt"
    p.write_text("nope")
    with pytest.raises(ValueError):
        with open_nwb(p):
            pass
