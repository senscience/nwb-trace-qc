"""Stage-2 incremental flush + bounded result holding."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest
import yaml

from nwb_trace_qc import pipeline as pl_mod
from nwb_trace_qc.config import load_config
from nwb_trace_qc.pipeline import run as pipeline_run


def _make_nwb(path: Path, identifier: str):
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=identifier,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    sweep = pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001",
        data=np.linspace(-0.07, 0.03, 500),
        electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
    )
    nwbfile.add_acquisition(sweep)
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


@pytest.fixture
def six_nwb_project(tmp_path: Path) -> Path:
    nwb_dir = tmp_path / "nwbs"
    nwb_dir.mkdir()
    for i in range(6):
        _make_nwb(nwb_dir / f"cell-{i}.nwb", f"cell-{i}")
    thresholds = tmp_path / "thr.yaml"
    thresholds.write_text(yaml.safe_dump({
        "metrics": {"vrest_mv": {"pass": {"min": -200, "max": 200}, "flag": {"min": -200, "max": 200}}}
    }))
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project_name": "six",
        "output_dir": str(tmp_path / "qc_out"),
        "nwb_sources": [{"dataset": "d", "path": str(nwb_dir), "glob": "*.nwb"}],
        "stimulus_protocols": {"ap_waveform": ["APWaveform"]},
        "thresholds_file": str(thresholds),
        "n_workers": 1,
    }))
    return cfg_path


def test_flush_every_triggers_multiple_cache_writes(six_nwb_project: Path, monkeypatch):
    """With flush_every=2 and 6 NWBs, we should see append_rows called ≥3 times.

    Patches the append_rows imported into the pipeline module (the same symbol the
    pipeline calls). Verifies the in-flight batch never exceeds flush_every.
    """
    cfg = load_config(six_nwb_project)
    calls: list[int] = []
    original = pl_mod.append_rows

    def spy(cache_path, rows):
        calls.append(len(rows))
        return original(cache_path, rows)

    monkeypatch.setattr(pl_mod, "append_rows", spy)
    pipeline_run(cfg, flush_every=2)
    # Each non-final flush is exactly flush_every; the final flush may be smaller (0..flush_every).
    assert calls, "no cache flushes recorded"
    assert all(n <= 2 for n in calls), f"flush batch exceeded flush_every: {calls}"
    # 6 NWBs / batch of 2 → at least 3 flushes total
    assert sum(calls) == 6
    assert len(calls) >= 3
