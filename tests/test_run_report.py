"""Pipeline emits a structured run_report.json with per-stage timing + counts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest
import yaml

from nwb_trace_qc.config import load_config
from nwb_trace_qc.pipeline import run as pipeline_run


def _make_minimal_nwbfile(identifier: str = "cell-1") -> pynwb.NWBFile:
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier=identifier,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="test electrode",
        device=nwbfile.create_device(name="amp1", description="test amp"),
    )
    sweep = pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001",
        data=np.linspace(-0.07, 0.03, 1000),
        electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
    )
    nwbfile.add_acquisition(sweep)
    return nwbfile


@pytest.fixture
def tiny_project(tmp_path: Path) -> Path:
    """Write two minimal NWBs + a project YAML + a permissive thresholds file."""
    nwb_dir = tmp_path / "nwbs"
    nwb_dir.mkdir()
    for cid in ("cell-1", "cell-2"):
        with pynwb.NWBHDF5IO(str(nwb_dir / f"{cid}.nwb"), mode="w") as io:
            io.write(_make_minimal_nwbfile(cid))
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(yaml.safe_dump({
        "metrics": {
            "vrest_mv": {"pass": {"min": -200, "max": 200}, "flag": {"min": -200, "max": 200}},
        },
    }))
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project_name": "tiny",
        "output_dir": str(tmp_path / "qc_out"),
        "nwb_sources": [{"dataset": "ds1", "path": str(nwb_dir), "glob": "*.nwb"}],
        "stimulus_protocols": {"ap_waveform": ["APWaveform"]},
        "thresholds_file": str(thresholds),
        "n_workers": 1,
    }))
    return cfg_path


def test_run_emits_run_report_json(tiny_project: Path):
    cfg = load_config(tiny_project)
    result = pipeline_run(cfg)
    rpt_path = Path(result["run_report"])
    assert rpt_path.exists()
    rpt = json.loads(rpt_path.read_text())
    # Top-level
    for key in ("started_at", "finished_at", "elapsed_s", "stages",
                "verdicts", "memory", "system", "manifest_stats", "compute_errors"):
        assert key in rpt, f"missing {key}"
    # Per-stage timing
    for stage in ("manifest_build", "metric_compute", "thresholds",
                  "thumbnails", "vision", "overrides", "report"):
        assert stage in rpt["stages"], f"missing stage {stage}"
        assert "elapsed_s" in rpt["stages"][stage]
        assert rpt["stages"][stage]["elapsed_s"] >= 0
    # Memory: peak RSS reported
    assert rpt["memory"]["peak_rss_mb"] > 0
    # Verdicts sum to total
    v = rpt["verdicts"]
    assert v["pass"] + v["flag"] + v["fail"] == v["total"]


def test_run_report_after_cache_hit_records_zero_new(tiny_project: Path):
    cfg = load_config(tiny_project)
    pipeline_run(cfg)  # warm the cache
    result = pipeline_run(cfg)  # second run hits cache
    rpt = json.loads(Path(result["run_report"]).read_text())
    assert rpt["stages"]["metric_compute"]["n_new"] == 0
    assert rpt["stages"]["metric_compute"]["n_cache_hits"] >= 2


def test_progress_callback_invoked(tiny_project: Path):
    cfg = load_config(tiny_project)
    calls: list[tuple[str, int, int]] = []
    pipeline_run(cfg, progress_callback=lambda s, d, t: calls.append((s, d, t)))
    # At least one callback per stage start
    stages_seen = {c[0] for c in calls}
    assert "manifest_build" in stages_seen
    assert "metric_compute" in stages_seen
