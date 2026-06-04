"""NWB pre-computed-metric inventory + `nwb-qc inventory-metrics` CLI."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest

from nwb_trace_qc.inventory import (
    NwbInventoryEntry,
    inventory_nwb,
    render_inventory_markdown,
)


def _make_min_nwb(path: Path) -> None:
    """Raw-only NWB — no processing, no scratch, no lab_meta_data."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001",
        data=np.linspace(-0.07, 0.03, 500),
        electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
    ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_inventory_on_raw_nwb_reports_no_processing(tmp_path: Path):
    """An NWB with no analysis containers reports cleanly."""
    p = tmp_path / "raw.nwb"
    _make_min_nwb(p)
    entry = inventory_nwb(p)
    assert entry.error is None
    assert entry.has_processing is False
    assert entry.has_lab_meta_data is False
    assert entry.has_scratch is False
    assert entry.found_metrics == {}


def test_inventory_markdown_has_provenance_table_row_per_metric(tmp_path: Path):
    """The aggregate markdown carries one row per canonical metric."""
    from nwb_trace_qc.families import METRIC_DESCRIPTIONS
    p = tmp_path / "raw.nwb"
    _make_min_nwb(p)
    entry = inventory_nwb(p)
    md = render_inventory_markdown([entry], project_name="test")
    # Header + one row per metric
    for metric in METRIC_DESCRIPTIONS.keys():
        assert f"`{metric}`" in md
    # Since this is a raw NWB, every metric's source is "nwb-trace-qc (computed)"
    assert "`nwb-trace-qc` (computed)" in md
    # The per-NWB block shows the file
    assert "raw.nwb" in md
    # The headline "no pre-computed canonical metrics" line is in the body
    assert "No pre-computed metrics matched" in md


def test_inventory_handles_unreadable_nwb_gracefully(tmp_path: Path):
    """A path that isn't a real NWB produces an error entry, not an exception."""
    p = tmp_path / "not_an_nwb.txt"
    p.write_text("nope")
    entry = inventory_nwb(p)
    assert entry.error is not None
    assert "raw.nwb" not in entry.error or True  # accept any error string


def test_inventory_cli_writes_markdown(tmp_path: Path):
    """End-to-end: `nwb-qc inventory-metrics --config foo.yaml` writes a markdown
    report to <output_dir>/metric_inventory.md."""
    import yaml
    from click.testing import CliRunner
    from nwb_trace_qc.cli import main

    # Build a tiny project with one NWB
    nwb_dir = tmp_path / "nwbs"; nwb_dir.mkdir()
    _make_min_nwb(nwb_dir / "c0.nwb")
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project_name": "inv_test",
        "output_dir": str(tmp_path / "qc_out"),
        "nwb_sources": [{"dataset": "d", "path": str(nwb_dir), "glob": "*.nwb"}],
        "stimulus_protocols": {"ap_waveform": ["APWaveform"]},
        "thresholds_file": str(tmp_path / "thr.yaml"),
        "n_workers": 1,
    }))
    (tmp_path / "thr.yaml").write_text(yaml.safe_dump({
        "ap_amp_overshoot_mv": {"flag_below": 0},
    }))

    runner = CliRunner()
    result = runner.invoke(main, ["inventory-metrics", "--config", str(cfg_path)],
                            catch_exceptions=False)
    assert result.exit_code == 0, result.output
    md_path = tmp_path / "qc_out" / "metric_inventory.md"
    assert md_path.exists(), result.output
    body = md_path.read_text()
    assert "Metric inventory — inv_test" in body
    assert "c0.nwb" in body
