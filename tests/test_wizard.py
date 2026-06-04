"""`nwb-qc start` wizard smoke test — drive the five prompts with scripted input."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest
from click.testing import CliRunner

from nwb_trace_qc.cli import main


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
def wizard_tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    ds_dir = root / "ds1"
    ds_dir.mkdir(parents=True)
    for i in range(2):
        _make_nwb(ds_dir / f"cell-{i}.nwb", f"cell-{i}")
    return root


def test_wizard_quits_at_inspect(wizard_tree: Path, tmp_path: Path):
    """Pressing 'q' at the first prompt exits cleanly with code 1 and no config written."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        input="q\n",
    )
    assert result.exit_code == 1, result.output
    assert "aborted at inspect" in result.output
    assert not out_yaml.exists()


def test_wizard_happy_path(wizard_tree: Path, tmp_path: Path):
    """accept config → run dry-run → run pipeline → done (no opening browser)."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    # 5 prompts: [y]es → [a]ccept → [r]un → (no prompt after run output) → [d]one
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        input="y\na\nr\nd\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert out_yaml.exists()
    # Run report was generated
    rpt = list((tmp_path).rglob("run_report.json"))
    assert rpt, f"no run_report.json under {tmp_path}; output:\n{result.output}"


def test_wizard_auto_writes_cohort_stats_and_suggested_thresholds(wizard_tree: Path,
                                                                     tmp_path: Path):
    """After a successful wizard run, cohort_stats.json (next to run_report.json)
    and `<stem>_thresholds_suggested.yaml` (next to the active thresholds) exist."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        input="y\na\nr\nd\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    cohort_files = list(tmp_path.rglob("cohort_stats.json"))
    assert cohort_files, f"no cohort_stats.json under {tmp_path}; output:\n{result.output}"
    import json as _json
    body = _json.loads(cohort_files[0].read_text())
    assert isinstance(body, dict) and body, "cohort_stats.json was written empty"

    suggested = list(tmp_path.rglob("*thresholds_suggested.yaml"))
    assert suggested, (
        f"no *_thresholds_suggested.yaml under {tmp_path}; output:\n{result.output}"
    )

    # The outcome stage advertised both files
    assert "suggested thresholds" in result.output
    assert "cohort stats" in result.output
