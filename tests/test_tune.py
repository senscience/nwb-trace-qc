"""Interactive threshold tuning + the standalone `nwb-qc tune` subcommand."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pytest
import yaml
from click.testing import CliRunner

from nwb_trace_qc.tune import (
    _per_metric_fail_counts,
    _preview_verdicts,
    tune_thresholds_interactive,
)


def test_per_metric_fail_counts_counts_only_failing_cells():
    df = pd.DataFrame({"vrest_mv": [-65, -50, -42, -70, -55]})
    thresholds = {"vrest_mv": {"flag_above": -55, "fail_above": -45}}
    # -65: pass, -50: flag, -42: fail, -70: pass, -55: pass (boundary, strict >)
    counts = _per_metric_fail_counts(df, thresholds)
    assert counts["vrest_mv"] == 2   # -50 flag + -42 fail


def test_per_metric_fail_counts_missing_column_returns_zero():
    df = pd.DataFrame({"vrest_mv": [-65]})
    counts = _per_metric_fail_counts(df, {"nonexistent_metric": {"flag_above": 1}})
    assert counts == {"nonexistent_metric": 0}


def test_preview_verdicts_picks_worst_across_metrics():
    """A cell that flags on one metric and fails on another rolls up to fail."""
    df = pd.DataFrame({
        "vrest_mv": [-65, -50, -42],    # pass / flag / fail
        "rs_drift_pct": [10, 50, 5],    # pass / fail / pass
    })
    thresholds = {
        "vrest_mv":     {"flag_above": -55, "fail_above": -45},
        "rs_drift_pct": {"flag_above": 20, "fail_above": 30},
    }
    counts = _preview_verdicts(df, thresholds)
    # row 0: pass+pass → pass
    # row 1: flag+fail → fail
    # row 2: fail+pass → fail
    assert counts == {"pass": 1, "flag": 0, "fail": 2}


def _make_min_nwb(path: Path, identifier: str = "c"):
    """Minimal NWB so we can exercise the tune flow against a real cache."""
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=identifier,
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


@pytest.fixture
def tuned_project(tmp_path: Path) -> Path:
    """Build a project that has been run once so the cache is warm."""
    nwb_dir = tmp_path / "nwbs"; nwb_dir.mkdir()
    for i in range(4):
        _make_min_nwb(nwb_dir / f"c{i}.nwb", f"c{i}")
    thresholds_path = tmp_path / "thr.yaml"
    thresholds_path.write_text(yaml.safe_dump({
        "vrest_mv": {"flag_above": -55, "fail_above": -45},
        "rs_drift_pct": {"flag_above": 20, "fail_above": 30},
    }))
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project_name": "tuned",
        "output_dir": str(tmp_path / "qc_out"),
        "nwb_sources": [{"dataset": "d", "path": str(nwb_dir), "glob": "*.nwb"}],
        "stimulus_protocols": {"ap_waveform": ["APWaveform"]},
        "thresholds_file": str(thresholds_path),
        "n_workers": 1,
    }))
    # Prime the cache by running the pipeline once
    from nwb_trace_qc.config import load_config
    from nwb_trace_qc.pipeline import run as pipeline_run
    pipeline_run(load_config(cfg_path))
    return cfg_path


def test_tune_subcommand_accepts_all_suggestions(tuned_project: Path):
    """Top-of-walk [a]ccept-all takes every suggested value in one keypress."""
    from nwb_trace_qc.cli import main

    runner = CliRunner()
    # Input: 'a' for accept-all, 'y' to save, 'n' to skip rerun (or whatever)
    input_str = "a\ny\nn\n"
    result = runner.invoke(main, ["tune", "--config", str(tuned_project), "--no-rerun"],
                            input=input_str, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    # One of: wrote thresholds, or "no changes to apply" if suggester matched bundled exactly
    assert ("wrote" in result.output) or ("no changes to apply" in result.output)


def test_tune_top_of_walk_cancel(tuned_project: Path):
    """Top-of-walk [c]ancel exits without writing anything."""
    from nwb_trace_qc.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["tune", "--config", str(tuned_project), "--no-rerun"],
                            input="c\n", catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "cancelled" in result.output


def test_tune_skip_metric_leaves_rule_unchanged(tmp_path: Path, monkeypatch, tuned_project: Path):
    """Typing 's' at the first rule of a metric skips the whole metric."""
    # Pre-read the thresholds file
    from nwb_trace_qc.config import load_config
    cfg = load_config(tuned_project)
    before = yaml.safe_load(cfg.thresholds_file.read_text())

    answers = iter([
        "w",        # top-of-walk: walk through (not accept-all)
        "s",        # rs_drift_pct first rule: skip → keep
        "s",        # vrest_mv first rule: skip → keep
    ])
    monkeypatch.setattr("click.prompt", lambda *a, **kw: next(answers))
    # We won't reach a confirm prompt since no rules change; assert no exception
    monkeypatch.setattr("click.confirm", lambda *a, **kw: False)

    changed = tune_thresholds_interactive(tuned_project, rerun=False)
    assert changed is False

    after = yaml.safe_load(cfg.thresholds_file.read_text())
    # File body matches before — no rules changed
    assert before == after


def test_tune_override_a_numeric_rule_writes_back(tuned_project: Path, monkeypatch):
    """Type a number at one prompt → that single rule is updated; the rest
    accept the suggested default (Enter)."""
    from nwb_trace_qc.config import load_config
    cfg = load_config(tuned_project)
    before = yaml.safe_load(cfg.thresholds_file.read_text())

    # NUMERIC_RULES iterates fail_above before flag_above. Metrics sort by
    # fail-count desc with ties broken by insertion order in the thresholds
    # YAML, so rs_drift_pct is walked first. The first prompt the user sees is
    # rs_drift_pct.fail_above — answer "99" here, accept defaults for the rest.
    answers = iter([
        "w",       # top-of-walk: walk per-rule
        "99",      # rs_drift_pct.fail_above → 99 (was 30)
        "",        # rs_drift_pct.flag_above → keep suggested
        "",        # vrest_mv.fail_above → keep suggested
        "",        # vrest_mv.flag_above → keep suggested
    ])
    monkeypatch.setattr("click.prompt", lambda *a, **kw: next(answers, ""))
    monkeypatch.setattr("click.confirm", lambda *a, **kw: True)

    changed = tune_thresholds_interactive(tuned_project, rerun=False)
    assert changed is True

    after = yaml.safe_load(cfg.thresholds_file.read_text())
    assert after["rs_drift_pct"]["fail_above"] == 99.0
    # vrest_mv.fail_above is unchanged (accepted suggested, which equals bundled)
    assert after["vrest_mv"]["fail_above"] == before["vrest_mv"]["fail_above"]
