"""calibrate.suggest_thresholds + render_suggested_yaml + write_cohort_stats_json (Part 7)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from nwb_trace_qc.calibrate import (
    compute_cohort_stats,
    percentiles,
    render_suggested_yaml,
    suggest_thresholds,
    write_cohort_stats_json,
)


def _cohort_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(100)],
        "vrest_mv": rng.normal(-65.0, 4.0, 100),   # cohort tighter than bundled default
        "rs_mohm_final": rng.normal(20.0, 3.0, 100),
        "ap_amp_overshoot_mv": rng.normal(30.0, 5.0, 100),
    })


def test_percentiles_returns_quartiles():
    s = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    p = percentiles(s, ps=(10, 50, 90))
    assert p["p10"] == pytest.approx(10, abs=0.5)
    assert p["p50"] == pytest.approx(50, abs=0.5)
    assert p["p90"] == pytest.approx(90, abs=0.5)
    assert p["n"] == 11


def test_percentiles_handles_nan_and_empty():
    assert percentiles(pd.Series([float("nan"), float("nan")])) == {}
    assert percentiles(pd.Series([], dtype=float)) == {}


def test_compute_cohort_stats_excludes_bookkeeping():
    df = _cohort_df()
    stats = compute_cohort_stats(df)
    assert "cell_id" not in stats
    assert "vrest_mv" in stats and "p90" in stats["vrest_mv"]


def test_suggest_thresholds_tightens_flag_above_when_cohort_p90_is_lower():
    """Bundled flag_above=-55; cohort P90 ≈ -60 (one-sigma above the mean). Suggested
    must be the cohort P90 (tighter)."""
    df = _cohort_df()
    bundled = {
        "vrest_mv": {"flag_above": -55, "fail_above": -45},
    }
    suggested = suggest_thresholds(df, bundled)
    # Cohort vrest P90 ≈ -65 + 1.28*4 ≈ -60
    flag_above = suggested["vrest_mv"]["flag_above"]
    assert -62 <= flag_above <= -58
    # fail_above never auto-suggested
    assert suggested["vrest_mv"]["fail_above"] == -45
    # Bundled rules preserved as _bundled
    assert suggested["vrest_mv"]["_bundled"]["flag_above"] == -55


def test_suggest_thresholds_keeps_bundled_when_cohort_p90_is_looser():
    """Bundled flag_above=10; cohort P90 ~+38 (way looser). Keep bundled (tighter)."""
    df = _cohort_df()
    bundled = {
        "ap_amp_overshoot_mv": {"flag_above": 10},
    }
    suggested = suggest_thresholds(df, bundled)
    # min(bundled=10, cohort_p90~37) = 10 → bundled wins
    assert suggested["ap_amp_overshoot_mv"]["flag_above"] == 10


def test_suggest_thresholds_preserves_boolean_rules():
    df = pd.DataFrame({"qc_protocol_coverage": [True, True, False]})
    bundled = {"qc_protocol_coverage": {"fail_if_false": True}}
    suggested = suggest_thresholds(df, bundled)
    assert suggested["qc_protocol_coverage"]["fail_if_false"] is True


def test_render_suggested_yaml_parses_cleanly():
    df = _cohort_df()
    bundled = {
        "vrest_mv": {"flag_above": -55, "fail_above": -45},
        "qc_protocol_coverage": {"fail_if_false": True},
    }
    suggested = suggest_thresholds(df, bundled)
    text = render_suggested_yaml(suggested, n_cells=len(df), source_count=1)
    # Body still parses as YAML
    parsed = yaml.safe_load(text)
    assert "vrest_mv" in parsed
    assert parsed["qc_protocol_coverage"]["fail_if_false"] is True
    # Comments tell the user what the bundled + cohort values were
    assert "# bundled" in text
    assert "# cohort" in text


def test_write_cohort_stats_json_round_trip(tmp_path: Path):
    df = _cohort_df()
    out = tmp_path / "cohort.json"
    write_cohort_stats_json(df, out)
    loaded = json.loads(out.read_text())
    assert "vrest_mv" in loaded
    assert "p50" in loaded["vrest_mv"]
