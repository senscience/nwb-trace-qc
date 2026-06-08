"""v0.8.0 curation-first workflow:
- upsert_override / delete_override roundtrip
- POST /api/curation writes the row + refreshes final_verdict in-memory
- report.render_html groups curated vs awaiting and embeds no <img> tags
"""
from __future__ import annotations

import csv
import json
from datetime import date as _date
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nwb_trace_qc import server as srv_mod
from nwb_trace_qc.overrides import (
    delete_override,
    init_overrides_file,
    load_overrides,
    upsert_override,
)
from nwb_trace_qc.report import render_html


# ─── upsert_override / delete_override ────────────────────────────

def test_upsert_override_creates_and_updates(tmp_path: Path):
    path = tmp_path / "qc_overrides.csv"
    upsert_override(path, cell_id="c1", override_verdict="pass",
                     note="looks clean", reviewer="cristina", date="2026-06-08")
    df = load_overrides(path)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["cell_id"] == "c1"
    assert r["override_verdict"] == "pass"
    assert r["note"] == "looks clean"
    assert r["reviewer"] == "cristina"
    assert r["date"] == "2026-06-08"
    # Re-upsert with new verdict — same cell_id, replaces, not duplicates
    upsert_override(path, cell_id="c1", override_verdict="fail",
                     note="actually bad on closer look", reviewer="cristina",
                     date="2026-06-09")
    df = load_overrides(path)
    assert len(df) == 1
    assert df.iloc[0]["override_verdict"] == "fail"
    assert df.iloc[0]["date"] == "2026-06-09"


def test_delete_override_removes_row(tmp_path: Path):
    path = tmp_path / "qc_overrides.csv"
    init_overrides_file(path)
    upsert_override(path, cell_id="c1", override_verdict="pass",
                     note="", reviewer="cristina", date="2026-06-08")
    upsert_override(path, cell_id="c2", override_verdict="fail",
                     note="", reviewer="cristina", date="2026-06-08")
    assert delete_override(path, cell_id="c1") is True
    df = load_overrides(path)
    assert len(df) == 1
    assert df.iloc[0]["cell_id"] == "c2"
    # Idempotent: deleting again is a no-op
    assert delete_override(path, cell_id="c1") is False


def test_delete_override_missing_file_is_safe(tmp_path: Path):
    assert delete_override(tmp_path / "nope.csv", cell_id="c1") is False


# ─── /api/curation endpoint roundtrip ─────────────────────────────

@pytest.fixture
def _viewer_state(tmp_path: Path, monkeypatch):
    """Pre-populate the server module's state as if `nwb-qc serve` had booted."""
    monkeypatch.setattr(srv_mod, "_viewer_cells", [
        {"cell_id": "c1", "dataset": "ds", "final_verdict": "flag",
         "computed_verdict": "flag", "triggered_metrics": []},
        {"cell_id": "c2", "dataset": "ds", "final_verdict": "fail",
         "computed_verdict": "fail", "triggered_metrics": []},
    ])
    monkeypatch.setattr(srv_mod, "_cell_sha", {"c1": "sha1", "c2": "sha2"})
    monkeypatch.setattr(srv_mod, "_manifest_lookup",
                          {"c1": str(tmp_path / "c1.nwb"), "c2": str(tmp_path / "c2.nwb")})
    monkeypatch.setattr(srv_mod, "_overrides_path", tmp_path / "qc_overrides.csv")
    monkeypatch.setattr(srv_mod, "_curator", "cristina")
    return tmp_path


def test_curation_save_writes_row_and_refreshes_in_memory(_viewer_state: Path):
    overrides_path = srv_mod._overrides_path
    # Simulate the POST handler body for cell c1 → PASS.
    body = {"cell_id": "c1", "verdict": "pass", "note": "looks clean"}
    # Drive the same logic the endpoint does:
    today = _date.today().isoformat()
    upsert_override(overrides_path, cell_id=body["cell_id"],
                     override_verdict=body["verdict"], note=body["note"],
                     reviewer=srv_mod._curator, date=today)
    refreshed = srv_mod._reapply_overrides_to_cells()
    by_id = {c["cell_id"]: c for c in refreshed}
    assert by_id["c1"]["final_verdict"] == "pass"
    assert by_id["c1"]["override_verdict"] == "pass"
    assert by_id["c1"]["override_reviewer"] == "cristina"
    assert by_id["c1"]["override_note"] == "looks clean"
    assert by_id["c1"]["override_date"] == today
    # The other cell is untouched
    assert by_id["c2"]["final_verdict"] == "fail"
    # CSV is well-formed
    df = pd.read_csv(overrides_path)
    assert set(df.columns) >= {"cell_id", "override_verdict", "note",
                                  "reviewer", "date"}


def test_curation_clear_drops_row_and_reverts_verdict(_viewer_state: Path):
    overrides_path = srv_mod._overrides_path
    # First save then clear
    upsert_override(overrides_path, cell_id="c1", override_verdict="pass",
                     note="oops", reviewer="cristina", date="2026-06-08")
    srv_mod._reapply_overrides_to_cells()
    assert next(c for c in srv_mod._viewer_cells if c["cell_id"] == "c1")["final_verdict"] == "pass"
    # Clear it
    delete_override(overrides_path, cell_id="c1")
    refreshed = srv_mod._reapply_overrides_to_cells()
    c1 = next(c for c in refreshed if c["cell_id"] == "c1")
    assert c1["final_verdict"] == "flag"  # back to computed_verdict
    assert "override_verdict" not in c1
    assert "override_reviewer" not in c1


def test_reapply_overrides_with_no_csv_returns_cells_unchanged(_viewer_state: Path):
    # No overrides file exists yet. The reapply call should leave cells intact.
    refreshed = srv_mod._reapply_overrides_to_cells()
    by_id = {c["cell_id"]: c for c in refreshed}
    assert by_id["c1"]["final_verdict"] == "flag"
    assert by_id["c2"]["final_verdict"] == "fail"
    assert "override_verdict" not in by_id["c1"]


# ─── Report shape — no thumbnails, two sections ───────────────────

def _tiny_report_df(curated: bool) -> pd.DataFrame:
    """Build a report_df fixture matching what pipeline.run would produce."""
    rows = []
    rows.append({
        "cell_id": "c-auto-fail", "dataset": "ds", "nwb_path": "/x.nwb",
        "nwb_sha256": "sha-a",
        "computed_verdict": "fail", "final_verdict": "fail",
        "triggered_metrics": json.dumps([
            {"metric": "vrest_mv", "verdict": "fail", "reason": "< -95",
             "critical": True, "value": -98.0},
        ]),
        "override_verdict": "", "override_note": "", "override_reviewer": "",
        "override_date": "",
        "vrest_mv": -98.0, "rs_drift_pct": 5.0, "ap_amp_overshoot_mv": 30.0,
        "n_sweeps_total": 50, "n_sweeps_clipped": 0, "n_sweeps_nan": 0,
        "qc_protocol_coverage": True, "baseline_rms_mv": 0.5,
    })
    rows.append({
        "cell_id": "c-curated-pass", "dataset": "ds", "nwb_path": "/y.nwb",
        "nwb_sha256": "sha-b",
        "computed_verdict": "flag", "final_verdict": "pass" if curated else "flag",
        "triggered_metrics": json.dumps([
            {"metric": "rin_mohm", "verdict": "flag", "reason": "nan",
             "critical": False, "value": float("nan")},
        ]),
        "override_verdict": "pass" if curated else "",
        "override_note": "verified clean — flag was nan" if curated else "",
        "override_reviewer": "cristina" if curated else "",
        "override_date": "2026-06-08" if curated else "",
        "vrest_mv": -70.0, "rs_drift_pct": 8.0, "ap_amp_overshoot_mv": 35.0,
        "n_sweeps_total": 60, "n_sweeps_clipped": 0, "n_sweeps_nan": 0,
        "qc_protocol_coverage": True, "baseline_rms_mv": 0.4,
    })
    return pd.DataFrame(rows)


def test_report_renders_without_img_tags():
    df = _tiny_report_df(curated=True)
    html_str = render_html(df, thumbnails={}, project_name="t",
                            pipeline_version="0.8.0")
    # Thumbnails are dropped from the v0.8.0 report — viewer owns sweep
    # exploration now. Make sure no <img> tags slipped through.
    assert "<img " not in html_str
    assert "data:image/png" not in html_str


def test_report_has_curated_and_awaiting_sections():
    df = _tiny_report_df(curated=True)
    html_str = render_html(df, thumbnails={}, project_name="t",
                            pipeline_version="0.8.0")
    # Two top-level sections
    assert "Awaiting review" in html_str
    assert "Curated" in html_str
    # Curated cell shows up with curator + date in the override block
    assert "cristina" in html_str
    assert "2026-06-08" in html_str
    # The curated row's cell_id appears under the Curated section heading,
    # and the auto-fail cell appears under Awaiting. Both must be present.
    assert "c-curated-pass" in html_str
    assert "c-auto-fail" in html_str


def test_report_empty_curated_section_shows_empty_state():
    df = _tiny_report_df(curated=False)  # no overrides
    html_str = render_html(df, thumbnails={}, project_name="t",
                            pipeline_version="0.8.0")
    assert "No decisions saved yet" in html_str


def test_report_shows_curate_in_viewer_link():
    df = _tiny_report_df(curated=False)
    html_str = render_html(df, thumbnails={}, project_name="t",
                            pipeline_version="0.8.0")
    # The deep link CTA text matches the v0.8.0 curation framing
    assert "Curate in viewer" in html_str
