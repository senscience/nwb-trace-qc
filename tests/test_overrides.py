import pandas as pd

from nwb_trace_qc.overrides import apply_overrides


def test_no_overrides_passes_computed():
    verdicts = pd.DataFrame([
        {"cell_id": "a", "computed_verdict": "pass"},
        {"cell_id": "b", "computed_verdict": "fail"},
    ])
    overrides = pd.DataFrame(columns=["cell_id", "override_verdict", "note", "reviewer", "date"])
    out = apply_overrides(verdicts, overrides)
    assert list(out["final_verdict"]) == ["pass", "fail"]


def test_override_replaces_verdict():
    verdicts = pd.DataFrame([{"cell_id": "a", "computed_verdict": "fail"}])
    overrides = pd.DataFrame([{"cell_id": "a", "override_verdict": "pass",
                                "note": "manually inspected", "reviewer": "cg", "date": "2026-01-01"}])
    out = apply_overrides(verdicts, overrides)
    assert out["final_verdict"].iloc[0] == "pass"
    assert out["override_note"].iloc[0] == "manually inspected"
    assert out["override_reviewer"].iloc[0] == "cg"


def test_empty_override_verdict_ignored():
    verdicts = pd.DataFrame([{"cell_id": "a", "computed_verdict": "flag"}])
    overrides = pd.DataFrame([{"cell_id": "a", "override_verdict": "", "note": "", "reviewer": "", "date": ""}])
    out = apply_overrides(verdicts, overrides)
    assert out["final_verdict"].iloc[0] == "flag"
