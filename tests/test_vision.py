"""Vision-judge mock-mode + selection + integration tests. No real API calls."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nwb_trace_qc.config import VisionJudgeConfig
from nwb_trace_qc.vision import (
    VisionVerdict,
    apply_vision_verdicts,
    run_vision_pass,
    select_borderline_cells,
)


def _verdicts_df():
    return pd.DataFrame([
        {"cell_id": "a", "nwb_sha256": "sha_a", "computed_verdict": "pass", "triggered_metrics": []},
        {"cell_id": "b", "nwb_sha256": "sha_b", "computed_verdict": "flag", "triggered_metrics": [{"metric": "rs", "verdict": "flag"}]},
        {"cell_id": "c", "nwb_sha256": "sha_c", "computed_verdict": "flag", "triggered_metrics": []},
        {"cell_id": "d", "nwb_sha256": "sha_d", "computed_verdict": "fail", "triggered_metrics": []},
    ])


def test_borderline_only_flags():
    sel = select_borderline_cells(_verdicts_df(), max_cells=10)
    assert list(sel["cell_id"]) == ["b", "c"]


def test_borderline_respects_cap():
    sel = select_borderline_cells(_verdicts_df(), max_cells=1)
    assert list(sel["cell_id"]) == ["b"]


def test_mock_provider_only_calls_flag_cells(tmp_path: Path):
    # Make a dummy thumbnail file
    png = tmp_path / "thumb.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    cfg = VisionJudgeConfig(enabled=True, provider="mock", max_borderline_cells=10)
    thumbs = {"a": [png], "b": [png], "c": [png], "d": [png]}
    metrics = {"sha_a": {}, "sha_b": {"vrest_mv": -45}, "sha_c": {}, "sha_d": {}}
    verdicts, stats = run_vision_pass(
        verdicts_df=_verdicts_df(),
        metrics_by_sha=metrics,
        thumbnails=thumbs,
        cfg=cfg,
    )
    # Only b and c (flag) were called
    called_cells = {v.cell_id for v in verdicts}
    assert called_cells == {"b", "c"}
    assert stats["n_called"] == 2
    assert stats["n_borderline"] == 2
    # Cost telemetry plumbing populated even for mock provider
    assert "tokens_in" in stats and stats["tokens_in"] > 0
    assert "tokens_out" in stats and stats["tokens_out"] > 0
    # Mock returns fixed (100, 20) tokens/call; default model is haiku ($1/M in, $5/M out)
    # → 2 calls × (100*1 + 20*5)/1e6 = 0.0004
    assert stats["estimated_usd"] == pytest.approx(0.0004, rel=1e-6)
    assert stats["stopped_by_budget"] is False


def test_soft_cap_stops_vision_pass(tmp_path: Path, monkeypatch):
    """With a tiny budget, stop after the first call; un-judged cells stay rule-based."""
    from nwb_trace_qc import vision as _vision
    png = tmp_path / "thumb.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    # Force a nonzero per-call cost by patching the mock provider's usage block
    def fake_mock(prompt, images, model, api_key):
        return ({"verdict": "flag", "confidence": 0.5, "notes": ""},
                {"tokens_in": 1000, "tokens_out": 200, "estimated_usd": 0.01})
    monkeypatch.setitem(_vision._PROVIDERS, "mock", fake_mock)
    cfg = VisionJudgeConfig(enabled=True, provider="mock", max_borderline_cells=10,
                            max_cost_usd=0.005)  # below per-call cost
    thumbs = {"a": [png], "b": [png], "c": [png], "d": [png]}
    metrics = {"sha_a": {}, "sha_b": {}, "sha_c": {}, "sha_d": {}}
    verdicts, stats = run_vision_pass(
        verdicts_df=_verdicts_df(),
        metrics_by_sha=metrics,
        thumbnails=thumbs,
        cfg=cfg,
    )
    # First call goes through (running cost is 0 going in); second is blocked.
    assert stats["n_called"] == 1
    assert stats["stopped_by_budget"] is True
    assert len(verdicts) == 1


def test_apply_vision_escalates_flag_to_fail():
    verdicts = _verdicts_df()
    vverdicts = [
        VisionVerdict(cell_id="b", verdict="fail", confidence=0.9, notes="bad", prompt_hash="h", nwb_sha256="sha_b"),
        VisionVerdict(cell_id="c", verdict="pass", confidence=0.8, notes="ok",  prompt_hash="h", nwb_sha256="sha_c"),
    ]
    out = apply_vision_verdicts(verdicts, vverdicts)
    rows = {r.cell_id: r for r in out.itertuples(index=False)}
    assert rows["b"].computed_verdict == "fail"
    assert rows["b"].vision_reason == "vision_escalated"
    # Vision pass does NOT auto-pass a flag cell
    assert rows["c"].computed_verdict == "flag"
    assert rows["c"].vision_reason == "vision_suggests_pass"
    # Non-borderline cells unchanged
    assert rows["a"].computed_verdict == "pass"
    assert rows["d"].computed_verdict == "fail"


def test_disabled_vision_returns_empty():
    cfg = VisionJudgeConfig(enabled=False)
    verdicts, stats = run_vision_pass(
        verdicts_df=_verdicts_df(),
        metrics_by_sha={},
        thumbnails={},
        cfg=cfg,
    )
    assert verdicts == []
    assert stats == {"enabled": False}


def test_missing_api_key_skips_real_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = VisionJudgeConfig(enabled=True, provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
    verdicts, stats = run_vision_pass(
        verdicts_df=_verdicts_df(),
        metrics_by_sha={"sha_b": {}, "sha_c": {}},
        thumbnails={"b": [Path("/nonexistent")], "c": [Path("/nonexistent")]},
        cfg=cfg,
    )
    assert verdicts == []
    assert "skipped_reason" in stats
