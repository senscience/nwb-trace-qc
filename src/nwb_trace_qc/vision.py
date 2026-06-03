"""Optional LLM vision judge: escalate borderline (`flag`) cells via image inspection.

Off by default. Three providers: 'anthropic', 'openai', 'mock' (deterministic, no API).
Reuses already-rendered thumbnail PNGs (no separate rendering pass). Caches responses
keyed by (nwb_sha256, pipeline_version, prompt_hash) inside the same cache parquet,
so re-runs over the same data + same prompt are free.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import PIPELINE_VERSION

log = logging.getLogger(__name__)

# Hard input cap regardless of config (sanity)
_HARD_MAX_CELLS = 500
_VERDICTS = {"pass", "flag", "fail"}


@dataclass
class VisionVerdict:
    cell_id: str
    verdict: str            # 'pass' | 'flag' | 'fail'
    confidence: float       # 0..1
    notes: str
    prompt_hash: str
    nwb_sha256: str


# ─────────────────────────────────────────────────────────────
#  Prompt loading
# ─────────────────────────────────────────────────────────────

def load_prompt_template(path: Path | None) -> str:
    if path is None:
        # Bundled default
        path = Path(__file__).parent / "templates" / "vision_prompt.md"
    with open(path) as f:
        return f.read()


def prompt_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def _format_prompt(template: str, metrics: dict[str, Any]) -> str:
    # Build a one-per-line metric snapshot, skipping NaNs / None
    lines = []
    keep = ["vrest_mv", "rs_mohm_final", "rs_drift_pct", "ap_amp_overshoot_mv",
            "ap_threshold_drift_mv", "baseline_rms_mv", "rac_decay_residual_rel",
            "vm_drift_within_sweep_mv_per_s", "ap_failure_fraction", "ap_amp_cv",
            "late_instability_index", "n_sweeps_total", "qc_protocol_coverage"]
    for k in keep:
        v = metrics.get(k)
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4g}")
        else:
            lines.append(f"  {k}: {v}")
    return template.replace("{metrics_block}", "\n".join(lines) if lines else "  (no metrics)")


# ─────────────────────────────────────────────────────────────
#  Borderline selection
# ─────────────────────────────────────────────────────────────

def select_borderline_cells(verdicts_df: pd.DataFrame, max_cells: int) -> pd.DataFrame:
    """Return the subset of cells whose rule-based verdict is `flag`, capped to max_cells.

    Selection order = manifest order (the caller passes verdicts_df already ordered).
    """
    if "computed_verdict" not in verdicts_df.columns:
        return verdicts_df.head(0)
    flag = verdicts_df[verdicts_df["computed_verdict"] == "flag"].copy()
    cap = min(int(max_cells), _HARD_MAX_CELLS)
    if len(flag) > cap:
        log.info("vision: %d flag cells > cap %d; truncating", len(flag), cap)
        flag = flag.head(cap)
    return flag


# ─────────────────────────────────────────────────────────────
#  Providers
# ─────────────────────────────────────────────────────────────

def _png_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _parse_response(text: str) -> dict[str, Any]:
    """Extract a valid JSON object from a model response.

    Strategy: find the first {...} that parses. If anything fails, return a fallback.
    """
    if not text:
        return {"verdict": "flag", "confidence": 0.0, "notes": "empty response"}
    # Best-effort: pull the first JSON object
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return {"verdict": "flag", "confidence": 0.0, "notes": "unparseable response"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "flag", "confidence": 0.0, "notes": "invalid JSON"}
    v = str(obj.get("verdict", "flag")).lower()
    if v not in _VERDICTS:
        v = "flag"
    try:
        c = float(obj.get("confidence", 0.0))
        c = max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        c = 0.0
    notes = str(obj.get("notes", ""))[:300]
    return {"verdict": v, "confidence": c, "notes": notes}


def _call_anthropic(prompt: str, images: list[Path], model: str, api_key: str) -> dict[str, Any]:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("vision_judge.provider='anthropic' requires `pip install anthropic`") from e
    client = anthropic.Anthropic(api_key=api_key)
    content: list[dict[str, Any]] = []
    for img in images[:4]:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _png_to_b64(img)},
        })
    content.append({"type": "text", "text": prompt})
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": content}],
    )
    text = ""
    if resp.content:
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                text += blk.text
    return _parse_response(text)


def _call_openai(prompt: str, images: list[Path], model: str, api_key: str) -> dict[str, Any]:
    try:
        import openai
    except ImportError as e:
        raise RuntimeError("vision_judge.provider='openai' requires `pip install openai`") from e
    client = openai.OpenAI(api_key=api_key)
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images[:4]:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png_to_b64(img)}"},
        })
    resp = client.chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": parts}],
    )
    text = resp.choices[0].message.content if resp.choices else ""
    return _parse_response(text or "")


def _call_mock(prompt: str, images: list[Path], model: str, api_key: str) -> dict[str, Any]:
    """Deterministic mock for tests: returns 'flag' with the cell-id-ish fingerprint
    encoded into the notes so test assertions can pattern-match. Never makes a network call.
    """
    fingerprint = ",".join(img.name for img in images)[:80]
    return {
        "verdict": "flag",
        "confidence": 0.5,
        "notes": f"mock provider; prompt_len={len(prompt)}; images=[{fingerprint}]",
    }


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "mock": _call_mock,
}


# ─────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────

def run_vision_pass(
    *,
    verdicts_df: pd.DataFrame,
    metrics_by_sha: dict[str, dict[str, Any]],
    thumbnails: dict[str, list[Path]],
    cfg,  # VisionJudgeConfig
    cached_responses: pd.DataFrame | None = None,
) -> tuple[list[VisionVerdict], dict[str, Any]]:
    """Run the vision judge against borderline cells, returning (verdicts, stats).

    `metrics_by_sha`: nwb_sha256 → metric dict (so we don't have to look up by cell_id).
    `thumbnails`: cell_id → list of PNG paths.
    `cached_responses`: DataFrame with columns [nwb_sha256, pipeline_version, prompt_hash,
                       vision_verdict, vision_confidence, vision_notes], or None.
    """
    if not cfg.enabled:
        return [], {"enabled": False}

    template = load_prompt_template(cfg.prompt_template)
    p_hash = prompt_hash(template)

    provider_fn = _PROVIDERS.get(cfg.provider)
    if provider_fn is None:
        raise ValueError(f"unknown provider {cfg.provider!r}")

    api_key = ""
    if cfg.provider in {"anthropic", "openai"}:
        api_key = os.environ.get(cfg.api_key_env, "")
        if not api_key:
            log.warning("vision: %s not set; skipping vision pass", cfg.api_key_env)
            return [], {"enabled": True, "skipped_reason": f"{cfg.api_key_env} not set"}

    # Build a fast cache lookup
    cache_key_to_row: dict[tuple[str, str], dict[str, Any]] = {}
    if cached_responses is not None and not cached_responses.empty:
        for _, r in cached_responses.iterrows():
            cache_key_to_row[(r["nwb_sha256"], r["prompt_hash"])] = r.to_dict()

    selected = select_borderline_cells(verdicts_df, cfg.max_borderline_cells)
    out: list[VisionVerdict] = []
    stats = {
        "enabled": True, "provider": cfg.provider, "model": cfg.model,
        "n_borderline": int(len(selected)),
        "n_cached": 0, "n_called": 0, "n_errors": 0,
    }

    for r in selected.itertuples(index=False):
        sha = r.nwb_sha256
        cell_id = r.cell_id
        cached = cache_key_to_row.get((sha, p_hash)) if cfg.cache_responses else None
        if cached:
            out.append(VisionVerdict(
                cell_id=cell_id,
                verdict=str(cached["vision_verdict"]),
                confidence=float(cached["vision_confidence"]),
                notes=str(cached["vision_notes"]),
                prompt_hash=p_hash,
                nwb_sha256=sha,
            ))
            stats["n_cached"] += 1
            continue
        imgs = thumbnails.get(cell_id, [])
        if not imgs:
            log.info("vision: %s has no thumbnails; skipping", cell_id)
            continue
        prompt = _format_prompt(template, metrics_by_sha.get(sha, {}))
        try:
            resp = provider_fn(prompt, imgs, cfg.model, api_key)
        except Exception as e:  # noqa: BLE001
            log.warning("vision call failed for %s: %s", cell_id, e)
            stats["n_errors"] += 1
            continue
        out.append(VisionVerdict(
            cell_id=cell_id,
            verdict=resp["verdict"],
            confidence=resp["confidence"],
            notes=resp["notes"],
            prompt_hash=p_hash,
            nwb_sha256=sha,
        ))
        stats["n_called"] += 1

    return out, stats


def apply_vision_verdicts(verdicts_df: pd.DataFrame,
                          vision_verdicts: list[VisionVerdict]) -> pd.DataFrame:
    """Integrate vision verdicts into the per-cell verdicts DataFrame.

    Precedence (the human-override step happens *after* this in pipeline.run):
      - rules `fail` → final `fail` (vision can't downgrade)
      - rules `flag` + vision `fail` → final `fail` ("vision_escalated")
      - rules `flag` + vision `pass` → final stays `flag` ("vision_suggests_pass")
      - otherwise → final = computed verdict

    Adds columns: vision_verdict, vision_confidence, vision_notes,
                  computed_verdict_with_vision, vision_reason.
    """
    df = verdicts_df.copy()
    by_cell = {v.cell_id: v for v in vision_verdicts}
    df["vision_verdict"] = df["cell_id"].map(lambda cid: by_cell[cid].verdict if cid in by_cell else None)
    df["vision_confidence"] = df["cell_id"].map(lambda cid: by_cell[cid].confidence if cid in by_cell else None)
    df["vision_notes"] = df["cell_id"].map(lambda cid: by_cell[cid].notes if cid in by_cell else None)
    new_verdicts = []
    reasons = []
    for r in df.itertuples(index=False):
        rule = getattr(r, "computed_verdict", "pass")
        vv = getattr(r, "vision_verdict", None)
        if rule == "fail":
            new_verdicts.append("fail"); reasons.append("")
        elif rule == "flag" and vv == "fail":
            new_verdicts.append("fail"); reasons.append("vision_escalated")
        elif rule == "flag" and vv == "pass":
            new_verdicts.append("flag"); reasons.append("vision_suggests_pass")
        else:
            new_verdicts.append(rule); reasons.append("")
    df["computed_verdict_with_vision"] = new_verdicts
    df["vision_reason"] = reasons
    # The downstream override layer reads `computed_verdict`; swap it in
    df["computed_verdict"] = new_verdicts
    return df
