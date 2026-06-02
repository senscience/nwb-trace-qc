"""YAML thresholds → per-cell verdict + triggered-metrics list."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


# Verdict precedence: any 'fail' wins, then any 'flag', then 'pass'.
PRECEDENCE = {"fail": 2, "flag": 1, "pass": 0}


def load_thresholds(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def evaluate_metric(value: Any, rules: dict[str, Any]) -> tuple[str, str | None]:
    """Evaluate one metric against its rules; return (verdict, triggered_label_or_None).

    Rule keys understood:
      fail_above / fail_below : numeric thresholds
      flag_above / flag_below : numeric thresholds
      fail_if_false           : if True, value must be truthy or it's a fail
      fail_if_true            : if True, value must be falsy or it's a fail
      flag_if_false / flag_if_true: analogous

    Numeric NaN values trigger 'flag' (insufficient data) unless an explicit fail/flag
    rule says otherwise.
    """
    # Boolean-style rules first
    if rules.get("fail_if_false") and not value:
        return "fail", "missing"
    if rules.get("fail_if_true") and value:
        return "fail", "set_true"
    if rules.get("flag_if_false") and not value:
        return "flag", "missing"
    if rules.get("flag_if_true") and value:
        return "flag", "set_true"

    # Numeric rules
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "pass", None

    if math.isnan(v):
        # NaN — insufficient data. Surface as a soft flag.
        return "flag", "nan"

    if "fail_above" in rules and v > rules["fail_above"]:
        return "fail", f"> {rules['fail_above']}"
    if "fail_below" in rules and v < rules["fail_below"]:
        return "fail", f"< {rules['fail_below']}"
    if "flag_above" in rules and v > rules["flag_above"]:
        return "flag", f"> {rules['flag_above']}"
    if "flag_below" in rules and v < rules["flag_below"]:
        return "flag", f"< {rules['flag_below']}"
    return "pass", None


def evaluate(metrics: dict[str, Any], thresholds: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (overall_verdict, triggered).

    `triggered` is a list of dicts: {metric, value, verdict, reason}.
    """
    triggered: list[dict[str, Any]] = []
    worst = "pass"
    for metric, rules in thresholds.items():
        if metric not in metrics:
            continue
        v = metrics[metric]
        verdict, reason = evaluate_metric(v, rules)
        if verdict != "pass":
            triggered.append({
                "metric": metric,
                "value": v,
                "verdict": verdict,
                "reason": reason,
            })
            if PRECEDENCE[verdict] > PRECEDENCE[worst]:
                worst = verdict
    return worst, triggered
