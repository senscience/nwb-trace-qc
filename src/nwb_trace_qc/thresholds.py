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


def load_thresholds_with_overrides(
    base_path: str | Path,
    overrides_path: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load base thresholds, then layer overrides on top per-metric.

    Returns ``(merged, overrides_only)``. `overrides_only` is empty if the file
    doesn't exist — useful for the viewer to know which keys were edited.

    Override semantics: if `metric` exists in the override file, its rule
    *replaces* the base rule wholesale. (Partial-rule merging would let half-
    edited override files silently mutate untouched bounds.)
    """
    merged = dict(load_thresholds(base_path))
    if overrides_path is None:
        return merged, {}
    p = Path(overrides_path)
    if not p.exists():
        return merged, {}
    with open(p) as f:
        ov_raw = yaml.safe_load(f) or {}
    overrides = ov_raw.get("metrics", ov_raw)
    if not isinstance(overrides, dict):
        return merged, {}
    for metric, rules in overrides.items():
        if isinstance(rules, dict):
            merged[metric] = rules
    return merged, overrides


def save_threshold_overrides(
    overrides_path: str | Path,
    overrides: dict[str, dict[str, Any]],
) -> None:
    """Atomically write the override map to ``overrides_path`` as YAML.

    Wraps the dict in ``{"metrics": …}`` so the file structure matches the base
    thresholds file exactly.
    """
    p = Path(overrides_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump({"metrics": overrides}, f, sort_keys=True)
    tmp.replace(p)


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


def evaluate(metrics: dict[str, Any], thresholds: dict[str, dict[str, Any]],
             critical_metrics: set[str] | frozenset[str] | None = None
             ) -> tuple[str, list[dict[str, Any]]]:
    """Return (overall_verdict, triggered).

    `triggered` is a list of dicts: ``{metric, value, verdict, reason, critical}``.

    v0.6.0 verdict logic: only fails on **critical** metrics promote to a
    cell-level fail. Anything outside ``critical_metrics`` is "advisory" — its
    triggered chip is still surfaced (with ``critical: False``) but the cell
    verdict is capped at ``flag`` for advisory-only triggers. Pass `None` to
    use the bundled `families.DEFAULT_CRITICAL_METRICS`.
    """
    if critical_metrics is None:
        from .families import DEFAULT_CRITICAL_METRICS
        critical_metrics = DEFAULT_CRITICAL_METRICS

    triggered: list[dict[str, Any]] = []
    worst_critical = "pass"
    worst_advisory = "pass"
    for metric, rules in thresholds.items():
        if metric not in metrics:
            continue
        v = metrics[metric]
        verdict, reason = evaluate_metric(v, rules)
        if verdict == "pass":
            continue
        is_critical = metric in critical_metrics
        triggered.append({
            "metric": metric,
            "value": v,
            "verdict": verdict,
            "reason": reason,
            "critical": is_critical,
        })
        if is_critical:
            if PRECEDENCE[verdict] > PRECEDENCE[worst_critical]:
                worst_critical = verdict
        else:
            if PRECEDENCE[verdict] > PRECEDENCE[worst_advisory]:
                worst_advisory = verdict

    # Cell-level verdict = critical's worst, but advisory fails demote to flag.
    if worst_critical != "pass":
        return worst_critical, triggered
    if worst_advisory == "fail":
        return "flag", triggered
    return worst_advisory, triggered
