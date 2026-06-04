"""Suggest QC thresholds from a cohort's metric distributions.

Pure functions so the CLI surface in `cli.py` stays thin and the suggester is
testable on a mock DataFrame.

The suggester is conservative: it only tightens *upper-bound* (`flag_above`)
rules where the cohort's P90 is below the bundled default (i.e. the cohort is
"better" than the lab-agnostic default and we can be stricter), and only loosens
*lower-bound* (`flag_below`) rules where the cohort's P10 is below the bundled
default. Fail-thresholds are NEVER auto-suggested — those should stay
laboratory-judgment calls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def percentiles(series: pd.Series, ps: Iterable[int] = (10, 50, 90, 99)) -> dict[str, float]:
    """Return {'p10': …, 'p50': …, ...} for a numeric series, dropping NaN.

    Empty / all-NaN series returns an empty dict.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    out: dict[str, float] = {}
    for p in ps:
        try:
            out[f"p{p}"] = float(clean.quantile(p / 100.0))
        except Exception:
            continue
    out["n"] = int(clean.size)
    return out


def compute_cohort_stats(cache_df: pd.DataFrame,
                         metrics: Iterable[str] | None = None) -> dict[str, dict[str, float]]:
    """For every numeric metric column in `cache_df`, compute its percentiles.

    Returns {metric_name: {'p10':…, 'p50':…, 'p90':…, 'p99':…, 'n':…}}.
    Used both by `calibrate` (to suggest thresholds) and by `pipeline.run` (to
    annotate report chips with cohort-percentile context).
    """
    if metrics is None:
        # Auto-pick numeric columns that aren't bookkeeping / paths / ids
        excluded = {
            "cell_id", "dataset", "nwb_path", "nwb_sha256", "pipeline_version",
            "computed_verdict", "final_verdict", "triggered_metrics",
            "override_note", "override_reviewer", "override_date", "compute_error",
        }
        metrics = [c for c in cache_df.columns if c not in excluded
                   and not c.endswith("_provenance")
                   and not c.endswith("_by_protocol")]
    out: dict[str, dict[str, float]] = {}
    for m in metrics:
        if m not in cache_df.columns:
            continue
        stats = percentiles(cache_df[m])
        if stats:
            out[m] = stats
    return out


def suggest_thresholds(cache_df: pd.DataFrame,
                        bundled: dict[str, dict[str, Any]],
                        quantiles: tuple[int, ...] = (10, 50, 90, 99)
                        ) -> dict[str, dict[str, Any]]:
    """Return a thresholds dict with suggested numeric values plus the original
    bundled rules preserved alongside in `_bundled` and `_cohort` annotations.

    Output schema per metric:
        { 'flag_above': <suggested>, '_bundled': {...}, '_cohort': {p10:…} }
    """
    stats = compute_cohort_stats(cache_df, metrics=list(bundled.keys()))
    suggested: dict[str, dict[str, Any]] = {}
    for metric, rules in bundled.items():
        m_stats = stats.get(metric, {})
        new_rules: dict[str, Any] = {}
        # Carry over boolean / coverage rules as-is
        for key in ("fail_if_false", "fail_if_true", "flag_if_false", "flag_if_true"):
            if key in rules:
                new_rules[key] = rules[key]

        # Numeric rules: keep bundled fails; suggest new flags from P90 / P10 when sensible
        for key in ("fail_above", "fail_below"):
            if key in rules:
                new_rules[key] = rules[key]   # never auto-suggest fails

        if "flag_above" in rules and "p90" in m_stats:
            # Tighter (smaller) than bundled is acceptable — pick the smaller
            new_rules["flag_above"] = round(min(rules["flag_above"], m_stats["p90"]), 4)
        elif "flag_above" in rules:
            new_rules["flag_above"] = rules["flag_above"]

        if "flag_below" in rules and "p10" in m_stats:
            # Looser (more negative) than bundled is acceptable — pick the larger
            new_rules["flag_below"] = round(max(rules["flag_below"], m_stats["p10"]), 4)
        elif "flag_below" in rules:
            new_rules["flag_below"] = rules["flag_below"]

        new_rules["_bundled"] = dict(rules)
        if m_stats:
            new_rules["_cohort"] = m_stats
        suggested[metric] = new_rules
    return suggested


def render_suggested_yaml(suggested: dict[str, dict[str, Any]],
                            n_cells: int,
                            source_count: int) -> str:
    """Render the suggested-thresholds dict as a YAML string with helpful comments."""
    lines: list[str] = []
    lines.append("# nwb-qc calibrate — suggested thresholds from cohort statistics")
    lines.append(f"# Cohort: {n_cells} cells across {source_count} source(s)")
    lines.append("# Per-metric P10 / P50 / P90 / P99 below; current bundled defaults shown")
    lines.append("# in comments for comparison. Edit as you see fit, then point")
    lines.append("# `thresholds_file:` in your project YAML at this file.")
    lines.append("#")
    lines.append("# Auto-suggester is conservative: it only tightens flag_above / loosens flag_below")
    lines.append("# when the cohort statistics warrant it; fail_* rules are kept as bundled.")
    lines.append("")
    for metric, rules in suggested.items():
        bundled = rules.pop("_bundled", {})
        cohort = rules.pop("_cohort", {})
        lines.append(f"{metric}:")
        if bundled:
            parts = [f"{k}: {v}" for k, v in bundled.items()]
            lines.append(f"  # bundled: {'  '.join(parts)}")
        if cohort:
            cohort_str = "  ".join(f"P{int(k[1:])}: {v:.3g}" if k.startswith("p") and k != "p"
                                    else f"{k}: {v}"
                                    for k, v in cohort.items())
            lines.append(f"  # cohort: {cohort_str}")
        for key, value in rules.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def write_cohort_stats_json(cache_df: pd.DataFrame, out_path: Path) -> Path:
    """Write the cohort percentile snapshot used by report chips."""
    stats = compute_cohort_stats(cache_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, default=str))
    return out_path
