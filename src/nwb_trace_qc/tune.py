"""Interactive threshold tuning — `nwb-qc tune` and the wizard's [t]une option.

Walks every metric that has a threshold rule, prompted with:
  - the current rule values
  - the cohort percentile context (from the cache)
  - the suggested value from the calibrate suggester
  - the current cohort failure count under this rule

For each rule the user can press Enter to accept the suggested default, type
a new number to override, or type `s` to skip the metric and keep its current
values. After the walk, previews the new verdict counts vs. the original and
prompts to save + re-run.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml

from .cache import filter_for_version, load_cache
from .calibrate import compute_cohort_stats, suggest_thresholds
from .thresholds import evaluate_metric, load_thresholds


NUMERIC_RULES = ("fail_above", "fail_below", "flag_above", "flag_below")


def _per_metric_fail_counts(cache_df: pd.DataFrame,
                              thresholds: dict[str, dict]) -> dict[str, int]:
    """For each metric in thresholds, count how many cached cells flag-or-fail
    on its rule alone (so the tune walk surfaces high-impact metrics first)."""
    out: dict[str, int] = {}
    for metric, rules in thresholds.items():
        if metric not in cache_df.columns:
            out[metric] = 0; continue
        count = 0
        for v in cache_df[metric]:
            verdict, _ = evaluate_metric(v, rules)
            if verdict in {"flag", "fail"}:
                count += 1
        out[metric] = count
    return out


def _preview_verdicts(cache_df: pd.DataFrame,
                      thresholds: dict[str, dict]) -> dict[str, int]:
    """Re-evaluate cohort with these thresholds; return overall verdict counts."""
    counts = {"pass": 0, "flag": 0, "fail": 0}
    for _, row in cache_df.iterrows():
        worst = "pass"
        for metric, rules in thresholds.items():
            verdict, _ = evaluate_metric(row.get(metric), rules)
            if verdict == "fail":
                worst = "fail"
            elif verdict == "flag" and worst == "pass":
                worst = "flag"
        counts[worst] += 1
    return counts


def _format_cohort(cohort: dict) -> str:
    parts = []
    for k in ("p10", "p50", "p90", "p99"):
        if k in cohort:
            parts.append(f"P{int(k[1:])}={cohort[k]:.3g}")
    return " · ".join(parts)


def _prompt_numeric_rule(rule: str, current: Any, suggested: Any) -> tuple[str, float | None]:
    """Returns (action, value). action is 'set' / 'keep' / 'skip'."""
    cur_disp = f"{current}" if current is not None else "—"
    if suggested is not None and suggested != current:
        prompt_label = f"  {rule} (current={cur_disp}, suggested={suggested})"
        default_str = f"{suggested}"
    else:
        prompt_label = f"  {rule} (current={cur_disp})"
        default_str = f"{current}" if current is not None else ""

    raw = click.prompt(prompt_label, default=default_str, show_default=True).strip()
    if not raw:
        # Use whatever the default was — suggested if it differs, else current
        if suggested is not None and suggested != current:
            return "set", float(suggested)
        return "keep", None
    if raw.lower() in ("s", "skip"):
        return "skip", None
    try:
        return "set", float(raw)
    except ValueError:
        click.secho(f"    not a number — keeping current", dim=True)
        return "keep", None


def tune_thresholds_interactive(config_path: Path, *, rerun: bool = True,
                                  only_failing: bool = False) -> bool:
    """Walk the user through tuning each threshold rule. Returns True if the
    thresholds file was modified.

    `only_failing` (default False): when True, only show metrics that currently
    have at least one cell failing/flagging — useful for tightening the verdict
    on a cohort with many ignored-but-passing metrics.
    """
    from .config import load_config
    cfg = load_config(config_path)
    cache_df = filter_for_version(load_cache(cfg.cache_path))
    if cache_df.empty:
        raise click.ClickException(
            f"cache empty at {cfg.cache_path}; run `nwb-qc run` first"
        )
    if cfg.thresholds_file is None or not cfg.thresholds_file.exists():
        raise click.ClickException(f"thresholds_file not found: {cfg.thresholds_file}")

    thresholds = load_thresholds(cfg.thresholds_file)
    cohort_stats = compute_cohort_stats(cache_df)
    suggested_all = suggest_thresholds(cache_df, thresholds)
    fail_counts = _per_metric_fail_counts(cache_df, thresholds)
    initial = _preview_verdicts(cache_df, thresholds)

    click.secho("═" * 72, dim=True)
    click.secho("Threshold tuning · ", bold=True, fg="cyan", nl=False)
    click.echo(str(cfg.thresholds_file))
    click.echo("─" * 72)
    click.echo(f"  {len(cache_df)} cells · current verdicts: ", nl=False)
    click.secho(f"pass={initial['pass']}", fg="green", nl=False); click.echo(" ", nl=False)
    click.secho(f"flag={initial['flag']}", fg="yellow", nl=False); click.echo(" ", nl=False)
    click.secho(f"fail={initial['fail']}", fg="red")
    # Walk order: highest-impact metrics first
    metrics_in_order = sorted(thresholds.keys(),
                                key=lambda m: fail_counts.get(m, 0), reverse=True)
    if only_failing:
        metrics_in_order = [m for m in metrics_in_order if fail_counts.get(m, 0) > 0]

    new_thresholds = {k: dict(v) for k, v in thresholds.items()}

    # Top-of-walk shortcut: accept-all takes the cohort-suggested defaults in
    # one keypress, skipping the per-rule prompts. Useful for a first iteration
    # where you trust the calibrate suggester.
    click.secho(
        f"  {len(metrics_in_order)} metric(s) to review. Pick an action:\n"
        "    [w] walk through each metric (one prompt per rule)\n"
        "    [a] accept all suggested values in one keystroke\n"
        "    [c] cancel (no changes)\n",
        dim=True)
    top_choice = click.prompt("  action", default="w", show_default=True).strip().lower()[:1]
    if top_choice == "c":
        click.secho("  cancelled — no changes written.", fg="red")
        return False
    if top_choice == "a":
        # Apply every cohort-suggested value (skip _bundled / _cohort metadata keys)
        n_applied = 0
        for metric in metrics_in_order:
            sugg = suggested_all.get(metric, {})
            for key, value in sugg.items():
                if key.startswith("_"):     # skip _bundled, _cohort metadata
                    continue
                if value != thresholds[metric].get(key):
                    new_thresholds[metric][key] = value
                    n_applied += 1
        if n_applied == 0:
            click.secho("  cohort suggestions match the current thresholds exactly — "
                         "no changes to apply.", dim=True)
            return False
        new_verdicts = _preview_verdicts(cache_df, new_thresholds)
        click.secho(
            f"\nPreview (after accepting all suggested): "
            f"pass={new_verdicts['pass']} flag={new_verdicts['flag']} "
            f"fail={new_verdicts['fail']}", bold=True, fg="cyan")
        if not click.confirm("Save these thresholds?", default=True):
            click.secho("  abandoned — no changes written.", fg="red")
            return False
        _write_thresholds_file(cfg.thresholds_file, new_thresholds)
        click.secho(f"  ✓ wrote {cfg.thresholds_file} ({n_applied} rule(s) updated)",
                     fg="green", bold=True)
        if rerun and click.confirm("Re-run pipeline now?", default=True):
            _rerun(cfg)
        return True

    # Per-rule walk (the [w] path)
    click.secho(
        "  For each rule: [Enter]=accept suggested · type a number to override\n"
        "  · type 's' to skip the metric (keep current values).\n",
        dim=True)
    n_changed = 0

    for i, metric in enumerate(metrics_in_order, start=1):
        rules = thresholds[metric]
        suggested = suggested_all.get(metric, {})
        cohort = cohort_stats.get(metric, {})

        click.echo("")
        click.secho(f"[{i}/{len(metrics_in_order)}] {metric}", bold=True, nl=False)
        n_fail = fail_counts.get(metric, 0)
        if n_fail:
            click.secho(f"  ({n_fail} cells affected)", fg="yellow")
        else:
            click.secho(f"  (0 cells affected — nothing currently triggers)",
                         dim=True)
        cohort_str = _format_cohort(cohort)
        if cohort_str:
            click.secho(f"    cohort: {cohort_str}", dim=True)

        # Numeric rules
        skipped = False
        for rule in NUMERIC_RULES:
            if rule not in rules:
                continue
            action, value = _prompt_numeric_rule(rule, rules[rule], suggested.get(rule))
            if action == "skip":
                skipped = True; break
            if action == "set":
                if value != rules[rule]:
                    new_thresholds[metric][rule] = value
                    n_changed += 1

        if skipped:
            new_thresholds[metric] = dict(rules)   # reset to original
            click.secho(f"  → skipped (kept current)", dim=True)

    if n_changed == 0:
        click.secho("\n  no rules changed.", dim=True)
        return False

    # Preview verdicts under the new thresholds
    new_verdicts = _preview_verdicts(cache_df, new_thresholds)
    click.echo("")
    click.secho("─" * 72, dim=True)
    click.secho(f"Preview · ", bold=True, fg="cyan", nl=False)
    click.secho(f"pass={new_verdicts['pass']}", fg="green", nl=False); click.echo(" ", nl=False)
    click.secho(f"flag={new_verdicts['flag']}", fg="yellow", nl=False); click.echo(" ", nl=False)
    click.secho(f"fail={new_verdicts['fail']}", fg="red")
    delta_fail = new_verdicts['fail'] - initial['fail']
    delta_pass = new_verdicts['pass'] - initial['pass']
    if delta_fail or delta_pass:
        click.secho(
            f"  Δ fail = {delta_fail:+d}    Δ pass = {delta_pass:+d}",
            fg="green" if delta_fail < 0 else "yellow",
        )

    if not click.confirm("\nSave these thresholds?", default=True):
        click.secho("  abandoned — no changes written.", fg="red")
        return False

    _write_thresholds_file(cfg.thresholds_file, new_thresholds)
    click.secho(f"  ✓ wrote {cfg.thresholds_file}", fg="green", bold=True)

    if rerun and click.confirm("Re-run pipeline now? (cache-fast — only threshold + report stages)",
                                default=True):
        _rerun(load_config(config_path))
    return True


def _write_thresholds_file(path: Path, new_thresholds: dict[str, dict]) -> None:
    """Write the new thresholds YAML, preserving any leading comment header."""
    original_text = path.read_text()
    header_lines = [l for l in original_text.splitlines() if l.startswith("#")]
    body = yaml.safe_dump(new_thresholds, sort_keys=False)
    out_text = ("\n".join(header_lines) + "\n\n" if header_lines else "") + body
    path.write_text(out_text)


def _rerun(cfg) -> None:
    """Re-run the pipeline and pretty-print the new verdicts."""
    from .pipeline import run as pipeline_run
    result = pipeline_run(cfg)
    click.echo("")
    click.secho(f"New verdicts: ", bold=True, nl=False)
    click.secho(f"pass={result['n_pass']}", fg="green", nl=False); click.echo(" ", nl=False)
    click.secho(f"flag={result['n_flag']}", fg="yellow", nl=False); click.echo(" ", nl=False)
    click.secho(f"fail={result['n_fail']}", fg="red")
    click.echo(f"Report: {result.get('report')}")
