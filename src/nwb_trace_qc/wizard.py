"""Five-stage interactive flow for `nwb-qc start <root>`.

Composes the existing CLI building blocks (inspect, init-config, list-cells, run)
with a pause-and-confirm prompt between every stage and a live progress line
during the metric-compute stage.

  1. Inspect      — print inventory; [Enter] continue, [q] quit
  2. Propose      — generate project YAML; [a]ccept / [e]dit in $EDITOR / [q]uit
  3. Dry-run      — list discovered cells; [r]un / [b]ack / [q]uit
  4. Run          — execute pipeline with progress; on completion show summary
  5. Outcome      — show paths; [s]erve viewer / [o]pen report / [Enter] done
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Callable

import click

from .config import load_config
from .manifest import build_manifest, unique_nwbs
from .pipeline import run as pipeline_run


def _hr(char: str = "─") -> None:
    click.secho(char * 72, dim=True)


def _stage_banner(n: int, total: int, title: str) -> None:
    """Render the bold STAGE N/M · title banner."""
    click.secho("═" * 72, dim=True)
    click.secho(f"STAGE {n}/{total} · {title}", bold=True, fg="cyan")
    _hr()


def _eta(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0 or total <= 0:
        return "—"
    rate = done / max(elapsed_s, 1e-6)
    remaining = (total - done) / max(rate, 1e-6)
    if remaining < 60:   return f"{remaining:.0f}s"
    if remaining < 3600: return f"{remaining/60:.1f}m"
    return f"{remaining/3600:.1f}h"


def _bar(done: int, total: int, width: int = 22) -> str:
    if total <= 0:
        return "░" * width
    frac = min(1.0, max(0.0, done / total))
    filled = int(round(width * frac))
    return "█" * filled + "░" * (width - filled)


def _make_progress_callback() -> tuple[Callable[[str, int, int], None], list[float]]:
    """Returns (callback, [stage_t0]). The callback prints a carriage-return
    progress line on TTY stdout. On non-TTY, stays silent (logging already
    covers per-stage announcements).
    """
    is_tty = sys.stdout.isatty()
    state: dict[str, float] = {}

    def cb(stage: str, done: int, total: int) -> None:
        now = time.time()
        if stage not in state:
            state[stage] = now
            if is_tty:
                click.echo("")
                click.secho("  ▸ ", fg="cyan", nl=False)
                click.secho(f"{stage}", fg="cyan", bold=True, nl=False)
                click.secho(" starting…", dim=True)
        if not is_tty:
            return
        elapsed = now - state[stage]
        stage_lbl = click.style(f"[{stage}]", fg="cyan")
        if total > 0:
            pct = 100.0 * done / total
            bar = click.style(_bar(done, total), fg="green" if pct >= 100 else "cyan")
            counts = click.style(f"{done}/{total}", bold=True)
            pct_str = click.style(f"{pct:5.1f}%", fg="green" if pct >= 100 else "yellow")
            eta = click.style(f"ETA {_eta(done, total, elapsed)}", dim=True)
            line = f"  {stage_lbl} {bar} {counts} {pct_str}  elapsed {elapsed:.1f}s  {eta}"
        else:
            line = f"  {stage_lbl} {done} done · elapsed {elapsed:.1f}s"
        # \033[K clears from cursor to end of line — handles ANSI-styled output
        # cleanly without having to measure visible-vs-raw length.
        end = "\n" if (total > 0 and done >= total) else ""
        click.echo("\r" + line + "\033[K", nl=bool(end))

    return cb, []


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Prompt the user for one of `choices` (single-letter on the first letter of each).
    Echoes a colored choice list like `[a]accept/[e]edit/[q]quit`.
    """
    parts = []
    for c in choices:
        letter = click.style(f"[{c[0]}]", fg="cyan", bold=True)
        parts.append(f"{letter}{c[1:]}")
    choice_str = "/".join(parts)
    valid = {c[0] for c in choices}
    while True:
        raw = click.prompt(f"{prompt} ({choice_str})",
                           default=default, show_default=bool(default)).strip().lower()
        if not raw and default:
            return default
        if raw and raw[0] in valid:
            return raw[0]
        click.secho(f"  please answer one of: {', '.join(choices)}", fg="red")


# Order matters — used as the menu order in the interactive mapping prompt
_FAMILIES_FOR_MAPPING = [
    "spontaneous_hold",
    "test_pulse",
    "iv_subthreshold",
    "ap_waveform",
    "rest_firing",
    "threshold_search",
]

_HEURISTIC_HINTS = [
    # Order matters — first match wins. ap_waveform comes before threshold_search
    # so "APThres" reads as an AP waveform metric, not a threshold scan; and
    # iv_subthreshold/test_pulse come before threshold_search so "subthres" /
    # "rheo" don't get mis-bucketed.
    ("ap_waveform",      ["apwave", "spike", "apthres", "apdrop"]),
    ("spontaneous_hold", ["hold", "rest_pot", "baseline", "starthold"]),
    ("rest_firing",      ["firepat", "idrest", "fire", "rest"]),
    ("iv_subthreshold",  ["iv", "hyperpol", "depol", "subthres"]),
    ("test_pulse",       ["rseal", "rpip", "test", "rac", "ampl", "rheo"]),
    ("threshold_search", ["thres", "idthr"]),
]


def _guess_family(token: str) -> str | None:
    """Best-effort family guess from the token name. None when uncertain."""
    lo = token.lower()
    for fam, patterns in _HEURISTIC_HINTS:
        if any(p in lo for p in patterns):
            return fam
    return None


def _parse_unmapped_block(yaml_text: str) -> list[tuple[str, int]]:
    """Extract (token, sweep_count) pairs from the `# ⚠ UNMAPPED tokens` comment
    block in the YAML header. Returns [] if no such block is present."""
    import re as _re
    out: list[tuple[str, int]] = []
    in_block = False
    for line in yaml_text.splitlines():
        if "UNMAPPED tokens" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if not line.startswith("#"):
            break
        # Skip non-token lines inside the block (the "Add these to…" footer)
        m = _re.match(r"#\s+(\S+)\s+\((\d+)\s+sweeps?\)", line)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def _interactive_map_unmapped(output_path: Path) -> bool:
    """Walk the user through assigning each unmapped stimulus token to a family.

    Reads the YAML on disk, surfaces every token in the UNMAPPED block, prompts
    once per token with a heuristic-derived default, then rewrites the YAML
    with the new assignments folded into `stimulus_protocols:` and the
    `# ⚠ UNMAPPED` block trimmed to whatever the user left unassigned.

    Returns True if any assignment was made (so the caller can re-display).
    """
    import yaml as _yaml

    text = output_path.read_text()
    unmapped = _parse_unmapped_block(text)
    if not unmapped:
        click.secho("  no UNMAPPED tokens found in the YAML header — nothing to map.",
                    dim=True)
        return False

    # Split file into header (comments + blank line) and body (YAML proper)
    header_lines, body_lines, in_body = [], [], False
    for line in text.splitlines():
        if not in_body and (line.startswith("#") or line.strip() == ""):
            header_lines.append(line)
        else:
            in_body = True
            body_lines.append(line)
    body = "\n".join(body_lines)
    cfg = _yaml.safe_load(body) or {}
    families = cfg.get("stimulus_protocols") or {}

    click.secho(
        f"\n  {len(unmapped)} unmapped token(s). Pick an action:",
        fg="cyan", bold=True)
    click.secho(
        "    [w] walk through each (one prompt per token, heuristic-suggested default)\n"
        "    [a] accept all heuristic-suggested mappings in one keystroke\n"
        "    [c] cancel (leave YAML unchanged)\n",
        dim=True)
    top_choice = click.prompt("  action", default="w", show_default=True).strip().lower()[:1]

    if top_choice == "c":
        click.secho("  cancelled — YAML unchanged.", fg="red")
        return False

    assigned: dict[str, str] = {}

    if top_choice == "a":
        # Bulk-assign every heuristic-derived guess; tokens with no guess stay unmapped.
        for token, _n_sweeps in unmapped:
            fam = _guess_family(token)
            if fam not in _FAMILIES_FOR_MAPPING:
                continue
            families.setdefault(fam, [])
            if token not in families[fam]:
                families[fam].append(token)
            assigned[token] = fam
    else:
        # Per-token walk (default [w])
        click.secho("    Tip: any letter that isn't a number cancels mapping for that token.",
                    dim=True)
        for token, n_sweeps in unmapped:
            guess = _guess_family(token)
            guess_idx = (_FAMILIES_FOR_MAPPING.index(guess) + 1) if guess in _FAMILIES_FOR_MAPPING else 0
            click.echo("")
            click.secho(f"  {token}", bold=True, nl=False)
            click.secho(f"  ({n_sweeps} sweeps)", dim=True)
            for i, fam in enumerate(_FAMILIES_FOR_MAPPING, start=1):
                tag = "  ← suggested" if guess == fam else ""
                click.echo(f"    {i}) {fam}{tag}")
            click.echo(f"    0) skip — leave as unmapped")
            default_str = str(guess_idx) if guess_idx else "0"
            raw = click.prompt(f"  assign {token}",
                                default=default_str, show_default=True).strip()
            try:
                choice = int(raw)
            except ValueError:
                click.secho(f"    cancelled — {token} remains unmapped.", dim=True)
                continue
            if not (0 <= choice <= len(_FAMILIES_FOR_MAPPING)):
                click.secho(f"    out of range — {token} remains unmapped.", dim=True)
                continue
            if choice == 0:
                continue
            fam = _FAMILIES_FOR_MAPPING[choice - 1]
            # Lazily create the family list only when we're actually populating it,
            # so untouched families don't appear as empty arrays in the saved YAML.
            families.setdefault(fam, [])
            if token not in families[fam]:
                families[fam].append(token)
            assigned[token] = fam

    if not assigned:
        click.secho("\n  no tokens were mapped — YAML unchanged.", dim=True)
        return False

    # Rebuild the YAML body with the updated families. Trim the UNMAPPED block
    # to whatever the user left unassigned (purely cosmetic — the YAML body is
    # what actually drives behavior).
    cfg["stimulus_protocols"] = families
    new_body = _yaml.safe_dump(cfg, sort_keys=False)

    remaining = [(t, n) for t, n in unmapped if t not in assigned]
    new_header = _trim_unmapped_block(header_lines, assigned, remaining)

    output_path.write_text("\n".join(new_header) + "\n" + new_body)
    click.secho(
        f"\n  ✓ updated {output_path.name}: assigned {len(assigned)} token(s); "
        f"{len(remaining)} still unmapped.",
        fg="green", bold=True)
    if assigned:
        for token, fam in assigned.items():
            click.secho(f"      {token} → {fam}", dim=True)
    return True


def _trim_unmapped_block(header_lines: list[str], assigned: dict[str, str],
                          remaining: list[tuple[str, int]]) -> list[str]:
    """Rewrite the `# ⚠ UNMAPPED tokens` comment block to reflect only the still-
    unmapped tokens. Other header lines pass through unchanged.
    """
    out: list[str] = []
    in_block = False
    skipping_listing = False
    for line in header_lines:
        if "UNMAPPED tokens" in line:
            in_block = True
            if remaining:
                out.append(f"# ⚠ UNMAPPED tokens ({len(remaining)} remaining):")
                for token, n in remaining:
                    out.append(f"#   {token}  ({n} sweeps)")
                out.append("#")
                out.append("# Add these to the appropriate family under stimulus_protocols: below.")
                out.append("#")
            else:
                out.append("# All discovered stimulus tokens are now mapped to a family.")
                out.append("#")
            skipping_listing = True
            continue
        if skipping_listing:
            # Skip the original token listing + "Add these to" footer
            if line.startswith("#"):
                continue
            else:
                skipping_listing = False
                in_block = False
        out.append(line)
    return out


def _stage_inspect(root: Path) -> bool:
    from .inspect import inspect_root, render_terminal
    _stage_banner(1, 6, "Inspect")
    click.echo(render_terminal(inspect_root(root)))
    _hr()
    ans = _prompt_choice("continue?", ["yes", "quit"], default="y")
    return ans == "y"


def _prompt_curator_into_yaml(config_path: Path) -> None:
    """One-line prompt for the curator name. If the YAML already has a non-empty
    `curator:`, skip. If the user enters a value, splice a `curator: <name>` line
    into the YAML so subsequent `nwb-qc serve` sessions stamp it onto decisions
    automatically. Empty answer ⇒ leave the YAML untouched (viewer prompts later).
    """
    try:
        import yaml as _yaml
        raw = _yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return
    if isinstance(raw, dict) and str(raw.get("curator", "")).strip():
        return  # already set
    click.secho(
        "  Optional: curator name (stamped onto every decision saved from the viewer).\n"
        "  Press Enter to skip — the viewer will prompt once on first save instead.",
        dim=True)
    name = click.prompt("  curator", default="", show_default=False).strip()
    if not name:
        return
    # Splice `curator: <name>` near the top of the YAML, preserving comments.
    text = config_path.read_text()
    if "\ncurator:" in text or text.startswith("curator:"):
        return  # don't double-write if a `curator:` key already exists with empty value
    # Insert after the `project_name:` line if present, else prepend
    lines = text.splitlines(keepends=True)
    new_line = f"curator: {name}\n"
    out: list[str] = []
    spliced = False
    for ln in lines:
        out.append(ln)
        if not spliced and ln.lstrip().startswith("project_name:"):
            out.append(new_line)
            spliced = True
    if not spliced:
        out.insert(0, new_line)
    config_path.write_text("".join(out))
    click.secho(f"  ✓ added curator: {name} to {config_path.name}", fg="green")


def _stage_propose(root: Path, output_path: Path,
                   *, name: str | None, guess_tables: bool) -> Path | None:
    """Generate YAML, show it, allow edit-in-$EDITOR, return accepted path or None."""
    from .cli import _build_starter_config
    yaml_text = _build_starter_config(root, name=name, guess_tables=guess_tables,
                                      output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_text)

    while True:
        click.secho("═" * 72, dim=True)
        click.secho(f"STAGE 2/6 · Propose config", bold=True, fg="cyan", nl=False)
        click.secho(f"  →  {output_path}", dim=True)
        _hr()
        text = output_path.read_text()
        has_unmapped = "⚠ UNMAPPED" in text
        if has_unmapped:
            click.secho(
                "  ⚠ Some stimulus protocols in your NWBs aren't mapped to a family.\n"
                "    Pick [m]ap-unmapped below to walk through them one by one\n"
                "    (with heuristic-based suggestions), or [e]dit to open the\n"
                "    YAML in $EDITOR. Otherwise qc_protocol_coverage will be\n"
                "    False for every cell.",
                fg="yellow", bold=True)
        click.echo(text)
        _hr()
        choices = ["accept", "edit", "quit"]
        if has_unmapped:
            choices = ["accept", "map-unmapped", "edit", "quit"]
        ans = _prompt_choice("review", choices, default="a")
        if ans == "a":
            _prompt_curator_into_yaml(output_path)
            return output_path
        if ans == "q":
            return None
        if ans == "m":
            _interactive_map_unmapped(output_path)
            continue   # re-display the updated YAML and re-prompt
        # edit
        edited = click.edit(filename=str(output_path))
        if edited is None:
            click.secho("  (no changes saved)", dim=True)


def _stage_review_thresholds(config_path: Path) -> str:
    """Stage 3/6: review (and optionally edit) the thresholds YAML before the
    first metric-compute run. No cohort data exists yet, so this is purely
    editorial — same UX as the propose-config stage, but pointed at the
    thresholds file referenced by the project YAML.

    Returns 'a' (accept), 'q' (quit). The 'e'-loop is internal — we don't return
    control to the caller until the user picks accept or quit.
    """
    cfg = load_config(config_path)
    thresholds_file = cfg.thresholds_file
    if thresholds_file is None or not thresholds_file.exists():
        click.secho(
            f"  (no thresholds_file configured for this project — skipping review)",
            dim=True,
        )
        return "a"

    while True:
        click.secho("═" * 72, dim=True)
        click.secho(f"STAGE 3/6 · Review thresholds", bold=True, fg="cyan", nl=False)
        click.secho(f"  →  {thresholds_file}", dim=True)
        _hr()
        click.echo(thresholds_file.read_text())
        _hr()
        click.secho(
            "  Bundled defaults shown above. [e]dit now to set initial rules\n"
            "  before the first run — after the run you can also use\n"
            "  [t]une-thresholds in the outcome menu (cohort-percentile aware),\n"
            "  or `nwb-qc tune` standalone.",
            dim=True)
        ans = _prompt_choice("review", ["accept", "edit", "quit"], default="a")
        if ans == "a":
            return "a"
        if ans == "q":
            return "q"
        # edit
        edited = click.edit(filename=str(thresholds_file))
        if edited is None:
            click.secho("  (no changes saved)", dim=True)


def _stage_dryrun(config_path: Path) -> str:
    """Run the manifest-build dry-run and print the same block as `list-cells`.

    Returns the user choice: 'r' (run), 'b' (back to propose), 'q' (quit).
    """
    cfg = load_config(config_path)
    manifest = build_manifest(cfg)
    uniq = unique_nwbs(manifest)
    _stage_banner(4, 6, "Dry-run")
    click.secho(f"Project: ", nl=False); click.secho(cfg.project_name, bold=True)
    click.echo(f"Sources: {len(cfg.nwb_sources)}")
    for s in cfg.nwb_sources:
        n = int((manifest['dataset'] == s.dataset).sum())
        loc = f"manifest {s.manifest}" if s.manifest is not None else f"at {s.path}"
        click.secho(f"  - {s.dataset}: ", nl=False)
        click.secho(f"{n} NWBs", fg="green", bold=True, nl=False)
        click.secho(f" ({loc})", dim=True)
    click.secho(f"Total NWB rows: ", nl=False); click.secho(str(len(manifest)), bold=True)
    click.secho(f"Unique by sha256: ", nl=False)
    click.secho(str(len(uniq)), bold=True, nl=False)
    click.secho(f" (dedup saves {len(manifest)-len(uniq)} compute steps)", dim=True)
    stats = list(manifest.attrs.get("manifest_stats", []))
    if stats:
        click.echo("\nManifest-source diagnostics:")
        for s in stats:
            click.echo(
                f"  - {s['dataset']}: {s['n_files_in_manifest']} files in manifest "
                f"({s['n_nwbs_in_manifest']} .nwb) · "
                f"{s['n_eligible_after_filter']} eligible · "
                f"{s['n_present_on_disk']} present · {s['n_missing_on_disk']} missing"
            )
    _hr()
    return _prompt_choice("proceed", ["run", "back", "quit"], default="r")


def _stage_run(config_path: Path, *, with_vision: bool | None,
               max_cost_usd: float | None) -> dict | None:
    cfg = load_config(config_path)
    if with_vision is not None:
        cfg.vision_judge.enabled = with_vision
    _stage_banner(5, 6, "Run")
    callback, _ = _make_progress_callback()
    try:
        return pipeline_run(cfg, progress_callback=callback,
                            max_cost_usd=max_cost_usd)
    except Exception as e:  # noqa: BLE001
        click.secho(f"\n  ✗ run failed: {e}", fg="red", bold=True)
        return None


def _auto_calibrate(config_path: Path) -> dict[str, str | None]:
    """Run calibrate as a side-effect after a successful pipeline run.

    Writes cohort_stats.json (consumed by the NEXT report's chip explanations
    for percentile context) and a `<stem>_thresholds_suggested.yaml` next to
    the active thresholds file. Both artifacts are produced unconditionally —
    they're cheap and useful even if the user never opts into the suggested
    thresholds. Returns paths for the outcome stage to display.

    Failures are non-fatal: if the cache is empty or thresholds are missing,
    we report None for the relevant path and continue.
    """
    from .cache import filter_for_version, load_cache
    from .calibrate import (
        render_suggested_yaml,
        suggest_thresholds,
        write_cohort_stats_json,
    )
    from .thresholds import load_thresholds

    out: dict[str, str | None] = {"cohort_stats": None, "suggested_thresholds": None}
    try:
        cfg = load_config(config_path)
        cache_df = filter_for_version(load_cache(cfg.cache_path))
        if cache_df.empty:
            return out
        cohort_path = cfg.output_dir / "cohort_stats.json"
        write_cohort_stats_json(cache_df, cohort_path)
        out["cohort_stats"] = str(cohort_path)

        if cfg.thresholds_file and cfg.thresholds_file.exists():
            bundled = load_thresholds(cfg.thresholds_file)
            suggested = suggest_thresholds(cache_df, bundled)
            yaml_text = render_suggested_yaml(suggested, n_cells=len(cache_df),
                                                source_count=len(cfg.nwb_sources))
            stem = cfg.thresholds_file.stem
            suggested_path = cfg.thresholds_file.parent / f"{stem}_suggested.yaml"
            suggested_path.write_text(yaml_text)
            out["suggested_thresholds"] = str(suggested_path)
    except Exception as e:  # noqa: BLE001
        click.secho(f"  (auto-calibrate skipped: {e})", dim=True)
    return out


def _reload_result_from_disk(config_path: Path, prev_result: dict) -> dict:
    """After tune-and-rerun, the pipeline already ran and the report CSV is
    fresh on disk. Recompute verdict counts so the next outcome prompt shows
    the post-tune numbers. Falls back to the previous result if anything fails.
    """
    try:
        import pandas as _pd
        cfg = load_config(config_path)
        if cfg.report_csv and cfg.report_csv.exists():
            df = _pd.read_csv(cfg.report_csv)
            counts = df["final_verdict"].value_counts().to_dict() if "final_verdict" in df.columns else {}
            return {
                **prev_result,
                "n_cells": int(len(df)),
                "n_pass": int(counts.get("pass", 0)),
                "n_flag": int(counts.get("flag", 0)),
                "n_fail": int(counts.get("fail", 0)),
                "report": str(cfg.report_html),
                "viewer": str(cfg.report_html.parent / "qc_viewer.html"),
                "run_report": str(cfg.output_dir / "run_report.json"),
            }
    except Exception:
        pass
    return prev_result


def _adopt_suggested_thresholds_and_rerun(config_path: Path, suggested_path: str,
                                            *, with_vision: bool | None,
                                            max_cost_usd: float | None) -> dict | None:
    """Rewrite the project YAML's `thresholds_file:` to the suggested file, then
    re-run the pipeline. Cache hits everything (only the threshold layer re-evaluates).
    """
    import yaml as _yaml
    raw = _yaml.safe_load(config_path.read_text()) or {}
    raw["thresholds_file"] = suggested_path
    # Preserve the header comments
    original = config_path.read_text()
    header = "\n".join(line for line in original.splitlines() if line.startswith("#"))
    body = _yaml.safe_dump(raw, sort_keys=False)
    config_path.write_text((header + "\n\n" if header else "") + body)
    click.secho(f"  → updated thresholds_file in {config_path.name} to use "
                 f"{Path(suggested_path).name}", fg="cyan")
    return _stage_run(config_path, with_vision=with_vision, max_cost_usd=max_cost_usd)


def _stage_outcome(result: dict, *, with_vision: bool | None,
                    max_cost_usd: float | None) -> None:
    _stage_banner(6, 6, "Outcome")
    n_pass = result.get('n_pass', 0)
    n_flag = result.get('n_flag', 0)
    n_fail = result.get('n_fail', 0)
    click.echo(f"  cells:      {result.get('n_cells', 0)} (", nl=False)
    click.secho(f"pass={n_pass}", fg="green", nl=False); click.echo(" ", nl=False)
    click.secho(f"flag={n_flag}", fg="yellow", nl=False); click.echo(" ", nl=False)
    click.secho(f"fail={n_fail}", fg="red", nl=False)
    click.echo(")")
    click.echo(f"  report:     ", nl=False); click.secho(str(result.get('report')), fg="cyan")
    click.echo(f"  viewer:     ", nl=False); click.secho(str(result.get('viewer')), fg="cyan")
    click.echo(f"  run report: ", nl=False); click.secho(str(result.get('run_report')), fg="cyan")

    suggested = result.get("suggested_thresholds")
    cohort_stats = result.get("cohort_stats")
    if cohort_stats:
        click.echo(f"  cohort stats: ", nl=False); click.secho(str(cohort_stats), fg="cyan")
    if suggested:
        click.echo(f"  suggested thresholds: ", nl=False)
        click.secho(str(suggested), fg="cyan")
        click.secho(
            f"  ↑ derived from cohort percentiles; review before adopting "
            f"(or pick [c]alibrate-and-re-run below to apply them now).",
            dim=True)
    _hr()

    # v0.8.0: curation lives in the viewer, so [s]erve is the primary action.
    # The static report is still useful as a curation log + shareable artifact,
    # but it's no longer where decisions happen — kept second.
    choices = ["serve", "open", "tune-thresholds", "done"]
    if suggested:
        choices = ["serve", "open", "tune-thresholds", "calibrate-and-rerun", "done"]
    ans = _prompt_choice("next", choices, default="s")
    if ans == "o":
        report = result.get("report")
        if report and Path(report).exists():
            webbrowser.open(f"file://{report}")
    elif ans == "s":
        from .server import serve
        cfg_path_str = result.get("config_path")
        click.secho("  starting interactive viewer (Ctrl-C to stop)…", dim=True)
        if cfg_path_str:
            serve(load_config(Path(cfg_path_str)))
    elif ans == "t":
        cfg_path_str = result.get("config_path")
        if not cfg_path_str:
            click.secho("  config path missing — can't tune", fg="red")
            return
        from .tune import tune_thresholds_interactive
        # The tune flow handles its own re-run + verdict preview. After it
        # returns, drop the user back into the outcome stage so they can keep
        # iterating (open report, tune again, serve viewer, etc.) — but rebuild
        # the result dict from the freshly re-rendered report.
        tune_thresholds_interactive(Path(cfg_path_str), rerun=True)
        # After tuning's own re-run, rebuild the outcome from the on-disk report
        # CSV so the user sees the updated verdict counts in the next prompt.
        new_result = _reload_result_from_disk(Path(cfg_path_str), result)
        new_result["config_path"] = cfg_path_str
        # Suggested thresholds are stale after tuning — re-derive so the next
        # [c]alibrate-and-rerun reflects the new defaults.
        new_result.update(_auto_calibrate(Path(cfg_path_str)))
        _stage_outcome(new_result, with_vision=with_vision, max_cost_usd=max_cost_usd)
    elif ans == "c" and suggested:
        cfg_path_str = result.get("config_path")
        if not cfg_path_str:
            click.secho("  config path missing — can't re-run", fg="red")
            return
        new_result = _adopt_suggested_thresholds_and_rerun(
            Path(cfg_path_str), suggested,
            with_vision=with_vision, max_cost_usd=max_cost_usd,
        )
        if new_result is None:
            click.secho("  re-run failed.", fg="red")
            return
        # Recurse once into a fresh outcome — auto-calibrate is skipped on the
        # second pass (cohort_stats.json is already current; suggested wasn't
        # changed by re-thresholding, and we don't want an infinite calibrate loop).
        new_result["config_path"] = cfg_path_str
        _stage_outcome(new_result, with_vision=with_vision, max_cost_usd=max_cost_usd)


def _existing_cache_for(output_path: Path) -> Path | None:
    """If a project YAML already exists at output_path AND its cache parquet has
    rows, return the config Path so the wizard can offer a short-circuit. Otherwise None.
    """
    if not output_path.exists():
        return None
    try:
        cfg = load_config(output_path)
        if cfg.cache_path and cfg.cache_path.exists() and cfg.cache_path.stat().st_size > 0:
            return output_path
    except Exception:
        pass
    return None


def _stage_short_circuit_prompt(config_path: Path) -> str:
    """When an existing config + warm cache is detected, ask the user whether to
    skip straight to tuning/outcome or restart from inspect.

    Returns 'continue' (use existing state, skip to outcome), 'restart' (run
    the full inspect → propose → dry-run → run flow), or 'quit'.
    """
    cfg = load_config(config_path)
    click.secho("═" * 72, dim=True)
    click.secho("Existing project detected", bold=True, fg="cyan")
    _hr()
    click.echo(f"  config:    {config_path}")
    click.echo(f"  cache:     {cfg.cache_path}")
    if cfg.report_csv and cfg.report_csv.exists():
        click.echo(f"  report:    {cfg.report_html}")
    _hr()
    click.secho(
        "  Skip straight to the outcome stage to tune thresholds, open the\n"
        "  report, serve the viewer, or calibrate-and-rerun? Or restart from\n"
        "  the inspect stage (re-builds the manifest + re-runs metric compute\n"
        "  for any new NWBs — existing rows are still cache-served).",
        dim=True)
    return _prompt_choice("action",
                            ["continue-existing", "restart", "quit"],
                            default="c")


def run_wizard(root: Path, *, output_path: Path, name: str | None = None,
               guess_tables: bool = True, with_vision: bool | None = None,
               max_cost_usd: float | None = None) -> int:
    """Drive the five-stage interactive flow. Returns a CLI exit code.

    Re-entry: if `output_path` already points at a project YAML whose cache has
    rows, offer a short-circuit straight into the outcome stage (so the user
    can tune thresholds / re-open report / re-serve viewer without re-walking
    inspect → propose → dry-run → run).
    """
    existing = _existing_cache_for(output_path)
    if existing is not None:
        choice = _stage_short_circuit_prompt(existing)
        if choice == "q":
            return 1
        if choice == "c":
            # Reconstruct an outcome-stage `result` from the existing report on
            # disk, then drop into the outcome prompt. No metric-compute, no
            # report re-render — pure interactive.
            result = _reload_result_from_disk(existing, {})
            if not result:
                click.secho("  couldn't reconstruct outcome from disk; "
                             "running full pipeline instead.", fg="yellow")
            else:
                result["config_path"] = str(existing)
                result.update(_auto_calibrate(existing))
                _stage_outcome(result, with_vision=with_vision,
                                max_cost_usd=max_cost_usd)
                return 0
        # 'r' → fall through to the full flow

    if not _stage_inspect(root):
        click.secho("aborted at inspect.", fg="red")
        return 1
    while True:
        accepted = _stage_propose(root, output_path,
                                  name=name, guess_tables=guess_tables)
        if accepted is None:
            click.secho("aborted at propose.", fg="red")
            return 1
        thr_choice = _stage_review_thresholds(accepted)
        if thr_choice == "q":
            click.secho("aborted at review-thresholds.", fg="red")
            return 1
        choice = _stage_dryrun(accepted)
        if choice == "q":
            click.secho("aborted at dry-run.", fg="red")
            return 1
        if choice == "b":
            continue
        break
    result = _stage_run(accepted, with_vision=with_vision, max_cost_usd=max_cost_usd)
    if result is None:
        return 2
    result["config_path"] = str(accepted)
    # Auto-calibrate side files — cheap, always useful (cohort_stats.json feeds
    # the next report's chip explanations; suggested YAML is opt-in).
    calibrate_paths = _auto_calibrate(accepted)
    result.update(calibrate_paths)
    _stage_outcome(result, with_vision=with_vision, max_cost_usd=max_cost_usd)
    return 0
