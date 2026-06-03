"""Five-stage interactive flow for `nwb-qc start <root>`.

Composes the existing CLI building blocks (inspect, init-config, list-cells, run)
with a pause-and-confirm prompt between every stage and a live progress line
during the metric-compute stage.

  1. Inspect      — print inventory; [Enter] continue, [q] quit
  2. Propose      — generate project YAML; [a]ccept / [e]dit in $EDITOR / [q]uit
  3. Dry-run      — list discovered cells; [r]un / [b]ack / [q]uit
  4. Run          — execute pipeline with progress; on completion show summary
  5. Outcome      — show paths; [o]pen report / [s]erve / [Enter] done
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
    click.echo(char * 72)


def _eta(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0 or total <= 0:
        return "—"
    rate = done / max(elapsed_s, 1e-6)
    remaining = (total - done) / max(rate, 1e-6)
    if remaining < 60:   return f"{remaining:.0f}s"
    if remaining < 3600: return f"{remaining/60:.1f}m"
    return f"{remaining/3600:.1f}h"


def _make_progress_callback() -> tuple[Callable[[str, int, int], None], list[float]]:
    """Returns (callback, [stage_t0]). The callback prints a carriage-return
    progress line on TTY stdout. On non-TTY, falls back to one INFO log line
    per stage start (logging already covers per-stage).
    """
    is_tty = sys.stdout.isatty()
    state: dict[str, float] = {}

    def cb(stage: str, done: int, total: int) -> None:
        now = time.time()
        if stage not in state:
            state[stage] = now
            if is_tty:
                click.echo(f"\n  [stage] {stage} starting…")
        if not is_tty:
            return
        elapsed = now - state[stage]
        if total > 0:
            pct = 100.0 * done / total
            line = (f"  [{stage}] {done}/{total} ({pct:.1f}%)  "
                    f"elapsed {elapsed:.1f}s  ETA {_eta(done, total, elapsed)}")
        else:
            line = f"  [{stage}] {done} done · elapsed {elapsed:.1f}s"
        # carriage return overwrite; newline when stage completes
        end = "\n" if (total > 0 and done >= total) else ""
        click.echo("\r" + line.ljust(72), nl=bool(end))

    return cb, []


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Prompt the user for one of `choices` (single-letter). Echoes the choice list."""
    choice_str = "/".join(f"[{c}]{label}" for c, label in zip("".join(choices), choices))
    while True:
        raw = click.prompt(f"{prompt} ({choice_str})",
                           default=default, show_default=bool(default)).strip().lower()
        if not raw and default:
            return default
        if raw and raw[0] in {c[0] for c in choices}:
            return raw[0]
        click.echo(f"  please answer one of: {', '.join(choices)}")


def _stage_inspect(root: Path) -> bool:
    from .inspect import inspect_root, render_terminal
    _hr("═")
    click.echo("STAGE 1/5 · Inspect")
    _hr()
    click.echo(render_terminal(inspect_root(root)))
    _hr()
    ans = _prompt_choice("continue?", ["yes", "quit"], default="y")
    return ans == "y"


def _stage_propose(root: Path, output_path: Path,
                   *, name: str | None, guess_tables: bool) -> Path | None:
    """Generate YAML, show it, allow edit-in-$EDITOR, return accepted path or None."""
    from .cli import _build_starter_config
    yaml_text = _build_starter_config(root, name=name, guess_tables=guess_tables,
                                      output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_text)

    while True:
        _hr("═")
        click.echo(f"STAGE 2/5 · Propose config  →  {output_path}")
        _hr()
        text = output_path.read_text()
        if "⚠ UNMAPPED" in text:
            click.secho(
                "  ⚠ Some stimulus protocols in your NWBs aren't mapped to a family.\n"
                "    Look for the 'UNMAPPED tokens' block below and edit\n"
                "    stimulus_protocols: to slot them into the right families\n"
                "    before accepting — otherwise qc_protocol_coverage will be\n"
                "    False for every cell.",
                fg="yellow")
        click.echo(text)
        _hr()
        ans = _prompt_choice("review", ["accept", "edit", "quit"], default="a")
        if ans == "a":
            return output_path
        if ans == "q":
            return None
        # edit
        edited = click.edit(filename=str(output_path))
        if edited is None:
            click.echo("  (no changes saved)")


def _stage_dryrun(config_path: Path) -> str:
    """Run the manifest-build dry-run and print the same block as `list-cells`.

    Returns the user choice: 'r' (run), 'b' (back to propose), 'q' (quit).
    """
    cfg = load_config(config_path)
    manifest = build_manifest(cfg)
    uniq = unique_nwbs(manifest)
    _hr("═")
    click.echo("STAGE 3/5 · Dry-run")
    _hr()
    click.echo(f"Project: {cfg.project_name}")
    click.echo(f"Sources: {len(cfg.nwb_sources)}")
    for s in cfg.nwb_sources:
        n = int((manifest['dataset'] == s.dataset).sum())
        loc = f"manifest {s.manifest}" if s.manifest is not None else f"at {s.path}"
        click.echo(f"  - {s.dataset}: {n} NWBs ({loc})")
    click.echo(f"Total NWB rows: {len(manifest)}")
    click.echo(f"Unique by sha256: {len(uniq)} (dedup saves {len(manifest)-len(uniq)} compute steps)")
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
    _hr("═")
    click.echo("STAGE 4/5 · Run")
    _hr()
    callback, _ = _make_progress_callback()
    try:
        return pipeline_run(cfg, progress_callback=callback,
                            max_cost_usd=max_cost_usd)
    except Exception as e:  # noqa: BLE001
        click.echo(f"\n  ✗ run failed: {e}")
        return None


def _stage_outcome(result: dict) -> None:
    _hr("═")
    click.echo("STAGE 5/5 · Outcome")
    _hr()
    click.echo(f"  cells:      {result.get('n_cells', 0)} "
               f"(pass={result.get('n_pass',0)} flag={result.get('n_flag',0)} fail={result.get('n_fail',0)})")
    click.echo(f"  report:     {result.get('report')}")
    click.echo(f"  viewer:     {result.get('viewer')}")
    click.echo(f"  run report: {result.get('run_report')}")
    _hr()
    ans = _prompt_choice("next", ["open", "serve", "done"], default="d")
    if ans == "o":
        report = result.get("report")
        if report and Path(report).exists():
            webbrowser.open(f"file://{report}")
    elif ans == "s":
        from .server import serve
        cfg_path_str = result.get("config_path")
        # The wizard saved config_path on `result` if available; otherwise leave to caller
        click.echo("  starting interactive viewer (Ctrl-C to stop)…")
        if cfg_path_str:
            serve(load_config(Path(cfg_path_str)))


def run_wizard(root: Path, *, output_path: Path, name: str | None = None,
               guess_tables: bool = True, with_vision: bool | None = None,
               max_cost_usd: float | None = None) -> int:
    """Drive the five-stage interactive flow. Returns a CLI exit code."""
    if not _stage_inspect(root):
        click.echo("aborted at inspect.")
        return 1
    while True:
        accepted = _stage_propose(root, output_path,
                                  name=name, guess_tables=guess_tables)
        if accepted is None:
            click.echo("aborted at propose.")
            return 1
        choice = _stage_dryrun(accepted)
        if choice == "q":
            click.echo("aborted at dry-run.")
            return 1
        if choice == "b":
            continue
        break
    result = _stage_run(accepted, with_vision=with_vision, max_cost_usd=max_cost_usd)
    if result is None:
        return 2
    result["config_path"] = str(accepted)
    _stage_outcome(result)
    return 0
