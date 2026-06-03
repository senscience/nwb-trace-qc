"""CLI entry points for nwb-qc."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import click
import pyarrow.parquet as pq
import yaml

from . import PIPELINE_VERSION, __version__
from .config import default_families, load_config
from .manifest import build_manifest, unique_nwbs
from .pipeline import run as pipeline_run
from .thresholds import load_thresholds


def _detect_key_format(values: list) -> str:
    """Heuristic: stem | basename | absolute."""
    sample = [str(v) for v in values[:10] if v]
    if not sample: return "stem"
    if any("/" in v or "\\" in v for v in sample): return "absolute"
    if any(v.endswith(".nwb") for v in sample): return "basename"
    return "stem"


def _find_source_manifest(subdir: Path) -> Path | None:
    """Look for a wrangler `source_material/source_manifest.json` inside a top-level
    subdir, or one level deeper (the wrap-in-a-project-dir layout).
    Returns the absolute path or None."""
    direct = subdir / "source_material" / "source_manifest.json"
    if direct.is_file():
        return direct
    for inner in subdir.iterdir():
        if inner.is_dir():
            nested = inner / "source_material" / "source_manifest.json"
            if nested.is_file():
                return nested
    return None


def _count_source_nwbs(s: dict) -> int:
    """Return the NWB count for a source dict in the init-config output."""
    if "manifest" in s:
        n, _ = _summarize_manifest_for_init(Path(s["manifest"]))
        return n
    if "path" in s:
        return len(list(Path(s["path"]).glob(s.get("glob", "**/*.nwb"))))
    return 0


def _summarize_manifest_for_init(manifest_path: Path) -> tuple[int, int]:
    """Read just enough of a source_manifest.json to display counts at init time."""
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        summary = data.get("summary") or {}
        n = int(summary.get("total_files") or len(data.get("files") or []))
        total = int(summary.get("total_size_bytes") or 0)
        return n, total
    except Exception:
        return 0, 0


def _sample_paths_for_source(s: dict, n_samples: int = 3) -> list[Path]:
    """Return up to `n_samples` NWB paths from a `nwb_sources[i]` dict."""
    from .nwb_io import is_nwb
    out: list[Path] = []
    if "manifest" in s:
        try:
            with open(s["manifest"]) as f:
                data = json.load(f)
        except Exception:
            return out
        for entry in (data.get("files") or []):
            orig = entry.get("original_location") or ""
            if not orig:
                continue
            p = Path(orig).expanduser()
            if p.exists() and is_nwb(p):
                out.append(p)
            if len(out) >= n_samples:
                break
    elif "path" in s:
        root = Path(s["path"])
        glob = s.get("glob", "**/*.nwb")
        for p in root.glob(glob):
            if is_nwb(p):
                out.append(p)
            if len(out) >= n_samples:
                break
        # Pick up sibling Zarr stores when the glob targets files only
        if len(out) < n_samples and ".nwb.zarr" not in glob:
            zarr_glob = glob.replace("*.nwb", "*.nwb.zarr") if "*.nwb" in glob else f"{glob.rstrip('/')}/*.nwb.zarr"
            for p in root.glob(zarr_glob):
                if is_nwb(p):
                    out.append(p)
                if len(out) >= n_samples:
                    break
    return out


def _sample_stimulus_tokens(sources: list[dict], n_per_source: int = 3) -> Counter:
    """Open a few NWBs per source and return a Counter of the stimulus tokens
    parsed by the same `name.split("__")[1]` rule the pipeline uses.

    Returns an empty Counter on any IO failure — purely a hint, not a hard step.
    """
    from .nwb_io import open_nwb
    tokens: Counter = Counter()
    for s in sources:
        for p in _sample_paths_for_source(s, n_per_source):
            try:
                with open_nwb(p) as f:
                    for name in f.acquisition.keys():
                        parts = name.split("__")
                        stim = parts[1] if len(parts) >= 3 else name
                        tokens[stim] += 1
            except Exception:
                continue
    return tokens


def _classify_tokens(tokens: Counter, families: dict[str, list[str]]) -> tuple[dict[str, Counter], Counter]:
    """Split tokens into matched (per family) vs. unmatched against `families`.

    Matching is case-insensitive, same as `StimulusFamilyMap`.
    """
    known_lc = {n.lower(): fam for fam, names in families.items() for n in names}
    matched: dict[str, Counter] = defaultdict(Counter)
    unmatched: Counter = Counter()
    for tok, n in tokens.items():
        fam = known_lc.get(tok.lower())
        if fam:
            matched[fam][tok] += n
        else:
            unmatched[tok] += n
    return dict(matched), unmatched


def _render_discovered_block(matched: dict[str, Counter], unmatched: Counter) -> str:
    """Build the commented `# discovered:` / `# UNMAPPED:` block for the YAML header."""
    if not matched and not unmatched:
        return ""
    lines: list[str] = ["#", "# Stimulus protocols discovered by sampling your NWBs:"]
    for fam in sorted(matched):
        toks = ", ".join(f"{t} ({n})" for t, n in matched[fam].most_common())
        lines.append(f"#   {fam}: {toks}")
    if unmatched:
        essentials = {"spontaneous_hold", "test_pulse", "ap_waveform"}
        missing_essentials = essentials - set(matched)
        lines.append("#")
        lines.append(f"# ⚠ UNMAPPED tokens ({len(unmatched)} unique, "
                     f"{sum(unmatched.values())} sweeps in sampled NWBs):")
        for tok, n in unmatched.most_common():
            lines.append(f"#   {tok}  ({n} sweeps)")
        lines.append("#")
        lines.append("# Add these to the appropriate family under stimulus_protocols: below.")
        if missing_essentials:
            lines.append(f"# Heads-up: no protocols mapped to {sorted(missing_essentials)} —")
            lines.append("# qc_protocol_coverage will be False for every cell until you assign")
            lines.append("# at least one unmapped token to each essential family.")
    lines.append("#\n")
    return "\n".join(lines)


def _build_starter_config(root_path: Path, *, name: str | None = None,
                          guess_tables: bool = True,
                          output_path: Path | None = None) -> str:
    """Build a starter project YAML string for ROOT_PATH.

    Returns the YAML text (with header comments) but does NOT write to disk.
    The caller is responsible for `output_path.write_text(...)`. `output_path`
    is used only to compute relative paths in the header (`Next:` hint).

    Side effect: copies the bundled `default_thresholds.yaml` next to
    `output_path` if a thresholds file isn't already there.
    """
    root_path = root_path.resolve()
    name = (name or root_path.name).lower().replace(" ", "_")
    if output_path is None:
        cwd = Path.cwd()
        out_dir = cwd / "configs" if (cwd / "configs").is_dir() else cwd
        output_path = out_dir / f"{name}_project.yaml"

    # 1. NWB sources
    sources = []
    top_level = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
    for sub in top_level:
        mpath = _find_source_manifest(sub)
        if mpath is not None:
            sources.append({
                "dataset": sub.name.lower(),
                "manifest": str(mpath.resolve()),
                "only_processed": False,
            })
            continue
        nwb_hdf5 = [p for p in sub.rglob("*.nwb") if p.is_file()]
        nwb_zarr = [p for p in sub.rglob("*.nwb.zarr") if p.is_dir()]
        nwbs = nwb_hdf5 + nwb_zarr
        if not nwbs: continue
        rel_depths = {len(p.relative_to(sub).parts) for p in nwbs}
        glob = "**/*.nwb" if max(rel_depths) > 1 else "*.nwb"
        sources.append({
            "dataset": sub.name.lower(),
            "path": str(sub.resolve()),
            "recursive": True,
            "glob": glob,
        })
    root_nwbs = [p for p in root_path.glob("*.nwb") if p.is_file()] + \
                [p for p in root_path.glob("*.nwb.zarr") if p.is_dir()]
    if root_nwbs and not sources:
        sources.append({
            "dataset": name, "path": str(root_path), "recursive": False, "glob": "*.nwb"
        })

    # 2. Acquisition tables
    tables = []
    if guess_tables:
        for pq_path in sorted(root_path.rglob("*.parquet")):
            try:
                schema = pq.read_schema(pq_path)
                cols = set(schema.names)
            except Exception:
                continue
            if "nwb_file" in cols and "stimulus_type" in cols:
                try:
                    sample = pq.read_table(pq_path, columns=["nwb_file"]).column("nwb_file").to_pylist()[:10]
                except Exception:
                    sample = []
                column_map = {
                    "stimulus_type": "stimulus_type",
                    "rate_hz": "rate_hz" if "rate_hz" in cols else None,
                    "n_samples": "n_samples" if "n_samples" in cols else None,
                    "clamp_mode": "clamp_mode" if "clamp_mode" in cols else None,
                    "sweep_number": "sweep_number" if "sweep_number" in cols else None,
                }
                tables.append({
                    "path": str(pq_path.resolve()),
                    "nwb_key_column": "nwb_file",
                    "nwb_key_format": _detect_key_format(sample),
                    "columns": {k: v for k, v in column_map.items() if v is not None},
                })

    # 3. Cell table
    cell_table = None
    for csv in sorted(root_path.glob("*.csv")):
        try:
            with open(csv) as f:
                header = f.readline().strip().split(",")
        except Exception:
            continue
        if "cell_id" in header:
            ds_cols = [c for c in header if c.startswith("in_") and c not in {"in_"}]
            cell_table = {
                "path": str(csv.resolve()),
                "cell_id_column": "cell_id",
                "dataset_columns": ds_cols,
            }
            break

    families = default_families()

    pkg_default = Path(__file__).parent.parent.parent / "configs" / "default_thresholds.yaml"
    thr_target = output_path.parent / f"{name}_thresholds.yaml"
    if pkg_default.exists() and not thr_target.exists():
        thr_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pkg_default, thr_target)

    output_dir = (output_path.parent / f"qc_output_{name}").resolve()

    cfg = {
        "project_name": name,
        "output_dir": str(output_dir),
        "nwb_sources": sources,
        "acquisition_tables": tables,
        "stimulus_protocols": families,
        "thresholds_file": str(thr_target.resolve() if thr_target.exists() else thr_target),
        "n_workers": max(1, (os.cpu_count() or 4) - 2),
        "cell_table": cell_table,
    }

    # Sample a few NWBs per source to surface lab-specific stimulus protocols
    # the LNMC/BBP default mapping doesn't cover.
    sampled = _sample_stimulus_tokens(sources, n_per_source=3) if sources else Counter()
    matched, unmatched = _classify_tokens(sampled, families)
    discovered_block = _render_discovered_block(matched, unmatched)

    header = (
        f"# nwb-trace-qc project config — auto-generated\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# Root path scanned: {root_path}\n"
        f"# NWBs discovered: {sum(_count_source_nwbs(s) for s in sources)} across {len(sources)} source(s)\n"
        f"# Acquisition parquets registered: {len(tables)}\n"
        f"# Cell table detected: {'yes' if cell_table else 'no'}\n"
        f"#\n"
        f"# Review (a) stimulus_protocols if your lab uses non-LNMC names and\n"
        f"#        (b) thresholds_file before running `nwb-qc run`.\n"
        f"{discovered_block}\n"
    )
    return header + yaml.safe_dump(cfg, sort_keys=False)


class _ColorFormatter(logging.Formatter):
    """Lightweight ANSI colorizer for stderr log lines. No deps."""
    _COLORS = {
        logging.DEBUG:    "\033[2m",                # dim grey
        logging.INFO:     "\033[36m",               # cyan
        logging.WARNING:  "\033[33m",               # yellow
        logging.ERROR:    "\033[31m",               # red
        logging.CRITICAL: "\033[1;31m",             # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, *args, use_color: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if not self._use_color:
            return s
        return f"{self._COLORS.get(record.levelno, '')}{s}{self._RESET}"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    use_color = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    handler.setFormatter(_ColorFormatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        use_color=use_color,
    ))
    root = logging.getLogger()
    # Avoid duplicating handlers when CLI is invoked multiple times in one process (tests)
    root.handlers = [handler]
    root.setLevel(level)


@click.group()
@click.version_option(__version__, prog_name="nwb-qc")
@click.option("--verbose", "-v", is_flag=True, help="Verbose (DEBUG) logging to stderr.")
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """Cohort-scale QC for patch-clamp NWB datasets."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command("inspect")
@click.argument("root_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Where to write the full Markdown (or JSON) inventory. Default: ./<root>_inventory.md")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON to stdout (and to --output if given), not Markdown.")
@click.option("--no-write", is_flag=True, help="Skip writing the inventory file; only print to stdout.")
def inspect_cmd(root_path: Path, output: Path | None, as_json: bool, no_write: bool):
    """Read-only inventory of a wrangler-output tree (before deciding to QC it)."""
    from .inspect import inspect_root, render_terminal, render_markdown, render_json
    result = inspect_root(root_path)
    if as_json:
        click.echo(render_json(result))
    else:
        click.echo(render_terminal(result))
    if no_write:
        return
    if output is None:
        ext = ".json" if as_json else ".md"
        output = Path.cwd() / f"{root_path.resolve().name}_inventory{ext}"
    output.write_text(render_json(result) if as_json else render_markdown(result))
    click.echo(f"\nFull inventory written to: {output}")


@main.command("start")
@click.argument("root_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", default=None, help="Project name. Default: root_path basename.")
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None,
              help="Where to save the project YAML. Default: <name>_project.yaml in cwd (or configs/).")
@click.option("--guess-tables/--no-guess-tables", default=True,
              help="Scan for acquisition parquets under root_path (default ON).")
@click.option("--with-vision/--no-vision", default=None,
              help="Force the vision judge on/off for this run, overriding the config.")
@click.option("--max-cost-usd", "max_cost_usd", type=float, default=None,
              help="Soft cap on vision-judge spend for this run (USD).")
def start_cmd(root_path: Path, name: str | None, output_path: Path | None,
              guess_tables: bool, with_vision: bool | None, max_cost_usd: float | None):
    """Guided wizard: inspect → propose config → dry-run → run → outcome."""
    from .wizard import run_wizard
    root_path = root_path.resolve()
    name = (name or root_path.name).lower().replace(" ", "_")
    if output_path is None:
        cwd = Path.cwd()
        out_dir = cwd / "configs" if (cwd / "configs").is_dir() else cwd
        output_path = out_dir / f"{name}_project.yaml"
    elif output_path.is_dir():
        output_path = output_path / f"{name}_project.yaml"
    code = run_wizard(root_path, output_path=output_path, name=name,
                      guess_tables=guess_tables, with_vision=with_vision,
                      max_cost_usd=max_cost_usd)
    sys.exit(code)


@main.command("init-config")
@click.argument("root_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", "name", default=None, help="Project name. Default: root_path basename.")
@click.option("--output", "output", type=click.Path(path_type=Path), default=None,
              help="Output YAML path. Default: <name>_project.yaml in cwd (or configs/<name>_project.yaml inside a repo).")
@click.option("--guess-tables/--no-guess-tables", default=True,
              help="Scan for acquisition parquets under root_path (default ON).")
def init_config(root_path: Path, name: str | None, output: Path | None, guess_tables: bool):
    """Auto-discover NWBs + parquets under ROOT_PATH and write a starter project YAML."""
    root_path = root_path.resolve()
    name = (name or root_path.name).lower().replace(" ", "_")
    if output is None:
        cwd = Path.cwd()
        out_dir = cwd / "configs" if (cwd / "configs").is_dir() else cwd
        output = out_dir / f"{name}_project.yaml"
    elif output.is_dir():
        output = output / f"{name}_project.yaml"
    yaml_text = _build_starter_config(root_path, name=name,
                                       guess_tables=guess_tables, output_path=output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml_text)
    click.echo(f"Wrote {output}")
    click.echo(f"Next: nwb-qc list-cells --config {output}")


@main.command("list-cells")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def list_cells(config_path: Path):
    """Dry-run: show discovered NWBs and dedup info, no compute."""
    cfg = load_config(config_path)
    manifest = build_manifest(cfg)
    uniq = unique_nwbs(manifest)
    click.echo(f"Project: {cfg.project_name}")
    click.echo(f"Sources: {len(cfg.nwb_sources)}")
    for s in cfg.nwb_sources:
        n = int((manifest['dataset'] == s.dataset).sum())
        loc = f"manifest {s.manifest}" if s.manifest is not None else f"at {s.path}"
        click.echo(f"  - {s.dataset}: {n} NWBs ({loc})")
    click.echo(f"Total NWB rows: {len(manifest)}")
    click.echo(f"Unique by sha256: {len(uniq)} (dedup saves {len(manifest)-len(uniq)} compute steps)")
    click.echo(f"Acquisition tables registered: {len(cfg.acquisition_tables)}")
    if cfg.cell_table:
        click.echo(f"Cell table: {cfg.cell_table.path}")
    stats = list(manifest.attrs.get("manifest_stats", []))
    if stats:
        click.echo("\nManifest-source diagnostics:")
        for s in stats:
            click.echo(
                f"  - {s['dataset']}: {s['n_files_in_manifest']} files in manifest "
                f"({s['n_nwbs_in_manifest']} .nwb) · "
                f"{s['n_eligible_after_filter']} eligible · "
                f"{s['n_present_on_disk']} present · {s['n_missing_on_disk']} missing · "
                f"sha256 reused {s['n_sha256_reused']} / recomputed {s['n_sha256_recomputed']}"
            )
            if s.get("only_processed") and s.get("n_filtered_unprocessed", 0) > 0:
                click.echo(
                    f"      ⚠ {s['n_filtered_unprocessed']} additional NWB entr"
                    f"{'y' if s['n_filtered_unprocessed']==1 else 'ies'} "
                    f"dropped by only_processed=true (was_processed=false in the manifest). "
                    f"Set only_processed: false in the YAML to include them."
                )
    click.echo(f"\nNext: nwb-qc run --config {config_path}")


@main.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--filter", "filter_arg", default=None,
              help="Restrict to one logical dataset, e.g. dataset=RN")
@click.option("--report-only", is_flag=True, help="Re-render report from cache without NWB I/O")
@click.option("--with-vision/--no-vision", default=None,
              help="Force the vision judge on/off this run, overriding the config's vision_judge.enabled.")
@click.option("--max-cost-usd", "max_cost_usd", type=float, default=None,
              help="Soft cap on vision-judge spend for this run (USD). "
                   "Overrides vision_judge.max_cost_usd from the config.")
def run_cmd(config_path: Path, filter_arg: str | None, report_only: bool,
            with_vision: bool | None, max_cost_usd: float | None):
    """Run the full pipeline: discover → cache → compute → threshold → (vision) → override → report."""
    cfg = load_config(config_path)
    filter_ds = None
    if filter_arg:
        if "=" not in filter_arg or not filter_arg.startswith("dataset="):
            raise click.BadParameter("--filter must look like 'dataset=NAME'")
        filter_ds = filter_arg.split("=", 1)[1]
    if with_vision is not None:
        cfg.vision_judge.enabled = with_vision
    result = pipeline_run(cfg, filter_dataset=filter_ds, report_only=report_only,
                          max_cost_usd=max_cost_usd)
    click.echo(json.dumps(result, indent=2, default=str))
    report = result.get("report")
    if report:
        click.echo(f"\nNext: open '{report}'   # static HTML report")
        click.echo(f"      nwb-qc serve --config {config_path}   # interactive trace viewer")


@main.command("serve")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--no-browser", is_flag=True, help="Don't auto-open the browser.")
def serve_cmd(config_path: Path, host: str, port: int, no_browser: bool):
    """Start the interactive trace viewer (requires `nwb-qc run` to have been executed)."""
    from .server import serve  # lazy import; avoids pulling pynwb just to run --help
    cfg = load_config(config_path)
    serve(cfg, host=host, port=port, open_browser=not no_browser)


@main.command("report")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def report_cmd(config_path: Path):
    """Re-render report from existing cache (no NWB I/O)."""
    cfg = load_config(config_path)
    result = pipeline_run(cfg, report_only=True)
    click.echo(json.dumps(result, indent=2, default=str))
    report = result.get("report")
    if report:
        click.echo(f"\nNext: open '{report}'")


@main.command("thresholds")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Show how the current thresholds would classify cached cells.")
def thresholds_cmd(config_path: Path, dry_run: bool):
    """Apply thresholds against the cache; --dry-run shows verdict counts."""
    cfg = load_config(config_path)
    if not dry_run:
        click.echo("Use --dry-run to preview verdict counts without writing.")
        return
    from .cache import filter_for_version, load_cache
    from .thresholds import evaluate
    cache = filter_for_version(load_cache(cfg.cache_path))
    if cache.empty:
        click.echo("Cache is empty; run `nwb-qc run` first.")
        click.echo(f"\nNext: nwb-qc run --config {config_path}")
        return
    thresholds = load_thresholds(cfg.thresholds_file)
    counts = {"pass": 0, "flag": 0, "fail": 0}
    for r in cache.itertuples(index=False):
        m = r._asdict()
        v, _ = evaluate(m, thresholds)
        counts[v] += 1
    click.echo(f"Verdict counts against {len(cache)} cached NWBs: {counts}")
    click.echo(
        f"\nNext: edit {cfg.thresholds_file.name if cfg.thresholds_file else '<thresholds-file>'} then "
        f"`nwb-qc run --config {config_path}`   # re-render uses the cache, no NWB I/O"
    )
