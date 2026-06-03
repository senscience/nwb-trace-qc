"""CLI entry points for nwb-qc."""
from __future__ import annotations

import json
import os
import shutil
import sys
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


@click.group()
@click.version_option(__version__, prog_name="nwb-qc")
def main():
    """Cohort-scale QC for patch-clamp NWB datasets."""
    pass


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
    # Where to write
    if output is None:
        cwd = Path.cwd()
        out_dir = cwd / "configs" if (cwd / "configs").is_dir() else cwd
        output = out_dir / f"{name}_project.yaml"
    elif output.is_dir():
        output = output / f"{name}_project.yaml"

    # 1. NWB sources: one per top-level subdir under root containing NWBs;
    #    if root itself has NWBs, register it as a single source.
    sources = []
    top_level = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
    for sub in top_level:
        # Find all NWBs under this subdir, with a glob that captures them
        nwbs = list(sub.rglob("*.nwb"))
        if not nwbs: continue
        # Detect if we need a nested glob
        # If any NWB is more than 1 level deep, use a more permissive glob
        rel_depths = {len(p.relative_to(sub).parts) for p in nwbs}
        glob = "**/*.nwb" if max(rel_depths) > 1 else "*.nwb"
        sources.append({
            "dataset": sub.name.lower(),
            "path": str(sub.resolve()),
            "recursive": True,
            "glob": glob,
        })
    # Root has NWBs directly?
    root_nwbs = list(root_path.glob("*.nwb"))
    if root_nwbs and not sources:
        sources.append({
            "dataset": name, "path": str(root_path), "recursive": False, "glob": "*.nwb"
        })

    # 2. Acquisition tables: scan parquets under root_path
    tables = []
    if guess_tables:
        for pq_path in sorted(root_path.rglob("*.parquet")):
            try:
                schema = pq.read_schema(pq_path)
                cols = set(schema.names)
            except Exception:
                continue
            if "nwb_file" in cols and "stimulus_type" in cols:
                # Sample first values to detect key format
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

    # 3. Cell table: any CSV in root_path with a cell_id column
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

    # 4. Stimulus families (default LNMC/BBP)
    families = default_families()

    # 5. Thresholds: copy bundled defaults next to the project YAML
    pkg_default = Path(__file__).parent.parent.parent / "configs" / "default_thresholds.yaml"
    thr_target = output.parent / f"{name}_thresholds.yaml"
    if pkg_default.exists() and not thr_target.exists():
        shutil.copy(pkg_default, thr_target)

    output_dir = (output.parent / f"qc_output_{name}").resolve()

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

    header = (
        f"# nwb-trace-qc project config — auto-generated by `nwb-qc init-config`\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# Root path scanned: {root_path}\n"
        f"# NWBs discovered: {sum(len(list(Path(s['path']).glob(s['glob']))) for s in sources)} across {len(sources)} source(s)\n"
        f"# Acquisition parquets registered: {len(tables)}\n"
        f"# Cell table detected: {'yes' if cell_table else 'no'}\n"
        f"#\n"
        f"# Review (a) stimulus_protocols if your lab uses non-LNMC names and\n"
        f"#        (b) thresholds_file before running `nwb-qc run`.\n"
        f"# Next: nwb-qc list-cells --config {output.name}\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + yaml.safe_dump(cfg, sort_keys=False))

    click.echo(f"Wrote {output}")
    click.echo(f"      ({len(sources)} sources, {len(tables)} acquisition tables, "
               f"thresholds at {thr_target.name})")
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
        click.echo(f"  - {s.dataset}: {n} NWBs at {s.path}")
    click.echo(f"Total NWB rows: {len(manifest)}")
    click.echo(f"Unique by sha256: {len(uniq)} (dedup saves {len(manifest)-len(uniq)} compute steps)")
    click.echo(f"Acquisition tables registered: {len(cfg.acquisition_tables)}")
    if cfg.cell_table:
        click.echo(f"Cell table: {cfg.cell_table.path}")


@main.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--filter", "filter_arg", default=None,
              help="Restrict to one logical dataset, e.g. dataset=RN")
@click.option("--report-only", is_flag=True, help="Re-render report from cache without NWB I/O")
@click.option("--with-vision/--no-vision", default=None,
              help="Force the vision judge on/off this run, overriding the config's vision_judge.enabled.")
def run_cmd(config_path: Path, filter_arg: str | None, report_only: bool, with_vision: bool | None):
    """Run the full pipeline: discover → cache → compute → threshold → (vision) → override → report."""
    cfg = load_config(config_path)
    filter_ds = None
    if filter_arg:
        if "=" not in filter_arg or not filter_arg.startswith("dataset="):
            raise click.BadParameter("--filter must look like 'dataset=NAME'")
        filter_ds = filter_arg.split("=", 1)[1]
    if with_vision is not None:
        cfg.vision_judge.enabled = with_vision
    result = pipeline_run(cfg, filter_dataset=filter_ds, report_only=report_only)
    click.echo(json.dumps(result, indent=2, default=str))


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
        return
    thresholds = load_thresholds(cfg.thresholds_file)
    counts = {"pass": 0, "flag": 0, "fail": 0}
    for r in cache.itertuples(index=False):
        m = r._asdict()
        v, _ = evaluate(m, thresholds)
        counts[v] += 1
    click.echo(f"Verdict counts against {len(cache)} cached NWBs: {counts}")
