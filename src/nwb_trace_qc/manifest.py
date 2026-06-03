"""Discover NWBs, hash them, link them to (dataset, cell_id) rows."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .config import NWBSource, ProjectConfig
from .nwb_io import is_nwb, is_zarr, nwb_mtime, nwb_sha256, nwb_size

log = logging.getLogger(__name__)

MTIME_TOLERANCE_S = 1.0   # filesystem mtime can wobble by ~1 s across copies


@dataclass
class CellRow:
    dataset: str
    cell_id: str
    nwb_path: Path
    nwb_sha256: str
    nwb_size: int
    nwb_mtime: float


@dataclass
class ManifestEntry:
    """One file entry pulled from a wrangler source_manifest.json."""

    nwb_path: Path          # `original_location`, ~-expanded
    size_bytes: int
    sha256: str             # pre-computed by the wrangler; may be empty if skipped above the wrangler's size threshold
    mtime: float
    was_processed: bool


@dataclass
class ManifestSourceStats:
    """Per-source diagnostics surfaced in run-result JSON + list-cells output."""

    dataset: str
    manifest_path: Path
    n_files_in_manifest: int = 0       # everything in manifest.files
    n_nwbs_in_manifest: int = 0        # files with .nwb extension
    n_filtered_unprocessed: int = 0    # NWBs dropped because was_processed=false AND only_processed=true
    n_eligible_after_filter: int = 0   # NWBs eligible to QC
    n_present_on_disk: int = 0
    n_missing_on_disk: int = 0
    n_sha256_reused: int = 0
    n_sha256_recomputed: int = 0
    only_processed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "manifest_path": str(self.manifest_path),
            "n_files_in_manifest": self.n_files_in_manifest,
            "n_nwbs_in_manifest": self.n_nwbs_in_manifest,
            "n_filtered_unprocessed": self.n_filtered_unprocessed,
            "n_eligible_after_filter": self.n_eligible_after_filter,
            "n_present_on_disk": self.n_present_on_disk,
            "n_missing_on_disk": self.n_missing_on_disk,
            "n_sha256_reused": self.n_sha256_reused,
            "n_sha256_recomputed": self.n_sha256_recomputed,
            "only_processed": self.only_processed,
        }


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    """Backwards-compat alias; dispatches to nwb_io.nwb_sha256 (HDF5 file hash, Zarr dir fingerprint)."""
    return nwb_sha256(path, buf)


def _discover_paths(source: NWBSource) -> list[Path]:
    """Discover both HDF5 (`*.nwb` files) and Zarr (`*.nwb.zarr/` directories) under source.path."""
    if source.path is None or not source.path.exists():
        return []
    glob = source.glob
    # Build candidate set: file glob + directory glob for the Zarr counterpart
    candidates: set[Path] = set()
    for p in source.path.glob(glob):
        if is_nwb(p):
            candidates.add(p)
    # If the configured glob only matches files, also pick up sibling *.nwb.zarr dirs
    # (the most common case: glob="**/*.nwb" — also walk for **/*.nwb.zarr).
    if ".nwb.zarr" not in glob:
        zarr_glob = glob.replace("*.nwb", "*.nwb.zarr") if "*.nwb" in glob else f"{glob.rstrip('/')}/*.nwb.zarr"
        for p in source.path.glob(zarr_glob):
            if is_zarr(p):
                candidates.add(p)
    return sorted(candidates)


def _load_source_manifest(
    manifest_path: Path, only_processed: bool = False
) -> tuple[list[ManifestEntry], int, int]:
    """Read a wrangler `source_manifest.json` and return its NWB file entries.

    Expands `~` in `original_location` and validates extension. Filters by
    `was_processed` when `only_processed`.
    Returns (entries, total_files_in_manifest, nwbs_in_manifest_pre_filter).
    Raises FileNotFoundError if the manifest itself isn't readable.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"source manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        data = json.load(f)
    files = data.get("files", []) or []
    out: list[ManifestEntry] = []
    nwb_count = 0
    for f in files:
        orig = f.get("original_location") or ""
        if not orig:
            continue
        path = Path(orig).expanduser()
        # Accept HDF5 (.nwb) or Zarr (.nwb.zarr) entries
        name_lower = path.name.lower()
        if not (name_lower.endswith(".nwb") or name_lower.endswith(".nwb.zarr")):
            continue
        nwb_count += 1
        was_processed = bool(f.get("was_processed", True))
        if only_processed and not was_processed:
            continue
        out.append(ManifestEntry(
            nwb_path=path,
            size_bytes=int(f.get("size_bytes", 0) or 0),
            sha256=str(f.get("sha256", "") or ""),
            mtime=float(f.get("mtime", 0.0) or 0.0),
            was_processed=was_processed,
        ))
    return out, len(files), nwb_count


def _key_value(path: Path, fmt: str) -> str:
    if fmt == "stem":
        return path.stem
    if fmt == "basename":
        return path.name
    return str(path.resolve())


def build_manifest(cfg: ProjectConfig) -> pd.DataFrame:
    """Walk every nwb_sources entry, hash each file, return one row per (dataset, cell_id, nwb_path).

    Supports two source modes per entry:
      - path/glob: walk a directory tree and sha256-hash every NWB ourselves.
      - manifest:  follow a wrangler `source_manifest.json`; reuse its sha256 when
                   size+mtime match (within ±1 s), otherwise recompute on disk.

    Per-source stats (ManifestSourceStats) are attached to the DataFrame as
    `df.attrs["manifest_stats"] = [ ... ]` so the pipeline can surface them in
    the run-result JSON without changing the row schema.
    """
    rows: list[CellRow] = []
    stats_list: list[ManifestSourceStats] = []
    for source in cfg.nwb_sources:
        if source.manifest is not None:
            rows.extend(_rows_from_manifest_source(source, stats_list))
        else:
            rows.extend(_rows_from_path_source(source))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["dataset", "cell_id", "nwb_path", "nwb_sha256", "nwb_size", "nwb_mtime"])
    df["nwb_path"] = df["nwb_path"].astype(str)
    df.attrs["manifest_stats"] = [s.to_dict() for s in stats_list]
    return df


def _rows_from_path_source(source: NWBSource) -> list[CellRow]:
    out: list[CellRow] = []
    for p in _discover_paths(source):
        out.append(CellRow(
            dataset=source.dataset,
            cell_id=_cell_id_from_path(p),
            nwb_path=p,
            nwb_sha256=nwb_sha256(p),
            nwb_size=nwb_size(p),
            nwb_mtime=nwb_mtime(p),
        ))
    return out


def _cell_id_from_path(p: Path) -> str:
    """Stem of an NWB path. For Zarr (`cell.nwb.zarr`), strip both extensions."""
    name = p.name
    if name.endswith(".nwb.zarr"):
        return name[: -len(".nwb.zarr")]
    return p.stem


def _rows_from_manifest_source(source: NWBSource, stats_list: list[ManifestSourceStats]) -> list[CellRow]:
    entries, total, nwbs_in_manifest = _load_source_manifest(
        source.manifest, only_processed=source.only_processed
    )
    stats = ManifestSourceStats(
        dataset=source.dataset, manifest_path=source.manifest,
        n_files_in_manifest=total,
        n_nwbs_in_manifest=nwbs_in_manifest,
        n_filtered_unprocessed=(nwbs_in_manifest - len(entries)) if source.only_processed else 0,
        n_eligible_after_filter=len(entries),
        only_processed=source.only_processed,
    )
    out: list[CellRow] = []
    for e in entries:
        if not e.nwb_path.exists():
            stats.n_missing_on_disk += 1
            log.warning("manifest source %s: file missing on disk → skipping: %s",
                        source.dataset, e.nwb_path)
            continue
        stats.n_present_on_disk += 1
        # Use NWB-aware stat helpers so Zarr directories work the same way as HDF5 files
        on_disk_size = nwb_size(e.nwb_path)
        on_disk_mtime = nwb_mtime(e.nwb_path)
        size_matches = (on_disk_size == e.size_bytes) if e.size_bytes else False
        mtime_matches = abs(on_disk_mtime - e.mtime) <= MTIME_TOLERANCE_S if e.mtime else False
        if source.reuse_sha256 and e.sha256 and size_matches and mtime_matches:
            sha = e.sha256
            stats.n_sha256_reused += 1
        else:
            sha = nwb_sha256(e.nwb_path)
            stats.n_sha256_recomputed += 1
        out.append(CellRow(
            dataset=source.dataset,
            cell_id=_cell_id_from_path(e.nwb_path),
            nwb_path=e.nwb_path,
            nwb_sha256=sha,
            nwb_size=on_disk_size,
            nwb_mtime=on_disk_mtime,
        ))
    stats_list.append(stats)
    return out


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def unique_nwbs(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per unique nwb_sha256 (with a representative nwb_path)."""
    if manifest.empty:
        return manifest.copy()
    return manifest.drop_duplicates(subset=["nwb_sha256"], keep="first").reset_index(drop=True)


def load_acquisition_index(cfg: ProjectConfig) -> dict[str, pd.DataFrame]:
    """For each configured acquisition table, return a dict[nwb_key -> sub-DataFrame].

    The returned dict is keyed by the *stem* (file basename without .nwb) so callers can
    look up acquisitions for a given NWB by stem regardless of how the table stores the key.
    """
    out: dict[str, pd.DataFrame] = {}
    for tbl in cfg.acquisition_tables:
        df = pq.read_table(tbl.path).to_pandas()
        if tbl.nwb_key_column not in df.columns:
            raise KeyError(f"{tbl.path}: missing key column {tbl.nwb_key_column!r}")
        keys = df[tbl.nwb_key_column].astype(str)
        if tbl.nwb_key_format == "stem":
            stems = keys.str.replace(r"\.nwb$", "", regex=True).str.split("/").str[-1]
        elif tbl.nwb_key_format == "basename":
            stems = keys.str.split("/").str[-1].str.replace(r"\.nwb$", "", regex=True)
        else:  # absolute
            stems = keys.apply(lambda x: Path(x).stem)
        df = df.assign(_stem=stems)
        for stem, sub in df.groupby("_stem"):
            # Merge if multiple tables describe the same stem (concat rows)
            if stem in out:
                out[stem] = pd.concat([out[stem], sub], ignore_index=True)
            else:
                out[stem] = sub.reset_index(drop=True)
    return out
