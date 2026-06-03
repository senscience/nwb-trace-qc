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
    n_files_in_manifest: int = 0
    n_eligible_after_filter: int = 0   # after only_processed
    n_present_on_disk: int = 0
    n_missing_on_disk: int = 0
    n_sha256_reused: int = 0
    n_sha256_recomputed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "manifest_path": str(self.manifest_path),
            "n_files_in_manifest": self.n_files_in_manifest,
            "n_eligible_after_filter": self.n_eligible_after_filter,
            "n_present_on_disk": self.n_present_on_disk,
            "n_missing_on_disk": self.n_missing_on_disk,
            "n_sha256_reused": self.n_sha256_reused,
            "n_sha256_recomputed": self.n_sha256_recomputed,
        }


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_paths(source: NWBSource) -> list[Path]:
    if source.path is None or not source.path.exists():
        return []
    return sorted(p for p in source.path.glob(source.glob) if p.is_file() and p.suffix == ".nwb")


def _load_source_manifest(manifest_path: Path, only_processed: bool = True) -> tuple[list[ManifestEntry], int]:
    """Read a wrangler `source_manifest.json` and return its NWB file entries.

    Expands `~` in `original_location` and validates extension. Filters by
    `was_processed` when `only_processed`. Returns (entries, total_files_in_manifest).
    Raises FileNotFoundError if the manifest itself isn't readable.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"source manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        data = json.load(f)
    files = data.get("files", []) or []
    out: list[ManifestEntry] = []
    for f in files:
        orig = f.get("original_location") or ""
        if not orig:
            continue
        path = Path(orig).expanduser()
        if path.suffix.lower() != ".nwb":
            continue
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
    return out, len(files)


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
        st = p.stat()
        out.append(CellRow(
            dataset=source.dataset,
            cell_id=p.stem,
            nwb_path=p,
            nwb_sha256=_sha256(p),
            nwb_size=st.st_size,
            nwb_mtime=st.st_mtime,
        ))
    return out


def _rows_from_manifest_source(source: NWBSource, stats_list: list[ManifestSourceStats]) -> list[CellRow]:
    entries, total = _load_source_manifest(source.manifest, only_processed=source.only_processed)
    stats = ManifestSourceStats(dataset=source.dataset, manifest_path=source.manifest,
                                n_files_in_manifest=total, n_eligible_after_filter=len(entries))
    out: list[CellRow] = []
    for e in entries:
        if not e.nwb_path.exists():
            stats.n_missing_on_disk += 1
            log.warning("manifest source %s: file missing on disk → skipping: %s",
                        source.dataset, e.nwb_path)
            continue
        stats.n_present_on_disk += 1
        st = e.nwb_path.stat()
        size_matches = (st.st_size == e.size_bytes) if e.size_bytes else False
        mtime_matches = abs(st.st_mtime - e.mtime) <= MTIME_TOLERANCE_S if e.mtime else False
        if source.reuse_sha256 and e.sha256 and size_matches and mtime_matches:
            sha = e.sha256
            stats.n_sha256_reused += 1
        else:
            sha = _sha256(e.nwb_path)
            stats.n_sha256_recomputed += 1
        out.append(CellRow(
            dataset=source.dataset,
            cell_id=e.nwb_path.stem,
            nwb_path=e.nwb_path,
            nwb_sha256=sha,
            nwb_size=st.st_size,
            nwb_mtime=st.st_mtime,
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
