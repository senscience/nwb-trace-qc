"""Discover NWBs, hash them, link them to (dataset, cell_id) rows."""
from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import NWBSource, ProjectConfig


@dataclass
class CellRow:
    dataset: str
    cell_id: str
    nwb_path: Path
    nwb_sha256: str
    nwb_size: int
    nwb_mtime: float


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_paths(source: NWBSource) -> list[Path]:
    if not source.path.exists():
        return []
    return sorted(p for p in source.path.glob(source.glob) if p.is_file() and p.suffix == ".nwb")


def _key_value(path: Path, fmt: str) -> str:
    if fmt == "stem":
        return path.stem
    if fmt == "basename":
        return path.name
    return str(path.resolve())


def build_manifest(cfg: ProjectConfig) -> pd.DataFrame:
    """Walk every nwb_sources entry, hash each file, return one row per (dataset, cell_id, nwb_path)."""
    rows: list[CellRow] = []
    for source in cfg.nwb_sources:
        for p in _discover_paths(source):
            st = p.stat()
            rows.append(CellRow(
                dataset=source.dataset,
                cell_id=p.stem,
                nwb_path=p,
                nwb_sha256=_sha256(p),
                nwb_size=st.st_size,
                nwb_mtime=st.st_mtime,
            ))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["dataset", "cell_id", "nwb_path", "nwb_sha256", "nwb_size", "nwb_mtime"])
    df["nwb_path"] = df["nwb_path"].astype(str)
    return df


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
