"""Per-NWB-sha256 QC results, keyed by (nwb_sha256, pipeline_version)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import PIPELINE_VERSION


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def cached_hashes(cache_df: pd.DataFrame, version: str = PIPELINE_VERSION) -> set[str]:
    if cache_df.empty: return set()
    sub = cache_df[cache_df["pipeline_version"] == version]
    return set(sub["nwb_sha256"].astype(str))


def append_rows(cache_path: Path, new_rows: list[dict]) -> None:
    """Atomic append: load existing, concatenate, write to a tmp file, rename."""
    if not new_rows:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cache(cache_path)
    fresh = pd.DataFrame(new_rows)
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    # Deduplicate on (nwb_sha256, pipeline_version), keep the latest
    if {"nwb_sha256", "pipeline_version"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["nwb_sha256", "pipeline_version"], keep="last")
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), tmp)
    os.replace(tmp, cache_path)


def filter_for_version(cache_df: pd.DataFrame, version: str = PIPELINE_VERSION) -> pd.DataFrame:
    if cache_df.empty: return cache_df
    return cache_df[cache_df["pipeline_version"] == version].copy()
