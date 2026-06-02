"""Sticky human overrides — survive re-runs and threshold edits."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

OVERRIDE_COLUMNS = ["cell_id", "override_verdict", "note", "reviewer", "date"]


def init_overrides_file(path: Path) -> None:
    """Create an empty overrides CSV with header (idempotent)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OVERRIDE_COLUMNS)


def load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    # tolerate missing optional columns
    for c in OVERRIDE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[OVERRIDE_COLUMNS]


def apply_overrides(verdicts: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Replace computed_verdict with override_verdict where one exists.

    Adds columns: final_verdict, override_note, override_reviewer.
    """
    df = verdicts.copy()
    if overrides.empty or "override_verdict" not in overrides.columns:
        df["final_verdict"] = df.get("computed_verdict", "pass")
        df["override_note"] = ""
        df["override_reviewer"] = ""
        df["override_date"] = ""
        return df
    ov = overrides[overrides["override_verdict"].str.strip().ne("")].copy()
    merged = df.merge(
        ov.rename(columns={"override_verdict": "_ov_verdict", "note": "_ov_note",
                           "reviewer": "_ov_rev", "date": "_ov_date"}),
        on="cell_id", how="left",
    )
    merged["final_verdict"] = merged["_ov_verdict"].where(
        merged["_ov_verdict"].notna() & merged["_ov_verdict"].str.strip().ne(""),
        merged.get("computed_verdict", "pass"),
    )
    merged["override_note"] = merged["_ov_note"].fillna("")
    merged["override_reviewer"] = merged["_ov_rev"].fillna("")
    merged["override_date"] = merged["_ov_date"].fillna("")
    return merged.drop(columns=[c for c in merged.columns if c.startswith("_ov_")])
