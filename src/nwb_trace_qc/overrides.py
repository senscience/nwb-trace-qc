"""Sticky human overrides — survive re-runs and threshold edits."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

OVERRIDE_COLUMNS = ["cell_id", "override_verdict", "note", "reviewer", "date"]

# v0.7.0: trim overrides — per-NWB-sha cutoff sweep index forced by the user
# from the viewer's trim slider. trim_at_sweep=0 means "no trim" (override the
# auto-detected bad-ending to keep the full recording).
TRIM_OVERRIDE_COLUMNS = ["nwb_sha256", "trim_at_sweep", "note", "reviewer", "date"]


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


# ─── Trim overrides (v0.7.0) ─────────────────────────────────────────

def init_trim_overrides_file(path: Path) -> None:
    """Create an empty trim-overrides CSV with header (idempotent)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRIM_OVERRIDE_COLUMNS)


def load_trim_overrides(path: Path) -> dict[str, int]:
    """Return ``{nwb_sha256: trim_at_sweep}`` from the override CSV.

    ``trim_at_sweep=0`` means "no trim" (caller passes ``force_trim_at=0``
    to compute_metrics, bypassing both auto-detection and any prior cutoff).
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"nwb_sha256": str}).fillna({"trim_at_sweep": -1})
    if "nwb_sha256" not in df.columns or "trim_at_sweep" not in df.columns:
        return {}
    out: dict[str, int] = {}
    for r in df.itertuples(index=False):
        try:
            v = int(getattr(r, "trim_at_sweep"))
        except (ValueError, TypeError):
            continue
        if v < 0:
            continue
        sha = str(getattr(r, "nwb_sha256")).strip()
        if sha:
            out[sha] = v
    return out


def upsert_trim_override(path: Path, *, nwb_sha256: str, trim_at_sweep: int,
                          note: str = "", reviewer: str = "", date: str = "") -> None:
    """Insert or update a per-NWB trim override row, atomically."""
    init_trim_overrides_file(path)
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["nwb_sha256"] != nwb_sha256]
    new_row = pd.DataFrame([{
        "nwb_sha256": nwb_sha256, "trim_at_sweep": str(int(trim_at_sweep)),
        "note": note, "reviewer": reviewer, "date": date,
    }])
    out = pd.concat([df, new_row], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
