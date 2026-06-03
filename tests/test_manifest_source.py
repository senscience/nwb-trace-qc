"""Tests for the source-manifest discovery path (NWBSource with manifest=, sha256 reuse, missing-file handling)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from nwb_trace_qc.config import NWBSource, ProjectConfig
from nwb_trace_qc.manifest import (
    _load_source_manifest,
    build_manifest,
)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_source_nwbs(tmp_path: Path, n: int = 3) -> list[Path]:
    """Create synthetic source NWB files; return their absolute paths."""
    out: list[Path] = []
    src_dir = tmp_path / "data"
    src_dir.mkdir()
    for i in range(n):
        p = src_dir / f"cell_{i:02d}.nwb"
        p.write_bytes(f"fake nwb {i}".encode() * 200)
        out.append(p)
    return out


def _make_manifest(tmp_path: Path, nwb_paths: list[Path], include_unprocessed: bool = False) -> Path:
    """Write a minimal source_manifest.json pointing at the given absolute paths."""
    src_material = tmp_path / "output" / "proj" / "source_material"
    src_material.mkdir(parents=True)
    files = []
    for i, p in enumerate(nwb_paths):
        stat = p.stat()
        files.append({
            "path": f"source_material/{p.name}",
            "relative_path": p.name,
            "original_location": str(p),
            "size_bytes": stat.st_size,
            "sha256": _sha256_bytes(p.read_bytes()),
            "mtime": stat.st_mtime,
            "role": "unknown",
            "was_processed": True,
        })
    if include_unprocessed:
        # An unprocessed entry that should be filtered with only_processed=True
        files.append({
            "path": "source_material/skipped.nwb",
            "original_location": str(tmp_path / "data" / "skipped.nwb"),
            "size_bytes": 0, "sha256": "",
            "mtime": 0.0, "was_processed": False,
        })
    manifest = src_material / "source_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 5,
        "generated_at": "2026-06-01T00:00:00Z",
        "input_source": {"input_path": "data", "is_remote": False, "scheme": "file"},
        "preservation": {"included": False, "subdirectory": "source_material"},
        "summary": {"total_files": len(files), "total_size_bytes": sum(f["size_bytes"] for f in files)},
        "files": files,
    }))
    return manifest


def test_load_source_manifest_basic(tmp_path: Path):
    nwbs = _make_source_nwbs(tmp_path, 3)
    m = _make_manifest(tmp_path, nwbs)
    entries, total, n_nwbs = _load_source_manifest(m, only_processed=False)
    assert total == 3
    assert n_nwbs == 3
    assert len(entries) == 3
    assert all(e.nwb_path.exists() for e in entries)
    assert all(e.sha256 and len(e.sha256) == 64 for e in entries)


def test_load_source_manifest_only_processed(tmp_path: Path):
    nwbs = _make_source_nwbs(tmp_path, 2)
    m = _make_manifest(tmp_path, nwbs, include_unprocessed=True)
    entries_filtered, total, n_nwbs = _load_source_manifest(m, only_processed=True)
    assert total == 3
    assert n_nwbs == 3                      # the unprocessed entry is an .nwb
    assert len(entries_filtered) == 2
    entries_all, _, _ = _load_source_manifest(m, only_processed=False)
    assert len(entries_all) == 3            # default (False) keeps all NWBs


def test_default_includes_unprocessed_nwbs(tmp_path: Path):
    """Regression for the OBI tarball case: NWBs with was_processed=False must still be eligible by default."""
    nwbs = _make_source_nwbs(tmp_path, 2)
    m = _make_manifest(tmp_path, nwbs, include_unprocessed=True)
    # Need the unprocessed file to exist on disk for the build_manifest path
    (tmp_path / "data" / "skipped.nwb").write_bytes(b"\0" * 100)
    src = NWBSource(dataset="d", manifest=m)  # only_processed defaults to False
    assert src.only_processed is False
    cfg = ProjectConfig(project_name="t", nwb_sources=[src], thresholds_file=None,
                        output_dir=tmp_path / "out")
    df = build_manifest(cfg)
    # All 3 NWBs in the manifest are now eligible
    assert len(df) == 3
    stats = df.attrs["manifest_stats"][0]
    assert stats["n_nwbs_in_manifest"] == 3
    assert stats["n_filtered_unprocessed"] == 0
    assert stats["n_eligible_after_filter"] == 3


def test_build_manifest_reuses_sha256_when_unchanged(tmp_path: Path):
    nwbs = _make_source_nwbs(tmp_path, 3)
    m = _make_manifest(tmp_path, nwbs)
    src = NWBSource(dataset="d", manifest=m, only_processed=True, reuse_sha256=True)
    cfg = ProjectConfig(project_name="t", nwb_sources=[src], thresholds_file=None,
                        output_dir=tmp_path / "out")
    df = build_manifest(cfg)
    assert len(df) == 3
    stats = df.attrs["manifest_stats"][0]
    assert stats["n_files_in_manifest"] == 3
    assert stats["n_present_on_disk"] == 3
    assert stats["n_missing_on_disk"] == 0
    assert stats["n_sha256_reused"] == 3
    assert stats["n_sha256_recomputed"] == 0


def test_build_manifest_recomputes_sha256_after_mtime_change(tmp_path: Path):
    nwbs = _make_source_nwbs(tmp_path, 2)
    m = _make_manifest(tmp_path, nwbs)
    # Bump the on-disk mtime of one file by >1 s so the size/mtime gate fails for it
    far_future = nwbs[0].stat().st_mtime + 1000.0
    os.utime(nwbs[0], (far_future, far_future))
    src = NWBSource(dataset="d", manifest=m, reuse_sha256=True)
    cfg = ProjectConfig(project_name="t", nwb_sources=[src], thresholds_file=None,
                        output_dir=tmp_path / "out")
    df = build_manifest(cfg)
    stats = df.attrs["manifest_stats"][0]
    assert stats["n_sha256_reused"] == 1
    assert stats["n_sha256_recomputed"] == 1


def test_build_manifest_handles_missing_file_gracefully(tmp_path: Path):
    nwbs = _make_source_nwbs(tmp_path, 3)
    m = _make_manifest(tmp_path, nwbs)
    nwbs[1].unlink()   # delete one file
    src = NWBSource(dataset="d", manifest=m)
    cfg = ProjectConfig(project_name="t", nwb_sources=[src], thresholds_file=None,
                        output_dir=tmp_path / "out")
    df = build_manifest(cfg)
    assert len(df) == 2
    stats = df.attrs["manifest_stats"][0]
    assert stats["n_present_on_disk"] == 2
    assert stats["n_missing_on_disk"] == 1


def test_nwbsource_requires_exactly_one_of_path_or_manifest(tmp_path: Path):
    # both set → invalid
    with pytest.raises(Exception):
        NWBSource(dataset="d", path=tmp_path, manifest=tmp_path / "m.json")
    # neither set → invalid
    with pytest.raises(Exception):
        NWBSource(dataset="d")
    # path only → ok
    NWBSource(dataset="d", path=tmp_path)
    # manifest only → ok
    NWBSource(dataset="d", manifest=tmp_path / "m.json")
