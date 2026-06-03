"""Storage-backend dispatch for NWB.

NWB supports two storage backends:
  - HDF5 — one `.nwb` file. Read with `pynwb.NWBHDF5IO`.
  - Zarr — one `*.nwb.zarr/` directory store (Zarr v2). Read with `hdmf_zarr.NWBZarrIO`.

Both expose the same in-memory NWBFile schema, so once a file is open the
rest of the pipeline doesn't care which backend produced it.

This module centralises:
  - `is_zarr(path)` / `is_hdf5(path)` detection
  - `open_nwb(path)` context manager — picks the right IO class
  - `nwb_size(path)` — single-file size for HDF5, recursive sum for Zarr dirs
  - `nwb_sha256(path)` — file hash for HDF5, sorted-manifest hash for Zarr dirs
  - `nwb_mtime(path)` — file mtime for HDF5, max-of-children mtime for Zarr
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Iterator

import pynwb


def is_zarr(path: Path) -> bool:
    """A path is treated as a Zarr-NWB store if it's a directory ending in `.nwb.zarr`
    (the convention used by `hdmf-zarr`'s default writer) or whose root contains `.zgroup`."""
    if not path.is_dir():
        return False
    if path.name.endswith(".nwb.zarr"):
        return True
    return (path / ".zgroup").is_file()


def is_hdf5(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".nwb"


def is_nwb(path: Path) -> bool:
    return is_hdf5(path) or is_zarr(path)


@contextlib.contextmanager
def open_nwb(path: str | Path) -> Iterator[pynwb.NWBFile]:
    """Open an NWB file (HDF5 or Zarr) and yield its NWBFile. Handles cleanup."""
    p = Path(path)
    if is_zarr(p):
        # Lazy import — only when actually needed
        from hdmf_zarr import NWBZarrIO  # type: ignore
        io = NWBZarrIO(str(p), mode="r", load_namespaces=True)
    elif is_hdf5(p):
        io = pynwb.NWBHDF5IO(str(p), mode="r", load_namespaces=True)
    else:
        raise ValueError(f"not a recognised NWB path (neither .nwb nor .nwb.zarr): {p}")
    try:
        yield io.read()
    finally:
        io.close()


def nwb_size(path: Path) -> int:
    """Bytes-on-disk for an NWB store. Single file size for HDF5; recursive sum for Zarr."""
    if is_zarr(path):
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return total
    return path.stat().st_size


def nwb_mtime(path: Path) -> float:
    """Mtime for an NWB store. For Zarr (a directory), uses the max mtime of the
    top-level `.zgroup` / `.zmetadata` markers as a stable freshness signal —
    avoids walking thousands of chunk files."""
    if is_zarr(path):
        mtimes: list[float] = []
        for marker in (".zgroup", ".zattrs", ".zmetadata"):
            mp = path / marker
            if mp.exists():
                try:
                    mtimes.append(mp.stat().st_mtime)
                except OSError:
                    pass
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
        return max(mtimes) if mtimes else 0.0
    return path.stat().st_mtime


def nwb_sha256(path: Path, buf: int = 1 << 20) -> str:
    """Cache key for an NWB store. Stable across renames/moves; changes when content changes.

    - HDF5: streaming sha256 of the file bytes (single file, fast).
    - Zarr: sha256 of a sorted `relpath:size` manifest of the directory tree.
            Chunks are not read — only stat'd. NWB-Zarr stores are write-once in
            practice, so file size + path is a reliable fingerprint and orders of
            magnitude cheaper than hashing thousands of chunks. If a chunk's bytes
            ever change without changing its size, the fingerprint won't notice —
            in that case set source.reuse_sha256 = false / clear the cache.

    Both return a stable 64-char hex digest the cache layer keys on directly.
    """
    if is_zarr(path):
        return _zarr_dir_fingerprint(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _zarr_dir_fingerprint(root: Path) -> str:
    """Cheap stable cache key for a Zarr store: sha256 of sorted `relpath:size` lines."""
    entries: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        entries.append(f"{rel}:{size}")
    fingerprint = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(fingerprint).hexdigest()
