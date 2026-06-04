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
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pynwb

log = logging.getLogger(__name__)


# SI prefix multipliers — used to normalize TimeSeries.unit strings to base SI.
# Anything unrecognised → multiplier 1.0 with a one-time warning per file.
_UNIT_PREFIXES: dict[str, float] = {
    "":  1.0,
    "k": 1e3, "K": 1e3,
    "m": 1e-3, "M": 1e6,  # 'M' = mega only when followed by a base unit (megavolts irrelevant in ephys)
    "u": 1e-6, "μ": 1e-6, "μ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

# Already-warned-about unit strings, to avoid spamming the log
_WARNED_UNITS: set[str] = set()


def _parse_unit_to_si(unit_str: str | None, base: str) -> float:
    """Return a multiplier that converts a value in `unit_str` to base SI (`base`).

    `base` is "V" (volts) or "A" (amperes) — the SI target.

    Recognises the common ephys forms: `volts` / `V` / `mV` / `millivolts` /
    `microvolts` / `uV` / `μV`, and analogous for current. Unknown / empty
    units are treated as already-SI (multiplier = 1.0) with a single warning
    per unknown string.
    """
    if not unit_str:
        return 1.0
    s = unit_str.strip()
    s_lower = s.lower()

    # Long forms
    long_forms = {
        ("volts", "V"): 1.0, ("volt", "V"): 1.0,
        ("millivolts", "V"): 1e-3, ("millivolt", "V"): 1e-3,
        ("microvolts", "V"): 1e-6, ("microvolt", "V"): 1e-6,
        ("nanovolts", "V"): 1e-9, ("kilovolts", "V"): 1e3,
        ("amperes", "A"): 1.0, ("ampere", "A"): 1.0, ("amps", "A"): 1.0, ("amp", "A"): 1.0,
        ("milliamperes", "A"): 1e-3, ("milliamps", "A"): 1e-3,
        ("microamperes", "A"): 1e-6, ("microamps", "A"): 1e-6,
        ("nanoamperes", "A"): 1e-9, ("nanoamps", "A"): 1e-9,
        ("picoamperes", "A"): 1e-12, ("picoamps", "A"): 1e-12,
    }
    if (s_lower, base) in long_forms:
        return long_forms[(s_lower, base)]

    # Short forms: <prefix><base>, e.g. "mV", "pA", "uA", "μV"
    # Match by stripping a known base suffix and consulting the prefix table.
    for base_char, base_target in (("V", "V"), ("v", "V"), ("A", "A"), ("a", "A")):
        if base_target != base:
            continue
        if s.endswith(base_char):
            prefix = s[:-1]
            if prefix in _UNIT_PREFIXES:
                # Guard: lower-case "m" is milli, capital "M" is mega — both fall into _UNIT_PREFIXES.
                # For ephys we treat the empty prefix as 1.0 (bare "V" or "A").
                return _UNIT_PREFIXES[prefix]
            # Fall through to unknown

    # Unknown: warn once, assume SI
    if s not in _WARNED_UNITS:
        _WARNED_UNITS.add(s)
        log.warning("nwb_io: unrecognised unit string %r (expected SI form for %s); "
                    "assuming value is already in base SI units.", s, base)
    return 1.0


def _ts_to_si(obj: Any, base: str) -> np.ndarray:
    """Apply NWB's `data * conversion + offset`, then unit-prefix normalize.

    Per the NWB spec the physical value of a sample is
        physical_value = data * conversion + offset
    where (conversion, offset, unit) live on the TimeSeries. Most files have
    conversion=1.0 and offset=0.0 and use SI units, but the spec allows scaled
    forms. This helper returns the trace as a 1-D float array in base SI.
    """
    raw = np.asarray(obj.data[:], dtype=np.float64).reshape(-1)
    conv = float(getattr(obj, "conversion", 1.0) or 1.0)
    offs = float(getattr(obj, "offset", 0.0) or 0.0)
    unit_mult = _parse_unit_to_si(getattr(obj, "unit", None), base)
    # If conversion already maps to SI (the modern NWB convention), unit_mult is
    # typically 1.0 (unit was 'volts' / 'amperes'). When conversion=1.0 but unit
    # is e.g. 'millivolts', unit_mult does the work. They compose multiplicatively.
    return raw * (conv * unit_mult) + offs


def voltage_si(obj: Any) -> np.ndarray:
    """Return the trace in volts (SI), regardless of stored unit/conversion/offset."""
    return _ts_to_si(obj, "V")


def current_si(obj: Any) -> np.ndarray:
    """Return the trace in amperes (SI), regardless of stored unit/conversion/offset."""
    return _ts_to_si(obj, "A")


def find_paired_stimulus(nwbfile: pynwb.NWBFile, acq_name: str, acq_obj: Any) -> Any | None:
    """Return the stimulus TimeSeries paired with an acquisition, or None.

    NWB stores stimuli in `nwbfile.stimulus`. Matching strategy in order of
    confidence:
      1. Exact name match: nwbfile.stimulus.get(acq_name)
      2. Common suffix/prefix variants ('_stim', 'stim_', 'Stimulus')
      3. Time-aligned match: same starting_time (±1 ms), same rate, same length

    Returns None cleanly when nothing matches — callers can fall back.
    """
    stim_map = getattr(nwbfile, "stimulus", None) or {}
    if not stim_map:
        return None

    # 1. Exact name
    if acq_name in stim_map:
        return stim_map[acq_name]

    # 2. Common suffix/prefix variants
    candidates = [
        acq_name + "_stim",
        "stim_" + acq_name,
        acq_name.replace("ic__", "cc__", 1) if acq_name.startswith("ic__") else None,
        acq_name + "Stimulus",
        # Some labs prefix stimuli with a different family token: ccs__ vs ic__
        acq_name.replace("ic__", "ccs__", 1) if acq_name.startswith("ic__") else None,
    ]
    for name in candidates:
        if name and name in stim_map:
            return stim_map[name]

    # 3. Time-aligned match (last resort — O(n) over stim list)
    acq_t0 = float(getattr(acq_obj, "starting_time", 0) or 0)
    acq_rate = float(getattr(acq_obj, "rate", 0) or 0)
    acq_len = int(acq_obj.data.shape[0]) if hasattr(acq_obj, "data") and acq_obj.data.shape else 0
    for name, stim in stim_map.items():
        stim_t0 = float(getattr(stim, "starting_time", 0) or 0)
        stim_rate = float(getattr(stim, "rate", 0) or 0)
        stim_len = int(stim.data.shape[0]) if hasattr(stim, "data") and stim.data.shape else 0
        if (abs(stim_t0 - acq_t0) < 1e-3 and
                abs(stim_rate - acq_rate) < 1e-6 and
                stim_len == acq_len):
            return stim
    return None


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
