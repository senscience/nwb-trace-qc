"""`nwb-qc serve` — tiny stdlib HTTP server backing the interactive viewer.

Serves:
  GET /                          → the viewer.html template (filled with project name)
  GET /qc_report.csv             → the verdicts CSV (canonical, all cells)
  GET /api/cells                 → JSON list of FLAG-verdict cells (per-cell NaNs stripped)
  GET /api/sweeps/<cell_id>      → JSON list of sweep summaries for the cell's NWB
                                     (cached per nwb_sha256 across requests)
  GET /api/trace/<cell_id>/<idx> → JSON-decimated voltage trace for sweep at index `idx`
                                     ?max_points=1500

Bound to localhost only. No auth — designed for single-user local use.

Performance: NWB handles are LRU-cached (capacity 4). The 4 default sweep fetches
that follow a cell click reuse one open handle instead of opening four. The
/api/sweeps response is also memoized per nwb_sha256.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import io as _io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import ProjectConfig
from .families import METRIC_TO_FAMILY, PSEUDO_METRIC_LABELS
from .nwb_io import is_zarr, nwb_sha256, open_nwb
from .stimuli import StimulusFamilyMap

log = logging.getLogger(__name__)

# In-memory caches keyed by cell_id (NWB paths from manifest)
_manifest_lookup: dict[str, str] = {}
_cell_sha: dict[str, str] = {}
_report_path: Path | None = None
_template_html: str = ""
_project_name: str = ""
_family_map: StimulusFamilyMap | None = None
_total_cells: int = 0
_viewer_cells: list[dict[str, Any]] = []   # flag + fail rows (non-pass)
_thumbnails_dir: Path | None = None
_thumb_disk_cache_enabled: bool = True


def _lttb(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsampling. Returns (x_out, y_out) of length n_out."""
    n = len(y)
    if n_out >= n or n_out < 3:
        return x, y
    bucket_size = (n - 2) / (n_out - 2)
    out_x = np.empty(n_out, dtype=x.dtype)
    out_y = np.empty(n_out, dtype=y.dtype)
    out_x[0] = x[0]; out_y[0] = y[0]
    a = 0
    for i in range(n_out - 2):
        start_next = int(np.floor((i + 1) * bucket_size)) + 1
        end_next = int(np.floor((i + 2) * bucket_size)) + 1
        end_next = min(end_next, n)
        avg_x = np.mean(x[start_next:end_next]) if end_next > start_next else x[start_next]
        avg_y = np.mean(y[start_next:end_next]) if end_next > start_next else y[start_next]
        start = int(np.floor(i * bucket_size)) + 1
        end = int(np.floor((i + 1) * bucket_size)) + 1
        end = min(end, n)
        x_a, y_a = x[a], y[a]
        xs = x[start:end]; ys = y[start:end]
        area = np.abs((x_a - avg_x) * (ys - y_a) - (x_a - xs) * (avg_y - y_a)) * 0.5
        if len(area) == 0:
            chosen = start
        else:
            chosen = start + int(np.argmax(area))
        out_x[i + 1] = x[chosen]; out_y[i + 1] = y[chosen]
        a = chosen
    out_x[-1] = x[-1]; out_y[-1] = y[-1]
    return out_x, out_y


# ─── NWB handle LRU (capacity 4) ─────────────────────────────────────
_HANDLE_LRU: "OrderedDict[str, Any]" = OrderedDict()
_HANDLE_LRU_LOCK = threading.Lock()
_HANDLE_LRU_MAX = 4


def _evict_oldest() -> None:
    while len(_HANDLE_LRU) > _HANDLE_LRU_MAX:
        _, old = _HANDLE_LRU.popitem(last=False)
        try:
            old["__cm__"].__exit__(None, None, None)
        except Exception:
            log.debug("LRU eviction: __exit__ raised", exc_info=True)


def _get_handle(nwb_path: Path):
    """Return an open NWBFile from the LRU cache, opening it (and entering its
    context manager) on miss. Caller must NOT close it.
    """
    key = str(nwb_path.resolve())
    with _HANDLE_LRU_LOCK:
        entry = _HANDLE_LRU.get(key)
        if entry is not None:
            _HANDLE_LRU.move_to_end(key)
            return entry["nwbfile"]
        cm = open_nwb(nwb_path)
        nwbfile = cm.__enter__()
        _HANDLE_LRU[key] = {"nwbfile": nwbfile, "__cm__": cm}
        _evict_oldest()
        return nwbfile


def _close_all_handles() -> None:
    with _HANDLE_LRU_LOCK:
        while _HANDLE_LRU:
            _, e = _HANDLE_LRU.popitem(last=False)
            try:
                e["__cm__"].__exit__(None, None, None)
            except Exception:
                log.debug("close-all: __exit__ raised", exc_info=True)


# ─── Per-sha sweep-list cache ────────────────────────────────────────
_SWEEPS_BY_SHA: dict[str, list[dict[str, Any]]] = {}


def _read_sweeps(nwb_path: Path, sha: str | None = None) -> list[dict[str, Any]]:
    if sha and sha in _SWEEPS_BY_SHA:
        return _SWEEPS_BY_SHA[sha]
    f = _get_handle(nwb_path)
    out: list[dict[str, Any]] = []
    for i, (name, obj) in enumerate(f.acquisition.items()):
        unit = (getattr(obj, "unit", "") or "").lower()
        if unit not in {"volts", "v", ""}:
            continue
        rate = float(getattr(obj, "rate", 0) or 0)
        n_samp = int(obj.data.shape[0]) if obj.data.shape else 0
        duration_s = n_samp / rate if rate > 0 else 0.0
        parts = name.split("__")
        stim = parts[1] if len(parts) >= 3 else name
        sweep_num = parts[2] if len(parts) >= 3 else ""
        fam = _family_map.family_of(stim) if _family_map else None
        out.append({
            "idx": i, "name": name, "family": fam, "stimulus_type": stim,
            "sweep_number": sweep_num, "n_samples": n_samp,
            "rate_hz": rate, "duration_s": round(duration_s, 4),
        })
    if sha:
        _SWEEPS_BY_SHA[sha] = out
    return out


_THUMB_LRU: "OrderedDict[tuple, bytes]" = OrderedDict()
_THUMB_LRU_LOCK = threading.Lock()
_THUMB_LRU_MAX = 256  # ~5 MB at ~20 KB/PNG


def _thumb_disk_path(sha: str, sweep_idx: int, w: int, h: int) -> Path | None:
    if not (_thumb_disk_cache_enabled and _thumbnails_dir):
        return None
    return _thumbnails_dir / "viewer" / f"{sha[:8]}__{sweep_idx}_{w}x{h}.png"


def _render_sweep_thumb(nwb_path: Path, sha: str, sweep_idx: int,
                        w: int, h: int) -> bytes:
    """Render a single decimated sweep to a small PNG. Two-layer cache: in-memory
    LRU + optional on-disk PNG. Returns the raw PNG bytes."""
    key = (sha, sweep_idx, w, h)
    with _THUMB_LRU_LOCK:
        if key in _THUMB_LRU:
            _THUMB_LRU.move_to_end(key)
            return _THUMB_LRU[key]
    disk_path = _thumb_disk_path(sha, sweep_idx, w, h)
    if disk_path and disk_path.is_file():
        data = disk_path.read_bytes()
        with _THUMB_LRU_LOCK:
            _THUMB_LRU[key] = data
            while len(_THUMB_LRU) > _THUMB_LRU_MAX:
                _THUMB_LRU.popitem(last=False)
        return data

    f = _get_handle(nwb_path)
    names = [(n, o) for n, o in f.acquisition.items()
             if (getattr(o, "unit", "") or "").lower() in {"volts", "v", ""}]
    if sweep_idx < 0 or sweep_idx >= len(names):
        raise IndexError(f"sweep_idx {sweep_idx} out of range (have {len(names)})")
    name, obj = names[sweep_idx]
    data = np.asarray(obj.data[:]).reshape(-1).astype(np.float64)
    rate = float(getattr(obj, "rate", 0) or 0)
    t = np.arange(len(data)) / rate if rate > 0 else np.arange(len(data), dtype=float)
    x_out, y_out = _lttb(t, data, 600) if len(data) > 600 else (t, data)
    # Render tiny PNG — no axes labels (the tile shows the title above the img)
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=72)
    ax = fig.add_axes([0.04, 0.06, 0.94, 0.92])
    ax.plot(x_out, y_out * 1000.0, lw=0.6, color="#222")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5); spine.set_color("#bbb")
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    plt.close(fig)
    png_bytes = buf.getvalue()
    with _THUMB_LRU_LOCK:
        _THUMB_LRU[key] = png_bytes
        while len(_THUMB_LRU) > _THUMB_LRU_MAX:
            _THUMB_LRU.popitem(last=False)
    if disk_path:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(png_bytes)
    return png_bytes


def _read_trace(nwb_path: Path, sweep_idx: int, max_points: int) -> dict[str, Any]:
    f = _get_handle(nwb_path)
    names = []
    for name, obj in f.acquisition.items():
        unit = (getattr(obj, "unit", "") or "").lower()
        if unit not in {"volts", "v", ""}:
            continue
        names.append((name, obj))
    if sweep_idx < 0 or sweep_idx >= len(names):
        raise IndexError(f"sweep_idx {sweep_idx} out of range (have {len(names)})")
    name, obj = names[sweep_idx]
    data = np.asarray(obj.data[:]).reshape(-1).astype(np.float64)
    rate = float(getattr(obj, "rate", 0) or 0)
    t = np.arange(len(data)) / rate if rate > 0 else np.arange(len(data), dtype=float)
    x_out, y_out = _lttb(t, data, max_points) if len(data) > max_points else (t, data)
    return {
        "name": name,
        "rate_hz": rate,
        "n_samples": int(len(data)),
        "x_seconds": x_out.tolist(),
        "y_mv": (y_out * 1000.0).tolist(),
    }


# ─── Cell-list (flag-only, NaN-stripped) ─────────────────────────────

def _is_nan(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _build_cells_for_viewer(report_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read qc_report.csv and return every NON-PASS cell (flag + fail), sorted
    fail-first so the worst recordings surface at the top of the list.

    Returns ``(cells, total_cells_in_cohort)``. Pass cells are excluded — they
    don't need inspection. The deep-link from the static report (any cell, any
    verdict) lands here; the cell list now includes both flag and fail rows so
    the link works for either.
    """
    if not report_path.exists():
        return [], 0
    df = pd.read_csv(report_path)
    total = int(len(df))
    if "final_verdict" not in df.columns:
        return [], total
    keep = df[df["final_verdict"].isin(["flag", "fail"])].copy()
    # Sort: FLAG first (the borderline cells that need human judgement to triage),
    # then FAIL (the clearly-bad ones that mostly just need to be excluded).
    # Ties broken by cell_id alpha so the order is stable across runs.
    verdict_rank = {"flag": 0, "fail": 1}
    keep = keep.assign(_rank=keep["final_verdict"].map(verdict_rank).fillna(2))
    keep = keep.sort_values(["_rank", "cell_id"]).drop(columns=["_rank"])
    out: list[dict[str, Any]] = []
    for r in keep.itertuples(index=False):
        d = r._asdict()
        # Strip NaN/null per cell so the per-cell metric table only shows present values.
        d = {k: v for k, v in d.items() if not _is_nan(v)}
        # Parse triggered_metrics if it came in as a JSON string
        tm = d.get("triggered_metrics")
        if isinstance(tm, str):
            try:
                d["triggered_metrics"] = json.loads(tm)
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(d)
    return out, total


# Backwards-compat alias (other tools may still import the old name)
_build_flag_cells = _build_cells_for_viewer


def _sanitise_for_json(obj: Any) -> Any:
    """Recursively replace NaN with None and convert numpy scalars to Python
    types so json.dumps never emits the invalid ``NaN`` token (which would crash
    the browser's JSON.parse on the receiving end).
    """
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    # numpy scalars (np.float64 NaN, np.int64, …) inherit from Python float/int
    # for floats above; ints just pass through cleanly.
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# ─── HTTP handler ────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    server_version = "nwb-qc-serve/0.3.0"

    def log_message(self, fmt, *args):
        log.debug("%s - %s", self.client_address[0], fmt % args)

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(_sanitise_for_json(obj), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, status: int, path: Path, mime: str) -> None:
        data = path.read_bytes()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_png(self, status: int, png_bytes: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_bytes)))
        # 1 day browser cache: same (sha,idx,w,h) → same image forever
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(png_bytes)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        try:
            self._route()
        except FileNotFoundError as e:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(e)})
        except IndexError as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            log.exception("server error")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})

    def _route(self):
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if p and "=" in p)

        if path in ("/", "/index.html", "/viewer.html"):
            html = (_template_html
                    .replace("{{PROJECT}}", _project_name)
                    .replace("{{N_FLAG}}", str(len(_viewer_cells)))
                    .replace("{{N_TOTAL}}", str(_total_cells)))
            self._send_html(HTTPStatus.OK, html)
            return

        if path == "/api/cells":
            # `n_flag` is kept in the payload for backward compatibility with
            # older viewer.html templates — it now actually counts flag+fail
            # (the cells the viewer surfaces). Newer templates can read
            # `n_attention` instead which makes the semantics explicit.
            n_fail = sum(1 for c in _viewer_cells if c.get("final_verdict") == "fail")
            n_flag = sum(1 for c in _viewer_cells if c.get("final_verdict") == "flag")
            self._send_json(HTTPStatus.OK, {
                "n_flag": len(_viewer_cells),     # legacy: total non-pass
                "n_attention": len(_viewer_cells),
                "n_fail_only": n_fail,
                "n_flag_only": n_flag,
                "n_total": _total_cells,
                "cells": _viewer_cells,
            })
            return

        if path == "/api/families":
            # Table the viewer uses to compute "implicated families" per cell —
            # mirrors families.py so the client doesn't need to duplicate it.
            self._send_json(HTTPStatus.OK, {
                "metric_to_family": METRIC_TO_FAMILY,
                "pseudo_labels": PSEUDO_METRIC_LABELS,
            })
            return

        if path.startswith("/api/thumb/"):
            rest = path[len("/api/thumb/"):]
            cell_id, _, idx_str = rest.partition("/")
            nwb = _manifest_lookup.get(cell_id)
            if not nwb:
                raise FileNotFoundError(f"cell_id not in manifest: {cell_id}")
            try:
                idx = int(idx_str)
            except ValueError as e:
                raise IndexError(f"bad sweep index {idx_str!r}") from e
            w = max(80, min(600, int(params.get("w", "220"))))
            h = max(40, min(400, int(params.get("h", "100"))))
            sha = _cell_sha.get(cell_id) or ""
            png = _render_sweep_thumb(Path(nwb), sha, idx, w, h)
            self._send_png(HTTPStatus.OK, png)
            return

        if path == "/qc_report.csv":
            if _report_path is None or not _report_path.exists():
                raise FileNotFoundError("qc_report.csv not found; run `nwb-qc run` first")
            self._send_file(HTTPStatus.OK, _report_path, "text/csv; charset=utf-8")
            return

        if path.startswith("/api/sweeps/"):
            cell_id = path[len("/api/sweeps/"):]
            nwb = _manifest_lookup.get(cell_id)
            if not nwb:
                raise FileNotFoundError(f"cell_id not in manifest: {cell_id}")
            sha = _cell_sha.get(cell_id)
            self._send_json(HTTPStatus.OK, _read_sweeps(Path(nwb), sha))
            return

        if path.startswith("/api/trace/"):
            rest = path[len("/api/trace/"):]
            cell_id, _, idx_str = rest.partition("/")
            nwb = _manifest_lookup.get(cell_id)
            if not nwb:
                raise FileNotFoundError(f"cell_id not in manifest: {cell_id}")
            try:
                idx = int(idx_str)
            except ValueError as e:
                raise IndexError(f"bad sweep index {idx_str!r}") from e
            max_points = int(params.get("max_points", "1500"))
            self._send_json(HTTPStatus.OK, _read_trace(Path(nwb), idx, max_points))
            return

        if _report_path is not None:
            candidate = _report_path.parent / path.lstrip("/")
            if candidate.is_file() and candidate.resolve().is_relative_to(_report_path.parent.resolve()):
                mime = "text/html; charset=utf-8" if candidate.suffix == ".html" else "application/octet-stream"
                self._send_file(HTTPStatus.OK, candidate, mime)
                return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "no route", "path": path})


def _viewer_html_template() -> str:
    return (Path(__file__).parent / "templates" / "viewer.html").read_text()


def serve(cfg: ProjectConfig, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Block while serving on host:port. Ctrl-C to stop."""
    global _manifest_lookup, _cell_sha, _report_path, _template_html, _project_name, _family_map
    global _viewer_cells, _total_cells, _thumbnails_dir, _thumb_disk_cache_enabled
    if not cfg.manifest_path or not cfg.manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {cfg.manifest_path}; run `nwb-qc run --config <yaml>` first")
    manifest = pq.read_table(cfg.manifest_path).to_pandas()
    _manifest_lookup = dict(zip(manifest["cell_id"].astype(str), manifest["nwb_path"].astype(str)))
    _cell_sha = dict(zip(manifest["cell_id"].astype(str), manifest["nwb_sha256"].astype(str)))
    _report_path = cfg.report_csv
    _template_html = _viewer_html_template()
    _project_name = cfg.project_name
    _family_map = StimulusFamilyMap(cfg.stimulus_protocols)
    _viewer_cells, _total_cells = (
        _build_cells_for_viewer(cfg.report_csv) if cfg.report_csv else ([], 0)
    )
    _thumbnails_dir = cfg.thumbnails_dir
    _thumb_disk_cache_enabled = bool(getattr(cfg, "viewer_cache_thumbnails", True))
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"nwb-qc serve · {cfg.project_name} · {url}")
    n_fail = sum(1 for c in _viewer_cells if c.get("final_verdict") == "fail")
    n_flag = sum(1 for c in _viewer_cells if c.get("final_verdict") == "flag")
    print(f"  cells: {_total_cells} total · {n_fail} fail · {n_flag} flag · report: {_report_path}")
    print("  press Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    finally:
        server.server_close()
        _close_all_handles()
    print("\nserver stopped.")
