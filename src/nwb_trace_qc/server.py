"""`nwb-qc serve` — tiny stdlib HTTP server backing the interactive viewer.

Serves:
  GET /                          → the viewer.html template (filled with project name)
  GET /qc_report.csv             → the verdicts CSV
  GET /api/sweeps/<cell_id>      → JSON list of sweep summaries for the cell's NWB
  GET /api/trace/<cell_id>/<idx> → JSON-decimated voltage trace for sweep at index `idx`
                                     ?max_points=2000

Bound to localhost only. No auth — designed for single-user local use.
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import ProjectConfig
from .nwb_io import open_nwb
from .stimuli import StimulusFamilyMap

log = logging.getLogger(__name__)

# In-memory caches keyed by cell_id (NWB paths from manifest)
_manifest_lookup: dict[str, str] = {}
_report_path: Path | None = None
_template_html: str = ""
_project_name: str = ""
_family_map: StimulusFamilyMap | None = None


def _lttb(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsampling. Returns (x_out, y_out) of length n_out.

    Pure-numpy, fast enough for sweeps with hundreds of thousands of samples.
    """
    n = len(y)
    if n_out >= n or n_out < 3:
        return x, y
    # First and last points are always retained
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
        # Largest-area triangle
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


def _read_sweeps(nwb_path: Path) -> list[dict[str, Any]]:
    """Open the NWB and list its voltage-trace acquisitions."""
    out: list[dict[str, Any]] = []
    with open_nwb(nwb_path) as f:
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
    return out


def _read_trace(nwb_path: Path, sweep_idx: int, max_points: int) -> dict[str, Any]:
    """Open the NWB, pull sweep #sweep_idx, decimate to max_points, return data."""
    with open_nwb(nwb_path) as f:
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
        # Convert V → mV for browser-friendly axes
        return {
            "name": name,
            "rate_hz": rate,
            "n_samples": int(len(data)),
            "x_seconds": x_out.tolist(),
            "y_mv": (y_out * 1000.0).tolist(),
        }


class _Handler(BaseHTTPRequestHandler):
    server_version = "nwb-qc-serve/0.2.0"

    def log_message(self, fmt, *args):
        # Quieter logging
        log.debug("%s - %s", self.client_address[0], fmt % args)

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
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
            self._send_html(HTTPStatus.OK, _template_html.replace("{{PROJECT}}", _project_name))
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
            self._send_json(HTTPStatus.OK, _read_sweeps(Path(nwb)))
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
            max_points = int(params.get("max_points", "2000"))
            self._send_json(HTTPStatus.OK, _read_trace(Path(nwb), idx, max_points))
            return

        # Fallback: try to serve a static file under output_dir (e.g. qc_report.html)
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
    global _manifest_lookup, _report_path, _template_html, _project_name, _family_map
    if not cfg.manifest_path or not cfg.manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {cfg.manifest_path}; run `nwb-qc run --config <yaml>` first")
    manifest = pq.read_table(cfg.manifest_path).to_pandas()
    _manifest_lookup = dict(zip(manifest["cell_id"].astype(str), manifest["nwb_path"].astype(str)))
    _report_path = cfg.report_csv
    _template_html = _viewer_html_template()
    _project_name = cfg.project_name
    _family_map = StimulusFamilyMap(cfg.stimulus_protocols)
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"nwb-qc serve · {cfg.project_name} · {url}")
    print(f"  cells: {len(_manifest_lookup)} · report: {_report_path}")
    print("  press Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    server.server_close()
    print("\nserver stopped.")
