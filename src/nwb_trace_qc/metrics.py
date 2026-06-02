"""Per-NWB trace-level QC metric computation.

Conservative, lab-agnostic implementations that lean on stimulus *families*
(configurable name mapping) so they work for any cohort. Returns a flat dict
of metric values; threshold logic and verdicts live elsewhere.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pynwb

from .stimuli import StimulusFamilyMap

warnings.filterwarnings("ignore", message=".*HDMF.*", category=UserWarning)


# Voltage rails outside which a current-clamp trace is almost certainly clipped
VOLT_RAIL_MIN_V = -0.150  # -150 mV
VOLT_RAIL_MAX_V = 0.080   #  +80 mV


def _name_to_stim(acq_name: str) -> str:
    """Pull the stimulus-protocol token from an NWB acquisition name.

    Many labs prefix the protocol name (e.g. 'ic__APWaveform__042', 'ccs__IDRest__013')
    or store it bare. Returns the middle token if there are >=3 '__'-split parts,
    else the whole name.
    """
    parts = acq_name.split("__")
    return parts[1] if len(parts) >= 3 else acq_name


def _iter_current_clamp_acqs(nwbfile):
    """Yield (name, CurrentClampSeries-like) — anything that exposes voltage data + rate."""
    for name, obj in nwbfile.acquisition.items():
        data = getattr(obj, "data", None)
        if data is None:
            continue
        # Voltage trace: stored in volts, neurodata_type ending with ClampSeries
        unit = (getattr(obj, "unit", "") or "").lower()
        # Accept volts (current-clamp recording trace). Skip current-clamp stimuli (amperes).
        if unit not in {"volts", "v", ""}:
            continue
        yield name, obj


def _trace_array(obj) -> np.ndarray:
    """Return the trace as a 1-D float numpy array."""
    data = np.asarray(obj.data[:])
    if data.ndim > 1:
        data = data.reshape(-1)
    return data.astype(np.float64, copy=False)


def _rate(obj) -> float:
    return float(getattr(obj, "rate", 0.0) or 0.0)


def _median_last_seconds(trace_v: np.ndarray, rate_hz: float, seconds: float = 0.5) -> float:
    if rate_hz <= 0 or len(trace_v) == 0:
        return float("nan")
    n = max(1, int(rate_hz * seconds))
    n = min(n, len(trace_v))
    return float(np.median(trace_v[-n:]))


def _rms(trace_v: np.ndarray) -> float:
    if len(trace_v) == 0:
        return float("nan")
    centered = trace_v - np.mean(trace_v)
    return float(np.sqrt(np.mean(centered ** 2)))


def _is_clipped(trace_v: np.ndarray, rate_hz: float, min_ms: float = 1.0) -> bool:
    """True if the trace stays at a voltage rail for at least min_ms."""
    if len(trace_v) == 0 or rate_hz <= 0:
        return False
    n_min = max(1, int(rate_hz * min_ms / 1000.0))
    hits = (trace_v <= VOLT_RAIL_MIN_V) | (trace_v >= VOLT_RAIL_MAX_V)
    if not np.any(hits):
        return False
    # Look for a run of consecutive hits >= n_min
    # Fast: count max run length
    run = 0; mx = 0
    for v in hits:
        run = run + 1 if v else 0
        if run > mx: mx = run
        if mx >= n_min: return True
    return False


def _has_nan(trace_v: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(trace_v)))


def _peak_overshoot_mv(trace_v: np.ndarray) -> float:
    """Peak voltage above 0 V, in mV (negative if no overshoot)."""
    if len(trace_v) == 0:
        return float("nan")
    return float(np.max(trace_v) * 1000.0)


def _ap_threshold_mv(trace_v: np.ndarray, rate_hz: float, slope_thresh: float = 20.0) -> float:
    """First crossing of dV/dt threshold (mV/ms) — returns voltage at that point in mV.

    NaN if no spike-like upstroke detected. slope_thresh in mV/ms (default 20 = standard).
    """
    if len(trace_v) < 3 or rate_hz <= 0:
        return float("nan")
    dt_ms = 1000.0 / rate_hz
    dvdt = np.gradient(trace_v * 1000.0) / dt_ms  # mV/ms
    above = np.where(dvdt > slope_thresh)[0]
    if len(above) == 0:
        return float("nan")
    return float(trace_v[above[0]] * 1000.0)


def _rs_from_test_pulse_mohm(trace_v: np.ndarray, rate_hz: float) -> float:
    """Coarse access-resistance estimate from a current-clamp test-pulse capacitive transient.

    Looks for the largest 5–50 ms step deflection in the trace, fits the peak as a
    voltage step delta_V, and assumes the standard test-pulse amplitude of 50 pA
    (configurable later). Rs[MΩ] ≈ |delta_V[V]| / 50e-12 / 1e6.

    Returns NaN if the trace is too short or no step-like deflection is found.

    Note: this is intentionally a rough cohort-comparison estimate — for proper Rs
    we'd need the stimulus current trace, but per the project requirement to reuse
    pre-computed values where possible, exact Rs may come from the wrangler tables.
    """
    if len(trace_v) < 10 or rate_hz <= 0:
        return float("nan")
    baseline = float(np.median(trace_v[: max(1, int(0.005 * rate_hz))]))  # first 5 ms
    n_total = len(trace_v)
    step_window = trace_v[max(1, int(0.005 * rate_hz)) : min(n_total, int(0.055 * rate_hz))]
    if len(step_window) == 0:
        return float("nan")
    peak_v = float(np.min(step_window) if np.abs(np.min(step_window) - baseline) > np.abs(np.max(step_window) - baseline) else np.max(step_window))
    delta_v = peak_v - baseline
    # Assume nominal 50 pA test pulse; user can override threshold to compensate
    nominal_I = 50e-12  # 50 pA
    return abs(delta_v) / nominal_I / 1e6


def compute_metrics(nwb_path: str | Path, family_map: StimulusFamilyMap) -> dict[str, Any]:
    """Open one NWB, compute per-cell QC metrics, return a flat dict.

    Always returns a row even when computations fail — failure modes show up as
    NaN/None values; downstream verdict logic can flag missing data.
    """
    out: dict[str, Any] = {
        "vrest_mv": float("nan"),
        "vrest_drift_mv": float("nan"),
        "rs_mohm_initial": float("nan"),
        "rs_mohm_final": float("nan"),
        "rs_drift_pct": float("nan"),
        "rin_mohm": float("nan"),
        "ap_amp_overshoot_mv": float("nan"),
        "ap_threshold_drift_mv": float("nan"),
        "baseline_rms_mv": float("nan"),
        "n_sweeps_total": 0,
        "n_sweeps_clipped": 0,
        "n_sweeps_nan": 0,
        "qc_protocol_coverage": False,
        "compute_error": None,
    }
    try:
        with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
            nwbfile = io.read()
            sponhold_vrest: list[float] = []
            rs_estimates_in_order: list[float] = []
            ap_overshoots: list[float] = []
            ap_thresholds: list[float] = []
            baseline_rms_vals: list[float] = []
            present_families: set[str] = set()
            n_total = 0
            n_clip = 0
            n_nan = 0

            for name, obj in _iter_current_clamp_acqs(nwbfile):
                n_total += 1
                rate = _rate(obj)
                trace = _trace_array(obj)
                stim_token = _name_to_stim(name)
                family = family_map.family_of(stim_token)
                if family:
                    present_families.add(family)

                if _has_nan(trace):
                    n_nan += 1
                    continue
                if _is_clipped(trace, rate):
                    n_clip += 1

                if family == "spontaneous_hold":
                    sponhold_vrest.append(_median_last_seconds(trace, rate, 0.5))
                    baseline_rms_vals.append(_rms(trace) * 1000.0)  # in mV
                elif family == "test_pulse":
                    rs_estimates_in_order.append(_rs_from_test_pulse_mohm(trace, rate))
                elif family == "ap_waveform":
                    ap_overshoots.append(_peak_overshoot_mv(trace))
                    th = _ap_threshold_mv(trace, rate)
                    if not math.isnan(th):
                        ap_thresholds.append(th)
                elif family == "rest_firing":
                    th = _ap_threshold_mv(trace, rate)
                    if not math.isnan(th):
                        ap_thresholds.append(th)
                # IV / threshold_search: skipped here; full Rin needs current trace too

            out["n_sweeps_total"] = n_total
            out["n_sweeps_clipped"] = n_clip
            out["n_sweeps_nan"] = n_nan

            if sponhold_vrest:
                vals = [v for v in sponhold_vrest if not math.isnan(v)]
                if vals:
                    out["vrest_mv"] = float(np.median(vals) * 1000.0)  # V -> mV
                    if len(vals) >= 2:
                        out["vrest_drift_mv"] = float((vals[-1] - vals[0]) * 1000.0)

            if baseline_rms_vals:
                out["baseline_rms_mv"] = float(np.median(baseline_rms_vals))

            rs_clean = [r for r in rs_estimates_in_order if not math.isnan(r)]
            if rs_clean:
                out["rs_mohm_initial"] = float(rs_clean[0])
                out["rs_mohm_final"] = float(rs_clean[-1])
                if out["rs_mohm_initial"] > 0:
                    out["rs_drift_pct"] = float(
                        (out["rs_mohm_final"] - out["rs_mohm_initial"]) / out["rs_mohm_initial"] * 100.0
                    )

            if ap_overshoots:
                out["ap_amp_overshoot_mv"] = float(np.median(ap_overshoots))
            if len(ap_thresholds) >= 2:
                out["ap_threshold_drift_mv"] = float(ap_thresholds[-1] - ap_thresholds[0])

            # Coverage: at least one of each of the essential families
            essential = {"spontaneous_hold", "test_pulse", "ap_waveform"}
            out["qc_protocol_coverage"] = essential.issubset(present_families)
    except Exception as e:  # noqa: BLE001
        out["compute_error"] = f"{type(e).__name__}: {e}"
    return out
