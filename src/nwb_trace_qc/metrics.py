"""Per-NWB trace-level QC metric computation.

Conservative, lab-agnostic implementations that lean on stimulus *families*
(configurable name mapping) so they work for any cohort. Returns a flat dict
of metric values; threshold logic and verdicts live elsewhere.
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pynwb

from .nwb_io import current_si, find_paired_stimulus, open_nwb, voltage_si
from .stimuli import StimulusFamilyMap

log = logging.getLogger(__name__)

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
    """Yield (name, CurrentClampSeries-like) — anything that exposes voltage data + rate.

    Voltage acceptance: the unit string normalises to volts. Common variants:
    `volts`, `V`, `mV`, `millivolts`, `μV`. The actual values are still raw —
    `voltage_si()` in nwb_io applies conversion + unit-prefix when the trace is
    read for analysis. We just gatekeep here so we don't try to interpret a
    current-clamp stimulus channel as a voltage trace.
    """
    voltage_units = {"volts", "v", "mv", "millivolts", "millivolt",
                     "microvolts", "microvolt", "uv", "μv", ""}
    for name, obj in nwbfile.acquisition.items():
        data = getattr(obj, "data", None)
        if data is None:
            continue
        unit = (getattr(obj, "unit", "") or "").lower()
        if unit not in voltage_units:
            continue
        yield name, obj


def _trace_array(obj) -> np.ndarray:
    """DEPRECATED alias for voltage_si — kept for any straggler imports.
    New code should call `voltage_si(obj)` from `nwb_io` directly.
    """
    return voltage_si(obj)


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


def _step_decay_residual_rel(trace_v: np.ndarray, rate_hz: float) -> float:
    """Relative exponential-decay-fit residual for a test-pulse/step sweep.

    Detects the largest |dV| as the step edge; takes the 50-500 ms recovery window;
    fits V(t) = A*exp(-t/tau) + V_inf via 3-parameter least squares (log-linearised);
    returns sqrt(SSE) / |A|. NaN if window too short or step not found.

    Clean exponential decay → values near 0 (< 0.05 typical).
    Glitchy/ringing recovery → larger values (>0.15 = flag, >0.30 = fail).
    """
    if len(trace_v) < int(0.6 * rate_hz) or rate_hz <= 0:
        return float("nan")
    dv = np.abs(np.diff(trace_v))
    if len(dv) == 0 or np.max(dv) == 0:
        return float("nan")
    edge = int(np.argmax(dv))
    start = edge + int(0.005 * rate_hz)       # skip 5 ms past the edge (purely capacitive)
    stop  = edge + int(0.500 * rate_hz)
    if stop - start < int(0.020 * rate_hz):  # need ≥20 ms of recovery
        return float("nan")
    window = trace_v[start:stop]
    if len(window) < 5:
        return float("nan")
    t = np.arange(len(window)) / rate_hz
    # Three-parameter exponential fit using scipy-free approach:
    # Iteratively estimate V_inf from the tail, then log-linear fit (A, tau).
    v_inf = float(np.median(window[-max(1, len(window) // 5):]))
    y = window - v_inf
    A0 = float(y[0])
    if A0 == 0 or not np.all(np.isfinite(y)):
        return float("nan")
    sign = 1.0 if A0 > 0 else -1.0
    y_signed = sign * y
    # Keep only positive samples for log fit
    mask = y_signed > 1e-12
    if mask.sum() < 5:
        return float("nan")
    try:
        coefs = np.polyfit(t[mask], np.log(y_signed[mask]), 1)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    tau = -1.0 / coefs[0] if coefs[0] != 0 else float("inf")
    A_fit = sign * float(np.exp(coefs[1]))
    pred = A_fit * np.exp(-t / tau) + v_inf
    sse = float(np.sum((window - pred) ** 2))
    rmse = math.sqrt(sse / len(window))
    return float(rmse / abs(A_fit)) if A_fit != 0 else float("nan")


def _vm_drift_slope_mv_per_s(trace_v: np.ndarray, rate_hz: float, min_seconds: float = 3.0) -> float:
    """Linear-regression slope of voltage vs time on a long spontaneous-hold sweep.

    Trims first/last 500 ms to avoid transient. Slope in mV/s.
    NaN if the sweep is shorter than min_seconds.
    """
    n = len(trace_v)
    if rate_hz <= 0 or n < int(min_seconds * rate_hz):
        return float("nan")
    trim = int(0.5 * rate_hz)
    seg = trace_v[trim:n - trim]
    if len(seg) < 10:
        return float("nan")
    t = np.arange(len(seg)) / rate_hz
    # Least-squares slope
    tm = t - t.mean()
    vm = seg - seg.mean()
    denom = float(np.sum(tm * tm))
    if denom == 0:
        return float("nan")
    slope_v_per_s = float(np.sum(tm * vm) / denom)
    return slope_v_per_s * 1000.0  # V/s → mV/s


def _failed_spike_fraction(trace_v: np.ndarray, rate_hz: float,
                            slope_thresh_mv_per_ms: float = 20.0,
                            window_ms: float = 5.0) -> float:
    """Fraction of spike initiations (dV/dt threshold crossings) that don't reach ≥0 mV.

    NaN if no initiations detected (so we don't conflate "quiescent sweep" with "all failed").
    """
    if len(trace_v) < 3 or rate_hz <= 0:
        return float("nan")
    dt_ms = 1000.0 / rate_hz
    dvdt = np.gradient(trace_v * 1000.0) / dt_ms  # mV/ms
    rising = dvdt > slope_thresh_mv_per_ms
    # Initiations = first sample of each rising run
    starts = np.where(rising[1:] & ~rising[:-1])[0] + 1
    if len(starts) == 0:
        return float("nan")
    win = int(max(1, window_ms * rate_hz / 1000.0))
    failed = 0
    for s in starts:
        peak_window = trace_v[s : min(len(trace_v), s + win)]
        if len(peak_window) == 0:
            continue
        if np.max(peak_window) < 0.0:   # never reaches 0 V (= 0 mV)
            failed += 1
    return float(failed) / float(len(starts))


def _ap_amplitude_cv(trace_v: np.ndarray, rate_hz: float, min_aps: int = 5) -> float:
    """Coefficient of variation (std/mean) of AP peak amplitudes within one sweep.

    Detects local maxima above 0 mV separated by ≥2 ms refractory; needs ≥`min_aps`.
    NaN if fewer than `min_aps` are detected.
    """
    if len(trace_v) < 3 or rate_hz <= 0:
        return float("nan")
    refract = int(max(1, 0.002 * rate_hz))  # 2 ms refractory
    peaks: list[float] = []
    i = 1
    n = len(trace_v)
    while i < n - 1:
        if trace_v[i] > 0.0 and trace_v[i] > trace_v[i - 1] and trace_v[i] >= trace_v[i + 1]:
            peaks.append(float(trace_v[i] * 1000.0))  # mV
            i += refract
        else:
            i += 1
    if len(peaks) < min_aps:
        return float("nan")
    arr = np.array(peaks)
    mean = float(np.mean(arr))
    if mean == 0:
        return float("nan")
    return float(np.std(arr) / abs(mean))


def _late_instability_index(trace_v: np.ndarray, rate_hz: float, min_seconds: float = 5.0) -> float:
    """Ratio of late-window vs early-window activity/variance, minus 1.

    Splits a long sweep into first vs last quartile. Computes spike rate
    (zero-crossings of dV/dt above threshold) and voltage variance in each.
    Index = max(rate_ratio, var_ratio) - 1.
    NaN if sweep shorter than min_seconds.
    """
    n = len(trace_v)
    if rate_hz <= 0 or n < int(min_seconds * rate_hz):
        return float("nan")
    q = n // 4
    early = trace_v[:q]
    late = trace_v[3 * q : 4 * q]
    if len(early) < 10 or len(late) < 10:
        return float("nan")
    def _rate(seg: np.ndarray) -> float:
        if len(seg) < 3:
            return 0.0
        dt_ms = 1000.0 / rate_hz
        dvdt = np.gradient(seg * 1000.0) / dt_ms
        crossings = int(np.sum((dvdt[1:] > 20.0) & (dvdt[:-1] <= 20.0)))
        return crossings / (len(seg) / rate_hz)
    eps = 1e-9
    rate_ratio = _rate(late) / (_rate(early) + eps)
    var_ratio = float(np.var(late)) / (float(np.var(early)) + eps)
    return float(max(rate_ratio, var_ratio) - 1.0)


NOMINAL_TEST_PULSE_AMPS = 50e-12  # 50 pA — used only when stimulus current is unavailable


def _rs_from_test_pulse_mohm(trace_v: np.ndarray, rate_hz: float,
                              stim_a: np.ndarray | None = None) -> tuple[float, bool]:
    """Access resistance (MΩ) from a current-clamp test-pulse.

    Returns (rs_mohm, used_fallback) where `used_fallback=True` indicates the
    50 pA nominal-amplitude hack was used (no paired stimulus current was passed
    in). When `stim_a` is provided, the step amplitude is read directly from the
    stimulus and Rs is correct.

    Rs = ΔV / ΔI in MΩ. Windows: baseline = first 5 ms; step plateau = 20–50 ms
    after the largest step transition (so we skip the capacitive transient).
    """
    if len(trace_v) < 10 or rate_hz <= 0:
        return float("nan"), False

    n = len(trace_v)
    n_pre = max(1, int(0.005 * rate_hz))   # 5 ms baseline
    n_post_start = max(n_pre + 1, int(0.020 * rate_hz))
    n_post_end = min(n, int(0.050 * rate_hz))

    if n_post_end <= n_post_start:
        return float("nan"), False

    v_baseline = float(np.median(trace_v[:n_pre]))
    v_plateau = float(np.mean(trace_v[n_post_start:n_post_end]))
    delta_v = v_plateau - v_baseline    # volts

    if stim_a is not None and len(stim_a) >= n_post_end:
        # Use the actual stimulus current — proper Rs computation
        i_baseline = float(np.mean(stim_a[:n_pre]))
        i_plateau = float(np.mean(stim_a[n_post_start:n_post_end]))
        delta_i = i_plateau - i_baseline   # amps
        if abs(delta_i) < 1e-15:           # less than 1 fA = no real step
            return float("nan"), False
        return abs(delta_v / delta_i) / 1e6, False

    # Fallback: assume nominal 50 pA test pulse (legacy behavior)
    return abs(delta_v) / NOMINAL_TEST_PULSE_AMPS / 1e6, True


def _holding_current_pa(stim_a: np.ndarray | None, rate_hz: float,
                          pre_step_seconds: float = 0.005) -> float:
    """Mean baseline current of a sweep in pA.

    Uses the first `pre_step_seconds` of the stimulus trace as the holding
    baseline. Returns NaN when no stimulus is available.
    """
    if stim_a is None or len(stim_a) == 0 or rate_hz <= 0:
        return float("nan")
    n_pre = max(1, int(pre_step_seconds * rate_hz))
    n_pre = min(n_pre, len(stim_a))
    return float(np.mean(stim_a[:n_pre]) * 1e12)


def _iv_subthreshold_pair(stim_a: np.ndarray | None, trace_v: np.ndarray,
                           rate_hz: float) -> tuple[float, float] | None:
    """For one IV sweep, return (I_step_pA, V_steady_mV) for use in Rin fitting.

    Requires the paired stimulus (so we know the step amplitude). V_steady is the
    median voltage in the last 20 ms of the step; I_step is the median current in
    the same window minus the holding baseline (first 5 ms).

    Returns None when the sweep is unusable (no stim, no clear step, suprathreshold).
    """
    if stim_a is None or rate_hz <= 0 or len(trace_v) < int(0.1 * rate_hz):
        return None
    n = min(len(stim_a), len(trace_v))
    if n < int(0.05 * rate_hz):
        return None

    n_pre = max(1, int(0.005 * rate_hz))
    n_tail_start = max(n_pre + 1, n - int(0.020 * rate_hz))
    n_tail_end = n

    i_baseline = float(np.mean(stim_a[:n_pre]))
    i_step = float(np.mean(stim_a[n_tail_start:n_tail_end])) - i_baseline
    v_steady = float(np.median(trace_v[n_tail_start:n_tail_end]))

    if abs(i_step) < 5e-12:    # <5 pA — not a real step
        return None
    if v_steady > -0.040:       # above AP threshold → suprathreshold; not subthreshold IV
        return None
    return (i_step * 1e12, v_steady * 1000.0)


def _rin_mohm_from_iv_pairs(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Linear regression V = Rin * I + offset over (I_pA, V_mV) pairs.

    Returns (Rin_mohm, r2). Requires ≥3 distinct subthreshold steps.
    Rin in MΩ: slope of (V in mV) / (I in pA) * 1000 = mV/pA * 1000 = MΩ.
    (1 V / 1 A = 1 Ω; 1 mV / 1 pA = 1e-3 / 1e-12 = 1e9 Ω = 1 GΩ = 1000 MΩ.)
    """
    if len(pairs) < 3:
        return float("nan"), float("nan")
    i_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    v_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    if len(np.unique(np.round(i_arr, 2))) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(i_arr, v_arr, 1)
    v_pred = slope * i_arr + intercept
    ss_res = float(np.sum((v_arr - v_pred) ** 2))
    ss_tot = float(np.sum((v_arr - np.mean(v_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope * 1000.0), float(r2)


def _test_pulse_edge_overshoot_mv(trace_v: np.ndarray, rate_hz: float,
                                    stim_a: np.ndarray | None = None) -> float:
    """Per-sweep test-pulse edge artifact: peak |dV| in the first 5 ms after the
    step minus the steady-state plateau (20-50 ms). Catches the "BAD Rac" pattern
    from the experimenter sketches: sharp transients at step edges that ring.

    Healthy "GOOD" traces show <5 mV; BAD traces in the sketches show 15-25 mV.
    Returns 0 when no step is detected (clean baseline), NaN when trace is too short.
    """
    if rate_hz <= 0 or len(trace_v) < int(0.08 * rate_hz):
        return float("nan")

    n_pre = max(1, int(0.005 * rate_hz))
    n_edge_end = max(n_pre + 1, int(0.010 * rate_hz))   # 5-10 ms post-edge: capture artifact peak
    n_plateau_start = int(0.020 * rate_hz)
    n_plateau_end = int(0.050 * rate_hz)

    if n_plateau_end >= len(trace_v):
        return float("nan")

    # Locate the step edge — use stimulus when available, voltage otherwise
    if stim_a is not None and len(stim_a) >= n_plateau_end:
        i_baseline = float(np.mean(stim_a[:n_pre]))
        i_plateau = float(np.mean(stim_a[n_plateau_start:n_plateau_end]))
        if abs(i_plateau - i_baseline) < 5e-12:
            return 0.0   # no step → no edge artifact possible
    # else: assume the step is at the standard position

    v_plateau = float(np.mean(trace_v[n_plateau_start:n_plateau_end]))
    v_edge = trace_v[n_pre:n_edge_end]
    if len(v_edge) == 0:
        return 0.0
    v_edge_extreme = float(np.max(v_edge) if abs(np.max(v_edge) - v_plateau) > abs(np.min(v_edge) - v_plateau)
                            else np.min(v_edge))
    return abs(v_edge_extreme - v_plateau) * 1000.0   # V → mV


def _provenance_first_last(values_and_names: list[tuple[float, str]]) -> dict[str, Any]:
    """For a metric reduced by first/last/median, record which sweeps contributed."""
    clean = [(v, n) for v, n in values_and_names if not math.isnan(v)]
    if not clean:
        return {}
    return {"first": clean[0][1], "last": clean[-1][1], "n": len(clean)}


def _halve_session(values_and_names: list[tuple[float, str]]) -> tuple[list[float], list[float]]:
    """Split a chronologically-ordered list into first-half / second-half values
    (NaNs stripped). Returns (first_half, second_half) — empty when there's not
    enough material for two halves."""
    clean = [v for v, _ in values_and_names if not math.isnan(v)]
    if len(clean) < 4:
        return [], []
    mid = len(clean) // 2
    return clean[:mid], clean[mid:]


def compute_metrics(nwb_path: str | Path, family_map: StimulusFamilyMap) -> dict[str, Any]:
    """Open one NWB, compute per-cell QC metrics, return a flat dict.

    Always returns a row even when computations fail — failure modes show up as
    NaN/None values; downstream verdict logic can flag missing data.

    When the NWB carries paired `CurrentClampStimulusSeries` traces for its
    acquisitions, Rs is computed from the actual injected current; Rin is
    computed from the IV-protocol slope; holding current is surfaced. When the
    paired stimulus is absent, Rs falls back to the legacy 50 pA assumption
    (and the per-cell `n_rs_fallback` count is incremented).
    """
    out: dict[str, Any] = {
        # Core (v0.1.x)
        "vrest_mv": float("nan"),
        "vrest_drift_mv": float("nan"),
        "rs_mohm_initial": float("nan"),
        "rs_mohm_final": float("nan"),
        "rs_drift_pct": float("nan"),
        "rin_mohm": float("nan"),
        "rin_r2": float("nan"),
        "ap_amp_overshoot_mv": float("nan"),
        "ap_threshold_drift_mv": float("nan"),
        "baseline_rms_mv": float("nan"),
        "n_sweeps_total": 0,
        "n_sweeps_clipped": 0,
        "n_sweeps_nan": 0,
        "qc_protocol_coverage": False,
        # Visual-defect metrics (v0.2.x)
        "rac_decay_residual_rel": float("nan"),
        "vm_drift_within_sweep_mv_per_s": float("nan"),
        "ap_failure_fraction": float("nan"),
        "ap_amp_cv": float("nan"),
        "late_instability_index": float("nan"),
        # Whole-cell QC additions (v0.3.0)
        "holding_current_pa": float("nan"),
        "holding_current_drift_pa": float("nan"),
        "vrest_session_drift_mv": float("nan"),
        "rs_session_drift_pct": float("nan"),
        "ap_overshoot_session_drift_mv": float("nan"),
        "test_pulse_edge_overshoot_mv": float("nan"),
        "ap_amp_overshoot_min_mv": float("nan"),
        "ap_amp_attenuation_frac": float("nan"),
        # Bookkeeping
        "n_rs_fallback_sweeps": 0,
        "compute_error": None,
    }
    try:
        with open_nwb(nwb_path) as nwbfile:
            # All accumulators carry (value, acq_name) pairs so we can record
            # provenance (which sweeps drove first/last/median) for the report.
            sponhold_vrest: list[tuple[float, str]] = []
            rs_estimates: list[tuple[float, str]] = []
            ap_overshoots: list[tuple[float, str]] = []
            ap_thresholds: list[tuple[float, str]] = []
            baseline_rms_vals: list[float] = []
            holding_currents: list[tuple[float, str]] = []
            iv_pairs: list[tuple[float, float]] = []
            present_families: set[str] = set()
            n_total = 0; n_clip = 0; n_nan = 0
            # Visual-defect accumulators
            decay_residuals: list[float] = []
            vm_drift_slopes: list[float] = []
            failure_fractions: list[float] = []
            ap_amp_cvs: list[float] = []
            late_instabilities: list[float] = []
            edge_overshoots: list[float] = []
            # Per-AP amplitudes for attenuation-fraction reduction
            per_ap_amplitudes: list[float] = []
            # Bookkeeping
            rs_fallback_used = 0

            for name, obj in _iter_current_clamp_acqs(nwbfile):
                n_total += 1
                rate = _rate(obj)
                trace = voltage_si(obj)
                stim_obj = find_paired_stimulus(nwbfile, name, obj)
                stim_a = current_si(stim_obj) if stim_obj is not None else None
                stim_token = _name_to_stim(name)
                family = family_map.family_of(stim_token)
                if family:
                    present_families.add(family)

                if _has_nan(trace):
                    n_nan += 1
                    continue
                if _is_clipped(trace, rate):
                    n_clip += 1

                # Holding current — applies to every sweep that has a paired stim
                ih = _holding_current_pa(stim_a, rate)
                if not math.isnan(ih):
                    holding_currents.append((ih, name))

                if family == "spontaneous_hold":
                    sponhold_vrest.append((_median_last_seconds(trace, rate, 0.5), name))
                    baseline_rms_vals.append(_rms(trace) * 1000.0)
                    s = _vm_drift_slope_mv_per_s(trace, rate)
                    if not math.isnan(s):
                        vm_drift_slopes.append(abs(s))
                elif family == "test_pulse":
                    rs_val, used_fallback = _rs_from_test_pulse_mohm(trace, rate, stim_a)
                    if not math.isnan(rs_val):
                        rs_estimates.append((rs_val, name))
                    if used_fallback:
                        rs_fallback_used += 1
                    r = _step_decay_residual_rel(trace, rate)
                    if not math.isnan(r):
                        decay_residuals.append(r)
                    e = _test_pulse_edge_overshoot_mv(trace, rate, stim_a)
                    if not math.isnan(e):
                        edge_overshoots.append(e)
                elif family == "ap_waveform":
                    overshoot = _peak_overshoot_mv(trace)
                    ap_overshoots.append((overshoot, name))
                    per_ap_amplitudes.append(overshoot)
                    th = _ap_threshold_mv(trace, rate)
                    if not math.isnan(th):
                        ap_thresholds.append((th, name))
                    f = _failed_spike_fraction(trace, rate)
                    if not math.isnan(f):
                        failure_fractions.append(f)
                    li = _late_instability_index(trace, rate)
                    if not math.isnan(li):
                        late_instabilities.append(li)
                elif family == "rest_firing":
                    overshoot = _peak_overshoot_mv(trace)
                    per_ap_amplitudes.append(overshoot)
                    th = _ap_threshold_mv(trace, rate)
                    if not math.isnan(th):
                        ap_thresholds.append((th, name))
                    f = _failed_spike_fraction(trace, rate)
                    if not math.isnan(f):
                        failure_fractions.append(f)
                    cv = _ap_amplitude_cv(trace, rate)
                    if not math.isnan(cv):
                        ap_amp_cvs.append(cv)
                    li = _late_instability_index(trace, rate)
                    if not math.isnan(li):
                        late_instabilities.append(li)
                elif family == "iv_subthreshold":
                    pair = _iv_subthreshold_pair(stim_a, trace, rate)
                    if pair is not None:
                        iv_pairs.append(pair)

            out["n_sweeps_total"] = n_total
            out["n_sweeps_clipped"] = n_clip
            out["n_sweeps_nan"] = n_nan
            out["n_rs_fallback_sweeps"] = rs_fallback_used

            # Vrest + session drift
            vrest_vals = [v for v, _ in sponhold_vrest if not math.isnan(v)]
            if vrest_vals:
                out["vrest_mv"] = float(np.median(vrest_vals) * 1000.0)
                if len(vrest_vals) >= 2:
                    out["vrest_drift_mv"] = float((vrest_vals[-1] - vrest_vals[0]) * 1000.0)
                a, b = _halve_session(sponhold_vrest)
                if a and b:
                    out["vrest_session_drift_mv"] = float((np.median(b) - np.median(a)) * 1000.0)
                out["vrest_mv_provenance"] = json.dumps(_provenance_first_last(sponhold_vrest))

            if baseline_rms_vals:
                out["baseline_rms_mv"] = float(np.median(baseline_rms_vals))

            # Rs + session drift
            rs_clean = [(v, n) for v, n in rs_estimates if not math.isnan(v)]
            if rs_clean:
                out["rs_mohm_initial"] = float(rs_clean[0][0])
                out["rs_mohm_final"] = float(rs_clean[-1][0])
                if out["rs_mohm_initial"] > 0:
                    out["rs_drift_pct"] = float(
                        (out["rs_mohm_final"] - out["rs_mohm_initial"]) / out["rs_mohm_initial"] * 100.0
                    )
                a, b = _halve_session(rs_estimates)
                if a and b and np.median(a) > 0:
                    out["rs_session_drift_pct"] = float(
                        (np.median(b) - np.median(a)) / np.median(a) * 100.0
                    )
                out["rs_mohm_provenance"] = json.dumps(_provenance_first_last(rs_estimates))

            # Rin
            if iv_pairs:
                rin, r2 = _rin_mohm_from_iv_pairs(iv_pairs)
                out["rin_mohm"] = rin
                out["rin_r2"] = r2

            # Holding current
            ih_vals = [v for v, _ in holding_currents if not math.isnan(v)]
            if ih_vals:
                out["holding_current_pa"] = float(np.median(ih_vals))
                if len(ih_vals) >= 2:
                    out["holding_current_drift_pa"] = float(ih_vals[-1] - ih_vals[0])

            # AP overshoot + session drift + worst-sweep + attenuation fraction
            overshoot_vals = [v for v, _ in ap_overshoots if not math.isnan(v)]
            if overshoot_vals:
                out["ap_amp_overshoot_mv"] = float(np.median(overshoot_vals))
                out["ap_amp_overshoot_min_mv"] = float(np.min(overshoot_vals))
                a, b = _halve_session(ap_overshoots)
                if a and b:
                    out["ap_overshoot_session_drift_mv"] = float(np.median(b) - np.median(a))
                out["ap_amp_overshoot_mv_provenance"] = json.dumps(_provenance_first_last(ap_overshoots))
            if per_ap_amplitudes:
                attenuated = sum(1 for v in per_ap_amplitudes if not math.isnan(v) and v < 15.0)
                total = sum(1 for v in per_ap_amplitudes if not math.isnan(v))
                if total > 0:
                    out["ap_amp_attenuation_frac"] = float(attenuated / total)

            if len(ap_thresholds) >= 2:
                th_clean = [v for v, _ in ap_thresholds if not math.isnan(v)]
                out["ap_threshold_drift_mv"] = float(th_clean[-1] - th_clean[0])

            # Visual-defect reductions
            if decay_residuals:
                out["rac_decay_residual_rel"] = float(np.median(decay_residuals))
            if vm_drift_slopes:
                out["vm_drift_within_sweep_mv_per_s"] = float(np.max(vm_drift_slopes))
            if failure_fractions:
                out["ap_failure_fraction"] = float(np.median(failure_fractions))
            if ap_amp_cvs:
                out["ap_amp_cv"] = float(np.median(ap_amp_cvs))
            if late_instabilities:
                out["late_instability_index"] = float(np.max(late_instabilities))
            if edge_overshoots:
                out["test_pulse_edge_overshoot_mv"] = float(np.max(edge_overshoots))

            # Coverage
            essential = {"spontaneous_hold", "test_pulse", "ap_waveform"}
            out["qc_protocol_coverage"] = essential.issubset(present_families)
    except Exception as e:  # noqa: BLE001
        out["compute_error"] = f"{type(e).__name__}: {e}"
    return out
