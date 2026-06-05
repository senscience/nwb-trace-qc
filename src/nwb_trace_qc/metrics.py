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

from .efel_features import (
    EFEL_AP_AMPLITUDE,
    EFEL_AP_AMPLITUDE_FROM_VBASE,
    EFEL_AP_BEGIN_VOLTAGE,
    EFEL_SPIKECOUNT,
    EFEL_VOLTAGE_BASE,
    efel_features_for_sweep,
    feature_scalar,
)
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


def _read_rs_compensation_pct(nwbfile) -> float:
    """Read the Rs compensation percentage from the NWB's icephys metadata.

    pynwb's IntracellularElectrode object exposes `resistance_comp_correction`
    (typically a float 0..1 representing the fraction of Rs compensated, or a
    percentage 0..100 depending on the lab convention). Returns the value as a
    percentage (multiplying by 100 if it looks like a fraction), or NaN when
    no electrode carries the field.

    Some labs put the value in `lab_meta_data` instead — we try that as a
    fallback by looking for any field whose lowercased name contains 'rs' or
    'resistance' and that resolves to a number.
    """
    try:
        electrodes = list(getattr(nwbfile, "icephys_electrodes", {}).values())
    except Exception:
        electrodes = []
    for elec in electrodes:
        val = getattr(elec, "resistance_comp_correction", None)
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        # pynwb sometimes stores this as a 0..1 fraction; normalise to percent
        return v * 100.0 if 0 <= v <= 1 else v
    # Fallback: lab_meta_data scan
    try:
        lm = getattr(nwbfile, "lab_meta_data", {}) or {}
        for container in lm.values():
            for attr in dir(container):
                if attr.startswith("_"):
                    continue
                name_lower = attr.lower()
                if "rs" in name_lower or "resistance" in name_lower or "compensation" in name_lower:
                    try:
                        v = float(getattr(container, attr))
                        return v * 100.0 if 0 <= v <= 1 else v
                    except (TypeError, ValueError, AttributeError):
                        continue
    except Exception:
        pass
    return float("nan")


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


# ── eFEL-first wrappers ───────────────────────────────────────────
# Each wrapper tries eFEL first; on None (eFEL absent, call raised, or no
# usable feature value) falls back to our custom helper. The fallback count
# bubbles up to the cell row as `n_efel_fallback`.

def _efel_or_fallback_vrest_mv(voltage_v: np.ndarray, current_a: np.ndarray | None,
                                  rate_hz: float, fallback_fn) -> tuple[float, bool]:
    """Returns (value_in_volts, used_fallback). value_in_volts so the call site
    can keep its existing `* 1000` scale-at-emit-time pattern."""
    feats = efel_features_for_sweep(voltage_v, current_a, rate_hz,
                                       features=[EFEL_VOLTAGE_BASE])
    if feats is not None:
        v = feature_scalar(feats.get(EFEL_VOLTAGE_BASE))
        if not math.isnan(v):
            return float(v) / 1000.0, False    # eFEL returns mV → back to V
    return float(fallback_fn()), True


def _efel_or_fallback_peak_overshoot_mv(voltage_v: np.ndarray, current_a: np.ndarray | None,
                                          rate_hz: float, fallback_fn) -> tuple[float, bool]:
    feats = efel_features_for_sweep(voltage_v, current_a, rate_hz,
                                       features=[EFEL_AP_AMPLITUDE_FROM_VBASE, EFEL_VOLTAGE_BASE])
    if feats is not None:
        amps = feats.get(EFEL_AP_AMPLITUDE_FROM_VBASE)
        vbase = feature_scalar(feats.get(EFEL_VOLTAGE_BASE))
        if amps and not math.isnan(vbase):
            # Peak overshoot in mV = vbase + median(AP_amplitude_from_voltagebase) (both eFEL-mV)
            peak_mv = vbase + feature_scalar(amps)
            if not math.isnan(peak_mv):
                return float(peak_mv), False
    return float(fallback_fn()), True


def _efel_or_fallback_ap_amplitude_mv(voltage_v: np.ndarray, current_a: np.ndarray | None,
                                         rate_hz: float) -> float:
    """Median AP amplitude (peak − threshold) across spikes in this sweep, in mV.

    This is the canonical AP amplitude as defined by LNMC experimenter guidance
    (see docs/metrics_reference.md): the difference between the AP peak voltage
    and the threshold voltage at which the sodium current activated. Distinct
    from `ap_amp_overshoot_mv` which measures peak above 0 mV.

    Returns NaN when eFEL refuses or no spikes are detected.
    """
    feats = efel_features_for_sweep(voltage_v, current_a, rate_hz,
                                      features=[EFEL_AP_AMPLITUDE])
    if feats is None:
        return float("nan")
    amps = feats.get(EFEL_AP_AMPLITUDE)
    return feature_scalar(amps)


def _efel_or_fallback_ap_threshold_mv(voltage_v: np.ndarray, current_a: np.ndarray | None,
                                        rate_hz: float, fallback_fn) -> tuple[float, bool]:
    feats = efel_features_for_sweep(voltage_v, current_a, rate_hz,
                                       features=[EFEL_AP_BEGIN_VOLTAGE])
    if feats is not None:
        v = feature_scalar(feats.get(EFEL_AP_BEGIN_VOLTAGE))
        if not math.isnan(v):
            return float(v), False
    return float(fallback_fn()), True


def _efel_or_fallback_ap_amplitude_cv(voltage_v: np.ndarray, current_a: np.ndarray | None,
                                         rate_hz: float, fallback_fn) -> tuple[float, bool]:
    feats = efel_features_for_sweep(voltage_v, current_a, rate_hz,
                                       features=[EFEL_AP_AMPLITUDE])
    if feats is not None:
        amps = feats.get(EFEL_AP_AMPLITUDE)
        if amps and len(amps) >= 5:
            arr = np.asarray(amps, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 5 and arr.mean() != 0:
                return float(np.std(arr) / abs(arr.mean())), False
    return float(fallback_fn()), True


def _detect_bad_ending(
    vrest_seq: list[tuple[float, int]],
    rs_seq: list[tuple[float, int]],
    overshoot_seq: list[tuple[float, int]],
    n_total_sweeps: int,
    *,
    vrest_jump_mv: float = 0.010,        # 10 mV depolarisation past running median
    rs_explosion_factor: float = 1.75,    # Rs grows >1.75x its running median
    overshoot_floor_mv: float = 10.0,     # AP overshoot drops below this after a healthy run
    overshoot_healthy_mv: float = 20.0,
) -> tuple[int | None, str | None]:
    """Detect the global sweep index where the recording started to degrade.

    Each input list is the per-sweep stream of (value, global_sweep_index)
    pairs for one of three quality channels: Vrest (V), Rs (MOhm),
    AP overshoot (mV in mV, ie the trace-units returned by the AP picker).

    Returns (cutoff_index, reason). cutoff_index is the FIRST bad sweep;
    everything ≥ that index is post-degradation.

    Guardrails:
      - cutoffs in the first 30% of the recording are ignored (different
        problem — cell wasn't healthy from the start)
      - cutoffs in the last 5% are ignored (single tail glitch, not a pattern)
    """
    if n_total_sweeps < 6:
        return None, None
    min_idx = max(2, int(0.30 * n_total_sweeps))
    max_idx = max(min_idx + 1, int(0.95 * n_total_sweeps))

    candidates: list[tuple[int, str]] = []

    # --- Vrest channel: first sweep depolarised vs running median ---
    clean = [(v, i) for v, i in vrest_seq if not math.isnan(v)]
    for k in range(2, len(clean)):
        prev_vals = [v for v, _ in clean[:k]]
        if not prev_vals:
            continue
        median_prev = float(np.median(prev_vals))
        v_here, i_here = clean[k]
        if v_here > median_prev + vrest_jump_mv:
            candidates.append((i_here, "vrest_depolarisation"))
            break

    # --- Rs channel: first sweep that explodes vs running median ---
    clean = [(v, i) for v, i in rs_seq if not math.isnan(v)]
    for k in range(2, len(clean)):
        prev_vals = [v for v, _ in clean[:k] if v > 0]
        if not prev_vals:
            continue
        median_prev = float(np.median(prev_vals))
        v_here, i_here = clean[k]
        if median_prev > 0 and v_here > rs_explosion_factor * median_prev:
            candidates.append((i_here, "rs_explosion"))
            break

    # --- AP overshoot channel: first sweep below floor after healthy run ---
    clean = [(v, i) for v, i in overshoot_seq if not math.isnan(v)]
    saw_healthy = False
    for v_here, i_here in clean:
        if v_here >= overshoot_healthy_mv:
            saw_healthy = True
        elif saw_healthy and v_here < overshoot_floor_mv:
            candidates.append((i_here, "ap_collapse"))
            break

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])     # earliest cutoff wins
    cutoff_idx, reason = candidates[0]
    if cutoff_idx < min_idx or cutoff_idx > max_idx:
        return None, None
    return cutoff_idx, reason


def _filter_before(stream: list, cutoff_idx: int | None,
                     family_of_interest: bool = True) -> list:
    """Drop entries whose global sweep index >= cutoff_idx. When cutoff_idx is
    None or family_of_interest is False, pass through unchanged.

    `stream` items can be either (value, name) or (value, name, global_idx) —
    handled by tuple length so we don't have to alter every accumulator shape.
    """
    if cutoff_idx is None or not family_of_interest:
        return stream
    out = []
    for item in stream:
        if len(item) >= 3 and item[2] >= cutoff_idx:
            continue
        out.append(item)
    return out


def compute_metrics(nwb_path: str | Path, family_map: StimulusFamilyMap,
                      *, use_efel: bool = True, trim_bad_ending: bool = True) -> dict[str, Any]:
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
        # Bad-ending detection (v0.4.0)
        "bad_ending_at_sweep": float("nan"),
        "n_sweeps_trimmed": 0,
        "bad_ending_reason": None,
        # Experimenter-guidance additions (v0.5.0)
        # Vrest split: true Vrest (no holding current) vs held-Vm under Ihld
        "held_vm_mv": float("nan"),
        # AP amplitude defined as peak − threshold (eFEL's AP_amplitude),
        # distinct from ap_amp_overshoot_mv (peak − 0 mV)
        "ap_amplitude_mv": float("nan"),
        # Rs compensation value read from NWB icephys metadata
        "rs_compensation_pct": float("nan"),
        # CV of per-Rac Rs estimates × 100 — catches variability that
        # rs_drift_pct (first vs last) doesn't see
        "rac_variability_pct": float("nan"),
        # Bookkeeping
        "n_rs_fallback_sweeps": 0,
        "n_efel_fallback_sweeps": 0,
        "compute_error": None,
    }
    try:
        with open_nwb(nwb_path) as nwbfile:
            # Read Rs compensation value from the icephys metadata (v0.5.0)
            out["rs_compensation_pct"] = _read_rs_compensation_pct(nwbfile)

            # All accumulators carry (value, acq_name) pairs so we can record
            # provenance (which sweeps drove first/last/median) for the report.
            # v0.5.0: distinguish no-hold vs held spontaneous traces; legacy
            # `spontaneous_hold` (single bucket) routes to held_vm for backward
            # compatibility.
            vrest_no_hold: list[tuple[float, str, int]] = []      # true Vrest source
            held_vm_vals: list[tuple[float, str, int]] = []        # held under Ihld
            sponhold_vrest: list[tuple[float, str, int]] = []      # legacy fallback (mixed)
            rs_estimates: list[tuple[float, str, int]] = []
            ap_overshoots: list[tuple[float, str, int]] = []
            ap_thresholds: list[tuple[float, str, int]] = []
            ap_amplitudes_pkthr: list[tuple[float, str, int]] = []   # v0.5.0: peak − threshold
            baseline_rms_vals: list[tuple[float, int]] = []
            holding_currents: list[tuple[float, str, int]] = []
            iv_pairs: list[tuple[float, float]] = []
            present_families: set[str] = set()
            n_total = 0; n_clip = 0; n_nan = 0
            # Visual-defect accumulators
            decay_residuals: list[tuple[float, int]] = []
            vm_drift_slopes: list[tuple[float, int]] = []
            failure_fractions: list[tuple[float, int]] = []
            ap_amp_cvs: list[tuple[float, int]] = []
            late_instabilities: list[tuple[float, int]] = []
            edge_overshoots: list[tuple[float, int]] = []
            # Per-AP amplitudes for attenuation-fraction reduction
            per_ap_amplitudes: list[tuple[float, int]] = []
            # Bookkeeping
            rs_fallback_used = 0
            efel_fallback_used = 0
            legacy_hold_used = False

            for sweep_idx, (name, obj) in enumerate(_iter_current_clamp_acqs(nwbfile)):
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
                    holding_currents.append((ih, name, sweep_idx))

                if family in ("spontaneous_hold", "spontaneous_no_hold", "spontaneous_held"):
                    # eFEL voltage_base on the steady-state baseline; falls back
                    # to our last-500ms median when eFEL is absent / refuses.
                    if use_efel:
                        vrest_v, used_fb = _efel_or_fallback_vrest_mv(
                            trace, stim_a, rate,
                            fallback_fn=lambda: _median_last_seconds(trace, rate, 0.5))
                    else:
                        vrest_v, used_fb = _median_last_seconds(trace, rate, 0.5), False
                    if used_fb:
                        efel_fallback_used += 1
                    # Route to the right accumulator per the new (v0.5.0) split.
                    # Legacy `spontaneous_hold` (single bucket) feeds the legacy
                    # accumulator which falls into held_vm at reduction time.
                    if family == "spontaneous_no_hold":
                        vrest_no_hold.append((vrest_v, name, sweep_idx))
                    elif family == "spontaneous_held":
                        held_vm_vals.append((vrest_v, name, sweep_idx))
                    else:
                        sponhold_vrest.append((vrest_v, name, sweep_idx))
                        legacy_hold_used = True
                    baseline_rms_vals.append((_rms(trace) * 1000.0, sweep_idx))
                    s = _vm_drift_slope_mv_per_s(trace, rate)
                    if not math.isnan(s):
                        vm_drift_slopes.append((abs(s), sweep_idx))
                elif family == "test_pulse":
                    rs_val, used_fallback = _rs_from_test_pulse_mohm(trace, rate, stim_a)
                    if not math.isnan(rs_val):
                        rs_estimates.append((rs_val, name, sweep_idx))
                    if used_fallback:
                        rs_fallback_used += 1
                    r = _step_decay_residual_rel(trace, rate)
                    if not math.isnan(r):
                        decay_residuals.append((r, sweep_idx))
                    e = _test_pulse_edge_overshoot_mv(trace, rate, stim_a)
                    if not math.isnan(e):
                        edge_overshoots.append((e, sweep_idx))
                elif family == "ap_waveform":
                    if use_efel:
                        overshoot, used_fb = _efel_or_fallback_peak_overshoot_mv(
                            trace, stim_a, rate,
                            fallback_fn=lambda: _peak_overshoot_mv(trace))
                        if used_fb:
                            efel_fallback_used += 1
                    else:
                        overshoot = _peak_overshoot_mv(trace)
                    ap_overshoots.append((overshoot, name, sweep_idx))
                    per_ap_amplitudes.append((overshoot, sweep_idx))
                    if use_efel:
                        th, used_fb = _efel_or_fallback_ap_threshold_mv(
                            trace, stim_a, rate,
                            fallback_fn=lambda: _ap_threshold_mv(trace, rate))
                        if used_fb:
                            efel_fallback_used += 1
                    else:
                        th = _ap_threshold_mv(trace, rate)
                    if not math.isnan(th):
                        ap_thresholds.append((th, name, sweep_idx))
                    # v0.5.0: AP_amplitude = peak − threshold (canonical AP amplitude)
                    if use_efel:
                        amp_pkthr = _efel_or_fallback_ap_amplitude_mv(trace, stim_a, rate)
                        if not math.isnan(amp_pkthr):
                            ap_amplitudes_pkthr.append((amp_pkthr, name, sweep_idx))
                    f = _failed_spike_fraction(trace, rate)
                    if not math.isnan(f):
                        failure_fractions.append((f, sweep_idx))
                    li = _late_instability_index(trace, rate)
                    if not math.isnan(li):
                        late_instabilities.append((li, sweep_idx))
                elif family == "rest_firing":
                    if use_efel:
                        overshoot, used_fb = _efel_or_fallback_peak_overshoot_mv(
                            trace, stim_a, rate,
                            fallback_fn=lambda: _peak_overshoot_mv(trace))
                        if used_fb:
                            efel_fallback_used += 1
                    else:
                        overshoot = _peak_overshoot_mv(trace)
                    per_ap_amplitudes.append((overshoot, sweep_idx))
                    if use_efel:
                        th, used_fb = _efel_or_fallback_ap_threshold_mv(
                            trace, stim_a, rate,
                            fallback_fn=lambda: _ap_threshold_mv(trace, rate))
                        if used_fb:
                            efel_fallback_used += 1
                    else:
                        th = _ap_threshold_mv(trace, rate)
                    if not math.isnan(th):
                        ap_thresholds.append((th, name, sweep_idx))
                    # v0.5.0: AP_amplitude = peak − threshold
                    if use_efel:
                        amp_pkthr = _efel_or_fallback_ap_amplitude_mv(trace, stim_a, rate)
                        if not math.isnan(amp_pkthr):
                            ap_amplitudes_pkthr.append((amp_pkthr, name, sweep_idx))
                    f = _failed_spike_fraction(trace, rate)
                    if not math.isnan(f):
                        failure_fractions.append((f, sweep_idx))
                    if use_efel:
                        cv, used_fb = _efel_or_fallback_ap_amplitude_cv(
                            trace, stim_a, rate,
                            fallback_fn=lambda: _ap_amplitude_cv(trace, rate))
                        if used_fb:
                            efel_fallback_used += 1
                    else:
                        cv = _ap_amplitude_cv(trace, rate)
                    if not math.isnan(cv):
                        ap_amp_cvs.append((cv, sweep_idx))
                    li = _late_instability_index(trace, rate)
                    if not math.isnan(li):
                        late_instabilities.append((li, sweep_idx))
                elif family == "iv_subthreshold":
                    pair = _iv_subthreshold_pair(stim_a, trace, rate)
                    if pair is not None:
                        iv_pairs.append(pair)

            out["n_sweeps_total"] = n_total
            out["n_sweeps_clipped"] = n_clip
            out["n_sweeps_nan"] = n_nan
            out["n_rs_fallback_sweeps"] = rs_fallback_used
            out["n_efel_fallback_sweeps"] = efel_fallback_used

            # ── Bad-ending detection + auto-trim (v0.4.0) ───────────────
            # Pull Vrest signal from whichever family populated it (preference
            # order: no_hold > held > legacy lumped). Same for Rs and overshoot.
            vrest_combined = vrest_no_hold or held_vm_vals or sponhold_vrest
            vrest_seq = [(v, idx) for v, _name, idx in vrest_combined]
            rs_seq    = [(v, idx) for v, _name, idx in rs_estimates]
            ov_seq    = [(v, idx) for v, _name, idx in ap_overshoots]
            cutoff_idx, reason = _detect_bad_ending(vrest_seq, rs_seq, ov_seq, n_total) \
                                  if trim_bad_ending else (None, None)
            if cutoff_idx is not None:
                out["bad_ending_at_sweep"] = int(cutoff_idx)
                out["bad_ending_reason"] = reason
                out["n_sweeps_trimmed"] = int(n_total - cutoff_idx)
                # Filter every per-sweep accumulator to keep only entries strictly
                # before the cutoff. Items here are 3-tuples (..., sweep_idx) or
                # 2-tuples (value, sweep_idx) — _filter_before handles both.
                vrest_no_hold = _filter_before(vrest_no_hold, cutoff_idx)
                held_vm_vals = _filter_before(held_vm_vals, cutoff_idx)
                sponhold_vrest = _filter_before(sponhold_vrest, cutoff_idx)
                rs_estimates = _filter_before(rs_estimates, cutoff_idx)
                ap_overshoots = _filter_before(ap_overshoots, cutoff_idx)
                ap_thresholds = _filter_before(ap_thresholds, cutoff_idx)
                ap_amplitudes_pkthr = _filter_before(ap_amplitudes_pkthr, cutoff_idx)
                holding_currents = _filter_before(holding_currents, cutoff_idx)
                baseline_rms_vals = _filter_before(baseline_rms_vals, cutoff_idx)
                vm_drift_slopes = _filter_before(vm_drift_slopes, cutoff_idx)
                decay_residuals = _filter_before(decay_residuals, cutoff_idx)
                edge_overshoots = _filter_before(edge_overshoots, cutoff_idx)
                failure_fractions = _filter_before(failure_fractions, cutoff_idx)
                ap_amp_cvs = _filter_before(ap_amp_cvs, cutoff_idx)
                late_instabilities = _filter_before(late_instabilities, cutoff_idx)
                per_ap_amplitudes = _filter_before(per_ap_amplitudes, cutoff_idx)

            # Vrest source priority (v0.5.0): spontaneous_no_hold > spontaneous_held
            # > legacy spontaneous_hold. vrest_mv is the canonical TRUE resting
            # potential; held_vm_mv is a separate metric for the held membrane
            # potential under Ihld.
            vrest_source_stream = vrest_no_hold if vrest_no_hold else (
                held_vm_vals if held_vm_vals else sponhold_vrest
            )
            if vrest_source_stream is sponhold_vrest and legacy_hold_used:
                log.info(
                    "vrest_mv sourced from legacy `spontaneous_hold` family "
                    "(no `spontaneous_no_hold` or `spontaneous_held` mapped). "
                    "Update project YAML's stimulus_protocols to use the v0.5.0 split."
                )
            vrest_vals = [item[0] for item in vrest_source_stream if not math.isnan(item[0])]
            if vrest_vals:
                out["vrest_mv"] = float(np.median(vrest_vals) * 1000.0)
                if len(vrest_vals) >= 2:
                    out["vrest_drift_mv"] = float((vrest_vals[-1] - vrest_vals[0]) * 1000.0)
                a, b = _halve_session([(item[0], item[1]) for item in vrest_source_stream])
                if a and b:
                    out["vrest_session_drift_mv"] = float((np.median(b) - np.median(a)) * 1000.0)
                out["vrest_mv_provenance"] = json.dumps(
                    _provenance_first_last([(item[0], item[1]) for item in vrest_source_stream])
                )

            # Held-Vm: only the held family or legacy fallback contributes; never
            # the no-hold family (semantically a different measurement).
            held_stream = held_vm_vals if held_vm_vals else sponhold_vrest
            held_vals = [item[0] for item in held_stream if not math.isnan(item[0])]
            if held_vals:
                out["held_vm_mv"] = float(np.median(held_vals) * 1000.0)

            if baseline_rms_vals:
                out["baseline_rms_mv"] = float(np.median([item[0] for item in baseline_rms_vals]))

            # Rs + session drift
            rs_clean = [item for item in rs_estimates if not math.isnan(item[0])]
            if rs_clean:
                out["rs_mohm_initial"] = float(rs_clean[0][0])
                out["rs_mohm_final"] = float(rs_clean[-1][0])
                if out["rs_mohm_initial"] > 0:
                    out["rs_drift_pct"] = float(
                        (out["rs_mohm_final"] - out["rs_mohm_initial"]) / out["rs_mohm_initial"] * 100.0
                    )
                a, b = _halve_session([(item[0], item[1]) for item in rs_estimates])
                if a and b and np.median(a) > 0:
                    out["rs_session_drift_pct"] = float(
                        (np.median(b) - np.median(a)) / np.median(a) * 100.0
                    )
                out["rs_mohm_provenance"] = json.dumps(
                    _provenance_first_last([(item[0], item[1]) for item in rs_estimates])
                )
                # v0.5.0: variability of Rac across reps — separate from drift.
                # CV (std/median) × 100 catches non-monotonic instability that
                # rs_drift_pct (first vs last) misses.
                rs_clean_vals = np.array([item[0] for item in rs_clean], dtype=np.float64)
                if len(rs_clean_vals) >= 3:
                    med = float(np.median(rs_clean_vals))
                    if med > 0:
                        out["rac_variability_pct"] = float(
                            np.std(rs_clean_vals) / med * 100.0
                        )

            # Rin
            if iv_pairs:
                rin, r2 = _rin_mohm_from_iv_pairs(iv_pairs)
                out["rin_mohm"] = rin
                out["rin_r2"] = r2

            # Holding current
            ih_vals = [item[0] for item in holding_currents if not math.isnan(item[0])]
            if ih_vals:
                out["holding_current_pa"] = float(np.median(ih_vals))
                if len(ih_vals) >= 2:
                    out["holding_current_drift_pa"] = float(ih_vals[-1] - ih_vals[0])

            # AP overshoot + session drift + worst-sweep + attenuation fraction
            overshoot_vals = [item[0] for item in ap_overshoots if not math.isnan(item[0])]
            if overshoot_vals:
                out["ap_amp_overshoot_mv"] = float(np.median(overshoot_vals))
                out["ap_amp_overshoot_min_mv"] = float(np.min(overshoot_vals))
                a, b = _halve_session([(item[0], item[1]) for item in ap_overshoots])
                if a and b:
                    out["ap_overshoot_session_drift_mv"] = float(np.median(b) - np.median(a))
                out["ap_amp_overshoot_mv_provenance"] = json.dumps(
                    _provenance_first_last([(item[0], item[1]) for item in ap_overshoots])
                )
            if per_ap_amplitudes:
                attenuated = sum(1 for v, _idx in per_ap_amplitudes if not math.isnan(v) and v < 15.0)
                total = sum(1 for v, _idx in per_ap_amplitudes if not math.isnan(v))
                if total > 0:
                    out["ap_amp_attenuation_frac"] = float(attenuated / total)

            if len(ap_thresholds) >= 2:
                th_clean = [item[0] for item in ap_thresholds if not math.isnan(item[0])]
                out["ap_threshold_drift_mv"] = float(th_clean[-1] - th_clean[0])

            # v0.5.0: AP_amplitude (peak − threshold) — canonical AP amplitude
            # per LNMC experimenter guidance, distinct from ap_amp_overshoot_mv.
            amp_pkthr_vals = [item[0] for item in ap_amplitudes_pkthr if not math.isnan(item[0])]
            if amp_pkthr_vals:
                out["ap_amplitude_mv"] = float(np.median(amp_pkthr_vals))

            # Visual-defect reductions
            if decay_residuals:
                out["rac_decay_residual_rel"] = float(np.median([item[0] for item in decay_residuals]))
            if vm_drift_slopes:
                out["vm_drift_within_sweep_mv_per_s"] = float(np.max([item[0] for item in vm_drift_slopes]))
            if failure_fractions:
                out["ap_failure_fraction"] = float(np.median([item[0] for item in failure_fractions]))
            if ap_amp_cvs:
                out["ap_amp_cv"] = float(np.median([item[0] for item in ap_amp_cvs]))
            if late_instabilities:
                out["late_instability_index"] = float(np.max([item[0] for item in late_instabilities]))
            if edge_overshoots:
                out["test_pulse_edge_overshoot_mv"] = float(np.max([item[0] for item in edge_overshoots]))

            # Coverage. v0.5.0: spontaneous_no_hold / spontaneous_held / legacy
            # spontaneous_hold all satisfy the "spontaneous" essential.
            spontaneous_present = bool(
                present_families & {"spontaneous_no_hold", "spontaneous_held", "spontaneous_hold"}
            )
            essential = {"test_pulse", "ap_waveform"}
            out["qc_protocol_coverage"] = (
                spontaneous_present and essential.issubset(present_families)
            )
    except Exception as e:  # noqa: BLE001
        out["compute_error"] = f"{type(e).__name__}: {e}"
    return out
