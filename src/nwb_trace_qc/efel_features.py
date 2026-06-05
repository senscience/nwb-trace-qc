"""Thin wrapper over the eFEL library (the BBP/LNMC-canonical feature extractor).

Used by `metrics.py` to source the canonical AP/Vrest features eFEL knows how
to compute. Our custom helpers stay as fallbacks for malformed sweeps and for
the metrics eFEL doesn't cover (Rs from current-clamp test pulse, Rin from
multi-sweep IV fit, holding current, our visual-defect metrics).

API:
    efel_features_for_sweep(voltage_v, current_a, rate_hz, *, features) ->
        {feature_name: scalar | list[float] | None}

`features` is a list of eFEL feature names. The wrapper builds the
{'T': times, 'V': voltage_mV, 'stim_start': […], 'stim_end': […]} dict that
eFEL expects, calls `efel.get_feature_values([trace], features)`, and returns
the flat per-feature result. None when eFEL couldn't compute it on this sweep.

We pass voltage in mV (eFEL's default unit) and times in ms.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Minimum pre-stim window (seconds) we promise eFEL — voltage_base computes
# its baseline over the LAST 10% of the pre-stim window, so we need enough
# room before stim_start for the feature to have data to average.
_MIN_PRE_STIM_S = 0.020   # 20 ms

# Minimum stim window (seconds) — guards against degenerate (start ≈ end) cases.
_MIN_STIM_LEN_S = 0.010   # 10 ms


# Common feature names sourced from eFEL — kept here so call sites can refer
# to symbolic names instead of stringly-typed magic.
EFEL_VOLTAGE_BASE = "voltage_base"
EFEL_AP_AMPLITUDE_FROM_VBASE = "AP_amplitude_from_voltagebase"
EFEL_AP_AMPLITUDE = "AP_amplitude"   # peak − threshold (canonical AP amplitude)
EFEL_PEAK_VOLTAGE = "peak_voltage"
EFEL_AP_BEGIN_VOLTAGE = "AP_begin_voltage"
EFEL_SPIKECOUNT = "Spikecount"
EFEL_MEAN_FREQUENCY = "mean_frequency"


def _build_trace_dict(voltage_v: np.ndarray, rate_hz: float,
                       stim_start_ms: float | None, stim_end_ms: float | None,
                       label: str = "sweep") -> dict[str, Any]:
    """Build the input dict eFEL's get_feature_values expects.

    Enforces a minimum pre-stim window: voltage_base computes its baseline
    over the last 10% of (0, stim_start), so stim_start must be far enough
    in for that 10% to span at least a handful of samples. We clamp to 20 ms.
    """
    n = len(voltage_v)
    times_ms = (np.arange(n, dtype=np.float64) / rate_hz) * 1000.0
    total_ms = float(times_ms[-1]) if n > 0 else 0.0
    voltage_mv = voltage_v.astype(np.float64) * 1000.0

    min_pre_ms = _MIN_PRE_STIM_S * 1000.0
    min_stim_len_ms = _MIN_STIM_LEN_S * 1000.0

    # Sensible defaults when no paired stim told us where the stimulus is:
    # treat the first quarter of the trace as pre-stim baseline (gives eFEL
    # voltage_base ~10% of trace length to compute over) and the rest as stim.
    if stim_start_ms is None:
        stim_start_ms = max(min_pre_ms, total_ms * 0.25)
    if stim_end_ms is None:
        stim_end_ms = max(stim_start_ms + min_stim_len_ms, total_ms * 0.75)

    # Clamp: stim_start needs enough lead-in for voltage_base, and
    # stim_end must follow stim_start by at least the minimum stim length.
    stim_start_ms = max(stim_start_ms, min_pre_ms)
    if stim_start_ms > total_ms - min_stim_len_ms:
        stim_start_ms = max(min_pre_ms, total_ms - min_stim_len_ms)
    stim_end_ms = max(stim_end_ms, stim_start_ms + min_stim_len_ms)
    stim_end_ms = min(stim_end_ms, total_ms)

    # eFEL trace dicts use 4 numeric keys: T (ms), V (mV), stim_start, stim_end.
    # Don't include 'label' or other string keys — eFEL >=5 iterates every key
    # and tries to convert its values to floats, which fails on strings.
    return {
        "T": times_ms,
        "V": voltage_mv,
        "stim_start": [stim_start_ms],
        "stim_end": [stim_end_ms],
    }


def _step_window_ms(current_a: np.ndarray | None, rate_hz: float) -> tuple[float | None, float | None]:
    """Locate the stimulus step edges from the paired current trace (when present).

    Returns (start_ms, end_ms) of the step, or (None, None) when no clear step
    is detected (or no current trace was provided). The window is used by eFEL
    for stim-relative features.
    """
    if current_a is None or len(current_a) == 0 or rate_hz <= 0:
        return None, None
    i = np.asarray(current_a, dtype=np.float64)
    baseline = float(np.median(i[: max(1, int(0.005 * rate_hz))]))
    delta = i - baseline
    threshold = 0.25 * float(np.max(np.abs(delta))) if np.max(np.abs(delta)) > 0 else 0.0
    if threshold == 0.0:
        return None, None
    above = np.abs(delta) > threshold
    if not above.any():
        return None, None
    idx = np.where(above)[0]
    start_idx = int(idx[0])
    end_idx = int(idx[-1])
    start_ms = (start_idx / rate_hz) * 1000.0
    end_ms = (end_idx / rate_hz) * 1000.0
    # Don't return a stim_start that leaves no pre-stim window — _build_trace_dict
    # will clamp it anyway, but signal the caller more honestly here.
    if start_ms < _MIN_PRE_STIM_S * 1000.0:
        return None, None
    return start_ms, end_ms


def efel_features_for_sweep(
    voltage_v: np.ndarray,
    current_a: np.ndarray | None,
    rate_hz: float,
    *,
    features: list[str],
    label: str = "sweep",
) -> dict[str, Any] | None:
    """Compute the requested eFEL features for one sweep.

    Returns a dict {feature_name: list[float] | None} (eFEL returns lists of
    per-spike values for spike-based features and one-element lists for scalar
    features). Returns None when the eFEL call raises (caller falls back to
    custom helpers).
    """
    try:
        import efel
    except ImportError:
        log.debug("efel not importable; falling back to custom helpers")
        return None

    if len(voltage_v) == 0 or rate_hz <= 0:
        return None

    stim_start_ms, stim_end_ms = _step_window_ms(current_a, rate_hz)
    trace = _build_trace_dict(voltage_v, rate_hz, stim_start_ms, stim_end_ms, label=label)

    try:
        efel.reset()
        # eFEL 5+ exposes get_feature_values; older versions only getFeatureValues
        getter = getattr(efel, "get_feature_values", None) or efel.getFeatureValues
        # eFEL emits a RuntimeWarning per feature it can't compute (e.g. when
        # there's no spike in a quiet sweep). We treat None results as silent
        # fallbacks — the warning chatter would just pollute the progress line.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*efel.*")
            results = getter([trace], features)
    except Exception as e:  # noqa: BLE001
        log.debug("efel call failed for %s: %s", label, e)
        return None
    if not results:
        return None
    out = results[0]
    # Normalise: convert numpy arrays to lists, leave None as-is for missing
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in out.items()}


def feature_scalar(values: Any, reducer=np.median) -> float:
    """Reduce a per-spike eFEL feature list to a single scalar (NaN on empty)."""
    if values is None or (hasattr(values, "__len__") and len(values) == 0):
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(reducer(arr))
