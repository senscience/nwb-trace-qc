"""Unit tests for the v0.2.0 visual-defect metrics.

Each metric gets a clean-case (near-zero / NaN) and a degraded-case (above threshold).
Built from in-memory numpy arrays — no NWB I/O needed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nwb_trace_qc.metrics import (
    _ap_amplitude_cv,
    _failed_spike_fraction,
    _late_instability_index,
    _step_decay_residual_rel,
    _vm_drift_slope_mv_per_s,
)

RATE = 10_000.0  # 10 kHz, typical


# ─── 1. Step-decay residual ─────────────────────────────────

def test_decay_residual_clean_exponential():
    t = np.arange(0, 1.0, 1 / RATE)
    pre = np.full(int(0.05 * RATE), -0.070)
    decay = -0.070 + 0.020 * np.exp(-t / 0.05)  # clean exponential, tau=50ms
    trace = np.concatenate([pre, decay]).astype(np.float64)
    r = _step_decay_residual_rel(trace, RATE)
    assert r < 0.05, f"clean exponential should fit tightly, got {r}"


def test_decay_residual_glitchy():
    t = np.arange(0, 1.0, 1 / RATE)
    pre = np.full(int(0.05 * RATE), -0.070)
    decay = -0.070 + 0.020 * np.exp(-t / 0.05)
    # Add high-frequency noise + a transient spike in the middle
    noise = np.random.RandomState(0).normal(0, 0.005, size=len(decay))
    decay = decay + noise
    decay[len(decay) // 3 : len(decay) // 3 + 20] += 0.015  # injected glitch
    trace = np.concatenate([pre, decay]).astype(np.float64)
    r = _step_decay_residual_rel(trace, RATE)
    assert r > 0.10, f"glitchy decay should have large residual, got {r}"


def test_decay_residual_too_short():
    trace = np.full(int(0.1 * RATE), -0.070)
    r = _step_decay_residual_rel(trace, RATE)
    assert math.isnan(r)


# ─── 2. Within-sweep Vm drift ────────────────────────────────

def test_vm_drift_stable_baseline():
    trace = np.full(int(5 * RATE), -0.070, dtype=np.float64)
    # Add small noise
    trace = trace + np.random.RandomState(0).normal(0, 0.0005, size=trace.size)
    slope = _vm_drift_slope_mv_per_s(trace, RATE)
    assert abs(slope) < 0.5, f"stable baseline should have ~0 slope, got {slope}"


def test_vm_drift_obvious_drift():
    # -70 mV → -20 mV over 5 seconds = 10 mV/s
    n = int(5 * RATE)
    trace = np.linspace(-0.070, -0.020, n)
    slope = _vm_drift_slope_mv_per_s(trace, RATE)
    assert 9.0 < slope < 11.0, f"expected ~10 mV/s, got {slope}"


def test_vm_drift_too_short_returns_nan():
    trace = np.full(int(1.0 * RATE), -0.070, dtype=np.float64)
    assert math.isnan(_vm_drift_slope_mv_per_s(trace, RATE))


# ─── 3. Failed-spike fraction ────────────────────────────────

def _spike(t_offset_s, peak_v=0.030, width_s=0.001, rate=RATE):
    # Triangular spike from -70 mV to peak_v and back, over 2*width
    pre = int(t_offset_s * rate)
    half = int(width_s * rate)
    return pre, half, peak_v


def test_failure_fraction_all_complete():
    trace = np.full(int(0.5 * RATE), -0.070, dtype=np.float64)
    # 5 healthy APs
    for i in range(5):
        s = int((0.05 + i * 0.07) * RATE)
        half = int(0.0015 * RATE)
        for j in range(half):
            trace[s + j] = -0.070 + (0.100) * j / half        # up to +30 mV
        for j in range(half):
            trace[s + half + j] = 0.030 - 0.100 * j / half
    f = _failed_spike_fraction(trace, RATE)
    assert f is not None and f < 0.05, f"all healthy → near-zero, got {f}"


def test_failure_fraction_mixed():
    trace = np.full(int(0.5 * RATE), -0.070, dtype=np.float64)
    half = int(0.0015 * RATE)
    # 5 spikes: 3 healthy (peak +30 mV), 2 failed (peak -20 mV, never crosses 0)
    for i, peak in enumerate([0.030, 0.030, -0.020, 0.030, -0.020]):
        s = int((0.05 + i * 0.07) * RATE)
        amp = peak - (-0.070)
        for j in range(half):
            trace[s + j] = -0.070 + amp * j / half
        for j in range(half):
            trace[s + half + j] = peak - amp * j / half
    f = _failed_spike_fraction(trace, RATE)
    assert 0.30 <= f <= 0.50, f"2 of 5 failed → ~0.40, got {f}"


def test_failure_fraction_no_initiations_is_nan():
    trace = np.full(int(0.5 * RATE), -0.070, dtype=np.float64)
    assert math.isnan(_failed_spike_fraction(trace, RATE))


# ─── 4. AP amplitude CV ───────────────────────────────────────

def test_ap_amp_cv_consistent_spikes():
    trace = np.full(int(0.5 * RATE), -0.070, dtype=np.float64)
    half = int(0.0015 * RATE)
    # 6 APs at +30 mV consistently
    for i in range(6):
        s = int((0.05 + i * 0.06) * RATE)
        amp = 0.030 - (-0.070)
        for j in range(half):
            trace[s + j] = -0.070 + amp * j / half
        for j in range(half):
            trace[s + half + j] = 0.030 - amp * j / half
    cv = _ap_amplitude_cv(trace, RATE)
    assert cv < 0.05, f"consistent → CV ≈ 0, got {cv}"


def test_ap_amp_cv_jittered_amplitudes():
    trace = np.full(int(0.5 * RATE), -0.070, dtype=np.float64)
    half = int(0.0015 * RATE)
    peaks = [0.030, 0.005, 0.025, 0.008, 0.028, 0.006]  # alternating tall/short
    for i, peak in enumerate(peaks):
        s = int((0.05 + i * 0.06) * RATE)
        amp = peak - (-0.070)
        for j in range(half):
            trace[s + j] = -0.070 + amp * j / half
        for j in range(half):
            trace[s + half + j] = peak - amp * j / half
    cv = _ap_amplitude_cv(trace, RATE)
    assert cv > 0.30, f"jittered → CV large, got {cv}"


# ─── 5. Late-recording instability ────────────────────────────

def test_late_instability_stable_recording():
    n = int(8 * RATE)
    trace = np.full(n, -0.070, dtype=np.float64)
    # Add uniform low-rate spikes throughout (positions chosen to fit in the trace)
    half = int(0.0015 * RATE)
    for i in range(8):
        s = int((0.5 + i * 0.9) * RATE)
        amp = 0.030 - (-0.070)
        for j in range(half):
            trace[s + j] = -0.070 + amp * j / half
        for j in range(half):
            trace[s + half + j] = 0.030 - amp * j / half
    idx = _late_instability_index(trace, RATE)
    # Roughly equal activity early vs late: index should be modest
    assert idx < 1.0, f"stable → index small, got {idx}"


def test_late_instability_runaway():
    n = int(8 * RATE)
    trace = np.full(n, -0.070, dtype=np.float64)
    # Early quarter: a few quiet spikes
    half = int(0.0015 * RATE)
    for i, t in enumerate([0.5, 1.0]):
        s = int(t * RATE); amp = 0.030 - (-0.070)
        for j in range(half):
            trace[s + j] = -0.070 + amp * j / half
        for j in range(half):
            trace[s + half + j] = 0.030 - amp * j / half
    # Late quarter: many fast spikes + high-amplitude oscillation
    osc = 0.030 * np.sin(np.linspace(0, 50 * np.pi, int(2 * RATE)))
    trace[int(6 * RATE) : int(6 * RATE) + len(osc)] += osc
    idx = _late_instability_index(trace, RATE)
    assert idx > 3.0, f"runaway → index > 3, got {idx}"
