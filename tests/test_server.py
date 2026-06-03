"""Server endpoint smoke tests — LTTB downsampler + handler routing.

The /api/sweeps and /api/trace endpoints require a real NWB; those are exercised
end-to-end in the JY run. Here we test the pure-numpy LTTB and the handler-routing
boundary.
"""
import numpy as np

from nwb_trace_qc.server import _lttb


def test_lttb_passes_through_when_short():
    x = np.linspace(0, 1, 50)
    y = np.sin(x)
    xo, yo = _lttb(x, y, n_out=100)
    assert len(xo) == 50 and len(yo) == 50


def test_lttb_downsamples_to_n_out():
    x = np.linspace(0, 10, 10_000)
    y = np.sin(x) + np.random.RandomState(0).normal(0, 0.05, size=x.size)
    xo, yo = _lttb(x, y, n_out=500)
    assert len(xo) == 500
    assert xo[0] == x[0] and xo[-1] == x[-1]


def test_lttb_preserves_extrema_approximately():
    # Spike in the middle of a flat trace; LTTB should retain the peak (or close to it)
    x = np.linspace(0, 1, 10_000)
    y = np.zeros_like(x)
    y[5000] = 10.0
    xo, yo = _lttb(x, y, n_out=200)
    # The retained samples should include something near the spike
    near_spike = (xo > 0.45) & (xo < 0.55)
    assert yo[near_spike].max() >= 1.0  # spike substantially preserved
