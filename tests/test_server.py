"""Server endpoint smoke tests — LTTB downsampler + handler routing.

The /api/sweeps and /api/trace endpoints require a real NWB; those are exercised
end-to-end in the JY run. Here we test the pure-numpy LTTB and the handler-routing
boundary.
"""
from pathlib import Path

import numpy as np

from nwb_trace_qc.server import _build_flag_cells, _is_nan, _lttb


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


def test_build_flag_cells_filters_to_flag_only_and_strips_nans(tmp_path: Path):
    csv = tmp_path / "qc_report.csv"
    csv.write_text(
        "cell_id,dataset,final_verdict,vrest_mv,rs_drift_pct,ap_amp_overshoot_mv\n"
        "c1,ds,pass,-65.0,2.0,80.0\n"
        "c2,ds,flag,-60.0,,75.0\n"          # rs_drift_pct missing → stripped per-cell
        "c3,ds,flag,-62.0,5.0,\n"           # ap_amp_overshoot_mv missing → stripped
        "c4,ds,fail,-50.0,12.0,40.0\n"
    )
    flag, total = _build_flag_cells(csv)
    assert total == 4
    assert {c["cell_id"] for c in flag} == {"c2", "c3"}
    c2 = next(c for c in flag if c["cell_id"] == "c2")
    c3 = next(c for c in flag if c["cell_id"] == "c3")
    assert "rs_drift_pct" not in c2          # NaN stripped
    assert "ap_amp_overshoot_mv" in c2       # present value retained
    assert "ap_amp_overshoot_mv" not in c3
    assert "rs_drift_pct" in c3


def test_is_nan_handles_python_floats_and_pandas_na():
    import pandas as pd
    assert _is_nan(None) is True
    assert _is_nan(float("nan")) is True
    assert _is_nan(pd.NA) is True
    assert _is_nan(0.0) is False
    assert _is_nan("") is False  # empty string is a value, not NaN
    assert _is_nan(-65.3) is False
