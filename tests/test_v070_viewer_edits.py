"""v0.7.0 — new metric, ephys_qc_score, threshold overrides, trim overrides,
recompute path, save/upsert helpers used by the viewer."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import pytest
import yaml

from nwb_trace_qc import server as srv_mod
from nwb_trace_qc.config import load_config
from nwb_trace_qc.metrics import (
    _count_successful_spikes,
    _detect_spike_initiations,
    _failed_spike_fraction,
    compute_metrics,
)
from nwb_trace_qc.overrides import (
    init_trim_overrides_file,
    load_trim_overrides,
    upsert_trim_override,
)
from nwb_trace_qc.pipeline import run as pipeline_run
from nwb_trace_qc.server import _ephys_qc_score
from nwb_trace_qc.stimuli import StimulusFamilyMap
from nwb_trace_qc.thresholds import (
    load_thresholds_with_overrides,
    save_threshold_overrides,
)


# ─── n_spikes_total ───────────────────────────────────────────────

def test_detect_spike_initiations_separates_success_and_failure():
    """Synthetic two-spike train: one reaches +20 mV, one tops out at -20 mV.
    Detector should report 2 initiations, 1 successful."""
    rate = 20_000.0
    n = 4_000
    t = np.arange(n) / rate
    v = np.full(n, -0.07, dtype=np.float64)  # baseline -70 mV
    # Successful AP at sample 1000: 1 ms rise to +20 mV
    rise = int(0.001 * rate)
    for i in range(rise):
        v[1000 + i] = -0.07 + (0.09) * (i / rise)
    v[1000 + rise: 1000 + rise + 5] = 0.02   # peak +20 mV
    # Failed AP at sample 2500: same rise rate but caps at -20 mV
    for i in range(rise):
        v[2500 + i] = -0.07 + (0.05) * (i / rise)
    v[2500 + rise: 2500 + rise + 5] = -0.02

    n_init, n_ok = _detect_spike_initiations(v, rate)
    assert n_init == 2
    assert n_ok == 1
    # _count_successful_spikes returns just the success count
    assert _count_successful_spikes(v, rate) == 1
    # _failed_spike_fraction stays semantically the same after the refactor.
    assert _failed_spike_fraction(v, rate) == pytest.approx(0.5)


def _make_nwb_with_ap_sweep(path: Path, name: str = "ic__APWaveform__001") -> None:
    """Tiny NWB with one AP-train sweep producing ≥1 successful spike."""
    rate = 20_000.0
    n = 4_000
    v = np.full(n, -0.07, dtype=np.float64)
    rise = int(0.001 * rate)
    # Three clean APs at evenly-spaced offsets
    for offset in (800, 2000, 3200):
        for i in range(rise):
            v[offset + i] = -0.07 + 0.09 * (i / rise)
        v[offset + rise: offset + rise + 5] = 0.02

    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
        name=name, data=v, electrode=elec, gain=1.0,
        starting_time=0.0, rate=rate, unit="volts",
    ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_n_spikes_total_aggregates_across_ap_sweeps(tmp_path: Path):
    nwb = tmp_path / "ap.nwb"
    _make_nwb_with_ap_sweep(nwb)
    fm = StimulusFamilyMap({"ap_waveform": ["APWaveform"]})
    out = compute_metrics(nwb, fm, use_efel=False, trim_bad_ending=False)
    assert "n_spikes_total" in out
    assert isinstance(out["n_spikes_total"], int)
    assert out["n_spikes_total"] >= 1


def test_force_trim_at_overrides_auto_detection(tmp_path: Path):
    """force_trim_at=N bypasses bad-ending detection and trims at exactly N."""
    nwb = tmp_path / "ap.nwb"
    _make_nwb_with_ap_sweep(nwb, name="ic__APWaveform__001")
    fm = StimulusFamilyMap({"ap_waveform": ["APWaveform"]})
    out = compute_metrics(nwb, fm, use_efel=False, force_trim_at=1)
    # We had 1 sweep — force_trim_at=1 with n_total=1 is "no-op" (boundary).
    # Re-test with two sweeps:
    nwb2 = tmp_path / "ap2.nwb"
    # Build a fresh NWB with TWO AP sweeps so force_trim_at=1 actually trims.
    rate = 20_000.0
    n = 4_000
    v = np.full(n, -0.07, dtype=np.float64)
    rise = int(0.001 * rate)
    for offset in (800, 2000):
        for i in range(rise):
            v[offset + i] = -0.07 + 0.09 * (i / rise)
        v[offset + rise: offset + rise + 5] = 0.02
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier="x",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    for i in range(2):
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=f"ic__APWaveform__{i:03d}", data=v, electrode=elec, gain=1.0,
            starting_time=0.0, rate=rate, unit="volts",
        ))
    with pynwb.NWBHDF5IO(str(nwb2), mode="w") as io:
        io.write(nwbfile)
    forced = compute_metrics(nwb2, fm, use_efel=False, force_trim_at=1)
    assert forced["bad_ending_at_sweep"] == 1
    assert forced["n_sweeps_trimmed"] == 1
    assert forced["bad_ending_reason"] == "manual_override"


# ─── ephys_qc_score ───────────────────────────────────────────────

def test_ephys_qc_score_clean_recording_is_1():
    """No triggered_metrics ⇒ perfect score."""
    assert _ephys_qc_score({"triggered_metrics": []}) == pytest.approx(1.0)
    assert _ephys_qc_score({}) == pytest.approx(1.0)


def test_ephys_qc_score_one_critical_fail_drops_by_1_over_N():
    """One critical fail drops the score by 1/N_critical."""
    from nwb_trace_qc.families import DEFAULT_CRITICAL_METRICS
    n = len(DEFAULT_CRITICAL_METRICS)
    s = _ephys_qc_score({
        "triggered_metrics": [
            {"metric": "vrest_mv", "verdict": "fail", "critical": True},
        ],
    })
    assert s == pytest.approx(1.0 - 1.0 / n)


def test_ephys_qc_score_advisory_fails_do_not_drop_score():
    """Advisory fails shouldn't affect the critical-only composite."""
    s = _ephys_qc_score({
        "triggered_metrics": [
            {"metric": "rin_mohm", "verdict": "fail", "critical": False},
            {"metric": "rac_decay_residual_rel", "verdict": "fail", "critical": False},
        ],
    })
    assert s == pytest.approx(1.0)


# ─── Threshold overrides ──────────────────────────────────────────

def test_threshold_overrides_merge_on_top_of_base(tmp_path: Path):
    base = tmp_path / "thresholds.yaml"
    base.write_text(yaml.safe_dump({
        "metrics": {
            "vrest_mv": {"flag_above": -50, "flag_below": -90, "fail_below": -95},
            "rs_drift_pct": {"fail_above": 30},
        }
    }))
    # Note: base might not have a "metrics" wrapper if it was authored flat —
    # the loader has to handle both. Test flat-mode here too:
    flat = tmp_path / "flat.yaml"
    flat.write_text(yaml.safe_dump({
        "vrest_mv": {"flag_above": -50, "fail_below": -95},
    }))
    overrides = tmp_path / "threshold_overrides.yaml"
    save_threshold_overrides(overrides, {"vrest_mv": {"flag_above": -45, "fail_below": -100}})
    # Merge against the flat-mode base
    merged, applied = load_thresholds_with_overrides(flat, overrides)
    assert merged["vrest_mv"] == {"flag_above": -45, "fail_below": -100}
    # Untouched key in base survives if it exists; we used a flat one so nothing
    # else to assert here, just that no other keys magically appeared.
    assert set(merged) == {"vrest_mv"}
    assert applied == {"vrest_mv": {"flag_above": -45, "fail_below": -100}}


def test_threshold_overrides_no_file_returns_base(tmp_path: Path):
    base = tmp_path / "th.yaml"
    base.write_text(yaml.safe_dump({"vrest_mv": {"fail_below": -95}}))
    merged, applied = load_thresholds_with_overrides(base, tmp_path / "missing.yaml")
    assert merged == {"vrest_mv": {"fail_below": -95}}
    assert applied == {}


def test_save_threshold_overrides_is_atomic_and_yaml_structured(tmp_path: Path):
    path = tmp_path / "th.yaml"
    save_threshold_overrides(path, {"vrest_mv": {"flag_above": -45}})
    body = yaml.safe_load(path.read_text())
    assert body == {"metrics": {"vrest_mv": {"flag_above": -45}}}


# ─── Trim overrides ───────────────────────────────────────────────

def test_trim_overrides_init_load_upsert_roundtrip(tmp_path: Path):
    path = tmp_path / "qc_trim_overrides.csv"
    init_trim_overrides_file(path)
    assert load_trim_overrides(path) == {}
    upsert_trim_override(path, nwb_sha256="abc", trim_at_sweep=12, note="bad ending")
    upsert_trim_override(path, nwb_sha256="def", trim_at_sweep=0, note="keep all")
    overrides = load_trim_overrides(path)
    assert overrides == {"abc": 12, "def": 0}
    # Re-upsert the same sha replaces, doesn't duplicate
    upsert_trim_override(path, nwb_sha256="abc", trim_at_sweep=7)
    overrides = load_trim_overrides(path)
    assert overrides == {"abc": 7, "def": 0}
    # CSV stays well-formed and contains expected columns
    df = pd.read_csv(path)
    assert set(df.columns) >= {"nwb_sha256", "trim_at_sweep", "note", "reviewer", "date"}
    assert len(df) == 2


def test_trim_overrides_missing_file_returns_empty(tmp_path: Path):
    assert load_trim_overrides(tmp_path / "nope.csv") == {}


# ─── End-to-end: pipeline picks up overrides ──────────────────────

@pytest.fixture
def tiny_project_with_ap(tmp_path: Path) -> Path:
    """Tiny project with a single NWB carrying one AP sweep."""
    nwb_dir = tmp_path / "nwbs"
    nwb_dir.mkdir()
    _make_nwb_with_ap_sweep(nwb_dir / "cell-1.nwb")
    th = tmp_path / "thresholds.yaml"
    th.write_text(yaml.safe_dump({
        "metrics": {
            "vrest_mv": {"pass": {"min": -200, "max": 200}, "flag": {"min": -200, "max": 200}},
        },
    }))
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project_name": "t",
        "output_dir": str(tmp_path / "out"),
        "nwb_sources": [{"dataset": "ds", "path": str(nwb_dir), "glob": "*.nwb"}],
        "stimulus_protocols": {"ap_waveform": ["APWaveform"]},
        "thresholds_file": str(th),
        "n_workers": 1,
    }))
    return cfg_path


def test_pipeline_emits_n_spikes_total_column(tiny_project_with_ap: Path):
    cfg = load_config(tiny_project_with_ap)
    pipeline_run(cfg)
    df = pd.read_csv(cfg.report_csv)
    assert "n_spikes_total" in df.columns
    assert (df["n_spikes_total"].fillna(0).astype(int) >= 0).all()


def test_iter_current_clamp_acqs_is_chronological(tmp_path: Path):
    """REGRESSION (v0.7.1 fix): _iter_current_clamp_acqs must walk sweeps in
    chronological order, NOT acquisition.items() dict-insertion order.

    Some NWB writers group sweeps by stimulus type rather than by recording
    time. With dict-insertion order, the bad-ending detector falsely trimmed
    every spontaneous sweep on JY171019_B_1 because they were grouped at the
    tail of the dict — yet chronologically they were sprinkled throughout
    the session, including the very first sweep recorded.
    """
    from nwb_trace_qc.metrics import _iter_current_clamp_acqs
    nwb_path = tmp_path / "chrono.nwb"
    rate = 10_000.0
    n = 500
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier="x",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    # Insert in non-chronological dict order: spontaneous (late) → test_pulse
    # (early) → ap (mid). starting_time tells the real story.
    samples = np.linspace(-0.07, -0.06, n)
    for name, start in [
        ("ic__SponNoHold30__001", 200.0),   # dict-idx 0, chrono-idx 1
        ("ic__SponNoHold30__002", 600.0),   # dict-idx 1, chrono-idx 4
        ("ic__APWaveform__001",   100.0),   # dict-idx 2, chrono-idx 0
        ("ic__APWaveform__002",   400.0),   # dict-idx 3, chrono-idx 3
        ("ic__APWaveform__003",   300.0),   # dict-idx 4, chrono-idx 2
    ]:
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=name, data=samples, electrode=elec, gain=1.0,
            starting_time=start, rate=rate, unit="volts",
        ))
    with pynwb.NWBHDF5IO(str(nwb_path), mode="w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r") as io:
        nwb = io.read()
        names = [name for name, _obj in _iter_current_clamp_acqs(nwb)]
    assert names == [
        "ic__APWaveform__001",     # t=100
        "ic__SponNoHold30__001",   # t=200
        "ic__APWaveform__003",     # t=300
        "ic__APWaveform__002",     # t=400
        "ic__SponNoHold30__002",   # t=600
    ], f"_iter_current_clamp_acqs is not chronological — got {names}"


def test_pipeline_applies_trim_override_and_busts_cache(tiny_project_with_ap: Path):
    """Adding a trim override after a first run forces a recompute, and the new
    cache row replaces the stale one (append_rows dedupes on sha+version)."""
    cfg = load_config(tiny_project_with_ap)
    pipeline_run(cfg)
    df1 = pd.read_csv(cfg.report_csv)
    sha = df1["nwb_sha256"].iloc[0]
    # Write a trim override that throws away every sweep (trim_at_sweep=1 with
    # n_total=1 would be a no-op; this NWB has 1 sweep so use force=0 to verify
    # the loader-recompute path even when there's nothing to trim).
    upsert_trim_override(cfg.trim_overrides_file, nwb_sha256=sha,
                          trim_at_sweep=0, note="test")
    overrides = load_trim_overrides(cfg.trim_overrides_file)
    assert overrides == {sha: 0}
    # Second run should recompute the overridden sha (we can't easily assert
    # "recompute happened" without instrumentation, but we can assert the
    # report still renders cleanly and the column exists).
    pipeline_run(cfg)
    df2 = pd.read_csv(cfg.report_csv)
    assert "n_spikes_total" in df2.columns
    assert len(df2) == len(df1)
