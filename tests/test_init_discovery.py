"""init-config samples NWBs and surfaces unmapped stimulus tokens in the YAML."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest
import yaml
from click.testing import CliRunner

from nwb_trace_qc.cli import _build_starter_config, _classify_tokens, main


def _write_nwb(path: Path, sweep_names: list[str]) -> None:
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=path.stem,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    for sn in sweep_names:
        nwbfile.add_acquisition(pynwb.icephys.CurrentClampSeries(
            name=sn, data=np.linspace(-0.07, 0.03, 200),
            electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
        ))
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_classify_tokens_separates_matched_and_unmatched():
    fams = {"ap_waveform": ["APWaveform"], "rest_firing": ["IDRest"]}
    matched, unmatched = _classify_tokens(
        Counter({"APWaveform": 30, "Spontaneous": 5, "idrest": 10, "Test_eCode": 2}),
        fams,
    )
    assert "ap_waveform" in matched
    assert matched["ap_waveform"]["APWaveform"] == 30
    assert matched["rest_firing"]["idrest"] == 10  # case-insensitive match
    assert "Spontaneous" in unmatched and "Test_eCode" in unmatched
    assert "APWaveform" not in unmatched


def test_build_starter_config_surfaces_unmapped_tokens(tmp_path: Path):
    """An NWB with a mix of known + unknown stimulus tokens produces both a
    discovered block AND a flagged UNMAPPED block in the YAML header."""
    root = tmp_path / "data"
    ds = root / "ds1"; ds.mkdir(parents=True)
    _write_nwb(ds / "cell-1.nwb", [
        "ic__APWaveform__001", "ic__APWaveform__002",      # known
        "ic__Spontaneous__001", "ic__Test_eCode__001",     # unknown
        "ic__IDRest__001",                                  # known
    ])
    out_yaml = tmp_path / "project.yaml"
    yaml_text = _build_starter_config(root, output_path=out_yaml)
    assert "Stimulus protocols discovered" in yaml_text
    assert "UNMAPPED tokens" in yaml_text
    # Unmapped tokens are listed with their counts
    assert "Spontaneous" in yaml_text
    assert "Test_eCode" in yaml_text
    # Known tokens appear under their family lines
    assert "ap_waveform:" in yaml_text
    assert "APWaveform" in yaml_text
    # Heads-up about missing essential families fires (no spontaneous_hold/test_pulse)
    assert "spontaneous_hold" in yaml_text
    assert "qc_protocol_coverage will be False" in yaml_text


def test_build_starter_config_no_unmapped_block_when_clean(tmp_path: Path):
    """If every protocol in the NWBs matches a default family, the warning block
    is omitted (so well-formed labs don't get noise)."""
    root = tmp_path / "data"
    ds = root / "ds1"; ds.mkdir(parents=True)
    # Use names that all map: APWaveform (ap_waveform), IDRest (rest_firing),
    # SponHold3 (spontaneous_hold), Rac (test_pulse)
    _write_nwb(ds / "cell-1.nwb", [
        "ic__APWaveform__001", "ic__IDRest__001",
        "ic__SponHold3__001", "ic__Rac__001",
    ])
    out_yaml = tmp_path / "project.yaml"
    yaml_text = _build_starter_config(root, output_path=out_yaml)
    assert "UNMAPPED tokens" not in yaml_text
    assert "qc_protocol_coverage will be False" not in yaml_text


def test_init_config_cli_writes_discovery_block(tmp_path: Path):
    """End-to-end smoke: `nwb-qc init-config` produces a YAML that parses cleanly
    AND contains the discovery comments above the body."""
    root = tmp_path / "data"
    ds = root / "ds1"; ds.mkdir(parents=True)
    _write_nwb(ds / "cell-1.nwb", ["ic__APWaveform__001", "ic__MysteryProto__001"])
    out_yaml = tmp_path / "project.yaml"
    runner = CliRunner()
    result = runner.invoke(main, ["init-config", str(root), "--output", str(out_yaml)])
    assert result.exit_code == 0, result.output
    text = out_yaml.read_text()
    assert "UNMAPPED tokens" in text
    assert "MysteryProto" in text
    # Body still parses as valid YAML (comments are stripped by safe_load)
    parsed = yaml.safe_load(text)
    assert parsed["project_name"]
    assert isinstance(parsed["stimulus_protocols"], dict)
