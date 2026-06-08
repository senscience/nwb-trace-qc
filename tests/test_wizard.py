"""`nwb-qc start` wizard smoke test — drive the five prompts with scripted input."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynwb
import pytest
from click.testing import CliRunner

from nwb_trace_qc.cli import main


def _make_nwb(path: Path, identifier: str):
    nwbfile = pynwb.NWBFile(
        session_description="t", identifier=identifier,
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    elec = nwbfile.create_icephys_electrode(
        name="elec0", description="d",
        device=nwbfile.create_device(name="a", description="d"),
    )
    sweep = pynwb.icephys.CurrentClampSeries(
        name="ic__APWaveform__001",
        data=np.linspace(-0.07, 0.03, 500),
        electrode=elec, gain=1.0, starting_time=0.0, rate=10000.0, unit="volts",
    )
    nwbfile.add_acquisition(sweep)
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


@pytest.fixture
def wizard_tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    ds_dir = root / "ds1"
    ds_dir.mkdir(parents=True)
    for i in range(2):
        _make_nwb(ds_dir / f"cell-{i}.nwb", f"cell-{i}")
    return root


def test_wizard_quits_at_inspect(wizard_tree: Path, tmp_path: Path):
    """Pressing 'q' at the first prompt exits cleanly with code 1 and no config written."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        input="q\n",
    )
    assert result.exit_code == 1, result.output
    assert "aborted at inspect" in result.output
    assert not out_yaml.exists()


def test_wizard_happy_path(wizard_tree: Path, tmp_path: Path):
    """accept config → run dry-run → run pipeline → done (no opening browser)."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    # 5 prompts: [y]es → [a]ccept → [r]un → (no prompt after run output) → [d]one
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        # 6 prompts (v0.4.x): [y]es → [a]ccept-config → [a]ccept-thresholds
        # → [r]un → (no prompt after run output) → [d]one
        # v0.8.0: a blank line after the first 'a' skips the curator prompt
        input="y\na\n\na\nr\nd\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert out_yaml.exists()
    # Run report was generated
    rpt = list((tmp_path).rglob("run_report.json"))
    assert rpt, f"no run_report.json under {tmp_path}; output:\n{result.output}"


def test_unmapped_block_parser_and_heuristic():
    """Parser pulls (token, n_sweeps) pairs out of the YAML comment block, and
    the family-guess heuristic resolves the obvious cases."""
    from nwb_trace_qc.wizard import _guess_family, _parse_unmapped_block

    yaml_text = (
        "# header\n"
        "# ⚠ UNMAPPED tokens (3 unique):\n"
        "#   SpikeRec  (74 sweeps)\n"
        "#   HyperDePol  (36 sweeps)\n"
        "#   ResetITC  (10 sweeps)\n"
        "#\n"
        "# Add these to the appropriate family under stimulus_protocols: below.\n"
        "\n"
        "project_name: foo\n"
    )
    parsed = _parse_unmapped_block(yaml_text)
    assert parsed == [("SpikeRec", 74), ("HyperDePol", 36), ("ResetITC", 10)]

    # Heuristic: confident cases get a guess; unrelated tokens get None
    assert _guess_family("SpikeRec") == "ap_waveform"
    assert _guess_family("HyperDePol") == "iv_subthreshold"
    assert _guess_family("IDThres") == "threshold_search"
    assert _guess_family("FirePattern") == "rest_firing"
    assert _guess_family("RSealOpen") == "test_pulse"
    # No obvious mapping → None (user picks 0 in interactive flow)
    assert _guess_family("ResetITC") is None
    assert _guess_family("xyz123") is None


def test_interactive_mapping_updates_yaml_in_place(tmp_path: Path, monkeypatch):
    """End-to-end: a YAML with UNMAPPED tokens gets rewritten with the chosen
    families folded into stimulus_protocols and the UNMAPPED block trimmed."""
    from nwb_trace_qc.wizard import _interactive_map_unmapped
    yaml_text = (
        "# header\n"
        "# Stimulus protocols discovered by sampling your NWBs:\n"
        "#   ap_waveform: APWaveform (54)\n"
        "#\n"
        "# ⚠ UNMAPPED tokens (2 unique):\n"
        "#   SpikeRec  (74 sweeps)\n"
        "#   FirePattern  (18 sweeps)\n"
        "#\n"
        "# Add these to the appropriate family under stimulus_protocols: below.\n"
        "\n"
        "stimulus_protocols:\n"
        "  ap_waveform:\n"
        "  - APWaveform\n"
        "  rest_firing:\n"
        "  - IDRest\n"
        "project_name: foo\n"
    )
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(yaml_text)

    # Auto-answer the click.prompt sequence: [w] at top-of-walk to enter the
    # per-token loop, then "4" for SpikeRec (→ap_waveform), "5" for FirePattern
    # (→rest_firing).
    answers = iter(["w", "4", "5"])
    monkeypatch.setattr("click.prompt", lambda *a, **kw: next(answers))

    changed = _interactive_map_unmapped(yaml_path)
    assert changed is True

    import yaml as _yaml
    updated = _yaml.safe_load("\n".join(
        line for line in yaml_path.read_text().splitlines() if not line.startswith("#")
    ))
    fams = updated["stimulus_protocols"]
    assert "SpikeRec" in fams["ap_waveform"]
    assert "FirePattern" in fams["rest_firing"]
    # Original entries preserved
    assert "APWaveform" in fams["ap_waveform"]
    assert "IDRest" in fams["rest_firing"]
    # Header trimmed — no more UNMAPPED block
    assert "UNMAPPED tokens" not in yaml_path.read_text()
    assert "All discovered stimulus tokens are now mapped" in yaml_path.read_text()


def test_interactive_mapping_accept_all_assigns_in_bulk(tmp_path: Path, monkeypatch):
    """Top-of-walk [a] applies every heuristic-derived suggestion in one keystroke
    and skips the per-token prompts entirely."""
    from nwb_trace_qc.wizard import _interactive_map_unmapped
    yaml_text = (
        "# header\n"
        "# ⚠ UNMAPPED tokens (3 unique):\n"
        "#   SpikeRec  (74 sweeps)\n"
        "#   HyperDePol  (36 sweeps)\n"
        "#   FirePattern  (18 sweeps)\n"
        "#\n"
        "\n"
        "stimulus_protocols:\n"
        "  ap_waveform:\n"
        "  - APWaveform\n"
    )
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(yaml_text)

    # Only one prompt should fire: the top-of-walk action. Anything beyond that
    # is a regression (per-token prompts should be skipped on [a]).
    prompts: list[tuple] = []

    def fake_prompt(*args, **kwargs):
        prompts.append(args)
        return "a"

    monkeypatch.setattr("click.prompt", fake_prompt)
    changed = _interactive_map_unmapped(yaml_path)
    assert changed is True
    assert len(prompts) == 1, f"expected only the top-of-walk prompt; got {len(prompts)}"

    import yaml as _yaml
    updated = _yaml.safe_load("\n".join(
        line for line in yaml_path.read_text().splitlines() if not line.startswith("#")
    ))
    fams = updated["stimulus_protocols"]
    # Heuristic guesses: SpikeRec→ap_waveform, HyperDePol→iv_subthreshold, FirePattern→rest_firing
    assert "SpikeRec" in fams["ap_waveform"]
    assert "HyperDePol" in fams["iv_subthreshold"]
    assert "FirePattern" in fams["rest_firing"]


def test_interactive_mapping_accept_all_leaves_unguessable_tokens(tmp_path: Path, monkeypatch):
    """Tokens with no heuristic guess (None) stay in the UNMAPPED block on [a]."""
    from nwb_trace_qc.wizard import _interactive_map_unmapped
    yaml_text = (
        "# ⚠ UNMAPPED tokens (2 unique):\n"
        "#   SpikeRec  (74 sweeps)\n"
        "#   ElecCal  (10 sweeps)\n"
        "#\n"
        "\n"
        "stimulus_protocols: {}\n"
    )
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(yaml_text)
    monkeypatch.setattr("click.prompt", lambda *a, **kw: "a")
    changed = _interactive_map_unmapped(yaml_path)
    assert changed is True
    body = yaml_path.read_text()
    # SpikeRec heuristic → ap_waveform; ElecCal has no guess so it stays unmapped
    assert "SpikeRec" in body
    assert "ElecCal" in body


def test_wizard_aborts_at_review_thresholds_with_q(wizard_tree: Path, tmp_path: Path):
    """Pressing 'q' at the new Review-thresholds stage exits cleanly with code 1
    and no metric-compute runs (no cache populated)."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    # y (inspect) → a (propose) → q (review-thresholds)
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        # v0.8.0: blank line skips the new curator prompt between accept-config and thresholds
        input="y\na\n\nq\n",
    )
    assert result.exit_code == 1, result.output
    assert "aborted at review-thresholds" in result.output
    # No run report — pipeline never ran
    rpt = list(tmp_path.rglob("run_report.json"))
    assert not rpt


def test_interactive_mapping_skip_leaves_unmapped(tmp_path: Path, monkeypatch):
    """Pressing 0 skips a token — it should stay in the UNMAPPED block, not move."""
    from nwb_trace_qc.wizard import _interactive_map_unmapped
    yaml_text = (
        "# ⚠ UNMAPPED tokens (1 unique):\n"
        "#   ElecCal  (30 sweeps)\n"
        "#\n"
        "\n"
        "stimulus_protocols: {}\n"
    )
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(yaml_text)

    monkeypatch.setattr("click.prompt", lambda *a, **kw: "0")
    changed = _interactive_map_unmapped(yaml_path)
    assert changed is False
    body = yaml_path.read_text()
    # Body unchanged when nothing was mapped
    assert "ElecCal" in body


def test_wizard_auto_writes_cohort_stats_and_suggested_thresholds(wizard_tree: Path,
                                                                     tmp_path: Path):
    """After a successful wizard run, cohort_stats.json (next to run_report.json)
    and `<stem>_thresholds_suggested.yaml` (next to the active thresholds) exist."""
    runner = CliRunner()
    out_yaml = tmp_path / "project.yaml"
    result = runner.invoke(
        main, ["start", str(wizard_tree), "--output", str(out_yaml)],
        # 6 prompts (v0.4.x): [y]es → [a]ccept-config → [a]ccept-thresholds
        # → [r]un → (no prompt after run output) → [d]one
        # v0.8.0: a blank line after the first 'a' skips the curator prompt
        input="y\na\n\na\nr\nd\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    cohort_files = list(tmp_path.rglob("cohort_stats.json"))
    assert cohort_files, f"no cohort_stats.json under {tmp_path}; output:\n{result.output}"
    import json as _json
    body = _json.loads(cohort_files[0].read_text())
    assert isinstance(body, dict) and body, "cohort_stats.json was written empty"

    suggested = list(tmp_path.rglob("*thresholds_suggested.yaml"))
    assert suggested, (
        f"no *_thresholds_suggested.yaml under {tmp_path}; output:\n{result.output}"
    )

    # The outcome stage advertised both files
    assert "suggested thresholds" in result.output
    assert "cohort stats" in result.output
