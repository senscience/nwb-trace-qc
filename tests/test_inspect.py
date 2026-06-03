"""Smoke tests for nwb-qc inspect: walks a synthetic tree, checks the structured result."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nwb_trace_qc.inspect import inspect_root, render_json, render_markdown, render_terminal


def _make_synthetic_tree(root: Path) -> None:
    """Build a fake wrangler-output layout:
       root/
         cohort_a/
           proj/
             fair2.json
             README.md
             run_state.json
             data_dictionary.csv
             data/parquet/{acqs.parquet, session_metadata.parquet}
             assets/nwb/cell1.nwb, cell2.nwb
             scripts/extract.py
         cohort_b/
           cell3.nwb     # NWB directly under top-level subfolder
         empty_dir/      # no NWB and no parquet → should be filtered
    """
    a = root / "cohort_a" / "proj"
    a.mkdir(parents=True)
    (a / "fair2.json").write_text(json.dumps({
        "@type": "https://schema.org/Dataset",
        "conformsTo": "https://mlcommons.org/croissant/1.1",
        "name": "Cohort A test dataset",
    }))
    (a / "README.md").write_text("# Cohort A\n\nA tiny synthetic dataset for tests.")
    (a / "run_state.json").write_text(json.dumps({
        "version": "0.4.2", "status": "OK", "timestamp": "2026-06-01T19:52:19"
    }))
    (a / "data_dictionary.csv").write_text("variable,definition\nfoo,a foo\nbar,a bar\n")
    pqdir = a / "data" / "parquet"; pqdir.mkdir(parents=True)
    qc_eligible = pa.table({"nwb_file": ["a.nwb", "b.nwb"], "stimulus_type": ["x", "y"], "extra": [1, 2]})
    pq.write_table(qc_eligible, pqdir / "acqs.parquet")
    other = pa.table({"session_id": ["a", "b"], "n_acqs": [10, 20]})
    pq.write_table(other, pqdir / "session_metadata.parquet")
    nwbdir = a / "assets" / "nwb"; nwbdir.mkdir(parents=True)
    (nwbdir / "cell1.nwb").write_bytes(b"\0" * 1024)
    (nwbdir / "cell2.nwb").write_bytes(b"\0" * 2048)
    (a / "scripts").mkdir(); (a / "scripts" / "extract.py").write_text("# script")

    b = root / "cohort_b"; b.mkdir()
    (b / "cell3.nwb").write_bytes(b"\0" * 512)

    # cohort_c — a wrangler output WITH source_material/source_manifest.json
    # but no NWBs locally (sources not preserved)
    c_proj = root / "cohort_c" / "proj"
    c_proj.mkdir(parents=True)
    (c_proj / "source_material").mkdir()
    # Make one fake source NWB that lives outside cohort_c
    src_nwb = root / "external_data" / "cell_42.nwb"
    src_nwb.parent.mkdir()
    src_nwb.write_bytes(b"\0" * 4096)
    manifest = {
        "schema_version": 5, "generated_at": "2026-06-01T00:00:00Z",
        "preservation": {"included": False},
        "summary": {"total_files": 1, "total_size_bytes": 4096, "processed_files": 1},
        "files": [{
            "path": "source_material/cell_42.nwb",
            "original_location": str(src_nwb), "size_bytes": 4096,
            "sha256": "deadbeef" * 8, "mtime": src_nwb.stat().st_mtime,
            "was_processed": True,
        }],
    }
    (c_proj / "source_material" / "source_manifest.json").write_text(json.dumps(manifest))

    (root / "empty_dir").mkdir()


def test_inspect_finds_expected_datasets(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    names = sorted(d.name for d in r.datasets)
    # cohort_a (NWBs + parquets), cohort_b (loose NWB), cohort_c (manifest-only, no local NWBs)
    # external_data/ is just a holder for the manifest's source files; it ends up in the listing too
    # since it has NWBs in it.
    assert "cohort_a" in names and "cohort_b" in names and "cohort_c" in names
    # empty_dir is filtered out
    assert "empty_dir" not in names


def test_inspect_picks_up_source_manifest(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    cohort_c = next(d for d in r.datasets if d.name == "cohort_c")
    sm = cohort_c.source_manifest
    assert sm is not None
    assert sm.schema_version == 5
    assert sm.total_files == 1
    assert sm.preservation_included is False
    assert sm.sample_size == 1
    assert sm.sample_present == 1
    # Renderers should include the manifest line
    out_t = render_terminal(r)
    assert "source_material/source_manifest.json" in out_t
    assert "schema v5" in out_t
    out_md = render_markdown(r)
    assert "source_manifest.json" in out_md
    out_json = json.loads(render_json(r))
    by_name = {d["name"]: d for d in out_json["datasets"]}
    assert by_name["cohort_c"]["source_manifest"]["total_files"] == 1


def test_inspect_classifies_qc_eligible_parquet(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    cohort_a = next(d for d in r.datasets if d.name == "cohort_a")
    by_name = {p.name: p for p in cohort_a.parquets}
    assert by_name["acqs.parquet"].qc_eligible is True
    assert by_name["session_metadata.parquet"].qc_eligible is False
    assert "missing" in by_name["session_metadata.parquet"].reason_not_eligible


def test_inspect_picks_up_summaries(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    cohort_a = next(d for d in r.datasets if d.name == "cohort_a")
    assert cohort_a.fair2 is not None and "Cohort A test dataset" in cohort_a.fair2.summary
    assert cohort_a.readme is not None and "Cohort A" in cohort_a.readme.summary
    assert cohort_a.run_state is not None and "OK" in cohort_a.run_state.summary
    assert cohort_a.data_dictionary is not None and "2 variable" in cohort_a.data_dictionary.summary
    assert cohort_a.scripts == ["extract.py"]
    assert cohort_a.nested_subroot is not None and cohort_a.nested_subroot.name == "proj"


def test_inspect_detects_nwbs_directly_in_subfolder(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    cohort_b = next(d for d in r.datasets if d.name == "cohort_b")
    assert cohort_b.n_nwbs_total == 1
    assert cohort_b.nested_subroot is None  # no inner project dir


def test_render_terminal_runs(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    out = render_terminal(r)
    assert "Summary" in out and "datasets" in out
    assert "✓ QC-eligible" in out


def test_render_markdown_runs(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    md = render_markdown(r)
    assert md.startswith("# Inventory of")
    assert "cohort_a" in md and "cohort_b" in md


def test_render_json_valid(tmp_path: Path):
    _make_synthetic_tree(tmp_path)
    r = inspect_root(tmp_path)
    obj = json.loads(render_json(r))
    # cohort_a has 2 NWBs, cohort_b has 1, plus 1 in external_data referenced by cohort_c's manifest = 4 total
    assert obj["total_nwbs"] == 4
    # cohort_a, cohort_b, cohort_c, external_data — all have NWBs or manifests
    assert len(obj["datasets"]) >= 3
