from pathlib import Path
import textwrap

from nwb_trace_qc.config import load_config


def test_minimal_config(tmp_path: Path):
    cfg_text = textwrap.dedent("""
    project_name: minimal
    output_dir: ./out
    nwb_sources:
      - dataset: d1
        path: ./nwb
    """)
    p = tmp_path / "project.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert cfg.project_name == "minimal"
    # Relative paths get resolved against the YAML directory
    assert cfg.output_dir == (tmp_path / "out").resolve()
    assert cfg.nwb_sources[0].path == (tmp_path / "nwb").resolve()
    # Sensible defaults
    assert cfg.cache_path.name == "_qc_cache.parquet"
    assert cfg.report_html.name == "qc_report.html"
    # Default stimulus families present
    assert "spontaneous_hold" in cfg.stimulus_protocols
    assert "test_pulse" in cfg.stimulus_protocols


def test_relative_path_resolution(tmp_path: Path):
    cfg_text = textwrap.dedent("""
    project_name: rel
    output_dir: out_sub
    nwb_sources:
      - dataset: x
        path: nwb_sub
    cache_path: out_sub/custom_cache.parquet
    """)
    p = tmp_path / "rel.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert cfg.cache_path == (tmp_path / "out_sub" / "custom_cache.parquet").resolve()
