"""Project-config loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class NWBSource(BaseModel):
    """A logical NWB-source dataset, either as a directory tree or a wrangler source-manifest."""

    dataset: str
    # Directory-tree mode (legacy default)
    path: Path | None = None
    recursive: bool = True
    glob: str = "**/*.nwb"
    # Source-manifest mode (alternative; mutually exclusive with `path`)
    manifest: Path | None = None
    only_processed: bool = False         # default: include every NWB listed in the manifest
                                         # (was_processed=False often just means "the wrangler
                                         # processed the parent archive, not this individual NWB",
                                         # e.g. extracting *.nwb from *.tar.gz)
    reuse_sha256: bool = True            # trust manifest hash if size+mtime match

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.path is None) == (self.manifest is None):
            raise ValueError(
                f"NWBSource {self.dataset!r}: exactly one of `path` or `manifest` must be set "
                f"(path={self.path!r}, manifest={self.manifest!r})"
            )
        return self


class AcquisitionColumnMap(BaseModel):
    stimulus_type: str = "stimulus_type"
    rate_hz: str | None = "rate_hz"
    n_samples: str | None = "n_samples"
    clamp_mode: str | None = "clamp_mode"
    sweep_number: str | None = "sweep_number"


class AcquisitionTable(BaseModel):
    path: Path
    nwb_key_column: str = "nwb_file"
    nwb_key_format: str = "stem"  # 'stem' | 'basename' | 'absolute'
    columns: AcquisitionColumnMap = AcquisitionColumnMap()

    @model_validator(mode="after")
    def _check_format(self):
        if self.nwb_key_format not in {"stem", "basename", "absolute"}:
            raise ValueError(f"nwb_key_format must be stem|basename|absolute, got {self.nwb_key_format!r}")
        return self


class CellTable(BaseModel):
    path: Path
    cell_id_column: str = "cell_id"
    dataset_columns: list[str] = Field(default_factory=list)


class VisionJudgeConfig(BaseModel):
    """Opt-in LLM vision-judge configuration. Off by default.

    `max_cost_usd` is a soft cap: the vision pass stops calling the provider once
    the running estimated cost reaches this value. The pipeline still finishes
    rendering the report with whatever vision verdicts were collected (un-judged
    flag cells keep their rule-based verdict).
    """

    enabled: bool = False
    provider: str = "anthropic"            # 'anthropic' | 'openai' | 'mock'
    model: str = "claude-haiku-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_borderline_cells: int = 100
    max_cost_usd: float = 1.0
    prompt_template: Path | None = None    # null = bundled default
    cache_responses: bool = True

    @model_validator(mode="after")
    def _check_provider(self):
        if self.provider not in {"anthropic", "openai", "mock"}:
            raise ValueError(f"vision_judge.provider must be anthropic|openai|mock, got {self.provider!r}")
        return self


# Canonical stimulus-protocol families. Keys are *family* names referenced by metrics.
#
# In v0.5.0 the legacy `spontaneous_hold` family is split into two semantically
# distinct families (per LNMC experimenter guidance):
#
#   - spontaneous_no_hold: NO holding current injected — this is the TRUE resting
#     membrane potential. Vrest_mv is sourced from these sweeps when present.
#   - spontaneous_held:    holding current applied (typically -100 pA to clamp at
#                          -70 mV). Source of held_vm_mv + holding_current_pa.
#
# Older project YAMLs that map protocols to `spontaneous_hold` continue to work —
# metrics.py treats that family as a legacy fallback for both no-hold and held
# semantics, with a notice logged once per run.
_DEFAULT_FAMILIES = {
    "spontaneous_no_hold": ["SponNonHold30", "SponNoHold30", "StartNoHold"],
    "spontaneous_held":    ["SponHold3", "SponHold30", "StartHold"],
    "test_pulse": ["Rac", "TestAmpl", "TestRheo"],
    "iv_subthreshold": ["IV"],
    "ap_waveform": ["APWaveform"],
    "rest_firing": ["IDRest"],
    "threshold_search": ["IDThres", "IDThreshold"],
}


class ProjectConfig(BaseModel):
    project_name: str = "unnamed_project"
    output_dir: Path = Path("./qc_output")
    nwb_sources: list[NWBSource] = Field(default_factory=list)
    acquisition_tables: list[AcquisitionTable] = Field(default_factory=list)
    stimulus_protocols: dict[str, list[str]] = Field(default_factory=lambda: dict(_DEFAULT_FAMILIES))
    thresholds_file: Path | None = None
    # v0.7.0: viewer-editable overlay files (written by `nwb-qc serve`'s threshold
    # pencils and trim slider). Resolved against output_dir if relative.
    threshold_overrides_file: Path | None = None
    trim_overrides_file: Path | None = None
    n_workers: int = 4
    cache_path: Path | None = None
    manifest_path: Path | None = None
    overrides_path: Path | None = None
    report_html: Path | None = None
    report_csv: Path | None = None
    thumbnails_dir: Path | None = None
    cell_table: CellTable | None = None
    vision_judge: VisionJudgeConfig = Field(default_factory=VisionJudgeConfig)
    # Interactive viewer settings
    viewer_url: str = "http://127.0.0.1:8765"
    viewer_cache_thumbnails: bool = True
    # v0.8.0: curator identity stamped onto qc_overrides.csv rows when the
    # viewer saves a per-cell decision. Empty ⇒ viewer prompts once per
    # session and stashes the answer in localStorage.
    curator: str = ""
    # Quality-of-recording controls (v0.4.0)
    trim_bad_ending: bool = True       # auto-detect + trim degraded tail sweeps
    use_efel: bool = True              # source AP/Vrest features from eFEL when available
    # v0.6.0: which metrics' fails cascade to a cell-level fail. Empty list ⇒
    # use the bundled DEFAULT_CRITICAL_METRICS. Anything outside this set is
    # demoted from fail to flag at the cell level (still surfaced as an
    # advisory chip in the report).
    critical_metrics: list[str] = Field(default_factory=list)
    # absolute base path used to resolve all other relative paths (set by loader)
    config_path: Path | None = None

    @model_validator(mode="after")
    def _resolve_defaults(self):
        # Resolve relative paths against the directory containing the YAML
        base = self.config_path.parent if self.config_path else Path.cwd()
        def _abs(p: Path | None) -> Path | None:
            if p is None: return None
            return p if p.is_absolute() else (base / p).resolve()
        self.output_dir = _abs(self.output_dir)  # type: ignore[assignment]
        for s in self.nwb_sources:
            if s.path is not None:
                s.path = _abs(s.path)  # type: ignore[assignment]
            if s.manifest is not None:
                s.manifest = _abs(s.manifest)  # type: ignore[assignment]
        for t in self.acquisition_tables:
            t.path = _abs(t.path)  # type: ignore[assignment]
        if self.cell_table:
            self.cell_table.path = _abs(self.cell_table.path)  # type: ignore[assignment]
        # Output paths default to output_dir/<sensible name>
        out = self.output_dir
        if self.cache_path is None:       self.cache_path = out / "_qc_cache.parquet"
        if self.manifest_path is None:    self.manifest_path = out / "_qc_manifest.parquet"
        if self.overrides_path is None:   self.overrides_path = out / "qc_overrides.csv"
        if self.report_html is None:      self.report_html = out / "qc_report.html"
        if self.report_csv is None:       self.report_csv = out / "qc_report.csv"
        if self.thumbnails_dir is None:   self.thumbnails_dir = out / "traces"
        if self.threshold_overrides_file is None: self.threshold_overrides_file = out / "threshold_overrides.yaml"
        if self.trim_overrides_file is None:      self.trim_overrides_file = out / "qc_trim_overrides.csv"
        for attr in ("cache_path", "manifest_path", "overrides_path", "report_html",
                      "report_csv", "thumbnails_dir",
                      "threshold_overrides_file", "trim_overrides_file"):
            v = getattr(self, attr)
            setattr(self, attr, _abs(v))
        if self.thresholds_file is not None:
            self.thresholds_file = _abs(self.thresholds_file)
        if self.vision_judge and self.vision_judge.prompt_template is not None:
            self.vision_judge.prompt_template = _abs(self.vision_judge.prompt_template)
        return self


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw["config_path"] = path
    return ProjectConfig.model_validate(raw)


def default_families() -> dict[str, list[str]]:
    return dict(_DEFAULT_FAMILIES)
