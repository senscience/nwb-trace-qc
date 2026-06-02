"""Project-config loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class NWBSource(BaseModel):
    dataset: str
    path: Path
    recursive: bool = True
    glob: str = "**/*.nwb"


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


# Canonical stimulus-protocol families. Keys are *family* names referenced by metrics.
_DEFAULT_FAMILIES = {
    "spontaneous_hold": ["SponHold3", "SponHold30", "SponNoHold30", "StartHold"],
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
    n_workers: int = 4
    cache_path: Path | None = None
    manifest_path: Path | None = None
    overrides_path: Path | None = None
    report_html: Path | None = None
    report_csv: Path | None = None
    thumbnails_dir: Path | None = None
    cell_table: CellTable | None = None
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
            s.path = _abs(s.path)  # type: ignore[assignment]
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
        for attr in ("cache_path", "manifest_path", "overrides_path", "report_html", "report_csv", "thumbnails_dir"):
            v = getattr(self, attr)
            setattr(self, attr, _abs(v))
        if self.thresholds_file is not None:
            self.thresholds_file = _abs(self.thresholds_file)
        return self


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw["config_path"] = path
    return ProjectConfig.model_validate(raw)


def default_families() -> dict[str, list[str]]:
    return dict(_DEFAULT_FAMILIES)
