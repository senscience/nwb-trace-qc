"""Per-cohort inventory of pre-computed metrics inside NWBs.

For every NWB sampled from the configured sources, inspects the standard NWB
containers that *could* carry pre-computed analysis (processing modules,
lab_meta_data, scratch, intervals) and cross-references with the canonical
metric list in `families.METRIC_DESCRIPTIONS`. Surfaces:

  - which of our metrics, if any, this lab already pre-computed inside the NWB
  - where to find them (`processing/<module>/<container>/<field>` path)
  - which of our metrics are computed by `nwb-trace-qc` itself

Output: a markdown report (`<output_dir>/metric_inventory.md`) and a console
summary.

Useful when (a) onboarding a new cohort to understand what's already in the
file, (b) deciding whether a pre-computed value should override our compute
(future work), and (c) documenting the QC pipeline's data sources per cohort.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .families import METRIC_DESCRIPTIONS, METRIC_TO_FAMILY
from .nwb_io import open_nwb

log = logging.getLogger(__name__)


@dataclass
class NwbInventoryEntry:
    """What's in one NWB that *could* be a pre-computed metric source."""
    nwb_path: Path
    has_processing: bool = False
    processing_modules: list[str] = field(default_factory=list)
    has_lab_meta_data: bool = False
    has_scratch: bool = False
    has_intervals: bool = False
    # Per-metric findings: metric_name → (source_path, value_or_None)
    found_metrics: dict[str, str] = field(default_factory=dict)
    # Anything in processing modules we didn't recognize as a known metric
    other_processing_contents: list[str] = field(default_factory=list)
    error: str | None = None


# Heuristic: NWB containers whose field names hint at canonical metric values.
# (We only check shallow names — if a lab buried a feature under a custom
# DynamicTable, the user can drill in by hand.)
_KNOWN_METRIC_NAMES = set(METRIC_DESCRIPTIONS.keys()) | {
    # eFEL feature names (when a lab has stored eFEL output directly)
    "voltage_base", "AP_amplitude", "AP_amplitude_from_voltagebase",
    "AP_begin_voltage", "Spikecount", "mean_frequency", "AHP_depth",
    "AP_height", "peak_voltage",
}


def _scan_processing_module(module, found: dict[str, str], others: list[str],
                              base_path: str) -> None:
    """Walk a single NWB processing module and record what we see."""
    for container_name, container in module.data_interfaces.items():
        path = f"{base_path}/{container_name}"
        # Many labs store features as DynamicTable rows or TimeSeries
        for attr in ("colnames", "fields"):
            cols = getattr(container, attr, None)
            if cols is None:
                continue
            try:
                col_iter = list(cols)
            except TypeError:
                continue
            for col in col_iter:
                if isinstance(col, str) and col in _KNOWN_METRIC_NAMES:
                    found[col] = f"{path}/{col}"
                elif isinstance(col, str):
                    others.append(f"{path}/{col}")


def inventory_nwb(nwb_path: Path) -> NwbInventoryEntry:
    """Inspect one NWB and report what could be pre-computed metric source."""
    entry = NwbInventoryEntry(nwb_path=nwb_path)
    try:
        with open_nwb(nwb_path) as nwbfile:
            # Processing modules
            try:
                modules = dict(nwbfile.processing)
            except Exception:
                modules = {}
            entry.has_processing = bool(modules)
            entry.processing_modules = list(modules.keys())
            for mod_name, mod in modules.items():
                _scan_processing_module(
                    mod, entry.found_metrics, entry.other_processing_contents,
                    base_path=f"processing/{mod_name}",
                )
            # Lab-specific metadata
            try:
                lm = dict(nwbfile.lab_meta_data)
                entry.has_lab_meta_data = bool(lm)
            except Exception:
                entry.has_lab_meta_data = False
            # Scratch space
            try:
                sc = dict(nwbfile.scratch)
                entry.has_scratch = bool(sc)
            except Exception:
                entry.has_scratch = False
            # Epoch / interval tables
            try:
                iv = nwbfile.intervals
                entry.has_intervals = iv is not None and bool(getattr(iv, "items", lambda: [])())
            except Exception:
                entry.has_intervals = False
    except Exception as e:  # noqa: BLE001
        entry.error = f"{type(e).__name__}: {e}"
    return entry


def render_inventory_markdown(entries: list[NwbInventoryEntry], project_name: str) -> str:
    """Build the markdown report — one section per metric, plus per-NWB findings."""
    lines: list[str] = []
    lines.append(f"# Metric inventory — {project_name}")
    lines.append("")
    lines.append(f"Sampled {len(entries)} NWB(s) from the project's sources. ")
    lines.append("For each canonical QC metric the table below shows whether any of the ")
    lines.append("sampled files already carried a pre-computed value, or whether ")
    lines.append("`nwb-trace-qc` will compute it from raw traces.")
    lines.append("")

    # Aggregate: which metrics are found in any NWB?
    found_anywhere: dict[str, list[str]] = {}
    for entry in entries:
        for m, source in entry.found_metrics.items():
            found_anywhere.setdefault(m, []).append(f"{entry.nwb_path.name} → {source}")

    lines.append("## Metric provenance")
    lines.append("")
    lines.append("| Metric | Source | Notes |")
    lines.append("|---|---|---|")
    for metric in sorted(METRIC_DESCRIPTIONS.keys()):
        desc = METRIC_DESCRIPTIONS.get(metric, {})
        family = METRIC_TO_FAMILY.get(metric, "—")
        if metric in found_anywhere:
            source = "NWB (pre-computed)"
            notes = "; ".join(found_anywhere[metric][:3])
            if len(found_anywhere[metric]) > 3:
                notes += f"; …and {len(found_anywhere[metric]) - 3} more"
        else:
            source = "`nwb-trace-qc` (computed)"
            notes = f"{desc.get('what', '')}".strip()
        lines.append(f"| `{metric}` | {source} | {notes} |")
    lines.append("")

    # Per-NWB block
    lines.append("## Per-NWB findings")
    lines.append("")
    for entry in entries:
        lines.append(f"### `{entry.nwb_path.name}`")
        if entry.error:
            lines.append(f"- ⚠ failed to open: `{entry.error}`")
            lines.append("")
            continue
        lines.append(f"- `processing` modules: "
                     f"{', '.join(entry.processing_modules) if entry.processing_modules else '(none)'}")
        lines.append(f"- `lab_meta_data`: {'yes' if entry.has_lab_meta_data else '(none)'}")
        lines.append(f"- `scratch`: {'yes' if entry.has_scratch else '(none)'}")
        lines.append(f"- `intervals`: {'yes' if entry.has_intervals else '(none)'}")
        if entry.found_metrics:
            lines.append("- Pre-computed metrics found:")
            for m, src in sorted(entry.found_metrics.items()):
                lines.append(f"   - `{m}` → `{src}`")
        else:
            lines.append("- No pre-computed metrics matched the canonical names.")
        if entry.other_processing_contents:
            lines.append(f"- Other processing-module contents (not matched): "
                          f"`{'`, `'.join(entry.other_processing_contents[:8])}`")
            if len(entry.other_processing_contents) > 8:
                lines.append(f"  …and {len(entry.other_processing_contents) - 8} more")
        lines.append("")

    return "\n".join(lines)


def render_inventory_console(entries: list[NwbInventoryEntry]) -> str:
    """Compact console summary — one line per NWB + headline stats."""
    n = len(entries)
    n_with_processing = sum(1 for e in entries if e.has_processing)
    n_with_metrics = sum(1 for e in entries if e.found_metrics)
    all_found_metrics: set[str] = set()
    for e in entries:
        all_found_metrics.update(e.found_metrics.keys())
    lines = []
    lines.append(f"Sampled {n} NWB(s); {n_with_processing} carry processing modules; "
                  f"{n_with_metrics} have at least one pre-computed canonical metric.")
    if all_found_metrics:
        lines.append(f"Pre-computed metrics found across the sample: "
                      f"{', '.join(sorted(all_found_metrics))}")
    else:
        lines.append("No pre-computed canonical metrics found — "
                      "nwb-trace-qc will compute everything from raw traces.")
    lines.append("")
    for e in entries:
        suffix = (f"  → {len(e.found_metrics)} pre-computed: "
                   f"{', '.join(sorted(e.found_metrics.keys())[:4])}"
                   if e.found_metrics else "  → no pre-computed metrics")
        lines.append(f"  - {e.nwb_path.name}{suffix}")
    return "\n".join(lines)
