"""`nwb-qc inspect` — read-only inventory of a wrangler-output tree.

Walks a root path and reports what's there in a structured way so you can decide
whether to point `init-config` at it. Reads parquet schemas + Croissant/README/run-state
metadata; never opens NWB files (it just counts and sizes them).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


# ─── Data classes ────────────────────────────────────────────

@dataclass
class ParquetInfo:
    path: Path
    rows: int
    cols: list[str]
    qc_eligible: bool
    reason_not_eligible: str = ""

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class NWBAssets:
    path: Path
    count: int
    total_bytes: int


@dataclass
class SimpleFile:
    path: Path
    size: int
    summary: str = ""           # one-line summary (e.g. first heading, name field, etc.)


@dataclass
class SourceManifestInfo:
    path: Path
    schema_version: int | None = None
    generated_at: str = ""
    total_files: int = 0
    total_bytes: int = 0
    n_processed: int = 0
    preservation_included: bool | None = None
    # Cheap presence check: stat a small sample of original_locations
    sample_size: int = 0
    sample_present: int = 0


@dataclass
class DatasetEntry:
    """One discovered top-level subfolder of the inspection root."""
    name: str
    path: Path
    size_bytes: int
    n_nwbs_total: int
    parquets: list[ParquetInfo] = field(default_factory=list)
    nwb_dirs: list[NWBAssets] = field(default_factory=list)
    swc_count: int = 0
    fair2: SimpleFile | None = None
    readme: SimpleFile | None = None
    data_dictionary: SimpleFile | None = None
    run_state: SimpleFile | None = None
    source_manifest: SourceManifestInfo | None = None
    scripts: list[str] = field(default_factory=list)
    nested_subroot: Path | None = None  # the inner project dir (e.g. jy_vpl_…)


@dataclass
class InspectResult:
    root: Path
    root_size_bytes: int
    datasets: list[DatasetEntry] = field(default_factory=list)
    loose_nwbs: list[Path] = field(default_factory=list)  # NWBs sitting at root or otherwise unattached

    def total_nwbs(self) -> int:
        return sum(d.n_nwbs_total for d in self.datasets) + len(self.loose_nwbs)


# ─── Helpers ─────────────────────────────────────────────────

_QC_REQUIRED_COLS = {"nwb_file", "stimulus_type"}


def _human_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024 ** 2: return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3: return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


_STIMULUS_ALIASES = ("stimulus_type", "stimulus_description", "protocol", "stim_type", "stimulus")
_NWB_KEY_ALIASES = ("nwb_file", "nwb_path", "nwb")


def _summarize_parquet(p: Path) -> ParquetInfo:
    try:
        schema = pq.read_schema(p)
        cols = list(schema.names)
        rows = pq.read_metadata(p).num_rows
    except Exception as e:  # noqa: BLE001
        return ParquetInfo(path=p, rows=0, cols=[], qc_eligible=False,
                           reason_not_eligible=f"unreadable: {type(e).__name__}")
    has_nwb = "nwb_file" in cols
    has_stim = "stimulus_type" in cols
    if has_nwb and has_stim:
        return ParquetInfo(path=p, rows=rows, cols=cols, qc_eligible=True)
    # Try to suggest a column mapping when canonical names are absent but candidates exist
    nwb_alt = next((c for c in _NWB_KEY_ALIASES if c in cols and c != "nwb_file"), None)
    stim_alt = next((c for c in _STIMULUS_ALIASES if c in cols and c != "stimulus_type"), None)
    if has_nwb and stim_alt:
        return ParquetInfo(path=p, rows=rows, cols=cols, qc_eligible=False,
                           reason_not_eligible=f"map columns.stimulus_type → '{stim_alt}'")
    if nwb_alt and has_stim:
        return ParquetInfo(path=p, rows=rows, cols=cols, qc_eligible=False,
                           reason_not_eligible=f"map nwb_key_column → '{nwb_alt}'")
    if nwb_alt and stim_alt:
        return ParquetInfo(path=p, rows=rows, cols=cols, qc_eligible=False,
                           reason_not_eligible=f"map nwb_key_column → '{nwb_alt}', stimulus_type → '{stim_alt}'")
    missing = []
    if not has_nwb: missing.append("nwb_file (no candidate column found)")
    if not has_stim: missing.append("stimulus_type (no candidate column found)")
    return ParquetInfo(path=p, rows=rows, cols=cols, qc_eligible=False,
                       reason_not_eligible=f"missing: {'; '.join(missing)}")


def _summarize_fair2(p: Path) -> SimpleFile:
    size = p.stat().st_size
    try:
        with open(p) as f:
            data = json.load(f)
        name = data.get("name") or data.get("title") or "(unnamed)"
        # Croissant: conformsTo / @type
        kind = ""
        ct = data.get("conformsTo") or data.get("@type", "")
        if isinstance(ct, str) and "croissant" in ct.lower():
            kind = "Croissant 1.x"
        summary = f"{kind + ', ' if kind else ''}{name!s}"[:140]
        return SimpleFile(path=p, size=size, summary=summary)
    except Exception:
        return SimpleFile(path=p, size=size, summary="(JSON parse failed)")


def _summarize_readme(p: Path) -> SimpleFile:
    size = p.stat().st_size
    try:
        with open(p) as f:
            for line in f:
                line = line.strip().lstrip("#").strip()
                if line:
                    return SimpleFile(path=p, size=size, summary=line[:140])
    except Exception:
        pass
    return SimpleFile(path=p, size=size, summary="")


def _summarize_run_state(p: Path) -> SimpleFile:
    size = p.stat().st_size
    try:
        with open(p) as f:
            data = json.load(f)
        bits = []
        if "version" in data: bits.append(f"v{data['version']}")
        elif "wrangler_version" in data: bits.append(f"v{data['wrangler_version']}")
        if "status" in data: bits.append(str(data["status"]))
        if "timestamp" in data: bits.append(str(data["timestamp"]))
        elif "run_at" in data: bits.append(str(data["run_at"]))
        return SimpleFile(path=p, size=size, summary=" · ".join(bits) or "(no recognized fields)")
    except Exception:
        return SimpleFile(path=p, size=size, summary="(JSON parse failed)")


def _summarize_source_manifest(p: Path, sample: int = 5) -> SourceManifestInfo:
    info = SourceManifestInfo(path=p)
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return info
    info.schema_version = data.get("schema_version")
    info.generated_at = str(data.get("generated_at", ""))
    pres = data.get("preservation") or {}
    info.preservation_included = pres.get("included") if "included" in pres else None
    summary = data.get("summary") or {}
    info.total_files = int(summary.get("total_files") or 0)
    info.total_bytes = int(summary.get("total_size_bytes") or 0)
    info.n_processed = int(summary.get("processed_files") or 0)
    # Sample-check: do the first few original_locations actually exist?
    files = data.get("files") or []
    sample_list = files[:sample]
    info.sample_size = len(sample_list)
    for f in sample_list:
        loc = (f.get("original_location") or "").strip()
        if loc and Path(loc).expanduser().exists():
            info.sample_present += 1
    return info


def _summarize_data_dictionary(p: Path) -> SimpleFile:
    size = p.stat().st_size
    try:
        with open(p) as f:
            n_lines = sum(1 for _ in f)
        n_vars = max(0, n_lines - 1)
        return SimpleFile(path=p, size=size, summary=f"{n_vars} variable definition{'s' if n_vars != 1 else ''}")
    except Exception:
        return SimpleFile(path=p, size=size, summary="")


def _scan_subroot(subroot: Path) -> DatasetEntry:
    """Build a DatasetEntry for a top-level subfolder of the inspection root.

    Looks for either of two layouts:
      - subroot/{data,assets,scripts,fair2.json,...}    (flat wrangler-output layout)
      - subroot/<one-inner-dir>/{data,assets,...}       (wrapped layout, one project dir inside)
    """
    inner = subroot
    nested = None
    immediate_dirs = [p for p in subroot.iterdir() if p.is_dir()]
    immediate_has_signal = any(
        (subroot / x).exists() for x in ("data", "assets", "fair2.json", "README.md", "run_state.json")
    )
    if not immediate_has_signal and len(immediate_dirs) == 1:
        inner = immediate_dirs[0]
        nested = inner

    entry = DatasetEntry(
        name=subroot.name,
        path=subroot,
        size_bytes=_dir_size(subroot),
        n_nwbs_total=0,
        nested_subroot=nested,
    )

    # Top-level files of interest
    for fname, attr, summarizer in (
        ("fair2.json", "fair2", _summarize_fair2),
        ("README.md", "readme", _summarize_readme),
        ("data_dictionary.csv", "data_dictionary", _summarize_data_dictionary),
        ("run_state.json", "run_state", _summarize_run_state),
    ):
        p = inner / fname
        if p.is_file():
            setattr(entry, attr, summarizer(p))

    # source_material/source_manifest.json
    smp = inner / "source_material" / "source_manifest.json"
    if smp.is_file():
        entry.source_manifest = _summarize_source_manifest(smp)

    # Scripts
    scripts_dir = inner / "scripts"
    if scripts_dir.is_dir():
        entry.scripts = sorted(p.name for p in scripts_dir.glob("*.py"))

    # Parquets (recursive under the inner dir)
    for pq_path in sorted(inner.rglob("*.parquet")):
        entry.parquets.append(_summarize_parquet(pq_path))

    # NWB asset directories (recursive) — count both HDF5 (.nwb files) and Zarr (.nwb.zarr dirs)
    nwb_dirs: dict[Path, list[tuple[Path, int]]] = {}     # parent → [(child, size_bytes)]
    # HDF5
    for nwb in inner.rglob("*.nwb"):
        if nwb.is_file():
            try:
                nwb_dirs.setdefault(nwb.parent, []).append((nwb, nwb.stat().st_size))
            except OSError:
                continue
    # Zarr (each *.nwb.zarr/ is one logical NWB; sum its inner bytes)
    for zd in inner.rglob("*.nwb.zarr"):
        if zd.is_dir():
            try:
                size = sum(p.stat().st_size for p in zd.rglob("*") if p.is_file())
            except OSError:
                size = 0
            nwb_dirs.setdefault(zd.parent, []).append((zd, size))
    for d, items in sorted(nwb_dirs.items()):
        total = sum(sz for _, sz in items)
        entry.nwb_dirs.append(NWBAssets(path=d, count=len(items), total_bytes=total))
    entry.n_nwbs_total = sum(d.count for d in entry.nwb_dirs)

    # SWC count (if any)
    entry.swc_count = sum(1 for _ in inner.rglob("*.swc"))

    return entry


def inspect_root(root: Path) -> InspectResult:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = InspectResult(root=root, root_size_bytes=_dir_size(root))
    top = [p for p in sorted(root.iterdir()) if p.is_dir()]
    for sub in top:
        entry = _scan_subroot(sub)
        # Skip subfolders that have no NWBs AND no parquets AND no metadata — probably noise
        if (entry.n_nwbs_total == 0 and not entry.parquets and not entry.source_manifest
                and not (entry.fair2 or entry.readme or entry.run_state)):
            continue
        result.datasets.append(entry)
    # Loose NWBs at root
    result.loose_nwbs = sorted(p for p in root.glob("*.nwb") if p.is_file())
    return result


# ─── Renderers ───────────────────────────────────────────────

def render_terminal(r: InspectResult, max_parquet_lines: int = 8) -> str:
    out: list[str] = []
    out.append(f"nwb-qc inspect {r.root}")
    out.append("─" * 60)
    out.append(f"Root: {r.root}   ({len(r.datasets)} dataset entries, {_human_bytes(r.root_size_bytes)})")
    out.append("")
    if not r.datasets and not r.loose_nwbs:
        out.append("  (no datasets, parquets, or NWBs detected)")
        return "\n".join(out)

    for i, d in enumerate(r.datasets, 1):
        out.append(f"[{i}] {d.name}/  ·  {_human_bytes(d.size_bytes)}")
        inner_label = (d.nested_subroot.name + "/") if d.nested_subroot else ""
        if inner_label:
            out.append(f"    └─ {inner_label}")
            prefix = "       "
        else:
            prefix = "    "
        if d.readme: out.append(f"{prefix}├─ README.md            {_human_bytes(d.readme.size):>9}  \"{d.readme.summary}\"")
        if d.fair2: out.append(f"{prefix}├─ fair2.json           {_human_bytes(d.fair2.size):>9}  {d.fair2.summary}")
        if d.data_dictionary: out.append(f"{prefix}├─ data_dictionary.csv  {_human_bytes(d.data_dictionary.size):>9}  {d.data_dictionary.summary}")
        if d.run_state: out.append(f"{prefix}├─ run_state.json       {_human_bytes(d.run_state.size):>9}  {d.run_state.summary}")
        if d.source_manifest:
            sm = d.source_manifest
            pres = "not copied" if sm.preservation_included is False else ("copied" if sm.preservation_included else "?")
            sample = f"sample {sm.sample_present}/{sm.sample_size} present on disk" if sm.sample_size else "(no files)"
            out.append(
                f"{prefix}├─ source_material/source_manifest.json   "
                f"schema v{sm.schema_version} · {sm.total_files} files · {_human_bytes(sm.total_bytes)} · "
                f"preservation: {pres} · {sample}"
            )
        if d.parquets:
            out.append(f"{prefix}├─ parquet/")
            for pq_info in d.parquets[:max_parquet_lines]:
                badge = "✓ QC-eligible" if pq_info.qc_eligible else f"({pq_info.reason_not_eligible})"
                out.append(f"{prefix}│  ├─ {pq_info.name:<40} {pq_info.rows:>9,} rows · {len(pq_info.cols):>2} cols  {badge}")
            if len(d.parquets) > max_parquet_lines:
                out.append(f"{prefix}│  └─ … and {len(d.parquets) - max_parquet_lines} more")
        if d.nwb_dirs:
            total_bytes = sum(n.total_bytes for n in d.nwb_dirs)
            if len(d.nwb_dirs) == 1:
                nd = d.nwb_dirs[0]
                rel = nd.path.relative_to(d.path) if nd.path != d.path else Path(".")
                out.append(f"{prefix}├─ {rel}/{'':<{max(0, 35-len(str(rel)))}} {nd.count:>5} NWBs · {_human_bytes(nd.total_bytes)}")
            else:
                out.append(f"{prefix}├─ NWBs in {len(d.nwb_dirs)} sub-directories  · {d.n_nwbs_total} files · {_human_bytes(total_bytes)}")
        if d.swc_count:
            out.append(f"{prefix}├─ assets/swc/                          {d.swc_count} SWC files")
        if d.scripts:
            out.append(f"{prefix}└─ scripts/                             {', '.join(d.scripts)}")
        out.append("")

    if r.loose_nwbs:
        out.append(f"Loose NWBs at root: {len(r.loose_nwbs)}")
        for p in r.loose_nwbs[:5]:
            out.append(f"  · {p.name}")
        if len(r.loose_nwbs) > 5:
            out.append(f"  · … and {len(r.loose_nwbs) - 5} more")
        out.append("")

    # Summary
    total_qc = sum(1 for d in r.datasets for p in d.parquets if p.qc_eligible)
    out.append("Summary")
    out.append("───────")
    out.append(f"  {len(r.datasets)} datasets · {r.total_nwbs():,} NWB files · {_human_bytes(r.root_size_bytes)} total")
    out.append(f"  {total_qc} acquisition-table parquet{'s' if total_qc != 1 else ''} would be registered by `init-config`")
    wranglers = [d.run_state.summary for d in r.datasets if d.run_state]
    if wranglers:
        out.append("  Wrangler run state:")
        for w in wranglers:
            out.append(f"    · {w}")
    out.append("")
    out.append(f"Next: nwb-qc init-config {r.root}")
    out.append("")
    return "\n".join(out)


def render_markdown(r: InspectResult) -> str:
    lines: list[str] = []
    lines.append(f"# Inventory of `{r.root}`")
    lines.append("")
    lines.append(f"- Total size: **{_human_bytes(r.root_size_bytes)}**")
    lines.append(f"- Datasets detected: **{len(r.datasets)}**")
    lines.append(f"- NWB files total: **{r.total_nwbs():,}**")
    lines.append("")
    for i, d in enumerate(r.datasets, 1):
        lines.append(f"## {i}. `{d.name}/`  ·  {_human_bytes(d.size_bytes)}")
        if d.nested_subroot:
            lines.append(f"_inner project directory_: `{d.nested_subroot.name}/`")
        lines.append("")
        if d.readme:
            lines.append(f"**README.md** ({_human_bytes(d.readme.size)}) — {d.readme.summary}")
        if d.fair2:
            lines.append(f"**fair2.json** ({_human_bytes(d.fair2.size)}) — {d.fair2.summary}")
        if d.data_dictionary:
            lines.append(f"**data_dictionary.csv** ({_human_bytes(d.data_dictionary.size)}) — {d.data_dictionary.summary}")
        if d.run_state:
            lines.append(f"**run_state.json** ({_human_bytes(d.run_state.size)}) — {d.run_state.summary}")
        if d.source_manifest:
            sm = d.source_manifest
            pres = "not copied" if sm.preservation_included is False else ("copied" if sm.preservation_included else "?")
            sample = f"sample {sm.sample_present}/{sm.sample_size} present on disk" if sm.sample_size else "(no files)"
            lines.append(
                f"**source_material/source_manifest.json** — schema v{sm.schema_version}; "
                f"{sm.total_files} files; {_human_bytes(sm.total_bytes)}; preservation: {pres}; {sample}"
            )
        lines.append("")
        if d.parquets:
            lines.append("### Parquet tables")
            lines.append("")
            lines.append("| file | rows | columns | QC-eligible |")
            lines.append("|---|---:|---|---|")
            for p in d.parquets:
                col_list = ", ".join(p.cols[:8]) + (f", …(+{len(p.cols)-8})" if len(p.cols) > 8 else "")
                badge = "✓" if p.qc_eligible else f"✗ ({p.reason_not_eligible})"
                lines.append(f"| `{p.name}` | {p.rows:,} | `{col_list}` | {badge} |")
            lines.append("")
        if d.nwb_dirs:
            lines.append("### NWB assets")
            lines.append("")
            lines.append("| directory | files | size |")
            lines.append("|---|---:|---:|")
            for nd in d.nwb_dirs:
                rel = nd.path.relative_to(d.path)
                lines.append(f"| `{rel}` | {nd.count:,} | {_human_bytes(nd.total_bytes)} |")
            lines.append("")
        if d.swc_count:
            lines.append(f"**Morphology files (SWC):** {d.swc_count}")
            lines.append("")
        if d.scripts:
            lines.append(f"**Scripts:** {', '.join('`'+s+'`' for s in d.scripts)}")
            lines.append("")
    if r.loose_nwbs:
        lines.append(f"## Loose NWBs at root ({len(r.loose_nwbs)})")
        for p in r.loose_nwbs:
            lines.append(f"- `{p.name}`")
        lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append(f"```bash\nnwb-qc init-config {r.root}\nnwb-qc list-cells --config <output>_project.yaml\n```")
    return "\n".join(lines) + "\n"


def render_json(r: InspectResult) -> str:
    """Machine-readable form for piping."""
    def _entry(d: DatasetEntry) -> dict[str, Any]:
        return {
            "name": d.name,
            "path": str(d.path),
            "size_bytes": d.size_bytes,
            "n_nwbs": d.n_nwbs_total,
            "nested_subroot": str(d.nested_subroot) if d.nested_subroot else None,
            "parquets": [
                {"name": p.name, "path": str(p.path), "rows": p.rows, "cols": p.cols,
                 "qc_eligible": p.qc_eligible, "reason_not_eligible": p.reason_not_eligible}
                for p in d.parquets
            ],
            "nwb_dirs": [
                {"path": str(nd.path), "count": nd.count, "bytes": nd.total_bytes}
                for nd in d.nwb_dirs
            ],
            "swc_count": d.swc_count,
            "fair2_summary": d.fair2.summary if d.fair2 else None,
            "readme_summary": d.readme.summary if d.readme else None,
            "data_dictionary_summary": d.data_dictionary.summary if d.data_dictionary else None,
            "run_state_summary": d.run_state.summary if d.run_state else None,
            "source_manifest": (
                None if not d.source_manifest else {
                    "path": str(d.source_manifest.path),
                    "schema_version": d.source_manifest.schema_version,
                    "generated_at": d.source_manifest.generated_at,
                    "total_files": d.source_manifest.total_files,
                    "total_bytes": d.source_manifest.total_bytes,
                    "n_processed": d.source_manifest.n_processed,
                    "preservation_included": d.source_manifest.preservation_included,
                    "sample_size": d.source_manifest.sample_size,
                    "sample_present": d.source_manifest.sample_present,
                }
            ),
            "scripts": d.scripts,
        }
    return json.dumps({
        "root": str(r.root),
        "root_size_bytes": r.root_size_bytes,
        "total_nwbs": r.total_nwbs(),
        "datasets": [_entry(d) for d in r.datasets],
        "loose_nwbs": [str(p) for p in r.loose_nwbs],
    }, indent=2)
