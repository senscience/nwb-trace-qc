# Usage — running `nwb-trace-qc` on a dataset

A step-by-step walkthrough from raw NWB folder to interactive report. Three commands cover 99% of the workflow; the rest of this page explains what each does, what to look at, and how to iterate.

## 1. Auto-discover and write a config

Point `init-config` at any folder that contains NWB files (and, optionally, wrangler parquet tables sharing that root). It walks the tree, groups NWBs by top-level subfolder, detects acquisition parquets by schema, and writes a ready-to-run YAML.

```bash
nwb-qc init-config /path/to/your/data
```

Output (example):

```
Wrote ./mydata_project.yaml
      (3 sources, 2 acquisition tables, thresholds at mydata_thresholds.yaml)
Next: nwb-qc list-cells --config ./mydata_project.yaml
```

Two files appear next to your current directory:

- `mydata_project.yaml` — the project config. NWB sources, acquisition tables, stimulus-protocol families, output paths, worker count, and a path to your thresholds file.
- `mydata_thresholds.yaml` — a copy of the bundled defaults so you can edit per-project without touching the global ones.

If your folder has subfolders (`cohort_a/`, `cohort_b/`), each subfolder containing NWBs becomes one `nwb_sources` entry, named after the subfolder. If your wrangler output is also under that root and includes parquets with `nwb_file` + `stimulus_type` columns, they get registered as `acquisition_tables` automatically.

**Options:**

- `--name myname` — override the project name (default = folder basename).
- `--output path/to/file.yaml` — override the YAML filename/location. A directory path is also accepted.
- `--no-guess-tables` — skip the parquet scan if you don't want auto-registration.

If you're running from inside the repo root (where a `configs/` folder exists), the default output location becomes `configs/<name>_project.yaml` instead of the current directory.

## 2. Sanity-check what will be processed (no compute)

```bash
nwb-qc list-cells --config mydata_project.yaml
```

Output:

```
Project: mydata
Sources: 3
  - cohort_a: 412 NWBs at /path/to/your/data/cohort_a
  - cohort_b: 18  NWBs at /path/to/your/data/cohort_b
  - cohort_c: 7   NWBs at /path/to/your/data/cohort_c
Total NWB rows: 437
Unique by sha256: 437 (dedup saves 0 compute steps)
Acquisition tables registered: 2
Cell table: /path/to/your/data/cells.csv
```

This is your moment to confirm the counts look right. If any source is missing or has the wrong count, edit `mydata_project.yaml` — most often you're adjusting the `glob:` for nested archive layouts (e.g. `"**/data/*/*.nwb"` for OBI-style archives) or removing a path that was wrongly picked up.

Also worth reviewing in the YAML before the first real run:

- `stimulus_protocols:` — verify your lab's protocol names appear in the right family. Default is LNMC/BBP; if your lab uses, say, `BL_hold` for the baseline recording, add it to the `spontaneous_hold` list.
- `thresholds_file:` — peek at `mydata_thresholds.yaml`; loosen or tighten if your preparation differs from juvenile rodent cortical/thalamic patch-clamp.

## 3. Run the pipeline

```bash
nwb-qc run --config mydata_project.yaml
```

What happens, in order:

1. **Manifest** — sha256-hash every NWB.
2. **Cache lookup** — skip any NWB whose hash is already in `_qc_cache.parquet` (empty on first run; populated thereafter).
3. **Metric compute** — open new NWBs in parallel (`n_workers`), extract Vrest / Rs / AP overshoot / etc., append to cache.
4. **Apply thresholds** → `pass` / `flag` / `fail` per cell.
5. **Apply overrides** from `qc_overrides.csv` (empty on first run).
6. **Render thumbnails** for non-pass cells (cached on disk, skipped on re-runs).
7. **Write report** (HTML + CSV).

Output is a single JSON line:

```json
{
  "n_cells": 437,
  "n_unique_nwbs": 437,
  "n_computed_this_run": 437,
  "n_nwb_opens": 437,
  "n_pass": 102,
  "n_flag": 88,
  "n_fail": 247,
  "elapsed_s": 1842.3,
  "report": "/path/to/qc_output_mydata/qc_report.html"
}
```

Rough timing: a few seconds per NWB single-threaded, divided by `n_workers`. The JY 2,302-NWB run took 2 h 11 m on 6 workers; a 30-cell smoke test takes ~90 s.

## 4. Open the report

```bash
open /path/to/qc_output_mydata/qc_report.html
```

Defaults to **Fail + Flag** only — passes are hidden until you toggle them. Per-cell expand panel shows the full metric table, the specific triggers, and inline trace thumbnails of the offending sweeps. No external network requests; you can copy the HTML to a colleague's machine and it still works.

## 5. Iterate

Three things you might do next, none requires touching code.

### a) Tune thresholds

Edit `mydata_thresholds.yaml`. Re-run:

```bash
nwb-qc run --config mydata_project.yaml
```

Manifest hashing happens again (~few minutes for thousands of NWBs) but **no NWBs are reopened** — the cache hits — so the bulk of the wall-clock is just report rendering. Verdicts update according to the new rules.

### b) Restrict to one dataset for a smoke test

```bash
nwb-qc run --config mydata_project.yaml --filter dataset=cohort_a
```

### c) Stick a human verdict for a cell

Append to `qc_output_mydata/qc_overrides.csv`:

```csv
cell_id,override_verdict,note,reviewer,date
sample_42,pass,manually inspected — overshoot loss is end-of-recording artefact,you,2026-06-03
```

That row's verdict becomes whatever you say it is on the next run, regardless of thresholds. The report renders an "Override active" banner with your note. The expand-row panel in the HTML shows a copyable template line per cell to make this easy.

## The shortest possible loop

Once everything's configured:

```bash
nwb-qc run --config mydata_project.yaml && open qc_output_mydata/qc_report.html
```

That's the whole workflow. Iterate on thresholds and overrides; the cache makes every run after the first one fast.

## Visual-defect metrics (v0.2.0)

In addition to the scalar metrics, the pipeline computes five trace-shape metrics that catch visual patterns scalar metrics miss:

| Metric | What it catches |
|---|---|
| `rac_decay_residual_rel` | Test-pulse / step-response recovery that's glitchy or ringing instead of a clean exponential |
| `vm_drift_within_sweep_mv_per_s` | Drifting seal — within-sweep Vm slope (e.g. −70 mV → −20 mV over one long sweep) |
| `ap_failure_fraction` | Spikes that initiate (dV/dt threshold crossing) but never reach overshoot |
| `ap_amp_cv` | AP peak amplitudes that vary inconsistently within one train |
| `late_instability_index` | Orderly firing degrading to runaway oscillation in the latter portion of long sweeps |

Their thresholds live in `default_thresholds.yaml` alongside the scalar ones; the rule grammar (`fail_above`, `flag_above`, etc.) is unchanged. See the file for current default values.

## Optional LLM vision judge

Off by default. When enabled, only **borderline** cells (rule-based verdict = `flag`) get sent to a vision model for a second opinion. Cells that clearly pass or fail by the numeric rules skip the vision pass entirely, so the API cost is bounded.

Enable per-project in your YAML:

```yaml
vision_judge:
  enabled: true
  provider: anthropic              # 'anthropic' | 'openai' | 'mock'
  model: claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY   # the env var holding your key
  max_borderline_cells: 100
  prompt_template: null            # null = bundled default
  cache_responses: true
```

Both the `anthropic` and `openai` SDKs ship with `nwb-trace-qc` by default, so no extra install is needed — just set the API key for whichever provider your `vision_judge.provider` points at and run:

```bash
export ANTHROPIC_API_KEY=...
nwb-qc run --config mydata_project.yaml --with-vision
```

**Verdict precedence** (each later step wins):

1. Rule-based verdict (vrest / Rs / AP / visual-defect metrics).
2. Vision judge:
   - rules `flag` + vision `fail` → final `fail` (reason: `vision_escalated`).
   - rules `flag` + vision `pass` → final stays `flag` (reason: `vision_suggests_pass`) — vision can't auto-pass a borderline cell, only flag for review.
   - rules `fail` is never downgraded; rules `pass` is never escalated by vision.
3. Human override in `qc_overrides.csv` — always wins.

The HTML report's per-cell expand panel shows both the vision verdict (blue banner with confidence + notes) and any override (yellow banner). Mock provider is bundled for tests — set `provider: mock` to exercise the integration without API calls.

## Interactive trace viewer (`nwb-qc serve`)

The static `qc_report.html` is shareable but only renders the auto-thumbnails of offending sweeps. For visual verification of any sweep on demand, run:

```bash
nwb-qc serve --config mydata_project.yaml
```

This starts a localhost-only HTTP server (default port 8765) and opens your browser to the interactive viewer. The viewer reads `qc_report.csv` for the cell list and lazy-loads sweep data from the underlying NWBs on demand — only what you click is decoded.

Layout:

- **Left**: cell list with verdict chips, filterable by `cell_id`, sorted fail/flag first.
- **Centre-left**: sweeps for the selected cell, grouped by stimulus family.
- **Right**: 2 × 2 grid of plot panels. Click a sweep to swap it into the next free slot; click "×" to clear a slot.

Plots are rendered with vanilla Canvas (no external JS), include a dashed 0 mV reference line, and decimate via LTTB to ~2,500 points per sweep regardless of source sampling rate. Trace data is fetched via `/api/trace/<cell_id>/<sweep_idx>?max_points=N` — you can hit the API directly for scripting:

```bash
curl http://127.0.0.1:8765/api/sweeps/JY160222_A_1   # list sweeps
curl 'http://127.0.0.1:8765/api/trace/JY160222_A_1/0?max_points=1000'
```

Stop the server with Ctrl-C. The viewer requires `nwb-qc run` to have been executed at least once (it reads the manifest + CSV the run writes); after that the underlying NWB files must remain accessible at their original paths.

## Other CLI subcommands

| Command | What it does |
|---|---|
| `nwb-qc report --config <file>` | Re-render the HTML/CSV from the existing cache without doing any NWB I/O. Useful after editing the report template or doing nothing-else-changed runs. |
| `nwb-qc thresholds --config <file> --dry-run` | Show how the current thresholds would classify cached cells (verdict counts only) without writing the report. |
| `nwb-qc serve --config <file> [--port N] [--no-browser]` | Interactive trace viewer (see above). |
| `nwb-qc --version` | Print the package version. |

## Troubleshooting

- **"thresholds_file not found"** — `init-config` couldn't find the bundled defaults to copy. Either run from a directory where the repo's `configs/default_thresholds.yaml` is reachable, or point `thresholds_file:` in your project YAML at an absolute path.
- **All cells flagged on `qc_protocol_coverage`** — your `stimulus_protocols` mapping doesn't match your data's protocol names. Open one NWB with `pynwb`, list `acquisition.keys()`, and add the relevant tokens to the right family in your project YAML.
- **Absolute Rs values look unrealistic** — the Rs estimator currently assumes a nominal 50 pA test pulse (it reads only the voltage trace, not the paired stimulus current). Trust `rs_drift_pct` over `rs_mohm_final` until proper Rs from the paired stimulus is implemented; relax or remove the `fail_above` rule on `rs_mohm_final`.
- **First run feels slow** — it's I/O-bound when NWBs aren't in the OS page cache. Re-runs hit the cache and skip metric compute entirely; only manifest hashing remains.

See [`jy_quickstart.md`](jy_quickstart.md) for a worked example with real numbers.
