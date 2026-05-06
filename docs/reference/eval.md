---
layout: default
title: LLM Evaluation Suite
parent: Reference
nav_order: 5
---

# LLM Evaluation Suite (`soak eval`)

The eval suite is a small, repeatable benchmark for the LLM contract that
soaking depends on. It exists because weaker models can silently mangle
hash references, drop required fields, or produce semantically poor
codes/themes — symptoms that don't show up as errors but degrade results.

Three things live under one roof:

- **Probes** — fixed templates (`probe_consolidate.sd`,
  `probe_long_context.sd`, etc.) run against any model, with
  deterministic checks (schema validity, hash fidelity).
- **Snapshots** — frozen single-stage cases captured from real runs,
  replayable against any model.
- **Reports** — markdown + HTML side-by-side comparison tables.

The same underlying machinery (`soak.evals.run_probe`) is callable from
the `soak` CLI, the soakresearch Django management commands, and the
pytest suite (`pytest -m llm tests/evals`).


## Quick start

```bash
# Cheapest possible probe against one model
soak eval run --probe schema --model gpt-5-mini

# Sweep multiple probes × multiple models in parallel
soak eval run --probe schema,consolidate \
  --model gpt-5-mini,gpt-4.1-mini,claude-haiku-4-5 \
  --max-concurrent 5

# Build the comparison report (markdown + HTML)
python -m soak.evals.report --out ./soaking-eval/llm_evals.md
# or
python scripts/eval_report.py
```

Outputs land in `./soaking-eval/` by default:

- `soaking-eval/results/<YYYY-MM-DD>.jsonl` — append-only history, one
  line per (probe, model) call
- `soaking-eval/llm_evals.md` — markdown comparison report
- `soaking-eval/llm_evals.html` — pandoc-rendered HTML (auto, if pandoc
  is on `PATH`)

Override the location with the `SOAK_EVAL_RESULTS=/path` env var or
the `--results` flag on the report command.


## `soak eval` subcommands

| Subcommand | Purpose |
|---|---|
| `soak eval run` | Run probes against models, record metrics |
| `soak eval replay` | Replay a snapshot against a model |
| `soak eval snapshot` | Build a snapshot from a `soak run` JSON output |
| `soak eval list` | List committed snapshots |


## `soak eval run` — running probes

### Specifying probes and models

```bash
soak eval run --probe schema --model gpt-5-mini
```

Both flags accept **CSV lists** for sweeps:

```bash
soak eval run \
  --probe schema,consolidate,long_context \
  --model gpt-5-mini,gpt-4.1-mini,claude-haiku-4-5
```

This runs the cartesian product (3 probes × 3 models = 9 calls).

### Available probes

| Probe | What it tests | Cost |
|---|---|---|
| `schema` | Single-call code extraction. Tool-calling smoke test. | Cheap (~$0.001) |
| `consolidate` | Codes → reference-mode consolidation → themes. Hash fidelity end-to-end. | Moderate (~$0.01) |
| `long_context` | Reference-mode consolidate from a 50-code fixture. Tests hash truncation under length. | Higher |
| `themes_long` | Theme generation from a 50-code fixture. Tests `Theme.code_hashes` fidelity at scale. | Higher |

Run `soak eval run --help` for the full current list (it's drawn from
`soak.evals.AVAILABLE_PROBES`).

### Reproducibility

Two flags control determinism:

```bash
soak eval run -p schema -m gpt-5-mini --seed 1 --temperature 0
```

- `--seed` (default `1`) is the model sampling seed. Pinning it makes
  re-runs reproducible — same prompt + same seed → same output → same
  joblib cache key → comparable cost/latency between runs.
- `--temperature` (default `0.0`) for deterministic decoding on models
  that honor it.

The seed and temperature are recorded in each JSONL row so the report
can show what produced the result.

### Parallelism

```bash
soak eval run -p schema -m m1,m2,m3,m4,m5 --max-concurrent 5
```

`--max-concurrent N` (default `5`) runs N (probe, model) pairs in
parallel via a thread pool. struckdown's internal MAX_CONCURRENCY
semaphore continues to bound in-call concurrency separately.

Set `--max-concurrent 1` for strictly serial execution if you're
debugging or hitting provider rate limits.

### Credentials

`soak eval run` reads `LLM_API_KEY` and `LLM_API_BASE` from the
environment, or accepts `--api-key` / `--api-base`. For the soakresearch
web app there's a sibling command, `manage.py eval_probe`, that resolves
keys from the encrypted `Credential` table.

### Recording

By default each run appends to
`./soaking-eval/results/<YYYY-MM-DD>.jsonl`. Pass `--no-record` to skip
recording (e.g. for ad-hoc one-offs you don't want polluting history).

### Exit codes

- `0` — every (probe, model) pair passed `schema_valid` and any hash checks
- `1` — at least one pair failed (sweep continues, every pair is still recorded)
- `2` — bad arguments (unknown probe, missing API key, etc.)


## Reports

```bash
# Markdown + HTML (HTML auto-generated if pandoc is available)
python scripts/eval_report.py

# Or via the Django command if you want it from soakresearch:
manage.py eval_report
```

Each row in the report is the **most recent** entry per (probe, model)
across all `*.jsonl` files in `./soaking-eval/results/`. Re-running a
probe overwrites the previous entry in the report.

The report includes:

- One table per probe with metrics (schema-valid, hash refs, coverage,
  cost, latency)
- A collapsible **Examples** section per probe with model-by-model
  side-by-side comparison: codes show name + description + supporting
  quotes; themes show name + description + the resolved code names for
  each `code_hash` reference.

### Output format

- The default markdown uses HTML tables for the examples section so
  pandoc / GitHub render the wide side-by-side comparison cleanly.
- Run `python scripts/eval_report.py --no-html` to skip the pandoc step.
- The markdown uses real Unicode curly quotes (U+201C/201D) and box
  characters so the raw source is human-readable too.


## Snapshots and replay

A *snapshot* is a frozen single-stage call: enough information to
re-run one slot of one node against any chosen model, without standing
up the full pipeline. Use snapshots to capture interesting cases from
production runs and replay them against future model candidates.

### Format

```
soaking-eval/snapshots/<name>/
  template.sd       # the prompt template (jinja, with [[type:slot]] markers)
  inputs.json       # context dict for complete()
  expected.json     # baseline outputs keyed by slot name
  metadata.json     # source ref, model, date, scrub status, schema version
  README.md         # why this case is interesting
```

### Building a snapshot

From a `soak run` output JSON:

```bash
soak eval snapshot \
  --from-cli /path/to/run_output.json \
  --template /path/to/template.sd \
  --as my-case --stage themes
```

From a soakresearch `AnalysisRun` (uses Django credentials, scrubs
emails/phones by default):

```bash
manage.py snapshot_run <run_uuid> --as my-case --stage theme_groups
manage.py snapshot_run <run_uuid> --as my-case --stage theme_groups --keep-pii
```

### Replaying

```bash
soak eval replay ./soaking-eval/snapshots/my-case --model gpt-5-mini
soak eval replay ./soaking-eval/snapshots/my-case --model gpt-4.1
```

Replay loads the snapshot, runs `complete()` against the named model,
and prints the actual outputs alongside a summary diff against
`expected.json`. No full pipeline is involved — this is intentionally
cheap.

### Listing snapshots

```bash
soak eval list
```


## File layout summary

```
./
├── soaking-eval/                # auto-created in CWD
│   ├── results/
│   │   └── 2026-05-06.jsonl     # one line per probe×model
│   ├── snapshots/
│   │   └── <name>/              # 5 files per snapshot
│   ├── llm_evals.md             # report markdown
│   └── llm_evals.html           # pandoc HTML
└── ...
```

To redirect: `SOAK_EVAL_RESULTS=/path/to/dir` env var, or pass
`--results /path` to the report tool.


## Pytest gate (`pytest -m llm`)

```bash
pytest -m llm tests/evals                    # gate: hard-asserts pass
pytest -m llm tests/evals --eval-mode        # sweep: record only, no asserts
SOAK_EVAL_MODELS=gpt-5-mini,gpt-4.1-mini \
  pytest -m llm tests/evals --eval-mode      # sweep specific models
```

Two flavours of test:

- `test_*_gate` runs against the pinned model in `SOAK_GATE_MODEL`
  (default `gpt-5.1-mini`) with hard assertions. Fails CI on
  prompt drift, schema bugs, struckdown changes.
- `test_probe_sweep` parametrises over every probe × every model in
  `SOAK_EVAL_MODELS`. Skipped unless `--eval-mode` (or
  `SOAK_EVAL_MODE=1`); records JSONL without asserting.


## Adding a new probe

1. Create `soak/evals/data/probe_<name>.sd` (a struckdown template).
2. In `soak/evals/probes.py`, add a `_run_<name>_checks()` function and
   register it in `AVAILABLE_PROBES` alongside the template name.
3. If the probe needs pre-loaded fixture data, add a
   `context_loader` callable that returns a `dict` of jinja vars.
4. (Optional) add a per-probe formatter in `soak/evals/report.py` so
   the markdown table shows the right columns.
