"""`soak eval` -- LLM capability eval suite (probes + snapshot/replay).

Subcommands:

- ``soak eval run --probe <name> --model <name>``
  Run an LLM capability probe (the Phase 1 entry point). Reads
  ``LLM_API_KEY`` / ``LLM_API_BASE`` from the environment.
- ``soak eval replay <dir> --model <name>``
  Replay a snapshot against a chosen model. Cheap -- no full pipeline.
- ``soak eval snapshot --from-cli <run_output.json> --as <name> [--stage X]``
  Build a snapshot directory from a ``soak run`` output JSON.

The snapshot format lives in ``soak.evals.snapshot``. See the docstring
there for the on-disk layout and provenance fields.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from soak.evals import (AVAILABLE_PROBES, SnapshotMetadata, list_snapshots,
                        load_snapshot, record_metrics, run_probe,
                        write_snapshot)

eval_app = typer.Typer(
    name="eval",
    help="LLM capability eval suite (probes + snapshots).",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


# --- shared helpers ----------------------------------------------------------


def _resolve_credentials(api_key: Optional[str], api_base: Optional[str]):
    """Build (LLM, LLMCredentials) from CLI args / env. Exit on missing key."""
    from struckdown.llm import LLM, LLMCredentials  # local import: keeps CLI fast

    key = api_key or os.environ.get("LLM_API_KEY")
    base = api_base or os.environ.get("LLM_API_BASE") or None
    if not key:
        typer.echo(
            "error: no API key. Set LLM_API_KEY or pass --api-key.",
            err=True,
        )
        raise typer.Exit(2)
    return key, base


# --- run (probe) -------------------------------------------------------------


@eval_app.command("run")
def run(
    probes: str = typer.Option(
        "consolidate",
        "--probe",
        "-p",
        help=(
            "Probe name (or CSV list). Available: "
            f"{', '.join(sorted(AVAILABLE_PROBES))}"
        ),
    ),
    models: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model name, or CSV list (e.g. 'gpt-5-mini,gpt-4.1-mini').",
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override LLM_API_KEY."),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="Override LLM_API_BASE."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON metrics to this path (default: stdout).",
    ),
    timeout: int = typer.Option(
        600, "--timeout", help="Wall-clock timeout in seconds.",
    ),
    seed: int = typer.Option(
        1,
        "--seed",
        help=(
            "Sampling seed -- pin to make re-runs reproducible "
            "(comparable cost/latency)."
        ),
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", help="Sampling temperature (default 0.0).",
    ),
    record: bool = typer.Option(
        True,
        "--record/--no-record",
        help=(
            "Append metrics to ./soaking-eval/results/<date>.jsonl so "
            "the report tool can pick them up. Default on."
        ),
    ),
    max_concurrent: int = typer.Option(
        5,
        "--max-concurrent",
        help=(
            "How many (probe, model) pairs to run in parallel. Default 5. "
            "Set to 1 for serial. Higher values risk provider rate limits."
        ),
    ),
):
    """Run probe(s) against model(s); cartesian product, recorded to JSONL."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from struckdown.llm import LLM, LLMCredentials

    key, base = _resolve_credentials(api_key, api_base)

    probe_list = [p.strip() for p in probes.split(",") if p.strip()]
    for p in probe_list:
        if p not in AVAILABLE_PROBES:
            typer.echo(
                f"error: unknown probe {p!r}. Available: "
                f"{', '.join(sorted(AVAILABLE_PROBES))}",
                err=True,
            )
            raise typer.Exit(2)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        typer.echo("error: --model is empty.", err=True)
        raise typer.Exit(2)

    work = [(p, m) for p in probe_list for m in model_list]
    max_concurrent = max(1, int(max_concurrent))
    out_lock = threading.Lock()
    all_metrics: list = []
    failed = False

    def _one(probe_name: str, model_name: str):
        with out_lock:
            typer.echo(f"-> probe={probe_name} model={model_name}...", err=True)
        llm_obj = LLM(model_name=model_name)
        creds_obj = LLMCredentials(api_key=key, base_url=base)
        result = run_probe(
            probe_name,
            llm=llm_obj,
            credentials=creds_obj,
            model_name=model_name,
            timeout_seconds=timeout,
            seed=seed,
            temperature=temperature,
        )
        metrics = result.metrics_dict()
        with out_lock:
            all_metrics.append(metrics)
            if record:
                path = record_metrics(metrics)
                typer.echo(f"   recorded [{model_name}] -> {path}", err=True)
            if result.error or not result.schema_valid or not result.hash_check_passed:
                typer.echo(
                    f"   FAIL [{model_name}/{probe_name}]: "
                    f"error={result.error[:120]!r}",
                    err=True,
                )
            else:
                typer.echo(f"   ok [{model_name}/{probe_name}]", err=True)
        return result

    typer.echo(
        f"Dispatching {len(work)} pair(s) max_concurrent={max_concurrent}...",
        err=True,
    )

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {
            pool.submit(_one, pn, mn): (pn, mn) for pn, mn in work
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                pn, mn = futures[fut]
                typer.echo(f"   THREAD-ERROR [{mn}/{pn}]: {exc}", err=True)
                failed = True
                continue
            if result.error or not result.schema_valid or not result.hash_check_passed:
                failed = True

    payload_obj = all_metrics[0] if len(all_metrics) == 1 else all_metrics
    payload = json.dumps(payload_obj, indent=2, default=str)
    if output:
        output.write_text(payload)
        typer.echo(f"Wrote metrics to {output}", err=True)
    else:
        sys.stdout.write(payload + "\n")

    if failed:
        raise typer.Exit(1)


# --- replay (snapshot) -------------------------------------------------------


def _diff_outputs(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Tiny structural diff between two slot-keyed output dicts.

    Snapshots aren't exact-match (different model, different sampling),
    so we just summarise: which slots are present / missing / changed at
    the top level. Detailed comparison is the job of the probe checks.
    """
    actual_keys = set(actual)
    expected_keys = set(expected)
    return {
        "slots_only_in_actual": sorted(actual_keys - expected_keys),
        "slots_only_in_expected": sorted(expected_keys - actual_keys),
        "slots_shared": sorted(actual_keys & expected_keys),
    }


@eval_app.command("replay")
def replay(
    snapshot_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Snapshot directory (see soak.evals.snapshot for the layout).",
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model to replay against (need not match the original).",
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override LLM_API_KEY."),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="Override LLM_API_BASE."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write replay JSON (model output + diff) to this path.",
    ),
):
    """Replay one snapshot against a chosen model.

    Loads ``snapshot_dir/template.sd`` and ``inputs.json``, calls
    ``struckdown.complete`` with the requested model, and prints the
    actual outputs alongside a summary diff against ``expected.json``.
    No full pipeline is involved -- this is intentionally cheap.
    """
    from struckdown import complete
    from struckdown.llm import LLM, LLMCredentials

    snapshot = load_snapshot(snapshot_dir)
    key, base = _resolve_credentials(api_key, api_base)

    llm = LLM(model_name=model)
    credentials = LLMCredentials(api_key=key, base_url=base)

    typer.echo(
        f"Replaying snapshot '{snapshot.metadata.name}' "
        f"(stage={snapshot.metadata.stage}, original={snapshot.metadata.original_model}) "
        f"against {model}...",
        err=True,
    )

    result = complete(
        snapshot.template, snapshot.inputs, model=llm, credentials=credentials
    )

    actual: Dict[str, Any] = {}
    for slot_name, slot_result in result.results.items():
        out = slot_result.output
        if hasattr(out, "model_dump"):
            actual[slot_name] = out.model_dump(mode="json")
        elif isinstance(out, list):
            actual[slot_name] = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in out
            ]
        else:
            actual[slot_name] = out

    diff = _diff_outputs(actual, snapshot.expected)
    payload = {
        "snapshot": snapshot.metadata.name,
        "stage": snapshot.metadata.stage,
        "model": model,
        "original_model": snapshot.metadata.original_model,
        "diff": diff,
        "actual": actual,
        "fresh_cost": getattr(result, "fresh_cost", 0) or 0,
    }
    body = json.dumps(payload, indent=2, default=str)

    if output:
        output.write_text(body)
        typer.echo(f"Wrote replay output to {output}", err=True)
    else:
        sys.stdout.write(body + "\n")


# --- snapshot (build from cli output) ----------------------------------------


@eval_app.command("snapshot")
def snapshot(
    name: str = typer.Option(
        ...,
        "--as",
        help="Slug for the snapshot directory (e.g. 'kimi_consolidate_failure').",
    ),
    from_cli: Optional[Path] = typer.Option(
        None,
        "--from-cli",
        exists=True,
        dir_okay=False,
        help="Build a snapshot from a `soak run` output JSON file.",
    ),
    template_path: Optional[Path] = typer.Option(
        None,
        "--template",
        exists=True,
        dir_okay=False,
        help=(
            "Optional path to the prompt template (.sd) that produced the "
            "snapshot's expected output. Required when --from-cli does not "
            "include the template inline."
        ),
    ),
    stage: str = typer.Option(
        "unknown",
        "--stage",
        help="Stage / node name to record on the snapshot metadata.",
    ),
    out_dir: Path = typer.Option(
        Path("tests/evals/data/snapshots"),
        "--out-dir",
        help="Parent directory under which the new snapshot is created.",
    ),
    keep_pii: bool = typer.Option(
        False,
        "--keep-pii",
        help="Mark the snapshot as un-scrubbed. Default is scrubbed.",
    ),
    notes: Optional[str] = typer.Option(
        None, "--notes", help="Optional notes recorded on the snapshot."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite an existing snapshot dir."
    ),
):
    """Build a snapshot from a `soak run` output JSON.

    The output JSON is the file ``soak run --output X`` writes (the
    same JSON as ``StruckdownResult.model_dump_json``). We pull out the
    template + per-slot results to populate the snapshot. The original
    model name is read from the result if available; otherwise it is
    recorded as ``unknown``.
    """
    if from_cli is None:
        typer.echo(
            "error: --from-cli is required (other sources may follow in later phases).",
            err=True,
        )
        raise typer.Exit(2)

    raw = json.loads(from_cli.read_text(encoding="utf-8"))

    # The CLI-output JSON shape is `StruckdownResult.model_dump`. We
    # accept either that shape or a reduced {"template", "context",
    # "expected"} shape so users can hand-build small snapshots.
    template_text: str
    inputs: Dict[str, Any] = {}
    expected: Dict[str, Any] = {}
    original_model: Optional[str] = None

    if "results" in raw and isinstance(raw["results"], dict):
        # StruckdownResult shape
        if template_path is None:
            typer.echo(
                "error: --template required when --from-cli is a StruckdownResult JSON.",
                err=True,
            )
            raise typer.Exit(2)
        template_text = template_path.read_text(encoding="utf-8")
        for slot_name, slot in raw["results"].items():
            expected[slot_name] = slot.get("output")
        # try to surface the model from the first completion entry
        completions = (raw.get("completions") or {})
        if isinstance(completions, dict):
            for c in completions.values():
                if isinstance(c, dict) and c.get("model"):
                    original_model = c["model"]
                    break
    elif {"template", "expected"}.issubset(raw):
        # Minimal hand-built shape
        template_text = (
            template_path.read_text(encoding="utf-8")
            if template_path
            else raw["template"]
        )
        inputs = raw.get("inputs") or {}
        expected = raw["expected"]
        original_model = raw.get("model")
    else:
        typer.echo(
            "error: could not recognise --from-cli JSON shape "
            "(expected StruckdownResult or {template, expected}).",
            err=True,
        )
        raise typer.Exit(2)

    metadata = SnapshotMetadata(
        name=name,
        stage=stage,
        source=f"cli:{from_cli.name}",
        original_model=original_model,
        scrubbed=not keep_pii,
        notes=notes,
    )

    target_dir = out_dir / name
    written = write_snapshot(
        target_dir,
        template=template_text,
        inputs=inputs,
        expected=expected,
        metadata=metadata,
        overwrite=overwrite,
    )
    typer.echo(f"Wrote snapshot to {written}", err=True)


@eval_app.command("list")
def list_(
    root: Path = typer.Option(
        Path("tests/evals/data/snapshots"),
        "--root",
        help="Snapshot corpus root.",
    ),
):
    """List committed snapshots under the corpus root."""
    snaps: List[Path] = list_snapshots(root)
    if not snaps:
        typer.echo(f"No snapshots under {root}", err=True)
        return
    for path in snaps:
        try:
            snap = load_snapshot(path)
            typer.echo(
                f"  {path.name:30s}  stage={snap.metadata.stage:14s} "
                f"original={snap.metadata.original_model or '?'}"
            )
        except Exception as exc:
            typer.echo(f"  {path.name:30s}  <load error: {exc}>", err=True)


# Backwards-compat shim: the old single-command form `soak eval --probe X
# --model Y` still works because Typer routes to the default sub-command
# when invoked with no subcommand and a known --probe/--model. To keep
# that behavior, expose ``eval_cmd`` as the legacy callable used by
# ``cli/__init__.py``; it is now an alias for the ``eval_app`` group.
eval_cmd = eval_app
