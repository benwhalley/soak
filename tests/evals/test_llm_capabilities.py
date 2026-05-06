"""LLM capability gate + sweep tests.

Two flavours of test live here:

1. **Gate** -- hard pass/fail against a pinned model
   (``test_*_gate``). Catch system regressions (prompt drift, schema
   bug, struckdown change). Default-on when ``-m llm`` runs.
2. **Eval sweep** -- parametrised over every model in
   ``SOAK_EVAL_MODELS``, opt-in via ``--eval-mode`` flag (or
   ``SOAK_EVAL_MODE=1``). Records JSON metrics; assertions are
   downgraded to record-only so a flaky model run doesn't break the
   test session.

All tests carry ``@pytest.mark.llm`` so plain ``pytest`` skips them.
Run with::

    pytest -m llm tests/evals
    pytest -m llm tests/evals --eval-mode    # sweep mode

Override the gate model with ``SOAK_GATE_MODEL``; override the sweep
list with ``SOAK_EVAL_MODELS=gpt-4.1-mini,gpt-5-mini``.
"""

from __future__ import annotations

import os

import pytest

from soak.evals import AVAILABLE_PROBES, record_metrics, run_probe


def _record(result_dict: dict):
    """Append metrics to the canonical results JSONL.

    Thin wrapper kept for clarity at the call sites; delegates to
    ``soak.evals.record_metrics`` (single source of truth for where
    history lands -- ``scripts/eval_report.py`` reads the same path).
    """
    return record_metrics(result_dict)


def _eval_mode_active(request) -> bool:
    """True if pytest was invoked with --eval-mode or SOAK_EVAL_MODE=1."""
    if request.config.getoption("--eval-mode", default=False):
        return True
    return os.environ.get("SOAK_EVAL_MODE", "").lower() in ("1", "true", "yes")


def _maybe_assert(condition: bool, msg: str, *, eval_mode: bool):
    """In gate mode, assert. In eval mode, just record (don't fail)."""
    if eval_mode:
        if not condition:
            print(f"[eval] soft-fail: {msg}")
        return
    assert condition, msg


# ---------------------------------------------------------------------------
# Gate tests (pinned model, hard assertions)
# ---------------------------------------------------------------------------


@pytest.mark.llm
def test_consolidate_gate(gate_model: str, llm_credentials, request):
    """Hard gate: schema + reference fidelity for the consolidate probe."""
    eval_mode = _eval_mode_active(request)
    llm, credentials = llm_credentials(gate_model)
    result = run_probe(
        "consolidate", llm=llm, credentials=credentials, model_name=gate_model,
    )
    _record(result.metrics_dict())

    _maybe_assert(not result.error, f"probe failed: {result.error}", eval_mode=eval_mode)
    _maybe_assert(
        result.schema_valid,
        f"schema invalid; slots={[s.slot_name for s in result.slots]}",
        eval_mode=eval_mode,
    )

    by_name = {c.name: c for c in result.checks}
    quote_check = by_name.get("Consolidated quote references valid")
    code_check = by_name.get("Theme code references valid")
    _maybe_assert(quote_check is not None, "missing quote-reference check", eval_mode=eval_mode)
    _maybe_assert(code_check is not None, "missing theme-code-reference check", eval_mode=eval_mode)
    if quote_check is not None:
        _maybe_assert(
            quote_check.passed,
            f"quote-ref check failed: {quote_check.detail}",
            eval_mode=eval_mode,
        )
    if code_check is not None:
        _maybe_assert(
            code_check.passed,
            f"theme-code-ref check failed: {code_check.detail}",
            eval_mode=eval_mode,
        )


@pytest.mark.llm
def test_schema_gate(gate_model: str, llm_credentials, request):
    """Cheap baseline gate: tool-calling produces valid Code objects."""
    eval_mode = _eval_mode_active(request)
    llm, credentials = llm_credentials(gate_model)
    result = run_probe(
        "schema", llm=llm, credentials=credentials, model_name=gate_model,
    )
    _record(result.metrics_dict())

    _maybe_assert(not result.error, f"probe failed: {result.error}", eval_mode=eval_mode)
    _maybe_assert(
        result.schema_valid,
        f"schema invalid; slots={[s.slot_name for s in result.slots]}",
        eval_mode=eval_mode,
    )
    for check in result.checks:
        _maybe_assert(check.passed, f"{check.name}: {check.detail}", eval_mode=eval_mode)


# ---------------------------------------------------------------------------
# Eval sweep tests (parametrised over SOAK_EVAL_MODELS, opt-in via --eval-mode)
# ---------------------------------------------------------------------------


def _eval_models_param() -> list[str]:
    """Read SOAK_EVAL_MODELS at collection time so parametrise() can use it."""
    raw = os.environ.get("SOAK_EVAL_MODELS", "gpt-4.1-mini,gpt-5-mini")
    return [m.strip() for m in raw.split(",") if m.strip()]


# Probes that are reasonable to sweep across many models. Order matters
# for cost / runtime: cheap baseline first, fat probes last.
SWEEP_PROBES = ["schema", "consolidate", "long_context", "themes_long"]


@pytest.mark.llm
@pytest.mark.parametrize("probe_name", SWEEP_PROBES)
@pytest.mark.parametrize("model_name", _eval_models_param())
def test_probe_sweep(probe_name: str, model_name: str, llm_credentials, request):
    """Run every (probe, model) pair, recording metrics.

    In gate mode this is skipped (SWEEP_PROBES is opt-in via --eval-mode
    so we don't accidentally rack up cost on every CI run). In eval mode
    we record the metrics and only soft-fail on probe errors.
    """
    if not _eval_mode_active(request):
        pytest.skip("sweep tests are opt-in; run with --eval-mode or SOAK_EVAL_MODE=1")

    if probe_name not in AVAILABLE_PROBES:
        pytest.skip(f"unknown probe {probe_name!r}")

    llm, credentials = llm_credentials(model_name)
    result = run_probe(
        probe_name, llm=llm, credentials=credentials, model_name=model_name,
    )
    _record(result.metrics_dict())

    if result.error:
        print(f"[eval] {probe_name} on {model_name} errored: {result.error}")
