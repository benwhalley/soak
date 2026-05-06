"""Eval suite pytest config.

Two run modes (per ``plans/LLM_CAPABILITY_EVALS.md``):

- **Gate** (default for the gate test): pinned model, hard assertions.
- **Eval**: parametrise over models from ``SOAK_EVAL_MODELS`` env var,
  record JSON instead of asserting. Phase 2.

The gate model is pinned to ``GATE_MODEL`` below. Override with
``SOAK_GATE_MODEL`` for one-off runs against a candidate replacement.

All eval tests carry the ``@pytest.mark.llm`` mark, which is excluded
from the default ``addopts`` so plain ``pytest`` skips them. Run with
``pytest -m llm tests/evals``.
"""

from __future__ import annotations

import os
from typing import List

import pytest

# Pinned gate model. Override with SOAK_GATE_MODEL to evaluate a
# candidate replacement, but only flip the default after multiple
# stable runs (see plan, "Decisions").
GATE_MODEL = os.environ.get("SOAK_GATE_MODEL", "gpt-5.1-mini")


def _eval_models() -> List[str]:
    """Models to sweep in eval mode (Phase 2). Default sweep matches plan."""
    raw = os.environ.get("SOAK_EVAL_MODELS", "gpt-4.1-mini,gpt-5-mini")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _eval_mode() -> bool:
    """Eval mode is opt-in via SOAK_EVAL_MODE=1 or --eval-mode flag."""
    return os.environ.get("SOAK_EVAL_MODE", "").lower() in ("1", "true", "yes")


def pytest_addoption(parser):
    parser.addoption(
        "--eval-mode",
        action="store_true",
        default=False,
        help=(
            "Run probes in eval mode: parametrise over SOAK_EVAL_MODELS "
            "and record metrics to JSON instead of asserting."
        ),
    )


@pytest.fixture(scope="session")
def gate_model() -> str:
    return GATE_MODEL


@pytest.fixture(scope="session")
def eval_models() -> List[str]:
    return _eval_models()


@pytest.fixture(scope="session")
def llm_credentials():
    """Build an LLM + LLMCredentials pair from ``LLM_API_KEY``/``LLM_API_BASE``.

    The eval suite reads env vars directly (per the plan). Tests that
    need a model factory call ``llm_factory(model_name)`` to get
    ``(llm, credentials)``.
    """
    from struckdown.llm import LLM, LLMCredentials

    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE") or None
    if not api_key:
        pytest.skip("LLM_API_KEY not set; skipping LLM probe tests.")

    def factory(model_name: str):
        return (
            LLM(model_name=model_name),
            LLMCredentials(api_key=api_key, base_url=api_base),
        )

    return factory
