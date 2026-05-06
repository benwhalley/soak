"""Probe implementation: run a struckdown template against a model and
extract deterministic checks + stats.

Relocated from ``django_soak/structured_output_probes.py`` so the eval
machinery doesn't depend on Django. The django admin view now calls
:func:`run_probe` and renders the returned :class:`ProbeRunResult`.

The deterministic check block (:func:`_run_checks`) is the part that
actually exercises the LLM contract:

- ``Consolidated quote references valid`` -- does ``QuoteReference.hash``
  always resolve to a real ``Quote.hash`` from the prior code stage?
- ``Theme code references valid`` -- does ``Theme.code_hashes`` always
  resolve to a ``Code.hash`` from the consolidated stage?

Phase 1 adds ``probe_consolidate``. ``probe_schema`` and
``probe_long_context`` are Phase 2.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# trigger registration of soak return types (code, theme, etc).
import soak.models.base  # noqa: F401
from soak.models.base import Code, Quote, QuoteReference, Theme

logger = logging.getLogger(__name__)


PROBES_DIR = Path(__file__).parent / "data"
# Slow models (kimi-k2.6 via openrouter) regularly take 4+ minutes for the
# full multi-stage probe. 600s gives headroom without masking real hangs.
DEFAULT_TIMEOUT_SECONDS = 600

# Probe registry. Each entry maps a logical probe name to:
#   template:        .sd filename under PROBES_DIR
#   description:     human-readable summary
#   check_fn:        callable(raw_outputs, context) -> (checks, stats)
#   context_loader:  optional callable() -> dict of pre-loaded jinja vars
#                    (e.g. long-context probes that pre-bake 50 codes
#                    into the prompt rather than asking the LLM to invent
#                    them). The same dict is also passed to check_fn so
#                    it can compare LLM output against the inputs.
#
# Phase 1: ``consolidate`` (django parity).
# Phase 2: ``schema``, ``long_context``, ``themes_long``.
AVAILABLE_PROBES: Dict[str, Dict[str, Any]] = {}  # populated below, after check_fns are defined

# Internal/system fields that we strip from displayed output JSON. These
# come from struckdown / pydantic but aren't part of what the LLM emitted.
_STRIP_FIELDS = {
    "llm_config",
    "slug",
    "resolved_quotes",
    "resolved_code_refs",
    "label",
    "type",
}

# Display tag per slot name -- used by the django admin view to label
# each result block. Keys must match the slot names in the .sd template.
SLOT_DISPLAY: Dict[str, str] = {
    "free": "[[boolean]]",
    "choice": "[[pick]]",
    "codes": "[[code*]]",
    "consolidated": "[[code*|quotes=reference]]",
    "themes": "[[theme{1,2}]]",
}


@dataclass
class SlotProbeResult:
    slot_name: str
    type_tag: str
    success: bool
    output_json: str
    error: str


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ProbeRunResult:
    probe: str
    model_name: str
    slots: List[SlotProbeResult]
    checks: List[Check]
    stats: Dict[str, Any]
    total_cost: float
    duration_seconds: float
    error: str
    raw_outputs: Dict[str, Any] = field(default_factory=dict)
    # ``total_cost`` is the cost of *this run* including cached calls
    # (i.e. what the API would have charged if no cache were involved).
    # ``fresh_cost`` excludes cache hits -- $0 for fully cached runs.
    # Both are recorded so the report can show the headline number while
    # preserving the cache-aware figure for cost-sensitive sweeps.
    fresh_cost: float = 0.0
    # Compact, human-readable summary of what the model emitted (top
    # codes/themes with name+description). Recorded into JSONL so the
    # markdown report can show qualitative samples alongside metrics.
    samples: Dict[str, Any] = field(default_factory=dict)
    # Sampling parameters used for this run -- recorded so the JSONL
    # makes clear which seed produced which result. Re-runs with the
    # same seed should hit the cache and produce identical metrics.
    seed: int = 1
    temperature: float = 0.0

    @property
    def schema_valid(self) -> bool:
        """All slots parsed cleanly into typed objects."""
        return bool(self.slots) and all(s.success for s in self.slots) and not self.error

    @property
    def hash_check_passed(self) -> bool:
        """All reference-validity checks (if any) pass.

        Probes vary: consolidate has both quote-ref and code-ref checks;
        long_context has only the quote-ref one; schema has none. We
        treat 'no reference checks present' as trivially passed -- the
        probe didn't try, so it can't have failed.
        """
        ref_checks = [
            c for c in self.checks
            if "reference" in c.name.lower() or "hash" in c.name.lower()
        ]
        if not ref_checks:
            return True
        return all(c.passed for c in ref_checks)

    def metrics_dict(self) -> Dict[str, Any]:
        """Flat dict suitable for JSON-recording in eval mode."""
        return {
            "probe": self.probe,
            "model": self.model_name,
            "schema_valid": self.schema_valid,
            "hash_check_passed": self.hash_check_passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "stats": self.stats,
            "duration_seconds": self.duration_seconds,
            "total_cost": self.total_cost,
            "fresh_cost": self.fresh_cost,
            "samples": self.samples,
            "seed": self.seed,
            "temperature": self.temperature,
            "error": self.error,
        }


def _clean_dict(d: dict) -> dict:
    return {k: _clean_value(v) for k, v in d.items() if k not in _STRIP_FIELDS}


def _clean_value(v):
    if isinstance(v, dict):
        return _clean_dict(v)
    if isinstance(v, list):
        return [_clean_value(item) for item in v]
    return v


def _clean_output(obj):
    """Render a struckdown output value to a display dict.

    ``Code`` objects get their hash + per-quote hashes attached so the
    user can cross-reference them with the ``QuoteReference`` hashes
    that show up in the next slot.
    """
    if isinstance(obj, Code):
        d = _clean_dict(obj.model_dump(mode="json"))
        d["hash"] = obj.hash()
        d["quotes"] = []
        for q in obj.quotes:
            if isinstance(q, Quote):
                qd = {"text": q.text, "hash": q.hash()}
            else:
                qd = _clean_output(q)
            d["quotes"].append(qd)
        return d
    if hasattr(obj, "model_dump"):
        return _clean_dict(obj.model_dump(mode="json"))
    if isinstance(obj, list):
        return [_clean_output(item) for item in obj]
    return obj


def _collect_quote_hashes(codes: List[Code]) -> Set[str]:
    hashes: Set[str] = set()
    for code in codes:
        for q in code.quotes:
            if isinstance(q, Quote):
                hashes.add(q.hash())
    return hashes


def _collect_code_hashes(codes: List[Code]) -> Set[str]:
    return {code.hash() for code in codes}


def _collect_referenced_quote_hashes(codes: List[Code]) -> Set[str]:
    hashes: Set[str] = set()
    for code in codes:
        for q in code.quotes:
            if isinstance(q, QuoteReference):
                hashes.add(q.hash)
    return hashes


_MAX_CODES_PER_SLOT = 8
_MAX_THEMES = 6
_MAX_QUOTES_PER_CODE = 5
_MAX_CODES_PER_THEME = 8
_QUOTE_TEXT_LIMIT = 240


def _extract_samples(
    raw_outputs: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Pull a compact, human-readable summary of what the model emitted.

    Designed for the markdown report's **Examples** section so reviewers
    can eyeball the actual codes / themes alongside the boolean metrics.

    Includes:

    - Codes: name + description + up to 5 supporting quotes (text +
      hash). For ``QuoteReference`` cases (consolidate / long_context
      probes), the hash is resolved to its underlying ``Quote.text``
      using the upstream ``codes`` slot or the loaded ``codes_in`` fixture.
    - Themes: name + description + the *resolved code names* for the
      first few referenced ``code_hashes``. Without this, a theme is
      just an opaque list of hashes and reviewers can't judge whether
      the grouping is sensible.

    Truncation caps keep each (probe, model) JSONL line ~10-30 KB.
    """
    # ---- build hash -> text / hash -> code name lookups -----------------
    quote_text_by_hash: Dict[str, str] = {}
    code_name_by_hash: Dict[str, str] = {}

    def _ingest_codes(items: Any) -> None:
        if not isinstance(items, list):
            return
        for c in items:
            if not isinstance(c, Code):
                continue
            code_name_by_hash[c.hash()] = c.name
            for q in c.quotes:
                if isinstance(q, Quote):
                    quote_text_by_hash[q.hash()] = q.text

    _ingest_codes(raw_outputs.get("codes"))
    _ingest_codes(raw_outputs.get("consolidated"))
    _ingest_codes(context.get("codes_in"))

    # ---- per-item summarisers ------------------------------------------
    def _summarise_code(c: Code) -> Dict[str, Any]:
        quotes_out = []
        for q in c.quotes[:_MAX_QUOTES_PER_CODE]:
            if isinstance(q, Quote):
                quotes_out.append({
                    "hash": q.hash(),
                    "text": q.text[:_QUOTE_TEXT_LIMIT],
                })
            else:  # QuoteReference -- only carries hash
                text = quote_text_by_hash.get(q.hash, "")
                quotes_out.append({
                    "hash": q.hash,
                    "text": (text or "_unresolved_")[:_QUOTE_TEXT_LIMIT],
                    "ref": True,
                })
        return {
            "hash": c.hash(),
            "name": c.name,
            "description": c.description,
            "n_quotes": len(c.quotes),
            "quotes": quotes_out,
        }

    def _summarise_theme(t: Theme) -> Dict[str, Any]:
        codes_out = []
        for h in list(t.code_hashes)[:_MAX_CODES_PER_THEME]:
            codes_out.append({
                "hash": h,
                "name": code_name_by_hash.get(h, "_unresolved_"),
            })
        return {
            "name": t.name,
            "description": t.description,
            "n_codes": len(t.code_hashes),
            "codes": codes_out,
        }

    samples: Dict[str, Any] = {}
    for slot_key in ("codes", "consolidated"):
        items = raw_outputs.get(slot_key)
        if isinstance(items, list):
            codes = [c for c in items if isinstance(c, Code)]
            if codes:
                samples[slot_key] = [_summarise_code(c) for c in codes[:_MAX_CODES_PER_SLOT]]

    themes = raw_outputs.get("themes")
    if isinstance(themes, list):
        ts = [t for t in themes if isinstance(t, Theme)]
        if ts:
            samples["themes"] = [_summarise_theme(t) for t in ts[:_MAX_THEMES]]

    return samples


def _run_consolidate_checks(
    raw_outputs: Dict[str, Any], context: Dict[str, Any]
) -> tuple[List[Check], Dict[str, Any]]:
    """Deterministic post-processing checks on raw slot outputs.

    Returns (checks, stats). The two reference-validity checks are the
    hard-fail metrics for gate mode; the rest are informational.
    """
    checks: List[Check] = []
    stats: Dict[str, Any] = {}

    codes_output = raw_outputs.get("codes")
    consolidated_output = raw_outputs.get("consolidated")
    themes_output = raw_outputs.get("themes")

    codes = [c for c in (codes_output or []) if isinstance(c, Code)]
    consolidated = [c for c in (consolidated_output or []) if isinstance(c, Code)]
    themes = [t for t in (themes_output or []) if isinstance(t, Theme)]

    original_quote_hashes = _collect_quote_hashes(codes)
    referenced_quote_hashes = _collect_referenced_quote_hashes(consolidated)
    invalid_quote_refs = referenced_quote_hashes - original_quote_hashes
    checks.append(
        Check(
            name="Consolidated quote references valid",
            passed=len(invalid_quote_refs) == 0 and len(referenced_quote_hashes) > 0,
            detail=(
                f"{len(referenced_quote_hashes)} references, all valid"
                if not invalid_quote_refs
                else f"{len(invalid_quote_refs)} hallucinated: {', '.join(sorted(invalid_quote_refs))}"
            ),
        )
    )

    consolidated_code_hashes = _collect_code_hashes(consolidated)
    theme_code_refs: Set[str] = set()
    for t in themes:
        theme_code_refs.update(t.code_hashes)
    invalid_code_refs = theme_code_refs - consolidated_code_hashes
    checks.append(
        Check(
            name="Theme code references valid",
            passed=len(invalid_code_refs) == 0 and len(theme_code_refs) > 0,
            detail=(
                f"{len(theme_code_refs)} references, all valid"
                if not invalid_code_refs
                else f"{len(invalid_code_refs)} hallucinated: {', '.join(sorted(invalid_code_refs))}"
            ),
        )
    )

    checks.append(
        Check(
            name="At least 1 theme",
            passed=len(themes) >= 1,
            detail=f"{len(themes)} theme(s)",
        )
    )
    checks.append(
        Check(
            name="Consolidated codes <= original codes",
            passed=len(consolidated) <= len(codes),
            detail=f"{len(codes)} original -> {len(consolidated)} consolidated",
        )
    )
    empty_themes = [t.name for t in themes if len(t.code_hashes) == 0]
    checks.append(
        Check(
            name="Every theme has at least 1 code",
            passed=len(empty_themes) == 0 and len(themes) > 0,
            detail=(
                "All themes have codes"
                if not empty_themes
                else f"Empty themes: {', '.join(empty_themes)}"
            ),
        )
    )

    used_quote_hashes = referenced_quote_hashes
    unused_quote_hashes = original_quote_hashes - used_quote_hashes
    stats["quotes_in_codes"] = sum(len(c.quotes) for c in codes)
    stats["quotes_in_consolidated"] = sum(len(c.quotes) for c in consolidated)
    stats["quotes_in_themes"] = sum(len(t.code_hashes) for t in themes)
    stats["unique_quotes"] = len(original_quote_hashes)
    stats["used_quotes"] = len(used_quote_hashes)
    stats["unused_quotes"] = len(unused_quote_hashes)
    stats["unused_quote_hashes"] = sorted(unused_quote_hashes)
    stats["num_codes"] = len(codes)
    stats["num_consolidated"] = len(consolidated)
    stats["num_themes"] = len(themes)
    # quote_coverage is tracked record-only in phase 1 (see plan).
    stats["quote_coverage"] = (
        len(used_quote_hashes) / len(original_quote_hashes)
        if original_quote_hashes
        else 0.0
    )

    return checks, stats


def _run_schema_checks(
    raw_outputs: Dict[str, Any], context: Dict[str, Any]
) -> tuple[List[Check], Dict[str, Any]]:
    """Cheap baseline: did the model produce *any* valid Code objects?

    Hard fail = no codes at all. We deliberately don't enforce minimum
    quote counts here -- ``probe_consolidate`` already exercises that
    contract. This probe is the canary for "tool-calling busted entirely".
    """
    codes = [c for c in (raw_outputs.get("codes") or []) if isinstance(c, Code)]
    quote_count = sum(len(c.quotes) for c in codes)

    checks: List[Check] = [
        Check(
            name="At least 1 code",
            passed=len(codes) >= 1,
            detail=f"{len(codes)} code(s)",
        ),
        Check(
            name="Each code has at least 1 quote",
            passed=bool(codes) and all(len(c.quotes) >= 1 for c in codes),
            detail=f"{quote_count} quote(s) across {len(codes)} code(s)",
        ),
    ]
    stats = {
        "num_codes": len(codes),
        "quotes_in_codes": quote_count,
    }
    return checks, stats


def _run_long_context_checks(
    raw_outputs: Dict[str, Any], context: Dict[str, Any]
) -> tuple[List[Check], Dict[str, Any]]:
    """Hash fidelity over a long, pre-baked code listing.

    The probe's prompt renders ``codes_in`` (the loaded fixture) into the
    user message and asks the model to consolidate using
    ``[[code*:consolidated|quotes=reference]]``. Every quote-hash the
    model emits MUST appear in ``codes_in``; anything else is a
    hallucination. Truncation is a 0-tolerance metric here.
    """
    codes_in: List[Code] = context.get("codes_in", []) or []
    consolidated_output = raw_outputs.get("consolidated")
    consolidated = [c for c in (consolidated_output or []) if isinstance(c, Code)]

    original_quote_hashes = _collect_quote_hashes(codes_in)
    referenced_quote_hashes = _collect_referenced_quote_hashes(consolidated)
    invalid_quote_refs = referenced_quote_hashes - original_quote_hashes

    truncated = sorted(
        h for h in referenced_quote_hashes if len(h) != 8 or any(c.isupper() for c in h)
    )

    checks: List[Check] = [
        Check(
            name="Consolidated quote references valid",
            passed=len(invalid_quote_refs) == 0 and len(referenced_quote_hashes) > 0,
            detail=(
                f"{len(referenced_quote_hashes)} references, all valid"
                if not invalid_quote_refs
                else f"{len(invalid_quote_refs)} hallucinated: {', '.join(sorted(invalid_quote_refs)[:6])}"
            ),
        ),
        Check(
            name="No truncated/malformed hashes",
            passed=len(truncated) == 0,
            detail=f"{len(truncated)} malformed: {truncated[:6]}" if truncated else "all 8-char lowercase",
        ),
        Check(
            name="Consolidated count <= original",
            passed=len(consolidated) <= len(codes_in),
            detail=f"{len(codes_in)} input -> {len(consolidated)} consolidated",
        ),
    ]
    stats = {
        "input_codes": len(codes_in),
        "input_quotes": len(original_quote_hashes),
        "consolidated_codes": len(consolidated),
        "referenced_quotes": len(referenced_quote_hashes),
        "hallucinated_refs": sorted(invalid_quote_refs),
        "truncated_refs": truncated,
        "quote_coverage": (
            len(referenced_quote_hashes & original_quote_hashes) / len(original_quote_hashes)
            if original_quote_hashes
            else 0.0
        ),
    }
    return checks, stats


def _run_themes_long_checks(
    raw_outputs: Dict[str, Any], context: Dict[str, Any]
) -> tuple[List[Check], Dict[str, Any]]:
    """Theme generation against a fat code listing.

    Tests that ``Theme.code_hashes`` is a subset of the pre-loaded code
    hashes, and that no theme is empty. Same hash-fidelity contract as
    consolidate, but the failure surface is the theme stage.
    """
    codes_in: List[Code] = context.get("codes_in", []) or []
    themes_output = raw_outputs.get("themes")
    themes = [t for t in (themes_output or []) if isinstance(t, Theme)]

    input_code_hashes = _collect_code_hashes(codes_in)
    theme_code_refs: Set[str] = set()
    for t in themes:
        theme_code_refs.update(t.code_hashes)
    invalid_code_refs = theme_code_refs - input_code_hashes
    empty_themes = [t.name for t in themes if len(t.code_hashes) == 0]

    truncated = sorted(
        h for h in theme_code_refs if len(h) != 8 or any(c.isupper() for c in h)
    )

    checks: List[Check] = [
        Check(
            name="Theme code references valid",
            passed=len(invalid_code_refs) == 0 and len(theme_code_refs) > 0,
            detail=(
                f"{len(theme_code_refs)} references, all valid"
                if not invalid_code_refs
                else f"{len(invalid_code_refs)} hallucinated: {', '.join(sorted(invalid_code_refs)[:6])}"
            ),
        ),
        Check(
            name="No truncated/malformed hashes",
            passed=len(truncated) == 0,
            detail=f"{len(truncated)} malformed: {truncated[:6]}" if truncated else "all 8-char lowercase",
        ),
        Check(
            name="At least 1 theme",
            passed=len(themes) >= 1,
            detail=f"{len(themes)} theme(s)",
        ),
        Check(
            name="Every theme has at least 1 code",
            passed=len(empty_themes) == 0 and len(themes) > 0,
            detail=(
                "All themes have codes"
                if not empty_themes
                else f"Empty themes: {', '.join(empty_themes[:4])}"
            ),
        ),
    ]
    stats = {
        "input_codes": len(codes_in),
        "num_themes": len(themes),
        "theme_code_refs": len(theme_code_refs),
        "hallucinated_refs": sorted(invalid_code_refs),
        "truncated_refs": truncated,
        "code_coverage": (
            len(theme_code_refs & input_code_hashes) / len(input_code_hashes)
            if input_code_hashes
            else 0.0
        ),
    }
    return checks, stats


def _load_codes_long() -> Dict[str, Any]:
    """Load the long-codes fixture (~50 Code objects) from disk.

    The fixture is committed JSON so the probes are reproducible across
    machines and over time. ``Code(**c)`` re-validates the dicts so we
    fail loudly if the fixture format drifts.
    """
    fixture_path = PROBES_DIR / "codes_long.json"
    raw = json.loads(fixture_path.read_text())
    codes = [Code(**c) for c in raw]
    return {"codes_in": codes}


# --- registry (populated after check_fns + loaders are defined) ---

AVAILABLE_PROBES.update({
    "consolidate": {
        "template": "probe_consolidate.sd",
        "description": (
            "Codes -> reference-mode consolidation -> themes. Exercises "
            "hash fidelity across the multi-stage prompt."
        ),
        "check_fn": _run_consolidate_checks,
        "context_loader": None,
    },
    "schema": {
        "template": "probe_schema.sd",
        "description": (
            "Cheap baseline: single-call code extraction from a short "
            "transcript. Tool-calling smoke test, no hash fidelity."
        ),
        "check_fn": _run_schema_checks,
        "context_loader": None,
    },
    "long_context": {
        "template": "probe_long_context.sd",
        "description": (
            "Reference-mode consolidate from a fat (50+) pre-loaded code "
            "listing. Tests hash truncation/confusion under length."
        ),
        "check_fn": _run_long_context_checks,
        "context_loader": _load_codes_long,
    },
    "themes_long": {
        "template": "probe_themes_long.sd",
        "description": (
            "Theme generation from a fat (50+) pre-loaded code listing. "
            "Tests Theme.code_hashes fidelity at scale."
        ),
        "check_fn": _run_themes_long_checks,
        "context_loader": _load_codes_long,
    },
})


def _resolve_template(probe: str) -> Path:
    if probe not in AVAILABLE_PROBES:
        raise ValueError(
            f"Unknown probe {probe!r}. Available: {sorted(AVAILABLE_PROBES)}"
        )
    template_path = PROBES_DIR / AVAILABLE_PROBES[probe]["template"]
    if not template_path.exists():
        raise FileNotFoundError(f"Probe template missing: {template_path}")
    return template_path


def _run_template_sync(
    template_text: str,
    *,
    probe: str,
    model_name: str,
    llm,
    credentials,
    seed: int = 1,
    temperature: float = 0.0,
) -> ProbeRunResult:
    from struckdown import complete

    entry = AVAILABLE_PROBES[probe]
    check_fn = entry.get("check_fn") or _run_consolidate_checks
    context_loader = entry.get("context_loader")
    context: Dict[str, Any] = context_loader() if context_loader else {}

    # Pin seed + temperature for reproducibility -- without these,
    # cost/latency comparisons across re-runs are noisy because the
    # joblib cache key and model output both change run-to-run.
    extra_kwargs = {"seed": seed, "temperature": temperature}

    start = time.monotonic()
    try:
        result = complete(
            template_text, context, model=llm, credentials=credentials,
            extra_kwargs=extra_kwargs,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.warning("Probe %s failed against %s: %s", probe, model_name, exc)
        return ProbeRunResult(
            probe=probe,
            model_name=model_name,
            slots=[],
            checks=[],
            stats={},
            total_cost=0,
            duration_seconds=round(elapsed, 2),
            error=str(exc)[:500],
        )

    elapsed = time.monotonic() - start
    # total_cost: what the API would have charged for this run regardless
    # of cache hits (the headline number for the report).
    # fresh_cost: cost paid this run only (zero for fully cached runs).
    total = getattr(result, "total_cost", 0) or 0
    fresh = getattr(result, "fresh_cost", 0) or 0

    raw_outputs: Dict[str, Any] = {}
    slots: List[SlotProbeResult] = []
    for slot_name, slot_result in result.results.items():
        raw_outputs[slot_name] = slot_result.output
        dumped = _clean_output(slot_result.output)
        slots.append(
            SlotProbeResult(
                slot_name=slot_name,
                type_tag=SLOT_DISPLAY.get(slot_name, f"[[{slot_name}]]"),
                success=True,
                output_json=json.dumps(dumped, indent=2, default=str),
                error="",
            )
        )

    checks, stats = check_fn(raw_outputs, context)
    samples = _extract_samples(raw_outputs, context)

    return ProbeRunResult(
        probe=probe,
        model_name=model_name,
        slots=slots,
        checks=checks,
        stats=stats,
        total_cost=total,
        fresh_cost=fresh,
        duration_seconds=round(elapsed, 2),
        error="",
        raw_outputs=raw_outputs,
        samples=samples,
        seed=seed,
        temperature=temperature,
    )


def run_probe(
    probe: str,
    *,
    llm,
    credentials,
    model_name: Optional[str] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    seed: int = 1,
    temperature: float = 0.0,
) -> ProbeRunResult:
    """Run one probe against one model.

    Args:
        probe: Logical probe name (key into :data:`AVAILABLE_PROBES`).
        llm: A struckdown ``LLM``.
        credentials: A struckdown ``LLMCredentials``.
        model_name: Display name to record on the result. Defaults to
            ``llm.model_name`` if available.
        timeout_seconds: Hard wall-clock timeout for the probe call.
            Translates to a ``ThreadPoolExecutor.future.result(timeout=)``.
        seed: Sampling seed passed to the model (default 1). Pinning
            this makes re-runs reproducible -- same prompt + same seed
            produce the same model output, the same struckdown cache
            key, and therefore comparable cost/latency between runs.
        temperature: Sampling temperature (default 0.0 -- deterministic
            decoding for models that honor it).

    Returns:
        :class:`ProbeRunResult`. Even on failure (LLM error, schema
        mismatch, timeout) we return a result with ``error`` populated --
        callers don't need to handle exceptions.

    Notes:
        We run the probe in a worker thread because struckdown's
        ``complete`` does ``anyio.run()`` internally; calling it on a
        thread that already owns an event loop (e.g. an async django
        view, or pytest-asyncio) raises ``RuntimeError``.
    """
    template_path = _resolve_template(probe)
    template_text = template_path.read_text()
    name = model_name or getattr(llm, "model_name", None) or "<unknown>"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_template_sync,
            template_text,
            probe=probe,
            model_name=name,
            llm=llm,
            credentials=credentials,
            seed=seed,
            temperature=temperature,
        )
        try:
            return future.result(timeout=timeout_seconds)
        except Exception as exc:
            logger.warning("Probe %s timed out / errored: %s", probe, exc)
            return ProbeRunResult(
                probe=probe,
                model_name=name,
                slots=[],
                checks=[],
                stats={},
                total_cost=0,
                duration_seconds=float(timeout_seconds),
                error=f"timeout/error: {exc}"[:500],
            )
