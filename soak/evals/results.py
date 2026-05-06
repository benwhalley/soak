"""Append probe metrics to ``soaking-eval/results/<date>.jsonl`` in the CWD.

Single source of truth for *where* probe results land, so the pytest
suite, ``soak eval run``, the django ``manage.py eval_probe`` command,
and the snapshot/replay flow all populate the same per-project history
that ``soak.evals.report.build_report`` reads.

The default is ``./soaking-eval/results/`` — project-local, like
``.pytest_cache``. Override with the ``SOAK_EVAL_RESULTS`` env var for
a shared location (e.g. ``~/.soak/eval-results/``) or with the
explicit ``results_dir`` argument.

JSONL keeps history without churning git on every run -- new entries
append, the report tool picks the latest line per (probe, model) pair.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _default_results_dir() -> Path:
    """Pick the default location at call time (not import time).

    Order:
    1. ``SOAK_EVAL_RESULTS`` env var (explicit override).
    2. ``./soaking-eval/results`` -- project-local default. Mirrors
       ``.pytest_cache`` / ``node_modules`` style; each project picks up
       its own eval history without writing into the soaking source tree.
    """
    env = os.environ.get("SOAK_EVAL_RESULTS")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "soaking-eval" / "results"


# Re-evaluated lazily; reading this attribute at module-import time will
# resolve once and miss later CWD changes -- prefer the function above.
DEFAULT_RESULTS_DIR = _default_results_dir()


def record_metrics(
    metrics: Dict[str, Any],
    *,
    results_dir: Optional[Path] = None,
) -> Path:
    """Append one JSON line to the per-day results file.

    Args:
        metrics: Flat dict (``ProbeRunResult.metrics_dict()`` shape, but
            any JSON-serialisable dict works).
        results_dir: Override the destination. Defaults to
            ``DEFAULT_RESULTS_DIR``.

    Returns:
        Path to the JSONL file written.
    """
    target_dir = Path(results_dir) if results_dir else _default_results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = target_dir / f"{today}.jsonl"

    line = {**metrics, "recorded_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return path
