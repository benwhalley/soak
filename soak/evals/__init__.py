"""LLM capability eval suite.

Two run modes share the same probe code:

- **Gate**: hard pass/fail of pinned model. Catches system regressions
  (prompt drift, schema bugs, struckdown changes). Used by the django
  admin "test structured output" view and by `pytest -m llm` against
  the gate model in CI.
- **Eval**: parametrised over many models, records metrics to JSON.
  Used to compare candidates manually.

Public API:

- :func:`run_probe` -- run one probe against one model, return a
  :class:`ProbeRunResult` with deterministic checks + raw outputs.
- :class:`ProbeRunResult`, :class:`SlotProbeResult`, :class:`Check`
- :data:`PROBES_DIR` -- where bundled .sd templates live.

See ``plans/LLM_CAPABILITY_EVALS.md`` in the soakresearch repo for the
roadmap. Phase 1 ships only ``probe_consolidate``; further probes
(``probe_schema``, ``probe_long_context``, ``probe_themes_long``,
judge-graded probes) follow in later phases.
"""

from .probes import (PROBES_DIR, AVAILABLE_PROBES, Check, ProbeRunResult,
                     SlotProbeResult, run_probe)
from .report import DEFAULT_OUT as REPORT_DEFAULT_OUT, build_report
from .results import DEFAULT_RESULTS_DIR, record_metrics
from .snapshot import (SCHEMA_VERSION as SNAPSHOT_SCHEMA_VERSION, Snapshot,
                       SnapshotMetadata, list_snapshots, load_snapshot,
                       write_snapshot)

__all__ = [
    "AVAILABLE_PROBES",
    "Check",
    "DEFAULT_RESULTS_DIR",
    "PROBES_DIR",
    "ProbeRunResult",
    "REPORT_DEFAULT_OUT",
    "SNAPSHOT_SCHEMA_VERSION",
    "SlotProbeResult",
    "Snapshot",
    "SnapshotMetadata",
    "build_report",
    "list_snapshots",
    "load_snapshot",
    "record_metrics",
    "run_probe",
    "write_snapshot",
]
