"""Snapshot / replay format for the eval suite.

A *snapshot* is a frozen single-stage call: enough information to re-run
one slot of one node against any chosen model, without standing up the
full pipeline. It is the unit of "interesting case found in production"
that we want to test future model candidates against.

Layout on disk::

    <snapshot_dir>/
      template.sd        # the prompt template (jinja, with [[type:slot]] markers)
      inputs.json        # context dict passed to complete()
      expected.json      # baseline output (JSON of slot_name -> dumped value)
      metadata.json      # source ref, gate model, date, scrub status, schema version
      README.md          # human description of why this case is interesting

This module is the **single source of truth** for the format. Both
``soak eval replay`` (in this repo) and ``manage.py snapshot_run`` (in
soakresearch) write/read snapshots through these helpers, so the format
can evolve without divergent serialisation logic.

No Django dependency. No HTTP. Pure model + filesystem.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Bump on incompatible format changes. ``load_snapshot`` warns loudly on
# mismatch but still tries to load -- be conservative when bumping.
SCHEMA_VERSION = 1

_INPUTS_FILE = "inputs.json"
_TEMPLATE_FILE = "template.sd"
_EXPECTED_FILE = "expected.json"
_METADATA_FILE = "metadata.json"
_README_FILE = "README.md"


class SnapshotMetadata(BaseModel):
    """Provenance for a snapshot.

    Aim is to make it possible to track a snapshot back to the run that
    produced it, decide if it has been scrubbed of PII, and detect
    schema drift in the format itself.
    """

    schema_version: int = SCHEMA_VERSION
    name: str = Field(..., description="Short slug identifying this snapshot.")
    stage: str = Field(..., description="Logical stage name, e.g. 'consolidate' or a node name.")
    source: str = Field(
        ...,
        description=(
            "Free-form provenance string. e.g. 'analysis_run:<uuid>', "
            "'cli:run_output.json', 'hand-built'."
        ),
    )
    original_model: Optional[str] = Field(
        None, description="Model that produced ``expected.json`` (if any)."
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when the snapshot was written.",
    )
    scrubbed: bool = Field(
        True,
        description="Whether PII has been scrubbed. Default True; opt out explicitly.",
    )
    notes: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class Snapshot(BaseModel):
    """In-memory representation of a snapshot directory."""

    metadata: SnapshotMetadata
    template: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected: Dict[str, Any] = Field(default_factory=dict)
    readme: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


def write_snapshot(
    snapshot_dir: Path,
    *,
    template: str,
    inputs: Dict[str, Any],
    expected: Dict[str, Any],
    metadata: SnapshotMetadata,
    readme: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Write a snapshot to disk in the canonical layout.

    ``snapshot_dir`` is created if it doesn't exist. Existing files are
    refused unless ``overwrite=True`` -- snapshots are immutable by
    design once committed.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        _TEMPLATE_FILE: template,
        _INPUTS_FILE: json.dumps(inputs, indent=2, default=str),
        _EXPECTED_FILE: json.dumps(expected, indent=2, default=str),
        _METADATA_FILE: metadata.model_dump_json(indent=2),
        _README_FILE: readme or _default_readme(metadata),
    }

    if not overwrite:
        for fname in targets:
            target = snapshot_dir / fname
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing snapshot file {target}; "
                    f"pass overwrite=True if you really mean it."
                )

    for fname, content in targets.items():
        (snapshot_dir / fname).write_text(content, encoding="utf-8")

    return snapshot_dir


def load_snapshot(snapshot_dir: Path) -> Snapshot:
    """Load a snapshot from disk.

    Missing required files raise ``FileNotFoundError``. Schema-version
    mismatch raises ``ValueError`` -- the caller can catch and decide.
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshot dir not found: {snapshot_dir}")

    meta_path = snapshot_dir / _METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {_METADATA_FILE} in {snapshot_dir}")
    metadata = SnapshotMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))

    if metadata.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema_version={metadata.schema_version} does not match "
            f"current SCHEMA_VERSION={SCHEMA_VERSION} (snapshot={snapshot_dir})"
        )

    template_path = snapshot_dir / _TEMPLATE_FILE
    if not template_path.exists():
        raise FileNotFoundError(f"Missing {_TEMPLATE_FILE} in {snapshot_dir}")
    template = template_path.read_text(encoding="utf-8")

    inputs = _load_optional_json(snapshot_dir / _INPUTS_FILE)
    expected = _load_optional_json(snapshot_dir / _EXPECTED_FILE)
    readme_path = snapshot_dir / _README_FILE
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None

    return Snapshot(
        metadata=metadata,
        template=template,
        inputs=inputs,
        expected=expected,
        readme=readme,
    )


def list_snapshots(root: Path) -> List[Path]:
    """Return all snapshot directories under ``root`` (one level deep).

    A directory is treated as a snapshot if it contains ``metadata.json``.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / _METADATA_FILE).exists())


# --- internal -----------------------------------------------------------------


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    return json.loads(text)


def _default_readme(metadata: SnapshotMetadata) -> str:
    return (
        f"# {metadata.name}\n\n"
        f"- **Stage**: `{metadata.stage}`\n"
        f"- **Source**: {metadata.source}\n"
        f"- **Original model**: {metadata.original_model or '_unknown_'}\n"
        f"- **Captured**: {metadata.created_at}\n"
        f"- **Scrubbed**: {metadata.scrubbed}\n"
        + (f"\n{metadata.notes}\n" if metadata.notes else "\n")
        + "\nDescribe why this case is interesting -- weak-model failure mode, "
        "regression, edge case in the prompt, etc.\n"
    )
