"""Aggregate ``tests/evals/results/*.jsonl`` into a markdown comparison table.

This module is the single source of truth for how probe results are
presented. Both ``scripts/eval_report.py`` and the Django
``manage.py eval_report`` command call into ``render_report`` here.

Each entry is one JSON line per (probe, model) call. ``render_report``
picks the **latest** line per pair across all date-stamped files and
emits one markdown section per probe. Columns are tailored per probe
so the table is meaningful (different metrics matter for ``schema``
vs. ``long_context``).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .results import _default_results_dir

logger = logging.getLogger(__name__)


def _default_out() -> Path:
    """Markdown report goes next to the results dir by default.

    e.g. ``./soaking-eval/results/`` paired with ``./soaking-eval/llm_evals.md``.
    """
    return _default_results_dir().parent / "llm_evals.md"


# Eager evaluation for callers that read it at import time (CLI default
# values etc.). Use ``_default_out()`` if you need the freshest CWD-aware
# value at call time.
DEFAULT_OUT = _default_out()


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load every JSON line from every ``*.jsonl`` under ``results_dir``."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("bad json in %s", path)
    return rows


def latest_per_pair(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Keep the most recent entry per (probe, model)."""
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        probe = row.get("probe", "?")
        model = row.get("model", "?")
        recorded_at = row.get("recorded_at") or "1970-01-01T00:00:00"
        key = (probe, model)
        prev = by_pair.get(key)
        if prev is None or recorded_at > prev.get("recorded_at", ""):
            by_pair[key] = row
    return by_pair


def _check(row: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for c in row.get("checks") or []:
        if c.get("name") == name:
            return c
    return None


def _render_table(
    headers: List[str],
    rows: List[List[str]],
    aligns: Optional[List[str]] = None,
) -> str:
    """Render a markdown table with padded columns so the raw source aligns.

    ``aligns`` is a list of ``"l" | "c" | "r"`` -- one per column. Default
    left for text columns. The alignment is encoded in the separator row
    (``|:--|``, ``|:--:|``, ``|--:|``) so renderers honour it AND the raw
    text lines up nicely in PR diffs and editors.
    """
    if not rows:
        rows = []
    aligns = aligns or ["l"] * len(headers)

    # column widths from headers + every row
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _pad(cell: str, width: int, align: str) -> str:
        cell = str(cell)
        if align == "r":
            return cell.rjust(width)
        if align == "c":
            return cell.center(width)
        return cell.ljust(width)

    def _sep(width: int, align: str) -> str:
        # Minimum 3 dashes per markdown spec; we add the alignment colons
        # at the ends so the rendered HTML aligns the same way the raw
        # source does.
        bar = "-" * max(width, 3)
        if align == "l":
            return ":" + bar[1:]
        if align == "r":
            return bar[:-1] + ":"
        if align == "c":
            return ":" + bar[1:-1] + ":"
        return bar

    head_line = "| " + " | ".join(_pad(h, widths[i], aligns[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join(_sep(widths[i], aligns[i]) for i in range(len(headers))) + " |"

    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(_pad(c, widths[i], aligns[i]) for i, c in enumerate(row))
            + " |"
        )

    return "\n".join([head_line, sep_line, *body])


def _bool_cell(b: Optional[bool]) -> str:
    if b is None:
        return "-"
    return "✅" if b else "❌"


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.0f}%"


def _money(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x == 0:
        return "$0.00"
    return f"${x:.4f}"


def _seconds(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.0f}s"


def _format_consolidate(rows: List[Dict[str, Any]]) -> str:
    """Consolidate table: full metric surface.

    Columns:
      model | schema | quote refs | code refs | coverage | unused | uniq Q |
      n codes | n cons | n themes | cost | latency
    """
    headers = [
        "model", "schema", "quote refs", "code refs", "coverage", "unused Q",
        "uniq Q", "n codes", "n cons", "n themes", "cost", "latency",
    ]
    aligns = ["l", "c", "c", "c", "r", "r", "r", "r", "r", "r", "r", "r"]
    body = []
    for row in sorted(rows, key=lambda r: r.get("model", "")):
        quote_check = _check(row, "Consolidated quote references valid")
        code_check = _check(row, "Theme code references valid")
        stats = row.get("stats") or {}
        body.append([
            row.get("model", "?"),
            _bool_cell(row.get("schema_valid")),
            _bool_cell(quote_check.get("passed") if quote_check else None),
            _bool_cell(code_check.get("passed") if code_check else None),
            _pct(stats.get("quote_coverage")),
            str(stats.get("unused_quotes", 0) or 0),
            str(stats.get("unique_quotes", 0) or 0),
            str(stats.get("num_codes", 0) or 0),
            str(stats.get("num_consolidated", 0) or 0),
            str(stats.get("num_themes", 0) or 0),
            _money(row.get("total_cost")),
            _seconds(row.get("duration_seconds")),
        ])
    return _render_table(headers, body, aligns)


def _format_schema(rows: List[Dict[str, Any]]) -> str:
    """Schema table: simple counts. Tool-calling smoke test."""
    headers = ["model", "schema", "n codes", "n quotes", "quotes/code", "cost", "latency"]
    aligns = ["l", "c", "r", "r", "r", "r", "r"]
    body = []
    for row in sorted(rows, key=lambda r: r.get("model", "")):
        stats = row.get("stats") or {}
        n_codes = stats.get("num_codes") or 0
        n_quotes = stats.get("quotes_in_codes") or 0
        qpc = (n_quotes / n_codes) if n_codes else 0
        body.append([
            row.get("model", "?"),
            _bool_cell(row.get("schema_valid")),
            str(n_codes),
            str(n_quotes),
            f"{qpc:.1f}",
            _money(row.get("total_cost")),
            _seconds(row.get("duration_seconds")),
        ])
    return _render_table(headers, body, aligns)


def _format_long_context(rows: List[Dict[str, Any]]) -> str:
    """Long-context consolidate table.

    Columns: model | schema | refs ok | trunc ok | halluc# | trunc# |
    in codes | in Q | out codes | refs Q | coverage | cost | latency
    """
    headers = [
        "model", "schema", "refs", "trunc", "halluc", "trunc#", "in codes",
        "in Q", "out codes", "refs Q", "coverage", "cost", "latency",
    ]
    aligns = ["l", "c", "c", "c", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
    body = []
    for row in sorted(rows, key=lambda r: r.get("model", "")):
        ref_check = _check(row, "Consolidated quote references valid")
        trunc_check = _check(row, "No truncated/malformed hashes")
        stats = row.get("stats") or {}
        body.append([
            row.get("model", "?"),
            _bool_cell(row.get("schema_valid")),
            _bool_cell(ref_check.get("passed") if ref_check else None),
            _bool_cell(trunc_check.get("passed") if trunc_check else None),
            str(len(stats.get("hallucinated_refs") or [])),
            str(len(stats.get("truncated_refs") or [])),
            str(stats.get("input_codes", 0) or 0),
            str(stats.get("input_quotes", 0) or 0),
            str(stats.get("consolidated_codes", 0) or 0),
            str(stats.get("referenced_quotes", 0) or 0),
            _pct(stats.get("quote_coverage")),
            _money(row.get("total_cost")),
            _seconds(row.get("duration_seconds")),
        ])
    return _render_table(headers, body, aligns)


def _format_themes_long(rows: List[Dict[str, Any]]) -> str:
    """Themes-long table.

    Columns: model | schema | refs ok | trunc ok | halluc# | trunc# |
    in codes | n themes | refs C | coverage | cost | latency
    """
    headers = [
        "model", "schema", "refs", "trunc", "halluc", "trunc#",
        "in codes", "n themes", "refs C", "coverage", "cost", "latency",
    ]
    aligns = ["l", "c", "c", "c", "r", "r", "r", "r", "r", "r", "r", "r"]
    body = []
    for row in sorted(rows, key=lambda r: r.get("model", "")):
        ref_check = _check(row, "Theme code references valid")
        trunc_check = _check(row, "No truncated/malformed hashes")
        stats = row.get("stats") or {}
        body.append([
            row.get("model", "?"),
            _bool_cell(row.get("schema_valid")),
            _bool_cell(ref_check.get("passed") if ref_check else None),
            _bool_cell(trunc_check.get("passed") if trunc_check else None),
            str(len(stats.get("hallucinated_refs") or [])),
            str(len(stats.get("truncated_refs") or [])),
            str(stats.get("input_codes", 0) or 0),
            str(stats.get("num_themes", 0) or 0),
            str(stats.get("theme_code_refs", 0) or 0),
            _pct(stats.get("code_coverage")),
            _money(row.get("total_cost")),
            _seconds(row.get("duration_seconds")),
        ])
    return _render_table(headers, body, aligns)


_FORMATTERS = {
    "consolidate": _format_consolidate,
    "schema": _format_schema,
    "long_context": _format_long_context,
    "themes_long": _format_themes_long,
}


def _html_escape_cell(s: str) -> str:
    """Minimal HTML escape for table cell content.

    We render the side-by-side examples as raw HTML inside a markdown
    document so pandoc / GitHub render them correctly. Pandoc does NOT
    process markdown inside raw HTML blocks, so we use ``<strong>`` /
    ``<em>`` / ``<code>`` directly rather than ``**``/``_``/backticks.
    """
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip()


def _format_code_cell(item: Dict[str, Any]) -> str:
    """Compact rendering of one Code (name + desc + quotes) for a table cell."""
    parts = [f"<strong>{_html_escape_cell(item.get('name', '?'))}</strong>"]
    desc = _html_escape_cell(item.get("description", ""))
    if desc:
        parts.append(f"<em>{desc}</em>")
    quotes = item.get("quotes") or []
    for q in quotes:
        text = _html_escape_cell(q.get("text", ""))
        if text:
            # Use real unicode curly quotes (U+201C/U+201D) so the
            # markdown source stays readable too.
            parts.append(f"“{text}”")
    return "<br/><br/>".join(parts)


def _format_theme_cell(item: Dict[str, Any]) -> str:
    """Compact rendering of one Theme (name + desc + resolved code names)."""
    parts = [f"<strong>{_html_escape_cell(item.get('name', '?'))}</strong>"]
    desc = _html_escape_cell(item.get("description", ""))
    if desc:
        parts.append(f"<em>{desc}</em>")
    codes = item.get("codes") or []
    if codes:
        bullets = "<br/>".join(
            f"• <code>{_html_escape_cell(c.get('hash','?'))}</code> &mdash; "
            f"{_html_escape_cell(c.get('name', '?'))}"
            for c in codes
        )
        parts.append(bullets)
    n_total = item.get("n_codes", len(codes))
    if n_total > len(codes):
        parts.append(f"<em>…and {n_total - len(codes)} more</em>")
    return "<br/><br/>".join(parts)


def _render_samples(rows: List[Dict[str, Any]]) -> str:
    """Render the ``samples`` payload as side-by-side comparison tables.

    For each *slot* present in the probe (``codes``, ``consolidated``,
    ``themes``), build one table whose **columns are models** and whose
    **rows are output positions** (1st item, 2nd item, ...). This lets a
    reviewer compare what each model said in the same position at a glance.

    The whole block is wrapped in ``<details>`` so the metrics tables
    stay above the fold.
    """
    rows_with_samples = [r for r in rows if r.get("samples")]
    if not rows_with_samples:
        return ""

    by_model = {r["model"]: r["samples"] for r in rows_with_samples}
    models = sorted(by_model)

    # collect all distinct slot names that appear in any model
    slot_keys: List[str] = []
    seen = set()
    for sample in by_model.values():
        for key in sample.keys():
            if key not in seen:
                seen.add(key)
                slot_keys.append(key)

    blocks: List[str] = []
    blocks.append("<details><summary>Side-by-side examples (click to expand)</summary>\n")

    for slot in slot_keys:
        # Per-model item lists for this slot
        per_model_items: Dict[str, List[Dict[str, Any]]] = {
            m: by_model[m].get(slot) or [] for m in models
        }
        max_rows = max((len(items) for items in per_model_items.values()), default=0)
        if max_rows == 0:
            continue

        formatter = _format_theme_cell if slot == "themes" else _format_code_cell

        blocks.append(f"\n#### `{slot}`\n")

        # Use HTML table -- markdown tables don't handle multi-line bold +
        # quotes cleanly across many columns, and the column widths can
        # vary wildly per cell.
        blocks.append("<table>")
        blocks.append("<thead><tr><th>#</th>"
                      + "".join(f"<th><code>{m}</code></th>" for m in models)
                      + "</tr></thead>")
        blocks.append("<tbody>")
        for i in range(max_rows):
            cells = []
            for m in models:
                items = per_model_items[m]
                if i < len(items):
                    cells.append(f"<td>{formatter(items[i])}</td>")
                else:
                    cells.append("<td><i>—</i></td>")
            blocks.append(f"<tr><td>{i+1}</td>{''.join(cells)}</tr>")
        blocks.append("</tbody></table>\n")

    blocks.append("</details>\n")
    return "\n".join(blocks)


def render_report(by_pair: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    """Render a markdown report from the latest-per-pair dict."""
    by_probe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (probe, _), row in by_pair.items():
        by_probe[probe].append(row)

    parts: List[str] = []
    parts.append("# LLM capability eval results")
    parts.append("")
    parts.append(
        f"_Auto-generated by `soak.evals.report` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._"
    )
    parts.append("")
    parts.append(
        "Each row is the most recent recorded run per (probe, model). "
        "Run `pytest -m llm tests/evals --eval-mode` or "
        "`manage.py eval_probe --all-models --probes <list>` to refresh."
    )
    parts.append("")

    probe_order = ["schema", "consolidate", "long_context", "themes_long"]
    seen = set(by_probe)
    ordered = [p for p in probe_order if p in seen] + sorted(seen - set(probe_order))

    for probe in ordered:
        formatter = _FORMATTERS.get(probe)
        rows = by_probe[probe]
        parts.append(f"## `{probe}`")
        parts.append("")
        if formatter is None:
            parts.append(f"_(no formatter registered for probe `{probe}`)_")
            parts.append("")
            continue
        parts.append(formatter(rows))
        parts.append("")

        # Qualitative samples for human review of model output quality.
        # Skipped if no row in this probe carries samples (older runs).
        samples_block = _render_samples(rows)
        if samples_block.strip():
            parts.append("### Examples (click to expand)")
            parts.append("")
            parts.append(samples_block)
            parts.append("")

    return "\n".join(parts) + "\n"


# Minimal CSS for the pandoc-rendered HTML so the wide side-by-side
# comparison tables stay readable: borders, column-cap widths, sticky
# header, narrow vertical layout for long cells.
_HTML_CSS = """
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       max-width: 1400px; margin: 2em auto; padding: 0 1em;
       color: #222; line-height: 1.45; }
h1, h2, h3, h4 { line-height: 1.2; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
table { border-collapse: collapse; margin: 1em 0; max-width: 100%; }
th, td { vertical-align: top; padding: 8px 10px; border: 1px solid #ddd;
         max-width: 360px; font-size: 0.9em; }
th { background: #fafafa; position: sticky; top: 0; }
details { margin: 1em 0; }
details > summary { cursor: pointer; padding: 4px 0; font-weight: 600; }
"""


def _try_render_html(md_path: Path) -> Optional[Path]:
    """Run pandoc on ``md_path`` if available; return the HTML path or None.

    Failures (pandoc missing, conversion error) are logged at INFO and
    the markdown write is treated as the canonical output. We don't
    re-raise: HTML is a convenience, not a hard requirement.
    """
    if shutil.which("pandoc") is None:
        logger.info("pandoc not on PATH; skipping HTML render")
        return None

    html_path = md_path.with_suffix(".html")
    title = md_path.stem.replace("_", " ").title()

    try:
        subprocess.run(
            [
                "pandoc",
                str(md_path),
                "-o", str(html_path),
                "--from", "gfm+raw_html",
                "--to", "html5",
                "--standalone",
                "--metadata", f"title={title}",
                "-V", f"header-includes=<style>{_HTML_CSS}</style>",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("pandoc failed: %s", exc.stderr.strip()[:300])
        return None

    return html_path


def build_report(
    results_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
    render_html: bool = True,
) -> Tuple[str, int, Optional[Path]]:
    """High-level helper: load + dedupe + render markdown (+ optional HTML).

    Args:
        results_dir: defaults to ``_default_results_dir()``.
        out_path: if given, the markdown is also written here. ``None``
            returns the markdown string only.
        render_html: if ``True`` (default) and ``out_path`` is given,
            also try to render an HTML sibling via pandoc. Silently
            skipped if pandoc is missing.

    Returns:
        ``(markdown, n_pairs, html_path)``. ``html_path`` is ``None``
        when pandoc is unavailable, ``out_path`` is missing, or html
        rendering was disabled.
    """
    src = Path(results_dir) if results_dir else _default_results_dir()
    if not src.exists():
        raise FileNotFoundError(f"results dir not found: {src}")

    rows = load_results(src)
    if not rows:
        raise FileNotFoundError(f"no result files in {src}")

    by_pair = latest_per_pair(rows)
    markdown = render_report(by_pair)

    html_path: Optional[Path] = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        if render_html:
            html_path = _try_render_html(out_path)

    return markdown, len(by_pair), html_path
