#!/usr/bin/env python3
"""CLI shim for ``soak.evals.report``.

The actual rendering lives in the ``soak.evals.report`` module so the
Django ``manage.py eval_report`` command (in soakresearch) and pytest
sessions can use the same code path.

Usage::

    python scripts/eval_report.py                    # writes docs/llm_evals.md
    python scripts/eval_report.py --out -            # stdout
    python scripts/eval_report.py --results <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from soak.evals.report import DEFAULT_OUT, build_report
from soak.evals.results import DEFAULT_RESULTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"Output path or '-' for stdout (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--no-html",
        dest="render_html",
        action="store_false",
        help="Skip the pandoc-based HTML render (default: render if pandoc available).",
    )
    parser.set_defaults(render_html=True)
    args = parser.parse_args()

    try:
        out_path = None if args.out == "-" else Path(args.out)
        markdown, n, html_path = build_report(
            results_dir=args.results,
            out_path=out_path,
            render_html=args.render_html,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.out == "-":
        sys.stdout.write(markdown)
    else:
        print(f"wrote {args.out} ({n} entries)", file=sys.stderr)
        if html_path:
            print(f"wrote {html_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
