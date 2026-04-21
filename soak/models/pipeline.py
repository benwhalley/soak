"""QualitativeAnalysisPipeline with HTML export capability."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .base import QualitativeAnalysis
from .dag import DAG

logger = logging.getLogger(__name__)

from pydantic import PrivateAttr


class QualitativeAnalysisPipeline(DAG):
    _cost_summary: dict = PrivateAttr(default_factory=dict)
    _cached_model_dump: Optional[dict] = PrivateAttr(default=None)

    name: Optional[str] = None

    def get_model_dump(self) -> dict:
        """Get cached model_dump(mode='json'), computing once if needed."""
        if self._cached_model_dump is None:
            self._cached_model_dump = self.model_dump(mode="json")
        return self._cached_model_dump

    def to_html(self, template_path: Optional[str] = None) -> str:
        """Render the analysis as HTML using Jinja2 template from file.

        Args:
            template_path: Path to the HTML template file. If None, uses default template.

        Returns:
            Rendered HTML string.
        """
        if template_path is None:
            # Use default template in soak/templates directory
            template_dir = Path(__file__).parent / "templates"
            template_name = "pipeline.html"
        else:
            # Use provided template path
            template_path = Path(template_path)
            template_dir = template_path.parent
            template_name = template_path.name

        # Create Jinja2 environment and load template
        env = Environment(
            loader=FileSystemLoader([template_dir, template_dir / "nodes"]),
            extensions=["jinja_markdown.MarkdownExtension"],
        )

        # Add custom filter to convert DataFrames to HTML
        def df_to_html(df, show_index=None):
            """Convert pandas DataFrame to HTML table.

            Args:
                df: DataFrame to convert
                show_index: Whether to show index. If None, auto-detects based on index name or type.
            """
            if df is None or (hasattr(df, "empty") and df.empty):
                return "<p><em>No data</em></p>"

            # Auto-detect if index should be shown
            if show_index is None:
                # Show index if it has a name or is not a simple RangeIndex
                show_index = df.index.name is not None or not isinstance(
                    df.index, pd.RangeIndex
                )

            return df.to_html(
                classes="table table-sm table-striped", index=show_index, escape=True
            )

        env.filters["df_to_html"] = df_to_html

        # Add enumerate filter for templates
        def enumerate_filter(iterable):
            """Enumerate filter for Jinja2."""
            return list(enumerate(iterable))

        env.filters["enumerate"] = enumerate_filter

        # Add safe JSON filter that handles DataFrames
        def safe_tojson(obj, indent=2):
            """Safely convert to JSON, converting DataFrames to records."""
            import json

            import pandas as pd

            def convert_value(v):
                if isinstance(v, pd.DataFrame):
                    return v.to_dict("records")
                elif isinstance(v, dict):
                    return {k: convert_value(val) for k, val in v.items()}
                elif isinstance(v, list):
                    return [convert_value(item) for item in v]
                else:
                    return v

            try:
                converted = convert_value(obj)
                return json.dumps(converted, indent=indent, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)}, indent=indent)

        env.filters["safe_tojson"] = safe_tojson

        def truncatewords(value, n=30):
            """Truncate text to n words."""
            words = str(value).split()
            return " ".join(words[:n]) + ("..." if len(words) > n else "")

        env.filters["truncatewords"] = truncatewords

        # Add custom function to render individual nodes
        def render_node(node):
            """Render a node using its type-specific template."""
            node_template_name = f"{node.type.lower()}.html"
            nodes_template_dir = Path(__file__).parent.parent / "templates" / "nodes"

            try:
                # Try to load node-specific template
                if (nodes_template_dir / node_template_name).exists():
                    node_template = env.get_template(node_template_name)
                else:
                    # Fall back to default node template
                    raise Exception(
                        f"Node template not found: {(nodes_template_dir / node_template_name)}"
                    )
                    node_template = env.get_template("default.html")

                # Get node result with metadata
                try:
                    node_result = node.result()
                except Exception as e:
                    logger.warning(f"Error getting result for node {node.name}: {e}")
                    node_result = {
                        "metadata": {"name": node.name, "type": node.type},
                        "error": str(e),
                    }

                return node_template.render(node=node, result=node_result)
            except Exception as e:
                logger.error(f"Error rendering node {node.name}: {e}")
                return f"<div class='alert alert-danger'>Error rendering node {node.name}: {e}</div>"

        env.globals["render_node"] = render_node

        template = env.get_template(template_name)

        # Get execution order for display
        execution_order = self.get_execution_order()

        # Render template with data
        # use cached model_dump to avoid recomputing for each template
        dd = self.get_model_dump().copy()
        dd["config"] = dd["config"].copy()  # shallow copy to avoid mutating cache
        dd["config"]["documents"] = []

        # Get soak version for footer
        try:
            import importlib.metadata

            soak_version = importlib.metadata.version("soaking")
        except importlib.metadata.PackageNotFoundError:
            soak_version = "dev"

        # compute self-similarity matrix for simple.html template
        self_similarity = None
        themes_json = "[]"
        self_similarity_json = "null"
        verification_data = None
        themes_with_codes = []

        if "simple" in template_name:
            import json

            try:
                analysis_result = self.result()
                themes_data = []

                # extract theme data for JSON embedding
                if analysis_result.themes:
                    for t in analysis_result.themes:
                        if hasattr(t, "model_dump"):
                            themes_data.append(t.model_dump())
                        elif isinstance(t, dict):
                            themes_data.append(t)
                        else:
                            themes_data.append({"name": str(t), "description": ""})

                themes_json = json.dumps(themes_data, default=str)

                # compute self-similarity if we have multiple themes
                if len(themes_data) > 1:
                    try:
                        from ..analysis.self_similarity import \
                            compute_self_similarity_matrix

                        self_similarity = compute_self_similarity_matrix(themes_data)
                        self_similarity_json = json.dumps(self_similarity, default=str)
                    except ImportError:
                        logger.debug("Self-similarity module not available")
                    except Exception as e:
                        logger.warning(f"Could not compute self-similarity: {e}")

                # extract verification data from checkquotes node if present
                checkquotes_node = self.nodes_dict.get("checkquotes")
                if checkquotes_node and checkquotes_node.output:
                    try:
                        matches = []
                        if hasattr(checkquotes_node, "sentence_matches"):
                            matches = checkquotes_node.sentence_matches or []
                        elif isinstance(checkquotes_node.output, dict):
                            matches = checkquotes_node.output.get(
                                "sentence_matches", []
                            )

                        if matches:
                            # count flagged quotes
                            flagged_count = sum(
                                1 for m in matches if m.get("llm_is_contained") is False
                            )
                            verification_data = {
                                "matches": matches,
                                "flagged": flagged_count,
                                "total": len(matches),
                            }
                    except Exception as e:
                        logger.debug(f"Could not extract verification data: {e}")

                # build themes with their related codes for expandable display
                if analysis_result.themes:
                    codes_by_ref = {}
                    if analysis_result.codes:
                        for code in analysis_result.codes:
                            # index by hash
                            code_hash = (
                                code.hash() if hasattr(code, "hash") else None
                            )
                            if code_hash:
                                codes_by_ref[code_hash] = code
                            # also index by slug for backward compat
                            slug = (
                                code.slug if hasattr(code, "slug") else code.get("slug")
                            )
                            if slug:
                                codes_by_ref[slug] = code

                    for theme in analysis_result.themes:
                        theme_dict = (
                            theme.model_dump()
                            if hasattr(theme, "model_dump")
                            else dict(theme)
                        )
                        code_refs = theme_dict.get(
                            "code_hashes", theme_dict.get("code_slugs", [])
                        )
                        related_codes = []
                        total_quotes = 0

                        for ref in code_refs:
                            code = codes_by_ref.get(ref)
                            if code:
                                code_dict = (
                                    code.model_dump()
                                    if hasattr(code, "model_dump")
                                    else dict(code)
                                )
                                quotes = code_dict.get(
                                    "all_quotes", code_dict.get("quotes", [])
                                )
                                if isinstance(quotes, list):
                                    total_quotes += len(quotes)
                                related_codes.append(code_dict)

                        themes_with_codes.append(
                            {
                                **theme_dict,
                                "related_codes": related_codes,
                                "total_quotes": total_quotes,
                            }
                        )

            except Exception as e:
                logger.debug(f"Could not prepare template data: {e}")

        return template.render(
            pipeline=self,
            result=self.result().model_dump(),
            detail=dd,
            execution_order=execution_order,
            soak_version=soak_version,
            self_similarity=self_similarity,
            themes_json=themes_json,
            self_similarity_json=self_similarity_json,
            verification_data=verification_data,
            themes_with_codes=themes_with_codes,
        )

    def result(self):
        """Extract QualitativeAnalysis result from pipeline execution."""

        def safe_get_output(name, key):
            """Extract output from node, handling both live and serialized formats.

            For live StruckdownResult: uses .outputs property
            For deserialized dict: navigates results[key]['output']
            For Reduce nodes: direct list of items (codes, themes)
            """
            try:
                from soak.models.base import Code, CodeList, Theme, Themes

                node = self.nodes_dict.get(name)
                if not node or not node.output:
                    return None

                # Handle list output (StruckdownResult or deserialized dict or direct items)
                if isinstance(node.output, list) and len(node.output) > 0:
                    first_item = node.output[0]

                    # Check if this is a direct list of Code/Theme items (from Reduce nodes)
                    # rather than a StruckdownResult wrapper
                    if isinstance(first_item, Code):
                        return node.output
                    if isinstance(first_item, Theme):
                        return node.output
                    if isinstance(first_item, dict):
                        # Check if it's a Code or Theme dict
                        if "quotes" in first_item and ("slug" in first_item or "name" in first_item):
                            # This is a list of Code dicts from Reduce node
                            return node.output
                        if "code_hashes" in first_item or "code_slugs" in first_item:
                            # This is a list of Theme dicts from Reduce node
                            return node.output

                    # Otherwise treat as StruckdownResult or dict wrapper
                    complete = first_item
                    segment_output = None

                    # Live StruckdownResult: use .outputs property
                    if hasattr(complete, "outputs"):
                        outputs = complete.outputs
                        if key in outputs:
                            segment_output = outputs[key]

                    # Deserialized dict: navigate results[key]['output']
                    elif isinstance(complete, dict) and "results" in complete:
                        results = complete.get("results", {})
                        if key in results:
                            segment_output = results[key].get("output")

                    if segment_output is None:
                        return None

                    # Handle CodeList: extract .codes attribute
                    if isinstance(segment_output, CodeList):
                        return segment_output.codes

                    # Handle Themes: extract .themes attribute
                    if isinstance(segment_output, Themes):
                        return segment_output.themes

                    # For dict format: {'codes': [...]} or {'themes': [...]}
                    if isinstance(segment_output, dict) and key in segment_output:
                        return segment_output[key]

                    # For narrative or other direct values
                    return segment_output

                return None
            except Exception as e:
                logger.debug(f"Error getting output {name}.{key}: {e}")
                return None

        codes = safe_get_output("codes", "codes") or []
        themes = safe_get_output("themes", "themes") or []
        narrative = safe_get_output("narrative", "report") or ""

        return QualitativeAnalysis(
            codes=codes,
            themes=themes,
            narrative=narrative,
            name=self.name or "analysis",
        )
