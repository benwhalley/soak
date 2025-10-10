"""QualitativeAnalysisPipeline with HTML export capability."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .base import QualitativeAnalysis
from .dag import DAG

logger = logging.getLogger(__name__)


class QualitativeAnalysisPipeline(DAG):
    name: Optional[str] = None

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
        # model_dump with mode='json' ensures all data is JSON-serializable
        dd = self.model_dump(mode="json")
        dd["config"]["documents"] = []
        return template.render(
            pipeline=self,
            result=self.result(),
            detail=dd,
            execution_order=execution_order,
        )

    def result(self):
        """Extract QualitativeAnalysis result from pipeline execution."""

        def safe_get_output(name):
            try:
                return self.nodes_dict.get(name).output.response
            except Exception:
                logger.warning(f"Error getting output {name}")
                return None

        try:
            codes = self.nodes_dict.get("codes").output.response["codes"]
        except Exception:
            codes = []
        try:
            themes = self.nodes_dict.get("themes").output.response["themes"]
        except Exception:
            themes = []

        try:
            narrative = self.nodes_dict.get("narrative").output.response.get(
                "report", ""
            )
        except Exception:
            narrative = ""

        return QualitativeAnalysis(
            codes=codes,
            themes=themes,
            narrative=narrative,
            name=self.name or "analysis",
        )
