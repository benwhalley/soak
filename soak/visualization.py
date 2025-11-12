"""Visualization utilities for DAG structures."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from soak.models import DAG

logger = logging.getLogger(__name__)


ICON_MAP = {
    "GroupBy": "⧉",
    "Scrub": "⦸",
    "Split": "⩤",
    "Map": "⠿",
    "Reduce": " ⩥",
    "Transform": "∂",
    "VerifyQuotes": "✓",
    "Batch": "⧉",
    "Classifier": "⊢",
    "Filter": "∈",
}

MERMAID_SHAPE_MAP = {
    "GroupBy": ("{", "}"),
    "Scrub": ("[[", "]]"),
    "Split": ("{", "}"),
    "Map": ("[[", "]]"),
    "Reduce": ("[", "]"),
    "Transform": ("[", "]"),
    "VerifyQuotes": ("[[", "]]"),
    "Batch": ("[[", "]]"),
    "Classifier": ("[[", "]]"),
    "Filter": ("[[", "]]"),   #
}

GRAPHVIZ_SHAPE_MAP = {
    "GroupBy": "diamond",
    "Batch": "diamond",
    "Split": "ellipse",
    # all others default to "box"
}


def dag_to_mermaid(dag: "DAG") -> str:
    """Generate a Mermaid diagram of the DAG structure with shapes by node type.

    Args:
        dag: The DAG instance to visualize

    Returns:
        A Mermaid flowchart definition string
    """
    lines = ["flowchart TD"]

    # Generate node definitions with appropriate shapes
    for node in dag.nodes:
        le, ri = MERMAID_SHAPE_MAP.get(node.type, ("[", "]"))  # fallback to rectangle
        label = f"{ICON_MAP.get(node.type, '')} <span class='node-type small'>{node.type}</span><br><code>{node.name}</code>"
        lines.append(f"    {node.name}{le}{label}{ri}")

    # Generate edges
    for edge in dag.edges:
        lines.append(f"    {edge.from_node} --> {edge.to_node}")

    return "\n".join(lines)


def dag_to_graphviz(dag: "DAG") -> str:
    """Generate a Graphviz DOT diagram of the DAG structure.

    Args:
        dag: The DAG instance to visualize

    Returns:
        A DOT format string for Graphviz
    """
    lines = ["digraph G {"]
    lines.append("    rankdir=TB;")
    lines.append('    node [fontname="Helvetica", fontsize=11, style=rounded];')
    lines.append("")

    # generate node definitions with appropriate shapes
    for node in dag.nodes:
        shape = GRAPHVIZ_SHAPE_MAP.get(node.type, "box")
        icon = ICON_MAP.get(node.type, "")
        label = f"{icon} {node.type}\\n{node.name}"
        lines.append(f'    {node.name} [label="{label}", shape={shape}];')

    lines.append("")

    # generate edges
    for edge in dag.edges:
        lines.append(f"    {edge.from_node} -> {edge.to_node};")

    lines.append("}")
    return "\n".join(lines)




def render_graphviz_to_pdf(dot_definition: str, output_path: Path) -> Optional[Path]:
    """Render a Graphviz DOT definition to a PDF file.

    Args:
        dot_definition: The DOT format string
        output_path: Path where the PDF should be saved

    Returns:
        Path to the rendered file if successful, None otherwise
    """
    try:
        import subprocess
        import tempfile

        # ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # write DOT definition to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_definition)
            dot_file = f.name

        try:
            # render with dot command
            result = subprocess.run(
                ['dot', '-Tpdf', '-o', str(output_path), dot_file],
                check=True,
                capture_output=True,
                text=True
            )

            logger.info(f"✓ Graphviz diagram saved to {output_path}")
            return output_path

        finally:
            # cleanup temp file
            Path(dot_file).unlink(missing_ok=True)

    except FileNotFoundError:
        logger.warning(
            "Graphviz 'dot' command not found. Install graphviz system package: "
            "brew install graphviz (macOS) or apt-get install graphviz (Linux)"
        )
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to render Graphviz diagram: {e.stderr}")
        return None
    except Exception as e:
        logger.warning(f"Failed to render Graphviz diagram: {e}")
        return None
