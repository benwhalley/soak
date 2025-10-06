"""Visualization utilities for DAG structures."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soak.models import DAG


# Shape mapping for different node types in Mermaid diagrams
MERMAID_SHAPE_MAP = {
    "Split": ("(", ")"),  # round edges
    "Map": ("[[", "]]"),  # standard rectangle
    "Reduce": ("{{", "}}"),  # hexagon
    "Transform": (">", "]"),  #
    "TransformReduce": (">", "]"),  #
    "VerifyQuotes": ("[[", "]]"),  #
    "Batch": ("[[", "]]"),  # subroutine shape
    "Classifier": ("[/", "\\]"),  # parallelogram (for input/output operations)
    "Filter": ("[[", "]]"),  # standard rectangle (filtering operation)
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
        label = f"{node.type}: {node.name}"
        lines.append(f"    {node.name}{le}{label}{ri}")

    # Generate edges
    for edge in dag.edges:
        lines.append(f"    {edge.from_node} --> {edge.to_node}")

    # Add styling
    lines.append("""classDef heavyDotted stroke-dasharray: 4 4, stroke-width: 2px;""")
    for node in dag.nodes:
        if node.type == "TransformReduce":
            lines.append("""class all_themes heavyDotted;""")

    return "\n".join(lines)
