"""Result classes for soak API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..models import (Code, QualitativeAnalysis,
                          QualitativeAnalysisPipeline, Theme)


@dataclass
class CostSummary:
    """Summary of LLM API costs."""

    total_cost: float = 0.0
    fresh_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    fresh_count: int = 0
    cached_count: int = 0
    has_unknown_costs: bool = False
    all_costs_unknown: bool = False
    by_node: dict[str, dict[str, Any]] = field(default_factory=dict)

    def format(self, include_breakdown: bool = True) -> str:
        """Format cost summary as string."""
        from struckdown import CostSummary as SDCostSummary

        sd_summary = SDCostSummary(
            total_cost=self.total_cost,
            fresh_cost=self.fresh_cost,
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            fresh_count=self.fresh_count,
            cached_count=self.cached_count,
            has_unknown_costs=self.has_unknown_costs,
            all_costs_unknown=self.all_costs_unknown,
        )
        lines = [sd_summary.format_summary(include_breakdown=include_breakdown)]

        total_calls = self.fresh_count + self.cached_count
        if total_calls > 0:
            lines.append(f"  Total API calls: {total_calls}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "CostSummary":
        """Create from pipeline cost summary dict."""
        return cls(
            total_cost=data.get("total_cost", 0.0),
            fresh_cost=data.get("fresh_cost", 0.0),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            fresh_count=data.get("fresh_count", 0),
            cached_count=data.get("cached_count", 0),
            has_unknown_costs=data.get("has_unknown_costs", False),
            all_costs_unknown=data.get("all_costs_unknown", False),
            by_node=data.get("by_node", {}),
        )


@dataclass
class RunResult:
    """Result from running a pipeline."""

    pipeline: "QualitativeAnalysisPipeline"
    output_folder: Optional[Path] = None
    errors: list[str] = field(default_factory=list)
    _cost_summary: Optional[dict] = field(default=None, repr=False)

    @property
    def analysis(self) -> "QualitativeAnalysis":
        """Get the QualitativeAnalysis result."""
        return self.pipeline.result()

    @property
    def themes(self) -> list["Theme"]:
        """Get themes from the analysis."""
        return self.analysis.themes

    @property
    def codes(self) -> list["Code"]:
        """Get codes from the analysis."""
        return self.analysis.codes

    @property
    def cost_summary(self) -> Optional[CostSummary]:
        """Get cost summary if available."""
        if self._cost_summary:
            return CostSummary.from_dict(self._cost_summary)
        return None

    def to_html(
        self, template: str = "simple", path: Optional[Union[str, Path]] = None
    ) -> str:
        """Render analysis to HTML.

        Args:
            template: Template name (e.g., "simple", "pipeline") or path
            path: If provided, save HTML to this path

        Returns:
            Rendered HTML string
        """
        from ..cli._common import resolve_template

        template_path = resolve_template(template)
        html = self.pipeline.to_html(template_path=str(template_path))

        if path is not None:
            Path(path).write_text(html)

        return html

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.pipeline.get_model_dump(), indent=indent)

    def to_dict(self) -> dict:
        """Get as dictionary."""
        return self.pipeline.get_model_dump()

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Save analysis to JSON file.

        Args:
            path: Output path. If None, uses output_folder/analysis.json

        Returns:
            Path to saved file
        """
        if path is None:
            if self.output_folder is None:
                raise ValueError("No output path specified and no output_folder set")
            path = self.output_folder / "analysis.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path


@dataclass
class CompareResult:
    """Result from comparing analyses."""

    comparison: Any  # SimilarityComparison object
    stats_text: list[str] = field(default_factory=list)

    def by_comparisons(self) -> dict:
        """Get pairwise comparison data."""
        return self.comparison.by_comparisons()

    def to_html(self, calibration_info: Optional[dict] = None) -> str:
        """Render comparison report to HTML."""
        from jinja2 import Environment, FileSystemLoader

        from ..cli._common import TEMPLATES_DIR, get_soak_version

        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
        env.globals["enumerate"] = enumerate
        template = env.get_template("comparison.html")
        return template.render(
            comparison=self.comparison,
            calibration_info=calibration_info,
            soak_version=get_soak_version(),
            text_output="\n\n".join(self.stats_text),
        )

    def to_text(self) -> str:
        """Get text summary of comparison statistics."""
        return "\n\n".join(self.stats_text)


@dataclass
class CoverageResult:
    """Result from coverage analysis."""

    result: Any  # ThemeCoverageResult
    heatmaps: dict[str, str] = field(default_factory=dict)

    @property
    def documents(self) -> list:
        """Get document coverage data."""
        return self.result.documents

    @property
    def theme_names(self) -> list[str]:
        """Get theme names."""
        return self.result.theme_names

    @property
    def coverage_matrix(self) -> list[list[float]]:
        """Get coverage matrix (documents x themes)."""
        return self.result.coverage_matrix

    def to_html(self, **template_kwargs) -> str:
        """Render coverage report to HTML."""
        from jinja2 import Environment, FileSystemLoader

        from ..cli._common import TEMPLATES_DIR

        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
        template = env.get_template("coverage.html")
        return template.render(result=self.result, **self.heatmaps, **template_kwargs)

    def to_csv(self) -> str:
        """Export as CSV."""
        return self.result.to_csv()

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON."""
        return self.result.model_dump_json(indent=indent)
