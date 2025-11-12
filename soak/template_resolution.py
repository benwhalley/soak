"""Template resolution for hybrid inline/external templates.

Provides Django-style template search paths:
1. Current working directory (where .soak file is)
2. templates/ subdirectory in CWD
3. User-specified template_dirs from YAML
4. Package templates (soak/templates/)
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class TemplateNotFoundError(Exception):
    """Raised when template file cannot be found in any search path."""

    pass


class TemplateResolver:
    """Resolves template files using Django-style search paths."""

    def __init__(self, pipeline_path: Path, extra_dirs: Optional[List[str]] = None):
        """Initialize template resolver with search paths.

        Args:
            pipeline_path: Path to the .soak file (used to determine CWD)
            extra_dirs: Additional directories from template_dirs YAML field
        """
        self.pipeline_path = Path(pipeline_path)
        self.search_paths = self._build_search_paths(extra_dirs or [])

        logger.debug(f"Template search paths: {[str(p) for p in self.search_paths]}")

    def _build_search_paths(self, extra_dirs: List[str]) -> List[Path]:
        """Build ordered list of directories to search for templates.

        Priority order:
        1. CWD where .soak file is located
        2. templates/ subdirectory in CWD
        3. User-specified directories (template_dirs)
        4. Package templates (soak/templates/)

        Returns:
            List of Path objects (only existing directories)
        """
        paths = []

        # CWD where .soak file is
        if self.pipeline_path.is_file():
            cwd = self.pipeline_path.parent
        else:
            cwd = self.pipeline_path
        paths.append(cwd)

        # templates/ in CWD
        templates_dir = cwd / "templates"
        if templates_dir.exists() and templates_dir.is_dir():
            paths.append(templates_dir)

        # User-specified directories
        for extra_dir in extra_dirs:
            extra_path = Path(extra_dir).expanduser().resolve()
            if extra_path.exists() and extra_path.is_dir():
                paths.append(extra_path)
            else:
                logger.warning(f"Template directory not found: {extra_dir}")

        # Package templates (lowest priority)
        package_templates = Path(__file__).parent / "templates"
        if package_templates.exists() and package_templates.is_dir():
            paths.append(package_templates)

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for p in paths:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(resolved)

        return unique_paths

    def resolve(self, template_name: str) -> Path:
        """Find template file in search paths.

        Args:
            template_name: Name of template file (e.g., "scrub.sd", "summaries.sd")

        Returns:
            Path to template file

        Raises:
            TemplateNotFoundError: If not found in any search path
        """
        for search_path in self.search_paths:
            candidate = search_path / template_name
            if candidate.exists() and candidate.is_file():
                logger.debug(f"Found template '{template_name}' at {candidate}")
                return candidate

        # Not found -- raise helpful error
        searched_locations = "\n  - ".join(
            str(p / template_name) for p in self.search_paths
        )
        raise TemplateNotFoundError(
            f"Template '{template_name}' not found.\n\n"
            f"Searched in:\n  - {searched_locations}\n\n"
            f"Suggestions:\n"
            f"  - Add an inline template with ---#{template_name.replace('.sd', '')}\n"
            f"  - Create {template_name} in one of the search paths\n"
            f"  - Specify a different template file in the node config"
        )

    def load(self, template_name: str) -> str:
        """Resolve and load template content.

        Args:
            template_name: Name of template file

        Returns:
            Template content as string
        """
        path = self.resolve(template_name)
        return path.read_text(encoding="utf-8")

    def try_load(self, template_name: str) -> Optional[str]:
        """Try to load template, returning None if not found.

        Args:
            template_name: Name of template file

        Returns:
            Template content or None if not found
        """
        try:
            return self.load(template_name)
        except TemplateNotFoundError:
            return None
