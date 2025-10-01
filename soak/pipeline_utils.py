"""Utilities for loading and running analysis pipelines."""

import os
import pkgutil
import sys

# Add pipelines directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def discover_pipelines() -> list[str]:
    """Discover available pipeline modules using pkgutil."""

    import pipelines

    return sorted(
        [name for _, name, ispkg in pkgutil.iter_modules(pipelines.__path__) if ispkg]
    )
