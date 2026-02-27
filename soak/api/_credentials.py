"""Credential management for soak API."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..helpers import load_env_file


class CredentialsError(Exception):
    """Raised when credentials are missing and cannot be obtained."""

    pass


@dataclass
class Credentials:
    """LLM API credentials."""

    api_key: str
    base_url: str


# module-level credential override
_credential_override: Optional[Credentials] = None


def set_credentials(api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
    """Set credentials globally for all API calls.

    Args:
        api_key: LLM API key
        base_url: LLM API base URL
    """
    global _credential_override
    _credential_override = Credentials(api_key=api_key, base_url=base_url)


def clear_credentials() -> None:
    """Clear any globally set credentials."""
    global _credential_override
    _credential_override = None


@contextmanager
def credentials(api_key: str, base_url: str = "https://api.openai.com/v1"):
    """Context manager for temporary credential override.

    Usage:
        with api.credentials(api_key="...", base_url="..."):
            result = api.run(...)
    """
    global _credential_override
    old = _credential_override
    _credential_override = Credentials(api_key=api_key, base_url=base_url)
    try:
        yield
    finally:
        _credential_override = old


def get_credentials(cwd: Optional[Path] = None) -> Credentials:
    """Get credentials from override, environment, or .env file.

    Resolution order:
    1. Global override (set via set_credentials() or credentials() context)
    2. Environment variables (LLM_API_KEY, LLM_API_BASE)
    3. .env file in cwd

    Args:
        cwd: Directory to look for .env file (default: current directory)

    Returns:
        Credentials object

    Raises:
        CredentialsError: If credentials cannot be found
    """
    global _credential_override

    if _credential_override is not None:
        return _credential_override

    cwd = cwd or Path.cwd()

    # check environment variables
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_BASE")

    # check .env file
    if not api_key or not base_url:
        env_path = cwd / ".env"
        env_vars = load_env_file(env_path)
        api_key = api_key or env_vars.get("LLM_API_KEY")
        base_url = base_url or env_vars.get("LLM_API_BASE")

    missing = []
    if not api_key:
        missing.append("LLM_API_KEY")
    if not base_url:
        missing.append("LLM_API_BASE")

    if missing:
        raise CredentialsError(
            f"Missing credentials: {', '.join(missing)}. "
            f"Set via environment variables, .env file, or api.set_credentials()"
        )

    return Credentials(api_key=api_key, base_url=base_url)
