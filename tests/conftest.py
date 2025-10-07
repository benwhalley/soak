"""Pytest configuration."""

import pytest


@pytest.fixture
def anyio_backend():
    """Force asyncio backend for all tests."""
    return "asyncio"
