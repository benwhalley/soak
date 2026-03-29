"""Tests for concurrency configuration and rate limit error handling."""

import asyncio
from unittest.mock import MagicMock

import anyio
import pytest
from litellm.exceptions import RateLimitError as LitellmRateLimitError
from struckdown import LLMError
from struckdown.errors import RateLimitError as SDRateLimitError

from soak.error_handlers import (
    _is_rate_limit_error,
    _is_retryable_exception,
    managed_llm_call,
)
from soak.models.base import get_semaphore, set_max_concurrency
from soak.models.dag import DAGConfig


class TestRetryableExceptions:
    """Test that rate limit errors are correctly identified as retryable."""

    def test_litellm_rate_limit_is_retryable(self):
        exc = LitellmRateLimitError("rate limited", "test-model", None, None)
        assert _is_retryable_exception(exc) is True

    def test_struckdown_rate_limit_is_retryable(self):
        original = LitellmRateLimitError("rate limited", "test-model", None, None)
        exc = SDRateLimitError(original, "test prompt", "test-model")
        assert _is_retryable_exception(exc) is True

    def test_wrapped_rate_limit_is_retryable(self):
        """LLMError wrapping a litellm RateLimitError should be retryable."""
        original = LitellmRateLimitError("rate limited", "test-model", None, None)
        exc = LLMError(original, "test prompt", "test-model")
        assert _is_retryable_exception(exc) is True

    def test_value_error_is_not_retryable(self):
        exc = ValueError("bad value")
        assert _is_retryable_exception(exc) is False


class TestIsRateLimitError:
    """Test _is_rate_limit_error helper."""

    def test_litellm_rate_limit(self):
        exc = LitellmRateLimitError("rate limited", "test-model", None, None)
        assert _is_rate_limit_error(exc) is True

    def test_struckdown_rate_limit(self):
        original = LitellmRateLimitError("rate limited", "test-model", None, None)
        exc = SDRateLimitError(original, "test prompt", "test-model")
        assert _is_rate_limit_error(exc) is True

    def test_wrapped_rate_limit(self):
        original = LitellmRateLimitError("rate limited", "test-model", None, None)
        exc = LLMError(original, "test prompt", "test-model")
        assert _is_rate_limit_error(exc) is True

    def test_connection_error_is_not_rate_limit(self):
        from litellm.exceptions import APIConnectionError
        exc = APIConnectionError("connection failed", "test-model", None)
        assert _is_rate_limit_error(exc) is False


class TestSetMaxConcurrency:
    """Test that set_max_concurrency replaces the semaphore."""

    def test_set_max_concurrency_creates_new_semaphore(self):
        old_sem = get_semaphore()
        set_max_concurrency(100)
        new_sem = get_semaphore()
        assert old_sem is not new_sem

    def test_set_max_concurrency_different_values(self):
        set_max_concurrency(5)
        sem5 = get_semaphore()
        set_max_concurrency(200)
        sem200 = get_semaphore()
        assert sem5 is not sem200

    def test_get_semaphore_returns_same_instance(self):
        """Calling get_semaphore twice without set should return the same object."""
        set_max_concurrency(10)  # reset to a known state
        a = get_semaphore()
        b = get_semaphore()
        assert a is b


class TestDAGConfigRateLimitCallback:
    """Test that DAGConfig supports rate_limit_callback."""

    def test_default_is_none(self):
        config = DAGConfig()
        assert config.rate_limit_callback is None

    def test_can_set_callback(self):
        callback = MagicMock()
        config = DAGConfig(rate_limit_callback=callback)
        assert config.rate_limit_callback is callback


@pytest.mark.anyio
async def test_rate_limit_callback_fires_on_retry():
    """Test that the rate_limit_callback is invoked when a rate limit error triggers a retry."""
    callback = MagicMock()
    config = DAGConfig(rate_limit_callback=callback)

    call_count = {"n": 0}

    async def flaky_llm_func():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise SDRateLimitError(
                LitellmRateLimitError("rate limited", "test-model", None, None),
                "test prompt",
                "test-model",
            )
        return "success"

    result = await managed_llm_call(
        node_name="test_node",
        config=config,
        llm_func=flaky_llm_func,
        item_index=0,
    )

    assert result == "success"
    assert call_count["n"] == 2  # called twice: first fails, second succeeds
    callback.assert_called_once_with("test_node", "test-model")


@pytest.mark.anyio
async def test_rate_limit_callback_not_fired_on_connection_error():
    """Rate limit callback should NOT fire for non-rate-limit retries."""
    from litellm.exceptions import APIConnectionError

    callback = MagicMock()
    config = DAGConfig(rate_limit_callback=callback)

    call_count = {"n": 0}

    async def flaky_llm_func():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise APIConnectionError("connection failed", "test-model", None)
        return "success"

    result = await managed_llm_call(
        node_name="test_node",
        config=config,
        llm_func=flaky_llm_func,
        item_index=0,
    )

    assert result == "success"
    callback.assert_not_called()
