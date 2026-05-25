"""Error handling utilities for LLM API errors in soak pipelines.

This module provides centralized error handling logic for dealing with various
LLM API exceptions that can occur during pipeline execution. Uses struckdown's
error types exclusively -- no direct dependency on litellm or pydantic-ai.
"""

import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from decouple import config as _decouple_config

_SOAK_DEBUG = _decouple_config("DEBUG", default=False, cast=bool)

from struckdown import LLMError
from struckdown.errors import AuthError, BadRequestError
from struckdown.errors import ConnectionError as SDConnectionError
from struckdown.errors import (ContentFilterError, ContextWindowError,
                               RateLimitError)
from tenacity import (before_sleep_log, retry, retry_if_exception,
                      stop_after_attempt, wait_exponential)

logger = logging.getLogger(__name__)


def _is_retryable_exception(exc: Exception) -> bool:
    """Check if exception should trigger a retry (handles wrapped LLMError)."""
    if isinstance(exc, LLMError):
        return exc.is_retryable
    return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a rate limit error (direct or wrapped)."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, LLMError):
        original = getattr(exc, "original_error", None)
        return isinstance(original, RateLimitError)
    return False


class MaxConsecutiveConnectionErrorsExceeded(Exception):
    """Raised when consecutive connection error threshold is exceeded"""

    pass


class NodeInvariantError(Exception):
    """Raised when a node produces output that violates a declared invariant.

    Triggered by DAGNode.validate_output() after a node's run() completes.
    Carries the failing node's name and a human-readable message describing
    which invariant failed and why.

    Downstream consumers (e.g. the django web app's auto-retry policy)
    classify this as a retryable trigger -- the output is wrong but a
    fresh stochastic attempt may succeed.
    """

    def __init__(self, node_name: str, message: str, invariant_name: Optional[str] = None):
        self.node_name = node_name
        self.invariant_name = invariant_name
        self.message = message
        prefix = f"Node '{node_name}'"
        if invariant_name:
            prefix += f" invariant '{invariant_name}'"
        super().__init__(f"{prefix}: {message}")


class ConnectionErrorCounter:
    """Thread-safe counter for tracking consecutive connection errors.

    Tracks consecutive connection error occurrences across the entire pipeline run.
    When the threshold is exceeded, raises MaxConsecutiveConnectionErrorsExceeded.
    """

    def __init__(self, threshold: Optional[int] = None):
        self.count = 0
        self.threshold = threshold or int(
            os.environ.get("SOAK_MAX_CONSECUTIVE_CONNECTION_ERRORS", "3")
        )
        self._lock = threading.Lock()

    def record_connection_error(self) -> None:
        """Record a connection error and check if threshold exceeded.

        Raises:
            MaxConsecutiveConnectionErrorsExceeded: If threshold is reached
        """
        with self._lock:
            self.count += 1
            if self.count >= self.threshold:
                raise MaxConsecutiveConnectionErrorsExceeded(
                    f"Pipeline failed after {self.count} consecutive API connection errors. "
                    f"This indicates persistent connection issues with the LLM API. "
                    f"Check your network connection and API endpoint configuration."
                )

    def reset(self) -> None:
        """Reset the counter to zero (called after successful LLM calls)"""
        with self._lock:
            self.count = 0


# global instance used across all pipeline runs
connection_error_counter = ConnectionErrorCounter()


class ErrorBehavior:
    """Error handling behavior types"""

    FAIL = "fail"  # fail entire pipeline
    SKIP = "skip"  # skip item, continue pipeline
    RETRY = "retry"  # let retry logic handle


# map struckdown error types to default behaviors
EXCEPTION_BEHAVIORS = {
    # fatal errors -- always fail pipeline
    AuthError: ErrorBehavior.FAIL,
    BadRequestError: ErrorBehavior.FAIL,
    MaxConsecutiveConnectionErrorsExceeded: ErrorBehavior.FAIL,
    # configurable errors
    ContentFilterError: ErrorBehavior.SKIP,
    ContextWindowError: ErrorBehavior.SKIP,
    # retryable errors -- handled by retry decorator
    RateLimitError: ErrorBehavior.RETRY,
    SDConnectionError: ErrorBehavior.RETRY,
}


def get_error_behavior(error: Exception, config) -> str:
    """Determine how to handle an error based on its type and config.

    Args:
        error: The exception to handle (may be wrapped in LLMError)
        config: DAGConfig instance with error handling settings

    Returns:
        ErrorBehavior constant (FAIL/SKIP/RETRY)
    """
    # unwrap LLMError to get the struckdown error subclass
    if isinstance(error, LLMError):
        error_type = type(error)
    else:
        error_type = type(error)

    # handle configurable errors
    if isinstance(error, ContentFilterError):
        return (
            ErrorBehavior.SKIP
            if config.skip_content_policy_violations
            else ErrorBehavior.FAIL
        )

    if isinstance(error, ContextWindowError):
        return (
            ErrorBehavior.SKIP
            if not config.fail_on_context_exceeded
            else ErrorBehavior.FAIL
        )

    # use default behavior from map, or SKIP as fallback for unknown errors
    return EXCEPTION_BEHAVIORS.get(error_type, ErrorBehavior.SKIP)


def log_error_to_stderr(
    error: Exception,
    node_name: str,
    item_index: Optional[int] = None,
    behavior: str = ErrorBehavior.SKIP,
    config=None,
) -> None:
    """Log detailed error information to stderr.

    Args:
        error: The exception (may be LLMError with context)
        node_name: Name of the node where error occurred
        item_index: Index of the item being processed (if applicable)
        behavior: ErrorBehavior constant indicating how error will be handled
        config: DAGConfig instance (for log_failed_prompts setting)
    """
    # unwrap LLMError if needed
    if isinstance(error, LLMError):
        original_error = error.original_error
        error_type = type(error).__name__
        model_name = error.model_name
        prompt = error.prompt
        prompt_length = len(prompt)
    else:
        original_error = error
        error_type = type(error).__name__
        model_name = "unknown"
        prompt = None
        prompt_length = 0

    # format item context
    item_str = f" (item {item_index})" if item_index is not None else ""

    # determine log level and prefix based on behavior
    if behavior == ErrorBehavior.FAIL:
        prefix = "PIPELINE FAILED"
        log_func = logger.critical
    elif behavior == ErrorBehavior.SKIP:
        prefix = "SKIPPING"
        log_func = logger.warning
    else:
        prefix = "RETRYING"
        log_func = logger.info

    # special handling for specific error types
    if isinstance(error, ContentFilterError):
        log_func(
            f"\n{prefix} [{error_type}] Content Policy Violation in node '{node_name}'{item_str}"
        )
        log_func(f"Model: {model_name}")
        log_func(f"Error: {original_error}")

        if config and config.log_failed_prompts and prompt:
            sys.stderr.write(f"\n{'='*60}\n")
            sys.stderr.write(f"Prompt that triggered violation:\n")
            sys.stderr.write(f"{'-'*60}\n")
            sys.stderr.write(f"{prompt}\n")
            sys.stderr.write(f"{'='*60}\n\n")
            sys.stderr.flush()

        if behavior == ErrorBehavior.SKIP:
            log_func("Skipping this item and continuing...")

    elif isinstance(error, ContextWindowError):
        if behavior == ErrorBehavior.FAIL:
            log_func(f"\n{'='*60}")
            log_func(f"{prefix} [{error_type}] CONTEXT WINDOW EXCEEDED")
            log_func(f"{'='*60}")
        else:
            log_func(f"\n{prefix} [{error_type}] CONTEXT WINDOW EXCEEDED")

        log_func(f"Node: '{node_name}'{item_str}")
        log_func(f"Model: {model_name}")
        if prompt_length:
            estimated_tokens = prompt_length // 4
            log_func(
                f"Prompt length: ~{estimated_tokens:,} tokens ({prompt_length:,} chars)"
            )
        log_func(f"Error: {original_error}")

        if behavior == ErrorBehavior.FAIL:
            log_func("This indicates the input is too large for the model.")
            log_func("PIPELINE FAILED.")
            log_func("=" * 60)
        else:
            log_func("Skipping this item. Consider chunking or reducing input size.")

    elif isinstance(error, RateLimitError):
        log_func(
            f"\n{prefix} [{error_type}] Rate limit in node '{node_name}'{item_str}"
        )
        log_func(f"Model: {model_name}")
        if behavior == ErrorBehavior.SKIP:
            log_func("Skipping this item and continuing...")

    else:
        # generic error logging
        log_func(f"\n{prefix} [{error_type}] in node '{node_name}'{item_str}")
        log_func(f"Model: {model_name}")
        log_func(f"Error: {original_error}")

        if prompt_length:
            log_func(f"Prompt length: {prompt_length} chars")

        if behavior == ErrorBehavior.FAIL:
            log_func("PIPELINE FAILED.")
        elif behavior == ErrorBehavior.SKIP:
            log_func("Skipping this item and continuing...")

    # log to stderr for visibility
    sys.stderr.write(
        f"\n[{prefix}] [{error_type}] in node '{node_name}'{item_str}: {original_error}\n"
    )
    sys.stderr.flush()


def should_continue_pipeline(error: Exception, config) -> bool:
    """Determine if pipeline should continue after this error."""
    behavior = get_error_behavior(error, config)
    return behavior != ErrorBehavior.FAIL


def handle_llm_error_in_node(
    error: Exception,
    node_name: str,
    config,
    item_index: Optional[int] = None,
    item_type: str = "item",
) -> bool:
    """Handle LLM API error in a node with logging and behavior determination.

    Returns:
        True if should skip this item and continue, False if should re-raise
    """
    # track consecutive connection errors
    if isinstance(error, SDConnectionError):
        connection_error_counter.record_connection_error()
    elif isinstance(error, LLMError) and isinstance(
        error.original_error, SDConnectionError
    ):
        connection_error_counter.record_connection_error()

    behavior = get_error_behavior(error, config)

    log_error_to_stderr(
        error=error,
        node_name=node_name,
        item_index=item_index,
        behavior=behavior,
        config=config,
    )

    if should_continue_pipeline(error, config):
        logger.info(
            f"Skipping {item_type}{f' {item_index}' if item_index is not None else ''} "
            f"in node '{node_name}' due to {type(error).__name__}"
        )
        return True
    else:
        return False


def _save_debug_prompt(
    node_name: str, prompt_text: str, item_index: Optional[int] = None
):
    """Save rendered prompt to .prompts/ when DEBUG is set."""
    if not _SOAK_DEBUG:
        return
    prompts_dir = Path(".prompts")
    prompts_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_{item_index}" if item_index is not None else ""
    filename = f"{ts}_{node_name}{suffix}.txt"
    (prompts_dir / filename).write_text(prompt_text)


async def managed_llm_call(
    node_name: str, config, llm_func, item_index: Optional[int] = None, *args, **kwargs
) -> Optional[Any]:
    """Centralized wrapper for LLM calls with error handling, retry, and connection tracking.

    Retry behaviour: 3 attempts, exponential backoff (max 60s).
    Uses struckdown's is_retryable property to determine retry eligibility.
    """
    # save rendered prompts to .prompts/ when DEBUG is set
    if _SOAK_DEBUG:
        from soak.models.dag import render_template_preserve_undefined

        prompt_text = kwargs.get("multipart_prompt") or kwargs.get("template")
        context = kwargs.get("context", {})
        if prompt_text and context:
            prompt_text = render_template_preserve_undefined(prompt_text, context)
        if prompt_text:
            _save_debug_prompt(node_name, prompt_text, item_index)

    def _before_sleep_with_rate_limit_tracking(retry_state):
        """Log retry and emit RateLimitHit event if the error is a rate limit."""
        before_sleep_log(logger, logging.WARNING)(retry_state)
        exc = retry_state.outcome.exception()
        if _is_rate_limit_error(exc):
            emit = getattr(config, "emit", None)
            if emit:
                from soak.events import RateLimitHit

                model_name = getattr(exc, "model_name", "unknown")
                emit(RateLimitHit(node_name, model_name))

    @retry(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        before_sleep=_before_sleep_with_rate_limit_tracking,
        reraise=True,
    )
    async def _call_with_retry():
        return await llm_func(*args, **kwargs)

    try:
        result = await _call_with_retry()
        connection_error_counter.reset()
        return result
    except LLMError as e:
        emit = getattr(config, "emit", None)
        if emit:
            from soak.events import NodeError

            emit(NodeError(node_name, item_index, e))
        if handle_llm_error_in_node(e, node_name, config, item_index):
            return None  # skip this item
        raise
