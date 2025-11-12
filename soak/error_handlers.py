"""Error handling utilities for LLM API errors in soak pipelines.

This module provides centralized error handling logic for dealing with various
LiteLLM/OpenAI API exceptions that can occur during pipeline execution.
"""

import logging
import os
import sys
import threading
from typing import Any, Optional

from litellm.exceptions import (APIConnectionError, APIResponseValidationError,
                                AuthenticationError, BadRequestError,
                                BudgetExceededError,
                                ContentPolicyViolationError,
                                ContextWindowExceededError,
                                InternalServerError, NotFoundError,
                                PermissionDeniedError, UnsupportedParamsError)
from struckdown import StruckdownLLMError

logger = logging.getLogger(__name__)


class MaxConsecutiveConnectionErrorsExceeded(Exception):
    """Raised when consecutive APIConnectionError threshold is exceeded"""

    pass


class ConnectionErrorCounter:
    """Thread-safe counter for tracking consecutive APIConnectionErrors.

    Tracks consecutive APIConnectionError occurrences across the entire pipeline run.
    When the threshold is exceeded, raises MaxConsecutiveConnectionErrorsExceeded.
    """

    def __init__(self, threshold: Optional[int] = None):
        self.count = 0
        self.threshold = threshold or int(
            os.environ.get("SOAK_MAX_CONSECUTIVE_CONNECTION_ERRORS", "3")
        )
        self._lock = threading.Lock()

    def record_connection_error(self) -> None:
        """Record an APIConnectionError and check if threshold exceeded.

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
    RETRY = "retry"  # let instructor retry


# map exception types to default behaviors
EXCEPTION_BEHAVIORS = {
    # fatal errors -- always fail pipeline
    AuthenticationError: ErrorBehavior.FAIL,
    PermissionDeniedError: ErrorBehavior.FAIL,
    NotFoundError: ErrorBehavior.FAIL,
    BadRequestError: ErrorBehavior.FAIL,
    UnsupportedParamsError: ErrorBehavior.FAIL,
    APIResponseValidationError: ErrorBehavior.FAIL,
    BudgetExceededError: ErrorBehavior.FAIL,
    MaxConsecutiveConnectionErrorsExceeded: ErrorBehavior.FAIL,
    # configurable errors
    ContentPolicyViolationError: ErrorBehavior.SKIP,  # configurable via skip_content_policy_violations
    ContextWindowExceededError: ErrorBehavior.SKIP,  # configurable via fail_on_context_exceeded
    # retryable errors -- instructor handles these
    # (these shouldn't reach us if instructor's retry works, but we handle them gracefully)
}


def get_error_behavior(error: Exception, config) -> str:
    """Determine how to handle an error based on its type and config.

    Args:
        error: The exception to handle (may be wrapped in StruckdownLLMError)
        config: DAGConfig instance with error handling settings

    Returns:
        ErrorBehavior constant (FAIL/SKIP/RETRY)
    """
    # unwrap StruckdownLLMError if needed
    original_error = (
        error.original_error if isinstance(error, StruckdownLLMError) else error
    )
    error_type = type(original_error)

    # handle configurable errors
    if error_type == ContentPolicyViolationError:
        return (
            ErrorBehavior.SKIP
            if config.skip_content_policy_violations
            else ErrorBehavior.FAIL
        )

    if error_type == ContextWindowExceededError:
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
        error: The exception (may be StruckdownLLMError with context)
        node_name: Name of the node where error occurred
        item_index: Index of the item being processed (if applicable)
        behavior: ErrorBehavior constant indicating how error will be handled
        config: DAGConfig instance (for log_failed_prompts setting)
    """
    # unwrap StruckdownLLMError if needed
    if isinstance(error, StruckdownLLMError):
        original_error = error.original_error
        error_type = type(original_error).__name__
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
        level = "CRITICAL"
        prefix = "⚠️  PIPELINE FAILED ⚠️"
        log_func = logger.critical
    elif behavior == ErrorBehavior.SKIP:
        level = "WARNING"
        prefix = "⚠️"
        log_func = logger.warning
    else:
        level = "INFO"
        prefix = "ℹ️"
        log_func = logger.info

    # special handling for specific error types
    if isinstance(original_error, ContentPolicyViolationError):
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
            sys.stderr.flush()  # ensure prompt is visible immediately

        if behavior == ErrorBehavior.SKIP:
            log_func("Skipping this item and continuing...")

    elif isinstance(original_error, ContextWindowExceededError):
        if behavior == ErrorBehavior.FAIL:
            log_func(f"\n{'='*60}")
            log_func(f"{prefix} [{error_type}] CONTEXT WINDOW EXCEEDED")
            log_func(f"{'='*60}")
        else:
            log_func(f"\n{prefix}⚠️⚠️ [{error_type}] CONTEXT WINDOW EXCEEDED ⚠️⚠️")

        log_func(f"Node: '{node_name}'{item_str}")
        log_func(f"Model: {model_name}")
        if prompt_length:
            estimated_tokens = prompt_length // 4  # rough estimate
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

    else:
        # generic error logging - lead with error type
        log_func(f"\n{prefix} [{error_type}] in node '{node_name}'{item_str}")
        log_func(f"Model: {model_name}")
        log_func(f"Error: {original_error}")

        if prompt_length:
            log_func(f"Prompt length: {prompt_length} chars")

        if behavior == ErrorBehavior.FAIL:
            log_func("PIPELINE FAILED.")
        elif behavior == ErrorBehavior.SKIP:
            log_func("Skipping this item and continuing...")

    # log to stderr as well for visibility - always include error type prominently
    if behavior == ErrorBehavior.FAIL:
        sys.stderr.write(
            f"\n[{level}] [{error_type}] in node '{node_name}'{item_str}: {original_error}\n"
        )
        sys.stderr.flush()  # ensure immediate visibility
    else:
        # for non-fatal errors, still log to stderr but less prominently
        sys.stderr.write(
            f"\n[{level}] [{error_type}] in node '{node_name}'{item_str}\n"
        )
        sys.stderr.flush()  # ensure immediate visibility


def should_continue_pipeline(error: Exception, config) -> bool:
    """Determine if pipeline should continue after this error.

    Args:
        error: The exception to handle
        config: DAGConfig instance with error handling settings

    Returns:
        True if pipeline should continue, False if it should fail
    """
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

    This is a helper function to eliminate repetitive error handling code across nodes.
    It handles the standard pattern: calculate behavior, log, determine whether to skip or fail.

    Args:
        error: The StruckdownLLMError or other exception
        node_name: Name of the node where error occurred
        config: DAGConfig instance with error handling settings
        item_index: Optional index of the item being processed
        item_type: Type of item ("item", "batch", "node") for log messages

    Returns:
        True if should skip this item and continue, False if should re-raise and fail pipeline

    Raises:
        MaxConsecutiveConnectionErrorsExceeded: If threshold of consecutive APIConnectionErrors is reached
    """
    # unwrap error to check for connection errors
    original_error = (
        error.original_error if isinstance(error, StruckdownLLMError) else error
    )

    # track consecutive connection errors (APIConnectionError or InternalServerError with connection error message)
    if isinstance(original_error, APIConnectionError):
        connection_error_counter.record_connection_error()
        # if threshold exceeded, this will raise MaxConsecutiveConnectionErrorsExceeded
    elif (
        isinstance(original_error, InternalServerError)
        and "connection error" in str(original_error).lower()
    ):
        connection_error_counter.record_connection_error()
        # if threshold exceeded, this will raise MaxConsecutiveConnectionErrorsExceeded

    # calculate behavior first
    behavior = get_error_behavior(error, config)

    # log error with full context
    log_error_to_stderr(
        error=error,
        node_name=node_name,
        item_index=item_index,
        behavior=behavior,
        config=config,
    )

    # determine if we should continue or fail
    if should_continue_pipeline(error, config):
        # log skip message
        if item_index is not None:
            logger.info(
                f"Skipping {item_type} {item_index} in node '{node_name}' "
                f"due to {type(error.original_error if isinstance(error, StruckdownLLMError) else error).__name__}"
            )
        else:
            logger.info(
                f"Skipping {item_type} in node '{node_name}' "
                f"due to {type(error.original_error if isinstance(error, StruckdownLLMError) else error).__name__}"
            )
        return True  # skip this item
    else:
        return False  # re-raise to fail pipeline


async def managed_llm_call(
    node_name: str, config, llm_func, item_index: Optional[int] = None, *args, **kwargs
) -> Optional[Any]:
    """Centralized wrapper for LLM calls with error handling and connection tracking.

    This function handles the standard pattern for all LLM calls:
    - Execute the LLM function
    - Reset connection error counter on success
    - Handle StruckdownLLMError with appropriate behavior (skip/fail)
    - Let other exceptions propagate

    Args:
        node_name: Name of the node making the call
        config: DAGConfig instance with error handling settings
        llm_func: Async callable to execute (e.g., chatter_async, default_map_task)
        item_index: Optional index for error logging
        *args: Positional arguments passed to llm_func
        **kwargs: Keyword arguments passed to llm_func

    Returns:
        Result from llm_func on success, None if error was skipped

    Raises:
        Exception: If error should fail the pipeline
    """
    try:
        result = await llm_func(*args, **kwargs)
        connection_error_counter.reset()  # centralized success handling
        return result
    except StruckdownLLMError as e:
        if handle_llm_error_in_node(e, node_name, config, item_index):
            return None  # skip this item
        else:
            raise  # re-raise to fail pipeline
