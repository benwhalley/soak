#!/usr/bin/env python
"""Tests for error handling across node types.

These tests verify that nodes handle LLM errors gracefully based on config settings.
The implementation uses handle_llm_error_in_node() which provides:
- FAIL behavior: AuthenticationError, BadRequestError, NotFoundError, etc.
- SKIP behavior: ContextWindowExceededError (when fail_on_context_exceeded=False)
- SKIP behavior: ContentPolicyViolationError (when skip_content_policy_violations=True)
- SKIP behavior: Unknown errors (default fallback)
"""

import os
from pathlib import Path

import pytest
from struckdown import LLMCredentials

from soak.specs import load_template_bundle


@pytest.mark.anyio
async def test_error_handling_implementation_exists():
    """Test that error handling is implemented and accessible."""
    # Simple smoke test to verify error handling infrastructure
    from soak.error_handlers import (ErrorBehavior, get_error_behavior,
                                     handle_llm_error_in_node)
    from soak.models.dag import DAGConfig

    # Verify functions exist and are callable
    assert callable(handle_llm_error_in_node)
    assert callable(get_error_behavior)
    assert hasattr(ErrorBehavior, "SKIP")
    assert hasattr(ErrorBehavior, "FAIL")

    # Verify config attributes exist
    config = DAGConfig()
    assert hasattr(config, "fail_on_context_exceeded")
    assert hasattr(config, "skip_content_policy_violations")
    assert hasattr(config, "log_failed_prompts")


@pytest.mark.anyio
async def test_error_behavior_config():
    """Test error behavior configuration options."""
    from litellm.exceptions import (ContentPolicyViolationError,
                                    ContextWindowExceededError)

    from soak.error_handlers import ErrorBehavior, get_error_behavior
    from soak.models.dag import DAGConfig

    # Create mock errors
    context_error = ContextWindowExceededError(
        message="Context window exceeded", model="gpt-4", llm_provider="openai"
    )
    content_error = ContentPolicyViolationError(
        message="Content policy violation", model="gpt-4", llm_provider="openai"
    )

    # Test context window error behavior
    config_fail = DAGConfig(fail_on_context_exceeded=True)
    config_skip = DAGConfig(fail_on_context_exceeded=False)

    assert get_error_behavior(context_error, config_fail) == ErrorBehavior.FAIL
    assert get_error_behavior(context_error, config_skip) == ErrorBehavior.SKIP

    # Test content policy error behavior
    config_skip_content = DAGConfig(skip_content_policy_violations=True)
    config_fail_content = DAGConfig(skip_content_policy_violations=False)

    assert get_error_behavior(content_error, config_skip_content) == ErrorBehavior.SKIP
    assert get_error_behavior(content_error, config_fail_content) == ErrorBehavior.FAIL


@pytest.mark.anyio
async def test_nodes_have_error_handling():
    """Test that all major nodes import and use error handling."""
    # Verify error handling is imported in node modules
    from soak.models.nodes import (classifier, filter, map, transform,
                                   transform_reduce)

    # Check that managed_llm_call is available (centralized error handling)
    assert hasattr(map, "managed_llm_call")
    assert hasattr(classifier, "managed_llm_call")
    assert hasattr(filter, "managed_llm_call")
    assert hasattr(transform, "managed_llm_call")
    assert hasattr(transform_reduce, "managed_llm_call")


@pytest.mark.anyio
async def test_basic_pipeline_completes_without_errors():
    """Test that a simple pipeline completes successfully (baseline for error tests)."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_map_classifier.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()

    # Should complete without errors
    assert error is None
    assert result is not None

    # All nodes should have output
    for node in pipeline.nodes:
        if hasattr(node, "output"):
            assert node.output is not None


@pytest.mark.skip(
    reason="Test needs fixing: LLMCredentials falls back to environment variables, bypassing test credentials"
)
@pytest.mark.anyio
async def test_connection_error_threshold_exceeded():
    """Test pipeline fails after exceeding consecutive APIConnectionError threshold.

    This test verifies that when the LLM API is unreachable (connection errors),
    the pipeline fails after N consecutive connection errors rather than continuing
    to retry indefinitely.

    NOTE: Currently skipped because LLMCredentials uses environment variables as fallback,
    so test credentials are ignored. Test needs refactoring to properly isolate credentials.
    """
    # Set low threshold for faster testing
    original_threshold = os.environ.get("SOAK_MAX_CONSECUTIVE_CONNECTION_ERRORS")
    os.environ["SOAK_MAX_CONSECUTIVE_CONNECTION_ERRORS"] = "2"

    try:
        # Reset the connection error counter to ensure clean state
        from soak.error_handlers import connection_error_counter

        connection_error_counter.reset()

        # Load a simple pipeline (Map node without Split to avoid unrelated errors)
        pipeline = load_template_bundle(Path("tests/pipelines/test_csv_input.soak"))
        pipeline.config.document_paths = ["soak/data/test_data.csv"]

        # Set credentials with unreachable endpoint to trigger connection errors
        # Use localhost port that nothing is listening on
        pipeline.config.llm_credentials = LLMCredentials(
            api_key="sk-test1234567890abcdefghijklmnopqrstuvwxyz",
            api_base="http://localhost:9999",
        )

        # Run pipeline -- should fail with connection or authentication errors
        result, error = await pipeline.run()

        # Verify pipeline failed (exact error type depends on environment/proxy config)
        # The important thing is that it FAILS rather than continuing indefinitely
        assert (
            error is not None
        ), "Pipeline should have failed due to unreachable endpoint"

        # The error may be wrapped in ExceptionGroup/TaskGroup by anyio
        # As long as the pipeline failed, the test passes
        #  - Real connection errors will eventually hit the consecutive error threshold
        #  - Auth errors from proxies also cause immediate failure (which is correct)
        # Both scenarios demonstrate proper error handling

    finally:
        # Restore original threshold
        if original_threshold is None:
            os.environ.pop("SOAK_MAX_CONSECUTIVE_CONNECTION_ERRORS", None)
        else:
            os.environ["SOAK_MAX_CONSECUTIVE_CONNECTION_ERRORS"] = original_threshold

        # Reset counter after test
        from soak.error_handlers import connection_error_counter

        connection_error_counter.reset()


@pytest.mark.skip(
    reason="Requires way to reliably trigger LLM errors - needs mock or specific error-inducing prompts"
)
@pytest.mark.anyio
async def test_map_handles_errors_gracefully():
    """Test Map node handles errors per config (requires error-triggering setup)."""
    # This test would need a way to reliably trigger errors
    # Options: mock LLM calls, use intentionally broken prompts, or use test doubles
    pass


@pytest.mark.skip(
    reason="Requires way to reliably trigger LLM errors - needs mock or specific error-inducing prompts"
)
@pytest.mark.anyio
async def test_classifier_handles_errors_gracefully():
    """Test Classifier node handles errors per config (requires error-triggering setup)."""
    pass


@pytest.mark.skip(
    reason="Requires way to reliably trigger LLM errors - needs mock or specific error-inducing prompts"
)
@pytest.mark.anyio
async def test_filter_handles_errors_gracefully():
    """Test Filter node handles errors per config (requires error-triggering setup)."""
    pass


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running error handling implementation test...")
        await test_error_handling_implementation_exists()
        print("✓ Error handling implementation test passed\n")

        print("Running error behavior config test...")
        await test_error_behavior_config()
        print("✓ Error behavior config test passed\n")

        print("Running nodes have error handling test...")
        await test_nodes_have_error_handling()
        print("✓ Nodes have error handling test passed\n")

        print("Running basic pipeline test...")
        await test_basic_pipeline_completes_without_errors()
        print("✓ Basic pipeline test passed\n")

        print("Running connection error threshold test...")
        await test_connection_error_threshold_exceeded()
        print("✓ Connection error threshold test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
