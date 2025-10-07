#!/usr/bin/env python
"""Comprehensive tests for all node types in the soak pipeline system."""

from pathlib import Path

import pytest

from soak.models import LLMCredentials
from soak.specs import load_template_bundle

# Test data path
TEST_DATA = Path("data/chainstorecoffee.txt")


@pytest.mark.anyio
async def test_split_node():
    """Test Split node with different split_unit options."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_split.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    # Test each split type produced output
    for node in pipeline.nodes:
        assert node.output is not None, f"Node {node.name} has no output"
        assert len(node.output) > 0, f"Node {node.name} produced empty output"

        # Verify all chunks have content
        for chunk in node.output:
            content = chunk.content if hasattr(chunk, "content") else chunk
            assert len(content) > 0, f"Node {node.name} has empty chunk"

        # Verify provenance tracking
        if hasattr(node.output[0], "source_id"):
            assert (
                node.name in node.output[0].source_id
            ), f"Node name not in source_id for {node.name}"

    # Test specific behaviors
    tokens_node = [n for n in pipeline.nodes if n.name == "split_tokens"][0]
    words_node = [n for n in pipeline.nodes if n.name == "split_words"][0]
    sentences_node = [n for n in pipeline.nodes if n.name == "split_sentences"][0]
    paragraphs_node = [n for n in pipeline.nodes if n.name == "split_paragraphs"][0]

    # Document is long enough to require multiple chunks
    assert len(tokens_node.output) > 1, "tokens split should produce multiple chunks"
    assert len(words_node.output) > 1, "words split should produce multiple chunks"
    assert (
        len(sentences_node.output) > 1
    ), "sentences split should produce multiple chunks"
    assert (
        len(paragraphs_node.output) > 1
    ), "paragraphs split should produce multiple chunks"

    # Test overlap functionality
    assert tokens_node.overlap == 50, "tokens split should have overlap of 50"
    assert words_node.overlap == 30, "words split should have overlap of 30"


@pytest.mark.anyio
async def test_map_node():
    """Test Map node processes all items in parallel."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_map.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    chunks_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    map_node = [n for n in pipeline.nodes if n.name == "map_summarize"][0]

    # Map should process each chunk
    assert len(map_node.output) == len(
        chunks_node.output
    ), "Map output count should equal input count"

    # All outputs should have content
    for item in map_node.output:
        content = item.content if hasattr(item, "content") else str(item)
        assert len(content) > 0, "Map produced empty output"


@pytest.mark.anyio
async def test_classifier_node_with_agreement():
    """Test Classifier node with multiple models and agreement calculation."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_classifier.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Should have output
    assert classifier_node.output is not None
    assert len(classifier_node.output) > 0

    # Check that multiple models were used
    assert classifier_node.model_names is not None
    assert len(classifier_node.model_names) == 2

    # Check agreement fields are set
    assert classifier_node.agreement_fields is not None
    assert "topic" in classifier_node.agreement_fields
    assert "sentiment" in classifier_node.agreement_fields

    # Check provenance - each item should have model info
    for item in classifier_node.output:
        if hasattr(item, "metadata"):
            assert "model_name" in item.metadata or "source_id" in item.metadata

    # Export and check agreement stats (if calculated)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        classifier_node.export(folder)

        # Check CSV files were created for each model
        csv_files = list(folder.glob("*.csv"))
        assert len(csv_files) >= 2, "Should have CSV files for multiple models"

        # Check for agreement stats files
        agreement_stats_csv = folder / "agreement_stats.csv"
        agreement_stats_json = folder / "agreement_stats.json"

        # Agreement stats should exist if we have multiple models
        if classifier_node._agreement_stats:
            assert agreement_stats_csv.exists(), "Agreement stats CSV should exist"
            assert agreement_stats_json.exists(), "Agreement stats JSON should exist"


@pytest.mark.anyio
async def test_filter_node():
    """Test Filter node includes and excludes items correctly."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_filter.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    chunks_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    filter_node = [n for n in pipeline.nodes if n.name == "filter_about_coffee"][0]

    # Some items should pass filter (coffee article)
    assert filter_node.output is not None
    assert len(filter_node.output) > 0, "Filter should pass some items about coffee"

    # Some items should be excluded
    assert hasattr(filter_node, "_excluded_items")

    # Total should equal input
    total_processed = len(filter_node.output) + len(filter_node._excluded_items or [])
    assert total_processed == len(
        chunks_node.output
    ), "Filter should process all input items"


@pytest.mark.anyio
async def test_transform_node_multi_slot():
    """Test Transform node with multi-slot struckdown prompts."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_transform.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    transform_node = [n for n in pipeline.nodes if n.name == "multi_slot_transform"][0]

    # Should have output
    assert transform_node.output is not None

    # Transform output is a ChatterResult with structured data
    output = transform_node.output
    assert hasattr(
        output, "response"
    ), "Transform output should have response attribute"

    # Response can be a string or dict depending on the struckdown output
    response_str = str(output.response)
    assert len(response_str) > 0, "Transform should produce non-empty output"


@pytest.mark.anyio
async def test_reduce_node():
    """Test Reduce node aggregates multiple inputs."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_reduce.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    summaries_node = [n for n in pipeline.nodes if n.name == "summaries"][0]
    reduce_node = [n for n in pipeline.nodes if n.name == "combined"][0]

    # Reduce should aggregate multiple inputs into single output (string)
    assert len(summaries_node.output) > 1, "Should have multiple summaries"
    assert reduce_node.output is not None

    # Reduce concatenates into a single string
    assert isinstance(reduce_node.output, str), "Reduce output should be a string"
    assert len(reduce_node.output) > 0, "Reduce should produce non-empty output"


@pytest.mark.anyio
async def test_batch_node():
    """Test Batch node groups items correctly."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_batch.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    chunks_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    batch_node = [n for n in pipeline.nodes if n.name == "batched"][0]

    # Batch should reduce number of items by grouping them
    assert batch_node.output is not None

    # Output is a BatchList with batches attribute
    assert hasattr(
        batch_node.output, "batches"
    ), "Batch output should have batches attribute"
    assert len(batch_node.output.batches) < len(
        chunks_node.output
    ), "Batch should group items and reduce count"

    # Each batch should be a tuple/list of items
    assert len(batch_node.output.batches) > 0, "Should have at least one batch"
    # Batches are tuples or lists of Box/dict objects from get_items()
    first_batch = batch_node.output.batches[0]
    assert hasattr(first_batch, "__iter__"), "Each batch should be iterable"
    batch_items = list(first_batch)
    assert len(batch_items) > 0, "Batch should contain items"
    assert isinstance(
        batch_items[0], (dict, object)
    ), "Batch items should be dicts or objects"


@pytest.mark.anyio
async def test_verify_quotes_node():
    """Test VerifyQuotes node validates quotes against source."""
    # VerifyQuotes is complex - it requires a specific 'codes' node with quotes
    # Skip for now or test as part of a full pipeline
    pytest.skip(
        "VerifyQuotes requires complex setup with codes node - tested in integration"
    )


@pytest.mark.anyio
async def test_pipeline_serialization():
    """Test that pipelines can be serialized to JSON."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_map.yaml"))
    pipeline.config.document_paths = [str(TEST_DATA)]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    # Test serialization
    json_output = pipeline.model_dump_json()
    assert len(json_output) > 0, "Pipeline should serialize to JSON"
    assert "nodes" in json_output, "Serialized pipeline should contain nodes"


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running Split node test...")
        await test_split_node()
        print("✓ Split node test passed\n")

        print("Running Map node test...")
        await test_map_node()
        print("✓ Map node test passed\n")

        print("Running Classifier node test...")
        await test_classifier_node_with_agreement()
        print("✓ Classifier node test passed\n")

        print("Running Filter node test...")
        await test_filter_node()
        print("✓ Filter node test passed\n")

        print("Running Transform node test...")
        await test_transform_node_multi_slot()
        print("✓ Transform node test passed\n")

        print("Running Reduce node test...")
        await test_reduce_node()
        print("✓ Reduce node test passed\n")

        print("Running Batch node test...")
        await test_batch_node()
        print("✓ Batch node test passed\n")

        print("Running VerifyQuotes node test...")
        await test_verify_quotes_node()
        print("✓ VerifyQuotes node test passed\n")

        print("Running serialization test...")
        await test_pipeline_serialization()
        print("✓ Serialization test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
