#!/usr/bin/env python
"""Integration tests for multi-node pipelines."""

from pathlib import Path

import pytest
from struckdown import LLMCredentials

from soak.specs import load_template_bundle


@pytest.mark.anyio
async def test_map_classifier_pipeline():
    """Test Map → Classifier pipeline with provenance tracking."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_map_classifier.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    map_node = [n for n in pipeline.nodes if n.name == "extract_themes"][0]
    classifier_node = [n for n in pipeline.nodes if n.name == "classify_themes"][0]

    # Map should have processed all documents
    assert map_node.output is not None
    assert len(map_node.output) > 0

    # Classifier should have processed all map outputs
    assert classifier_node.output is not None

    # Check provenance preservation
    if hasattr(map_node.output[0], "source_id"):
        # TrackedItem provenance should be preserved
        assert map_node.output[0].source_id is not None


@pytest.mark.anyio
async def test_split_map_reduce_pipeline():
    """Test Split → Map → Reduce full pipeline."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_split_map_reduce.soak"))
    # Use a document that will split into multiple chunks
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    split_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    map_node = [n for n in pipeline.nodes if n.name == "summarize_chunks"][0]
    reduce_node = [n for n in pipeline.nodes if n.name == "combine"][0]

    # Split should create multiple chunks
    assert len(split_node.output) > 1

    # Map should process each chunk
    assert len(map_node.output) == len(split_node.output)

    # Reduce should combine into single output
    assert reduce_node.output is not None
    assert isinstance(reduce_node.output, str)

    # Check provenance through pipeline
    if hasattr(split_node.output[0], "source_id"):
        # Source IDs should have split node name
        assert split_node.name in split_node.output[0].source_id


@pytest.mark.anyio
async def test_split_map_classifier_pipeline():
    """Test Split → Map → Classifier with metadata propagation."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_split_map_classifier.soak")
    )
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    split_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    map_node = [n for n in pipeline.nodes if n.name == "extract"][0]
    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # All nodes should have output
    assert split_node.output is not None
    assert map_node.output is not None
    assert classifier_node.output is not None

    # Check metadata in classifier output
    result_dict = classifier_node.result()
    model_dfs = result_dict["model_dfs"]

    if model_dfs:
        first_model_df = list(model_dfs.values())[0]
        # Should have item_id or index column (from TrackedItem)
        assert "item_id" in first_model_df.columns or "index" in first_model_df.columns


@pytest.mark.anyio
async def test_filter_map_pipeline():
    """Test Filter → Map pipeline with item selection."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_filter_map.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    chunks_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    filter_node = [n for n in pipeline.nodes if n.name == "filter_relevant"][0]
    map_node = [n for n in pipeline.nodes if n.name == "analyze"][0]

    # Filter should reduce item count
    assert len(filter_node.output) < len(chunks_node.output)

    # Map should only process filtered items
    assert len(map_node.output) == len(filter_node.output)


@pytest.mark.anyio
async def test_batch_map_pipeline():
    """Test Batch → Map pipeline with batch processing."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_batch_map.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    chunks_node = [n for n in pipeline.nodes if n.name == "chunks"][0]
    batch_node = [n for n in pipeline.nodes if n.name == "batched"][0]
    map_node = [n for n in pipeline.nodes if n.name == "process_batches"][0]

    # Batch should group items
    from soak.models.nodes.batch import BatchList

    assert isinstance(batch_node.output, BatchList)
    assert len(batch_node.output.batches) < len(chunks_node.output)

    # Map should process batches and return BatchList
    assert isinstance(map_node.output, BatchList)


@pytest.mark.anyio
async def test_reduce_verify_pipeline():
    """Test Map → VerifyQuotes pipeline with code extraction."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_reduce_verify.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    map_node = [n for n in pipeline.nodes if n.name == "extract_codes"][0]
    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # Map should extract codes from documents
    assert map_node.output is not None
    assert len(map_node.output) > 0

    # VerifyQuotes should validate quotes from map output
    assert verify_node.sentence_matches is not None
    assert len(verify_node.sentence_matches) > 0

    # Verify statistics should be present
    assert verify_node.stats is not None
    assert verify_node.stats["n_quotes"] > 0
    assert "mean_bm25_score" in verify_node.stats
    assert "mean_cosine" in verify_node.stats

    # All quotes should have been verified
    for match in verify_node.sentence_matches:
        assert match["bm25_score"] > 0
        assert "cosine_similarity" in match


@pytest.mark.anyio
async def test_provenance_through_complex_pipeline():
    """Test source_id format through complex multi-node pipeline."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_provenance.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    # Check each node preserves/extends source_id
    for node in pipeline.nodes:
        if node.output and isinstance(node.output, list) and len(node.output) > 0:
            first_item = node.output[0]

            # Check TrackedItem has source_id
            if hasattr(first_item, "source_id"):
                source_id = first_item.source_id
                assert source_id is not None

                # Source ID should contain node name for Split nodes
                if node.type == "Split":
                    assert node.name in source_id

                # Check metadata
                if hasattr(first_item, "metadata") and first_item.metadata:
                    assert isinstance(first_item.metadata, dict)


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running Map → Classifier test...")
        await test_map_classifier_pipeline()
        print("✓ Map → Classifier test passed\n")

        print("Running Split → Map → Reduce test...")
        await test_split_map_reduce_pipeline()
        print("✓ Split → Map → Reduce test passed\n")

        print("Running Split → Map → Classifier test...")
        await test_split_map_classifier_pipeline()
        print("✓ Split → Map → Classifier test passed\n")

        print("Running Filter → Map test...")
        await test_filter_map_pipeline()
        print("✓ Filter → Map test passed\n")

        print("Running Batch → Map test...")
        await test_batch_map_pipeline()
        print("✓ Batch → Map test passed\n")

        print("Running Map → Reduce → Verify test...")
        await test_reduce_verify_pipeline()
        print("✓ Map → Reduce → Verify test passed\n")

        print("Running provenance test...")
        await test_provenance_through_complex_pipeline()
        print("✓ Provenance test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
