#!/usr/bin/env python
"""Tests for spreadsheet (CSV/Excel) input handling."""

from pathlib import Path

import pytest
from struckdown import LLMCredentials

from soak.specs import load_template_bundle


@pytest.mark.anyio
async def test_csv_input_basic():
    """Test basic CSV input creates TrackedItems with metadata."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_csv_input.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    # Documents should be loaded from CSV
    assert len(pipeline.config.documents) > 0

    # Each document should be TrackedItem with metadata
    for doc in pipeline.config.documents:
        assert hasattr(doc, "metadata")
        assert doc.metadata is not None

        # Should have column data in metadata
        assert "participant_id" in doc.metadata or "age" in doc.metadata


@pytest.mark.anyio
async def test_csv_column_access_in_template():
    """Test CSV columns are accessible in templates via metadata."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_csv_columns.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    map_node = [n for n in pipeline.nodes if n.name == "process"][0]

    # Should have processed each row
    assert len(map_node.output) == len(pipeline.config.documents)


@pytest.mark.anyio
async def test_csv_classifier_with_ground_truth():
    """Test CSV input with ground truth validation."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_classifier_ground_truth.soak")
    )
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Ground truth should work with CSV columns
    result_dict = classifier_node.result()
    assert result_dict["ground_truth_stats"] is not None

    # Should compare against 'condition' column
    assert "condition_pred" in result_dict["ground_truth_stats"]


@pytest.mark.anyio
async def test_csv_provenance_tracking():
    """Test provenance tracking with CSV input."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_csv_provenance.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    map_node = [n for n in pipeline.nodes if n.name == "process"][0]

    # Check source_id includes row identifier
    for idx, item in enumerate(map_node.output):
        if hasattr(item, "metadata"):
            # Should have original CSV metadata
            assert "participant_id" in item.metadata or "index" in item.metadata


@pytest.mark.anyio
async def test_csv_multiple_columns_in_template():
    """Test accessing multiple CSV columns in template."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_csv_multi_column.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    map_node = [n for n in pipeline.nodes if n.name == "analyze"][0]

    # Should have processed each row with all columns
    assert len(map_node.output) > 0


@pytest.mark.anyio
async def test_csv_export_preserves_metadata():
    """Test export preserves CSV metadata in output files."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_csv_export.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Export to temp directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        classifier_node.export(folder)

        # Check CSV export includes original columns
        import pandas as pd

        csv_files = list(folder.glob("*.csv"))
        assert len(csv_files) > 0

        for csv_file in csv_files:
            if "classifications" in csv_file.name:
                df = pd.read_csv(csv_file)

                # Should have item_id or similar identifier
                assert "item_id" in df.columns or "index" in df.columns

                # Should have classification outputs
                assert "sentiment" in df.columns


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running CSV input basic test...")
        await test_csv_input_basic()
        print("✓ CSV input basic test passed\n")

        print("Running CSV column access test...")
        await test_csv_column_access_in_template()
        print("✓ CSV column access test passed\n")

        print("Running CSV ground truth test...")
        await test_csv_classifier_with_ground_truth()
        print("✓ CSV ground truth test passed\n")

        print("Running CSV provenance test...")
        await test_csv_provenance_tracking()
        print("✓ CSV provenance test passed\n")

        print("Running CSV multi-column test...")
        await test_csv_multiple_columns_in_template()
        print("✓ CSV multi-column test passed\n")

        print("Running CSV export test...")
        await test_csv_export_preserves_metadata()
        print("✓ CSV export test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
