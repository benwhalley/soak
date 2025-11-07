#!/usr/bin/env python
"""Tests for ground truth validation in Classifier node."""

from pathlib import Path

import pytest
from struckdown import LLMCredentials

from soak.specs import load_template_bundle


@pytest.mark.anyio
async def test_classifier_ground_truth_basic():
    """Test Classifier with ground truth validation against existing column."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_classifier_ground_truth.soak")
    )
    # Use test_data.csv which has 'condition' column
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Check ground truth config was set
    assert classifier_node.ground_truths is not None
    assert "condition_pred" in classifier_node.ground_truths

    # Check metrics were calculated
    result_dict = classifier_node.result()
    assert result_dict["ground_truth_stats"] is not None

    # Check expected metrics
    gt_stats = result_dict["ground_truth_stats"]
    assert "condition_pred" in gt_stats

    # Should have metrics for each model
    for model_name, metrics in gt_stats["condition_pred"].items():
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

        # Metrics should be in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1


@pytest.mark.anyio
async def test_classifier_ground_truth_export():
    """Test ground truth metrics export to CSV/JSON."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_classifier_ground_truth.soak")
    )
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

        # Check ground truth files were created
        assert (folder / "ground_truth_metrics.csv").exists()
        assert (folder / "ground_truth_metrics.json").exists()

        # Check confusion matrix files
        confusion_files = list(folder.glob("confusion_matrix_*.csv"))
        assert len(confusion_files) > 0


@pytest.mark.anyio
async def test_classifier_ground_truth_multiple_fields():
    """Test Classifier with multiple ground truth fields."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_classifier_ground_truth_multi.soak")
    )
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Check multiple ground truth fields
    assert len(classifier_node.ground_truths) >= 2

    result_dict = classifier_node.result()
    gt_stats = result_dict["ground_truth_stats"]

    # Should have stats for all ground truth fields
    for field_name in classifier_node.ground_truths.keys():
        assert field_name in gt_stats


@pytest.mark.anyio
async def test_classifier_ground_truth_and_agreement():
    """Test Classifier with both ground truth and agreement metrics."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_classifier_ground_truth_and_agreement.soak")
    )
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Check both metrics were calculated
    result_dict = classifier_node.result()

    # Ground truth metrics
    assert result_dict["ground_truth_stats"] is not None
    assert len(result_dict["ground_truth_stats"]) > 0

    # Agreement metrics (requires multiple models)
    if len(classifier_node.model_names) >= 2:
        assert result_dict["agreement_stats"] is not None
        assert len(result_dict["agreement_stats"]) > 0


@pytest.mark.anyio
async def test_classifier_ground_truth_confusion_matrix():
    """Test confusion matrix structure and content."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_classifier_ground_truth.soak")
    )
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]
    result_dict = classifier_node.result()
    gt_stats = result_dict["ground_truth_stats"]

    # Check confusion matrix structure
    for field_name, model_metrics in gt_stats.items():
        for model_name, metrics in model_metrics.items():
            cm = metrics["confusion_matrix"]

            # Should be 2D array for binary classification
            assert isinstance(cm, list)
            assert len(cm) > 0
            assert isinstance(cm[0], list)

            # Should have labels
            assert "labels" in metrics
            assert len(metrics["labels"]) > 0


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running basic ground truth test...")
        await test_classifier_ground_truth_basic()
        print("✓ Basic ground truth test passed\n")

        print("Running ground truth export test...")
        await test_classifier_ground_truth_export()
        print("✓ Ground truth export test passed\n")

        print("Running multiple fields test...")
        await test_classifier_ground_truth_multiple_fields()
        print("✓ Multiple fields test passed\n")

        print("Running ground truth + agreement test...")
        await test_classifier_ground_truth_and_agreement()
        print("✓ Ground truth + agreement test passed\n")

        print("Running confusion matrix test...")
        await test_classifier_ground_truth_confusion_matrix()
        print("✓ Confusion matrix test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
