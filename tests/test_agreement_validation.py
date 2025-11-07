#!/usr/bin/env python
"""Tests for agreement metrics validation in Classifier node."""

from pathlib import Path

import pytest
from struckdown import LLMCredentials

from soak.specs import load_template_bundle


@pytest.mark.anyio
async def test_agreement_metrics_calculation():
    """Test agreement metrics are calculated correctly."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_agreement_basic.soak"))
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Check agreement metrics were calculated
    result_dict = classifier_node.result()
    assert result_dict["agreement_stats"] is not None

    # Should have stats for each agreement field
    for field in classifier_node.agreement_fields:
        assert field in result_dict["agreement_stats"]

        field_stats = result_dict["agreement_stats"][field]

        # Check expected metrics
        assert "Fleiss_Kappa" in field_stats
        assert "Percent_Agreement" in field_stats
        assert "n_items" in field_stats
        assert "n_raters" in field_stats

        # Kappa should be in valid range [-1, 1]
        assert -1 <= field_stats["Fleiss_Kappa"] <= 1

        # Percent agreement should be in [0, 100]
        assert 0 <= field_stats["Percent_Agreement"] <= 100

        # Should have correct number of raters
        assert field_stats["n_raters"] == len(classifier_node.model_names)


@pytest.mark.anyio
async def test_agreement_auto_detection():
    """Test agreement fields are auto-detected from template."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_agreement_auto_detect.soak")
    )
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    # Agreement fields should be auto-detected (not explicitly set in YAML)
    assert classifier_node.agreement_fields is not None
    assert len(classifier_node.agreement_fields) > 0

    # Should match output keys from template
    assert "sentiment" in classifier_node.agreement_fields


@pytest.mark.skip(reason="Agreement export features not fully implemented yet")
@pytest.mark.anyio
async def test_agreement_export():
    """Test agreement statistics are exported correctly."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_agreement_basic.soak"))
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

        # Check agreement stats files
        assert (folder / "agreement_stats.csv").exists()
        assert (folder / "agreement_stats.json").exists()

        # Read CSV and validate structure
        import pandas as pd

        df = pd.read_csv(folder / "agreement_stats.csv")
        assert "field" in df.columns
        assert "fleiss_kappa" in df.columns
        assert "percent_agreement" in df.columns


@pytest.mark.skip(reason="Agreement script generation not implemented yet")
@pytest.mark.anyio
async def test_agreement_script_generation():
    """Test agreement calculation scripts are generated."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_agreement_basic.soak"))
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

        # Check agreement script was generated
        script_files = list(folder.glob("*agreement*.py"))
        assert len(script_files) > 0

        # Script should be valid Python
        for script_file in script_files:
            script_content = script_file.read_text()
            assert "import pandas" in script_content or "import pyirr" in script_content


@pytest.mark.skip(reason="Human rater template generation not implemented yet")
@pytest.mark.anyio
async def test_human_rater_template_generation():
    """Test human rater template is generated for manual coding."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_agreement_basic.soak"))
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

        # Check human rater template was created
        template_files = list(folder.glob("*human_rater_template*.csv"))
        assert len(template_files) > 0

        # Template should have required columns
        import pandas as pd

        for template_file in template_files:
            df = pd.read_csv(template_file)

            # Should have source columns
            assert "item_id" in df.columns or "index" in df.columns

            # Should have empty coding columns for human rater
            for field in classifier_node.agreement_fields:
                assert field in df.columns


@pytest.mark.anyio
async def test_agreement_multiple_fields():
    """Test agreement calculation for multiple fields."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_agreement_multi_field.soak")
    )
    pipeline.config.document_paths = ["soak/data/test_data.csv"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    classifier_node = [n for n in pipeline.nodes if n.name == "classify"][0]

    result_dict = classifier_node.result()
    agreement_stats = result_dict["agreement_stats"]

    # Should have stats for all fields
    assert len(agreement_stats) >= 2

    # Each field should have separate metrics
    for field in classifier_node.agreement_fields:
        assert field in agreement_stats
        # Metrics can vary by field
        assert "Fleiss_Kappa" in agreement_stats[field]


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running agreement metrics calculation test...")
        await test_agreement_metrics_calculation()
        print("✓ Agreement metrics calculation test passed\n")

        print("Running auto-detection test...")
        await test_agreement_auto_detection()
        print("✓ Auto-detection test passed\n")

        print("Running agreement export test...")
        await test_agreement_export()
        print("✓ Agreement export test passed\n")

        print("Running script generation test...")
        await test_agreement_script_generation()
        print("✓ Script generation test passed\n")

        print("Running human rater template test...")
        await test_human_rater_template_generation()
        print("✓ Human rater template test passed\n")

        print("Running multiple fields test...")
        await test_agreement_multiple_fields()
        print("✓ Multiple fields test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
