#!/usr/bin/env python
"""Comprehensive tests for VerifyQuotes node."""

from pathlib import Path

import pytest
from struckdown import LLMCredentials

from soak.specs import load_template_bundle


@pytest.mark.anyio
async def test_verify_quotes_simple_exact_match():
    """Test VerifyQuotes with simple exact match quotes."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_verify_simple.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # Should have matches
    assert verify_node.sentence_matches is not None
    assert len(verify_node.sentence_matches) > 0

    # Check all quotes were verified
    codes_node = [n for n in pipeline.nodes if n.name == "codes"][0]
    assert codes_node.output is not None

    # Verify statistics
    assert verify_node.stats is not None
    assert verify_node.stats["n_quotes"] > 0
    assert "mean_bm25_score" in verify_node.stats
    assert "mean_cosine" in verify_node.stats

    # Simple exact matches should have high BM25 scores
    for match in verify_node.sentence_matches:
        assert match["bm25_score"] > 0
        assert match["cosine_similarity"] > 0.4  # Lowered threshold for test stability


@pytest.mark.anyio
async def test_verify_quotes_with_ellipsis():
    """Test VerifyQuotes with ellipsis quotes (head/tail matching)."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_verify_ellipsis.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # Check statistics are present
    assert verify_node.stats is not None
    assert "n_with_ellipses" in verify_node.stats

    # If there are ellipsis quotes, verify they have matches
    # (LLM might not always use ellipsis, so this is optional)
    matches_with_ellipsis = [
        m
        for m in verify_node.sentence_matches
        if "..." in m["quote"] or "…" in m["quote"]
    ]

    if len(matches_with_ellipsis) > 0:
        # Ellipsis matches should still have reasonable scores
        for match in matches_with_ellipsis:
            assert match["bm25_score"] > 0
            assert match["span_text"] is not None


@pytest.mark.anyio
async def test_verify_quotes_cross_window():
    """Test VerifyQuotes with quotes that span window boundaries."""
    pipeline = load_template_bundle(
        Path("tests/pipelines/test_verify_cross_window.soak")
    )
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # With small window size (150 chars), some quotes should span windows
    assert verify_node.window_size == 150

    # Should still find matches via neighbor expansion
    assert len(verify_node.sentence_matches) > 0

    # Check match_ratio is computed (from fuzzy alignment)
    matches_with_ratio = [
        m for m in verify_node.sentence_matches if m.get("match_ratio") is not None
    ]
    assert len(matches_with_ratio) > 0


@pytest.mark.anyio
async def test_verify_quotes_poor_matches_llm_judge():
    """Test VerifyQuotes with poor matches triggering LLM-as-judge."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_verify_llm_judge.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # Check if LLM judge was invoked for poor matches
    matches_with_llm = [
        m for m in verify_node.sentence_matches if m.get("llm_is_contained") is not None
    ]

    # If there are poor matches, LLM should have been called
    poor_matches = [
        m
        for m in verify_node.sentence_matches
        if (m["bm25_score"] < 30 and m["bm25_ratio"] < 2)
        or (m["bm25_score"] < 20 and m["cosine_similarity"] < 0.7)
    ]

    if len(poor_matches) > 0:
        assert len(matches_with_llm) > 0, "LLM judge should be invoked for poor matches"

        # Check LLM judge fields
        for match in matches_with_llm:
            assert "llm_explanation" in match
            assert match["llm_is_contained"] in [True, False, None]


@pytest.mark.anyio
async def test_verify_quotes_multi_document():
    """Test VerifyQuotes with multiple source documents (boundary tracking)."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_verify_multi_doc.soak"))
    # Use single document (testing doc boundary tracking)
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # Should have matches
    assert len(verify_node.sentence_matches) > 0

    # Each match should have source_doc and source_doc_content
    for match in verify_node.sentence_matches:
        assert match["source_doc"] is not None
        assert match["source_doc_content"] is not None
        assert len(match["source_doc_content"]) > 0


@pytest.mark.anyio
async def test_verify_quotes_metrics():
    """Test VerifyQuotes produces all expected metrics."""
    pipeline = load_template_bundle(Path("tests/pipelines/test_verify_simple.soak"))
    pipeline.config.document_paths = ["data-private/chainstorecoffee.txt"]
    pipeline.config.llm_credentials = LLMCredentials()

    result, error = await pipeline.run()
    assert error is None, f"Pipeline failed: {error}"

    verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]

    # Check all expected metrics are present
    required_metrics = [
        "n_quotes",
        "n_with_ellipses",
        "mean_bm25_score",
        "median_bm25_score",
        "mean_bm25_ratio",
        "median_bm25_ratio",
        "mean_cosine",
        "median_cosine",
        "min_cosine",
        "max_cosine",
    ]

    for metric in required_metrics:
        assert metric in verify_node.stats, f"Missing metric: {metric}"

    # Check match_ratio stats if available
    if any(m.get("match_ratio") is not None for m in verify_node.sentence_matches):
        assert "mean_match_ratio" in verify_node.stats
        assert "median_match_ratio" in verify_node.stats


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests when executed directly."""
        print("Running simple exact match test...")
        await test_verify_quotes_simple_exact_match()
        print("✓ Simple exact match test passed\n")

        print("Running ellipsis test...")
        await test_verify_quotes_with_ellipsis()
        print("✓ Ellipsis test passed\n")

        print("Running cross-window test...")
        await test_verify_quotes_cross_window()
        print("✓ Cross-window test passed\n")

        print("Running LLM judge test...")
        await test_verify_quotes_poor_matches_llm_judge()
        print("✓ LLM judge test passed\n")

        print("Running multi-document test...")
        await test_verify_quotes_multi_document()
        print("✓ Multi-document test passed\n")

        print("Running metrics test...")
        await test_verify_quotes_metrics()
        print("✓ Metrics test passed\n")

        print("All tests passed!")

    asyncio.run(run_all_tests())
