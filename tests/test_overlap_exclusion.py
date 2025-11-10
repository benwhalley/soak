"""Test overlap exclusion in Split→Reduce pipeline."""

from soak.models.base import TrackedItem
from soak.models.nodes.base import Split
from soak.models.nodes.batch import BatchList

# Rebuild model to resolve forward references
Split.model_rebuild()


def test_overlap_exclusion():
    """Test that Reduce excludes overlap when rejoining chunks."""
    # Create a simple test document
    test_text = "The quick brown fox jumps over the lazy dog and runs away fast"
    doc = TrackedItem(
        content=test_text,
        id="test_doc",
        sources=["test_doc"],
        metadata={"filename": "test.txt"}
    )

    # Create Split node
    split_node = Split(
        name="chunks",
        split_unit="words",
        chunk_size=5,  # 5 words per chunk
        overlap=2      # 2 words overlap
    )

    # Split the document
    chunks = split_node.split_tracked_document(doc)

    print(f"\nOriginal text ({len(test_text)} chars):")
    print(f"  '{test_text}'")

    print(f"\nChunks created (overlap={split_node.overlap}):")
    for idx, chunk in enumerate(chunks):
        overlap_region = chunk.content_excluding_overlap
        core = chunk.get_core_content()
        print(f"  Chunk {idx}: '{chunk.content}' (len={len(chunk.content)})")
        print(f"    Overlap region: {overlap_region}")
        print(f"    Core content: '{core}' (len={len(core)})")

    # Rejoin with core content only
    rejoined_core = "".join([chunk.get_core_content() for chunk in chunks])

    # Rejoin with full content (including overlap)
    rejoined_full = "".join([str(chunk.content) for chunk in chunks])

    print(f"\nRejoined with core content only ({len(rejoined_core)} chars):")
    print(f"  '{rejoined_core}'")

    print(f"\nRejoined with full content ({len(rejoined_full)} chars):")
    print(f"  '{rejoined_full}'")

    # Verify results
    print("\n" + "="*60)
    if len(rejoined_core) <= len(test_text):
        print("✓ SUCCESS: Rejoined core content is not longer than original")
    else:
        print("✗ FAIL: Rejoined core content is longer than original (has duplicates)")

    if len(rejoined_full) > len(rejoined_core):
        print("✓ SUCCESS: Rejoined full content is longer (as expected)")
    else:
        print("✗ FAIL: Rejoined full content is not longer")

    if rejoined_core == test_text:
        print("✓ SUCCESS: Rejoined core content matches original exactly!")
    else:
        print(f"⚠ WARNING: Rejoined core content differs from original")
        print(f"  Expected: '{test_text}'")
        print(f"  Got:      '{rejoined_core}'")


if __name__ == "__main__":
    test_overlap_exclusion()
