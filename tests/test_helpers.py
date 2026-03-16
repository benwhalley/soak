"""Tests for soak.helpers module."""

import pytest
from pathlib import Path
import numpy as np

from soak.helpers import (
    derive_input_source,
    print_comparison_stats,
    format_exception_concise,
    resolve_pipeline,
    hash_run_config,
    sanitize_for_filename,
)


class TestDeriveInputSource:
    """Tests for derive_input_source()."""

    def test_empty_list_returns_empty_string(self):
        """Empty input returns empty string."""
        assert derive_input_source([]) == ""

    def test_single_file_returns_pattern(self):
        """Single file returns directory with wildcard extension."""
        result = derive_input_source([Path("data/file.txt")])
        assert result == "data/*.txt"

    def test_multiple_files_same_dir_same_ext(self):
        """Multiple files in same dir with same extension."""
        files = [Path("data/a.txt"), Path("data/b.txt"), Path("data/c.txt")]
        result = derive_input_source(files)
        assert result == "data/*.txt"

    def test_multiple_files_same_dir_mixed_ext(self):
        """Multiple files with different extensions returns *."""
        files = [Path("data/a.txt"), Path("data/b.csv"), Path("data/c.json")]
        result = derive_input_source(files)
        assert result == "data/*"

    def test_tuple_input_extracts_path(self):
        """Works with (path, metadata) tuples."""
        files = [("data/a.txt", {}), ("data/b.txt", {"key": "value"})]
        result = derive_input_source(files)
        assert result == "data/*.txt"

    def test_mixed_directories_finds_common_parent(self):
        """Files in different subdirs finds common parent."""
        files = [Path("data/sub1/a.txt"), Path("data/sub2/b.txt")]
        result = derive_input_source(files)
        assert "data" in result

    def test_no_extension_files(self):
        """Files without extension return *."""
        files = [Path("data/file1"), Path("data/file2")]
        result = derive_input_source(files)
        assert result == "data/*"


class TestPrintComparisonStats:
    """Tests for print_comparison_stats()."""

    @pytest.fixture
    def minimal_result(self):
        """Minimal comparison result dict."""
        return {
            "similarity_metric": "angular",
            "hit_rate_a": 0.8,
            "hit_rate_b": 0.7,
            "jaccard": 0.5,
            "mean_max_sim_a_to_b": 0.75,
            "mean_max_sim_b_to_a": 0.72,
            "fidelity": 0.73,
            "hungarian": {
                "thresholded_metrics": {
                    "coverage_a": 0.6,
                    "coverage_b": 0.5,
                    "true_jaccard": 0.4,
                }
            },
            "ot_by_k": {
                0.25: {
                    "ot": {
                        "shared_mass": 0.65,
                        "avg_cost": 0.3,
                        "transport_plan": [[0.5, 0.3], [0.1, 0.1]],
                    }
                }
            },
            "default_k": 0.25,
            "selected_similarity_matrix": [[0.9, 0.7], [0.6, 0.85]],
        }

    def test_returns_string(self, minimal_result):
        """Returns a formatted string."""
        result = print_comparison_stats(
            minimal_result,
            name_a="Analysis A",
            name_b="Analysis B",
            list_a=["Theme 1", "Theme 2"],
            list_b=["Theme X", "Theme Y"],
            threshold=0.6,
            embedding_model="test-model",
            shepard_k=1.0,
            ot_k_values=[0.25],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_names(self, minimal_result):
        """Output contains analysis names."""
        result = print_comparison_stats(
            minimal_result,
            name_a="Analysis A",
            name_b="Analysis B",
            list_a=["T1", "T2"],
            list_b=["TX", "TY"],
            threshold=0.6,
            embedding_model="test-model",
            shepard_k=1.0,
            ot_k_values=[],
        )
        assert "Analysis A" in result
        assert "Analysis B" in result

    def test_contains_statistics(self, minimal_result):
        """Output contains key statistics."""
        result = print_comparison_stats(
            minimal_result,
            name_a="A",
            name_b="B",
            list_a=["T1", "T2"],
            list_b=["TX", "TY"],
            threshold=0.6,
            embedding_model="test-model",
            shepard_k=1.0,
            ot_k_values=[],
        )
        assert "Hit Rate" in result
        assert "Jaccard" in result
        assert "Fidelity" in result
        assert "HUNGARIAN" in result
        assert "OPTIMAL TRANSPORT" in result

    def test_contains_embedding_model(self, minimal_result):
        """Output contains embedding model name."""
        result = print_comparison_stats(
            minimal_result,
            name_a="A",
            name_b="B",
            list_a=["T1"],
            list_b=["TX"],
            threshold=0.6,
            embedding_model="my-special-model",
            shepard_k=1.0,
            ot_k_values=[],
        )
        assert "my-special-model" in result


class TestFormatExceptionConcise:
    """Tests for format_exception_concise()."""

    def test_formats_basic_exception(self):
        """Formats a basic exception."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = format_exception_concise(e)
            assert "ValueError" in result
            assert "test error" in result

    def test_includes_file_info(self):
        """Includes file and line info when available."""
        try:
            raise RuntimeError("file test")
        except RuntimeError as e:
            result = format_exception_concise(e)
            # should mention the file
            assert "test_helpers.py" in result or "File:" in result


class TestResolvePipeline:
    """Tests for resolve_pipeline()."""

    def test_finds_builtin_pipeline(self, tmp_path):
        """Can find built-in pipelines."""
        from soak.cli._common import PIPELINE_DIR

        # list actual pipelines
        existing = list(PIPELINE_DIR.glob("**/*.soak"))
        if existing:
            name = existing[0].stem
            result = resolve_pipeline(name, tmp_path, PIPELINE_DIR)
            assert result.exists()
            assert result.suffix == ".soak"

    def test_local_takes_precedence(self, tmp_path):
        """Local pipeline takes precedence over built-in."""
        from soak.cli._common import PIPELINE_DIR

        # create local file
        local_file = tmp_path / "test.soak"
        local_file.write_text("local content")

        result = resolve_pipeline("test", tmp_path, PIPELINE_DIR)
        assert result == local_file

    def test_not_found_raises_error(self, tmp_path):
        """FileNotFoundError raised when not found."""
        from soak.cli._common import PIPELINE_DIR

        with pytest.raises(FileNotFoundError):
            resolve_pipeline("nonexistent_xyz123", tmp_path, PIPELINE_DIR)


class TestHashRunConfig:
    """Tests for hash_run_config()."""

    def test_returns_string(self):
        """Returns a string hash."""
        result = hash_run_config(input_files=["a.txt", "b.txt"])
        assert isinstance(result, str)

    def test_default_length_is_4(self):
        """Default hash length is 4."""
        result = hash_run_config(input_files=["a.txt"])
        assert len(result) == 4

    def test_custom_length(self):
        """Can specify custom hash length."""
        result = hash_run_config(input_files=["a.txt"], length=8)
        assert len(result) == 8

    def test_same_inputs_same_hash(self):
        """Same inputs produce same hash."""
        h1 = hash_run_config(input_files=["a.txt", "b.txt"])
        h2 = hash_run_config(input_files=["a.txt", "b.txt"])
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        """Different inputs produce different hash."""
        h1 = hash_run_config(input_files=["a.txt"])
        h2 = hash_run_config(input_files=["b.txt"])
        assert h1 != h2

    def test_includes_model_in_hash(self):
        """Model name affects hash."""
        h1 = hash_run_config(input_files=["a.txt"], model_name="gpt-4")
        h2 = hash_run_config(input_files=["a.txt"], model_name="gpt-3.5")
        assert h1 != h2


class TestSanitizeForFilename:
    """Tests for sanitize_for_filename()."""

    def test_removes_slashes(self):
        """Removes forward and backward slashes."""
        assert "/" not in sanitize_for_filename("path/to/file")
        assert "\\" not in sanitize_for_filename("path\\to\\file")

    def test_removes_special_chars(self):
        """Removes special characters."""
        result = sanitize_for_filename('file:name*with?special<chars>"here|now')
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result
        assert "|" not in result

    def test_replaces_spaces(self):
        """Replaces spaces with underscores."""
        result = sanitize_for_filename("file with spaces")
        assert " " not in result
        assert "_" in result

    def test_safe_characters_preserved(self):
        """Safe characters are preserved."""
        safe = "abcABC123-_."
        result = sanitize_for_filename(safe)
        # dots, dashes, underscores, and alphanumeric should be preserved
        assert "abc" in result
        assert "123" in result
