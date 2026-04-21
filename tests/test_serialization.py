"""Tests for soak.serialization -- round-trip and display extraction."""

import pytest
# import StruckdownResult/SlotResult from results module directly
# to avoid the top-level struckdown __init__ which pulls in litellm
from struckdown.results import SlotResult, StruckdownResult

from soak.models.base import (BatchList, Code, CodeList, Quote, QuoteReference,
                              Theme, Themes, TrackedItem)
from soak.serialization import (deserialize_node_output, deserialize_value,
                                extract_display_output, serialize_node_output,
                                serialize_value)

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _make_code(**overrides) -> Code:
    defaults = {
        "slug": "test-code-slug",
        "name": "Test Code",
        "description": "A test code for serialization.",
        "quotes": [],
    }
    defaults.update(overrides)
    return Code(**defaults)


def _make_theme(**overrides) -> Theme:
    defaults = {
        "name": "A test theme for serialization",
        "description": "Describes the test theme in detail",
        "code_slugs": ["test-code-slug"],
    }
    defaults.update(overrides)
    return Theme(**defaults)


def _make_complete_result(slots: dict) -> StruckdownResult:
    """Build a StruckdownResult from {slot_name: output_value}."""
    results = {}
    for name, output in slots.items():
        results[name] = SlotResult(
            name=name,
            prompt="test prompt",
            output=output,
            action="code" if isinstance(output, (Code, list)) else "respond",
        )
    return StruckdownResult(results=results)


# --------------------------------------------------------------------------- #
#  Round-trip tests
# --------------------------------------------------------------------------- #


class TestCodeRoundTrip:
    def test_basic(self):
        code = _make_code()
        serialized = serialize_value(code)
        assert serialized["__type__"] == "Code"
        assert serialized["slug"] == "test-code-slug"

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, Code)
        assert deserialized.slug == code.slug
        assert deserialized.name == code.name
        assert deserialized.description == code.description

    def test_with_quotes(self):
        code = _make_code(
            quotes=[
                Quote(text="Example quote", source="doc1"),
                QuoteReference(hash="abcdefgh"),
            ]
        )
        serialized = serialize_value(code)
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, Code)
        assert len(deserialized.quotes) == 2
        assert isinstance(deserialized.quotes[0], Quote)
        assert isinstance(deserialized.quotes[1], QuoteReference)
        assert deserialized.quotes[0].text == "Example quote"


class TestThemeRoundTrip:
    def test_basic(self):
        theme = _make_theme()
        serialized = serialize_value(theme)
        assert serialized["__type__"] == "Theme"

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, Theme)
        assert deserialized.name == theme.name
        assert deserialized.code_slugs == theme.code_slugs


class TestQuoteRoundTrip:
    def test_quote(self):
        q = Quote(text="hello world", source="doc1", metadata={"key": "val"})
        serialized = serialize_value(q)
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, Quote)
        assert deserialized.text == "hello world"
        assert deserialized.source == "doc1"

    def test_quote_reference(self):
        qr = QuoteReference(hash="abcdefgh")
        serialized = serialize_value(qr)
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, QuoteReference)
        assert deserialized.hash == "abcdefgh"


class TestTrackedItemRoundTrip:
    def test_string_content(self):
        item = TrackedItem(
            content="hello world",
            id="doc1__split__0",
            sources=["doc1"],
            metadata={"filename": "test.txt"},
            content_excluding_overlap=(5, 11),
        )
        serialized = serialize_value(item)
        assert serialized["__type__"] == "TrackedItem"
        assert serialized["id"] == "doc1__split__0"

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, TrackedItem)
        assert deserialized.content == "hello world"
        assert deserialized.id == "doc1__split__0"
        assert deserialized.sources == ["doc1"]
        assert deserialized.metadata == {"filename": "test.txt"}
        assert deserialized.content_excluding_overlap == (5, 11)

    def test_complete_content(self):
        """TrackedItem whose content is a StruckdownResult (after Map node)."""
        code = _make_code()
        cr = _make_complete_result({"codes": [code]})
        item = TrackedItem(content=cr, id="doc1__coded", sources=["doc1"])

        serialized = serialize_value(item)
        assert serialized["__type__"] == "TrackedItem"
        assert serialized["content"]["__type__"] == "StruckdownResult"

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, TrackedItem)
        assert isinstance(deserialized.content, StruckdownResult)
        codes = deserialized.content.results["codes"].output
        assert isinstance(codes[0], Code)


class TestStruckdownResultRoundTrip:
    def test_single_slot(self):
        cr = _make_complete_result({"narrative": "A story about data"})
        serialized = serialize_value(cr)
        assert serialized["__type__"] == "StruckdownResult"

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, StruckdownResult)
        assert deserialized.results["narrative"].output == "A story about data"

    def test_with_code_outputs(self):
        codes = [_make_code(slug="code-one-slug-a"), _make_code(slug="code-two-slug-b")]
        cr = _make_complete_result({"codes": codes})

        serialized = serialize_value(cr)
        deserialized = deserialize_value(serialized)

        assert isinstance(deserialized, StruckdownResult)
        output = deserialized.results["codes"].output
        assert len(output) == 2
        assert all(isinstance(c, Code) for c in output)
        assert output[0].slug == "code-one-slug-a"

    def test_multi_slot(self):
        codes = [_make_code()]
        themes = [_make_theme()]
        cr = _make_complete_result({"codes": codes, "themes": themes})

        serialized = serialize_value(cr)
        deserialized = deserialize_value(serialized)

        assert isinstance(deserialized, StruckdownResult)
        assert isinstance(deserialized.results["codes"].output[0], Code)
        assert isinstance(deserialized.results["themes"].output[0], Theme)

    def test_strips_debug_data(self):
        """Serialized StruckdownResult should not contain prompts."""
        cr = _make_complete_result({"text": "hello"})
        serialized = serialize_value(cr)
        # should not have prompt in the serialized data
        result_data = serialized["results"]["text"]
        assert "prompt" not in result_data or result_data.get("prompt") is None


class TestContainerRoundTrip:
    def test_code_list(self):
        cl = CodeList(codes=[_make_code(slug="code-in-list-a")])
        serialized = serialize_value(cl)
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, CodeList)
        assert len(deserialized.codes) == 1

    def test_themes(self):
        t = Themes(themes=[_make_theme()])
        serialized = serialize_value(t)
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, Themes)
        assert len(deserialized.themes) == 1


class TestMapOutputRoundTrip:
    def test_list_of_complete_results(self):
        """Map output: list of StruckdownResults."""
        items = [
            _make_complete_result({"code": _make_code(slug="map-code-one-a")}),
            _make_complete_result({"code": _make_code(slug="map-code-two-b")}),
        ]
        serialized = serialize_value(items)
        assert isinstance(serialized, list)
        assert len(serialized) == 2

        deserialized = deserialize_value(serialized)
        assert all(isinstance(item, StruckdownResult) for item in deserialized)
        assert deserialized[0].results["code"].output.slug == "map-code-one-a"


class TestSplitOutputRoundTrip:
    def test_list_of_tracked_items(self):
        items = [
            TrackedItem(content="chunk 1", id="doc__s__0", sources=["doc"]),
            TrackedItem(content="chunk 2", id="doc__s__1", sources=["doc"]),
        ]
        serialized = serialize_value(items)
        deserialized = deserialize_value(serialized)

        assert len(deserialized) == 2
        assert all(isinstance(item, TrackedItem) for item in deserialized)
        assert deserialized[0].id == "doc__s__0"
        assert deserialized[1].content == "chunk 2"


# --------------------------------------------------------------------------- #
#  Legacy / graceful handling
# --------------------------------------------------------------------------- #


class TestLegacyData:
    def test_plain_dict_passthrough(self):
        """Dicts without __type__ should pass through unchanged."""
        data = {"slug": "abc", "name": "Test", "description": "Desc"}
        result = deserialize_value(data)
        assert isinstance(result, dict)
        assert result == data

    def test_unknown_type_returns_dict(self):
        """Unknown __type__ values should return dict without crashing."""
        data = {"__type__": "FutureType", "field": "value"}
        result = deserialize_value(data)
        assert isinstance(result, dict)
        assert result == {"field": "value"}

    def test_legacy_list_of_dicts(self):
        """Legacy format: list of plain dicts (no __type__)."""
        data = [
            {"slug": "a", "name": "A", "description": "desc a"},
            {"slug": "b", "name": "B", "description": "desc b"},
        ]
        result = deserialize_value(data)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_primitives_passthrough(self):
        assert deserialize_value(None) is None
        assert deserialize_value("hello") == "hello"
        assert deserialize_value(42) == 42
        assert deserialize_value(3.14) == 3.14
        assert deserialize_value(True) is True


# --------------------------------------------------------------------------- #
#  Display extraction
# --------------------------------------------------------------------------- #


class TestExtractDisplayOutput:
    def test_complete_single_slot(self):
        """Single-slot StruckdownResult envelope extracts to just the output."""
        data = {
            "__type__": "StruckdownResult",
            "results": {
                "codes": {
                    "action": "code",
                    "output": [
                        {"__type__": "Code", "slug": "abc-test-code", "name": "Test"},
                    ],
                }
            },
        }
        display = extract_display_output(data)
        # should be the list of codes, not the StruckdownResult envelope
        assert isinstance(display, list)
        assert display[0]["slug"] == "abc-test-code"

    def test_complete_multi_slot(self):
        """Multi-slot StruckdownResult extracts to {slot_name: output}."""
        data = {
            "__type__": "StruckdownResult",
            "results": {
                "codes": {
                    "action": "code",
                    "output": [{"__type__": "Code", "slug": "a-test-slug"}],
                },
                "themes": {
                    "action": "theme",
                    "output": [{"__type__": "Theme", "name": "T"}],
                },
            },
        }
        display = extract_display_output(data)
        assert isinstance(display, dict)
        assert "codes" in display
        assert "themes" in display

    def test_tracked_item_display(self):
        """TrackedItem envelope shows id and stringified content."""
        data = {
            "__type__": "TrackedItem",
            "content": "Some text here",
            "id": "doc__split__0",
            "sources": ["doc"],
            "metadata": {},
        }
        display = extract_display_output(data)
        assert display["id"] == "doc__split__0"
        assert display["content"] == "Some text here"

    def test_list_of_items(self):
        """Lists are extracted element by element."""
        data = [
            {
                "__type__": "Code",
                "slug": "a-code-slug",
                "name": "A",
                "description": "D",
            },
            {
                "__type__": "Code",
                "slug": "b-code-slug",
                "name": "B",
                "description": "D",
            },
        ]
        display = extract_display_output(data)
        assert len(display) == 2
        # __type__ key still present (harmless)
        assert display[0]["slug"] == "a-code-slug"

    def test_string_passthrough(self):
        assert extract_display_output("hello") == "hello"

    def test_none_passthrough(self):
        assert extract_display_output(None) is None

    def test_plain_dict_passthrough(self):
        data = {"matches": [{"quote_hash": "abc", "quote": "text"}]}
        display = extract_display_output(data)
        assert display == data
