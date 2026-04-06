"""Integration tests for serialization with pipeline execution and restart."""

import pytest
from struckdown.results import StruckdownResult, SlotResult

from soak.models.base import Code, CodeList, Theme, Themes, TrackedItem
from soak.models.nodes.base import _TemplateProxy, _collect_codes_from_dag, _prepare_for_template
from soak.serialization import (
    deserialize_node_output,
    deserialize_value,
    extract_display_output,
    serialize_node_output,
    serialize_value,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _make_code(slug: str = "test-code-slug", **kw) -> Code:
    return Code(
        slug=slug,
        name=kw.get("name", f"Code {slug}"),
        description=kw.get("description", "A test code."),
        quotes=kw.get("quotes", []),
    )


def _make_theme(name: str = "A test theme name here", **kw) -> Theme:
    return Theme(
        name=name,
        description=kw.get("description", "Describes the theme."),
        code_slugs=kw.get("code_slugs", ["test-code-slug"]),
    )


def _make_complete(slots: dict) -> StruckdownResult:
    results = {}
    for name, output in slots.items():
        results[name] = SlotResult(
            name=name,
            prompt="test prompt",
            output=output,
            action="code" if "code" in name else "respond",
        )
    return StruckdownResult(results=results)


class FakeNode:
    """Minimal node-like object for serialize_node_output."""

    def __init__(self, node_type: str, output):
        self.type = node_type
        self.output = output
        self.name = "test_node"


class FakeDAG:
    """Minimal DAG-like object for _collect_codes_from_dag."""

    def __init__(self, nodes: dict):
        self.nodes_dict = nodes


# --------------------------------------------------------------------------- #
#  Node-level serialize/deserialize round-trips
# --------------------------------------------------------------------------- #


class TestSerializeNodeOutputByType:
    def test_split_preserves_tracked_items(self):
        """Split output should serialize and round-trip TrackedItems."""
        items = [
            TrackedItem(content="chunk 1", id="doc__s__0", sources=["doc"]),
            TrackedItem(content="chunk 2", id="doc__s__1", sources=["doc"]),
        ]
        node = FakeNode("Split", items)
        serialized = serialize_node_output(node)

        deserialized = deserialize_node_output(serialized, "Split")
        assert len(deserialized) == 2
        assert all(isinstance(i, TrackedItem) for i in deserialized)
        assert deserialized[0].content == "chunk 1"
        assert deserialized[1].id == "doc__s__1"

    def test_transform_single_complete(self):
        """Transform output: single StruckdownResult unwrapped from list."""
        codes = [_make_code("code-alpha-one"), _make_code("code-beta-two")]
        cr = _make_complete({"codes": codes})
        node = FakeNode("Transform", [cr])

        serialized = serialize_node_output(node)
        assert serialized["__type__"] == "StruckdownResult"

        deserialized = deserialize_node_output(serialized, "Transform")
        assert isinstance(deserialized, StruckdownResult)
        output_codes = deserialized.results["codes"].output
        assert len(output_codes) == 2
        assert isinstance(output_codes[0], Code)
        assert output_codes[0].slug == "code-alpha-one"

    def test_map_single_slot_flattened(self):
        """Map output: single-slot StruckdownResults are flattened."""
        items = [
            _make_complete({"code": _make_code("code-map-one")}),
            _make_complete({"code": _make_code("code-map-two")}),
        ]
        node = FakeNode("Map", items)

        serialized = serialize_node_output(node)
        assert isinstance(serialized, list)
        assert len(serialized) == 2
        # single-slot is flattened -- output is the Code dict, not StruckdownResult
        assert serialized[0].get("__type__") == "Code"

    def test_map_multi_slot_preserved(self):
        """Map output: multi-slot StruckdownResults are preserved as StruckdownResult."""
        items = [
            _make_complete({"codes": [_make_code()], "summary": "Some text"}),
        ]
        node = FakeNode("Map", items)

        serialized = serialize_node_output(node)
        assert isinstance(serialized, list)
        assert serialized[0]["__type__"] == "StruckdownResult"

    def test_reduce_string(self):
        """Reduce output: plain string passes through."""
        node = FakeNode("Reduce", "combined text output")
        serialized = serialize_node_output(node)
        assert serialized == "combined text output"

        deserialized = deserialize_node_output(serialized, "Reduce")
        assert deserialized == "combined text output"

    def test_reduce_tracked_item(self):
        """Reduce output: TrackedItem is serialized and round-tripped."""
        item = TrackedItem(
            content="combined", id="all_codes", sources=["doc1", "doc2"]
        )
        node = FakeNode("Reduce", item)

        serialized = serialize_node_output(node)
        deserialized = deserialize_node_output(serialized, "Reduce")
        assert isinstance(deserialized, TrackedItem)
        assert deserialized.id == "all_codes"
        assert deserialized.sources == ["doc1", "doc2"]

    def test_reduce_code_list(self):
        """Reduce with items_field=codes returns list of Code objects."""
        codes = [_make_code("reduce-code-a"), _make_code("reduce-code-b")]
        node = FakeNode("Reduce", codes)

        serialized = serialize_node_output(node)
        deserialized = deserialize_node_output(serialized, "Reduce")
        assert len(deserialized) == 2
        assert all(isinstance(c, Code) for c in deserialized)

    def test_verify_dict_passthrough(self):
        """Verify/checkquotes output: plain dict passes through."""
        output = {
            "matches": [{"quote_hash": "abc", "quote": "text", "bm25_score": 5.0}],
            "doc_boundaries": [{"name": "doc1", "start": 0, "end": 100}],
        }
        node = FakeNode("VerifyQuotes", output)

        serialized = serialize_node_output(node)
        deserialized = deserialize_node_output(serialized, "VerifyQuotes")
        assert isinstance(deserialized, dict)
        assert deserialized["matches"][0]["quote_hash"] == "abc"

    def test_none_output(self):
        """None output stays None."""
        node = FakeNode("Transform", None)
        assert serialize_node_output(node) is None
        assert deserialize_node_output(None, "Transform") is None


# --------------------------------------------------------------------------- #
#  Template proxy with deserialized data
# --------------------------------------------------------------------------- #


class TestTemplateProxyWithDeserializedData:
    def test_proxy_accesses_typed_code_list(self):
        """_TemplateProxy wraps list of Codes in CodeList for __str__."""
        codes = [_make_code("proxy-code-a"), _make_code("proxy-code-b")]
        cr = _make_complete({"codes": codes})
        proxy = _TemplateProxy(cr)

        result = proxy.codes
        assert isinstance(result, CodeList)
        assert len(result) == 2
        assert result.codes[0].slug == "proxy-code-a"

    def test_proxy_accesses_typed_themes(self):
        """_TemplateProxy wraps list of Themes in Themes container."""
        themes = [_make_theme("Theme Alpha Description")]
        cr = _make_complete({"themes": themes})
        proxy = _TemplateProxy(cr)

        result = proxy.themes
        assert isinstance(result, Themes)
        assert len(result) == 1

    def test_proxy_resolves_theme_codes(self):
        """_TemplateProxy resolves theme code_slugs against collected codes."""
        code = _make_code("theme-linked-code")
        theme = _make_theme(code_slugs=["theme-linked-code"])
        cr = _make_complete({"themes": [theme]})
        proxy = _TemplateProxy(cr, all_codes=[code])

        result = proxy.themes
        resolved = result.themes[0].resolved_code_refs
        assert resolved is not None
        assert len(resolved) == 1
        assert resolved[0]["slug"] == "theme-linked-code"

    def test_proxy_str_returns_complete_str(self):
        """_TemplateProxy __str__ delegates to StruckdownResult.__str__."""
        cr = _make_complete({"text": "hello world"})
        proxy = _TemplateProxy(cr)
        assert "hello world" in str(proxy)

    def test_proxy_missing_slot_raises(self):
        """Accessing non-existent slot raises AttributeError."""
        cr = _make_complete({"text": "hello"})
        proxy = _TemplateProxy(cr)
        with pytest.raises(AttributeError):
            _ = proxy.nonexistent


class TestPrepareForTemplate:
    def test_transform_unwraps_single_list(self):
        """Transform output: single-element list is unwrapped to one proxy."""
        cr = _make_complete({"text": "hello"})
        result = _prepare_for_template([cr], source_node_type="Transform")
        assert isinstance(result, _TemplateProxy)

    def test_map_returns_list_of_proxies(self):
        """Map output: list of StruckdownResults becomes list of proxies."""
        items = [_make_complete({"code": _make_code()}) for _ in range(3)]
        result = _prepare_for_template(items, source_node_type="Map")
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(p, _TemplateProxy) for p in result)

    def test_non_complete_passthrough(self):
        """Non-StruckdownResult output passes through unchanged."""
        items = [TrackedItem(content="text", id="doc", sources=["doc"])]
        result = _prepare_for_template(items, source_node_type="Split")
        assert result is items


class TestCollectCodesFromDag:
    def test_collects_codes_from_complete_results(self):
        """_collect_codes_from_dag finds Codes inside StruckdownResult slots."""
        code = _make_code("dag-code-alpha")
        cr = _make_complete({"codes": [code]})
        node = FakeNode("Transform", cr)

        dag = FakeDAG({"transform": node})
        codes = _collect_codes_from_dag(dag)
        assert len(codes) == 1
        assert codes[0].slug == "dag-code-alpha"

    def test_collects_codes_from_list_output(self):
        """_collect_codes_from_dag finds Codes in Reduce output (raw list)."""
        codes = [_make_code("reduce-a"), _make_code("reduce-b")]
        node = FakeNode("Reduce", codes)

        dag = FakeDAG({"reduce": node})
        found = _collect_codes_from_dag(dag)
        assert len(found) == 2

    def test_skips_none_output(self):
        """_collect_codes_from_dag skips nodes with None output."""
        node = FakeNode("Transform", None)
        dag = FakeDAG({"transform": node})
        codes = _collect_codes_from_dag(dag)
        assert codes == []


# --------------------------------------------------------------------------- #
#  Full serialize → display extraction pipeline
# --------------------------------------------------------------------------- #


class TestSerializeToDisplayPipeline:
    def test_transform_codes_display(self):
        """Transform(codes) -> serialize -> extract_display -> list of code dicts."""
        codes = [_make_code("display-code-a"), _make_code("display-code-b")]
        cr = _make_complete({"codes": codes})
        node = FakeNode("Transform", [cr])

        serialized = serialize_node_output(node)
        display = extract_display_output(serialized)

        # single-slot StruckdownResult extracts to the output list
        assert isinstance(display, list)
        assert len(display) == 2
        assert display[0]["slug"] == "display-code-a"

    def test_transform_multi_slot_display(self):
        """Transform(codes+themes) -> serialize -> extract_display -> {slot: output}."""
        cr = _make_complete({
            "codes": [_make_code()],
            "themes": [_make_theme()],
        })
        node = FakeNode("Transform", [cr])

        serialized = serialize_node_output(node)
        display = extract_display_output(serialized)

        assert isinstance(display, dict)
        assert "codes" in display
        assert "themes" in display

    def test_map_display(self):
        """Map(codes) -> serialize -> extract_display -> list of code dicts."""
        items = [
            _make_complete({"code": _make_code("map-alpha-a")}),
            _make_complete({"code": _make_code("map-beta-bb")}),
        ]
        node = FakeNode("Map", items)

        serialized = serialize_node_output(node)
        display = extract_display_output(serialized)

        assert isinstance(display, list)
        assert len(display) == 2
        assert display[0]["slug"] == "map-alpha-a"

    def test_split_display(self):
        """Split -> serialize -> extract_display -> list of TrackedItem dicts."""
        items = [
            TrackedItem(content="chunk 1", id="doc__0", sources=["doc"]),
            TrackedItem(content="chunk 2", id="doc__1", sources=["doc"]),
        ]
        node = FakeNode("Split", items)

        serialized = serialize_node_output(node)
        display = extract_display_output(serialized)

        assert isinstance(display, list)
        assert len(display) == 2
        assert display[0]["id"] == "doc__0"
        assert display[0]["content"] == "chunk 1"

    def test_verify_display_passthrough(self):
        """Verify output: dict passes through display extraction unchanged."""
        output = {"matches": [{"quote_hash": "abc"}]}
        node = FakeNode("VerifyQuotes", output)

        serialized = serialize_node_output(node)
        display = extract_display_output(serialized)
        assert display["matches"][0]["quote_hash"] == "abc"


# --------------------------------------------------------------------------- #
#  VerifyQuotes node input normalization (restart scenarios)
# --------------------------------------------------------------------------- #


class TestVerifyNormalization:
    """Test that VerifyQuotes._normalize_to_outputs_list handles all input formats."""

    def _get_normalizer(self):
        """Get the normalization method without constructing a full VerifyQuotes node."""
        from soak.models.nodes.verify import VerifyQuotes

        # create minimal instance -- we only need the method
        node = VerifyQuotes.__new__(VerifyQuotes)
        return node._normalize_to_outputs_list

    def test_list_of_code_objects(self):
        """Direct list of Code objects is wrapped into CodeList."""
        normalize = self._get_normalizer()
        codes = [_make_code("verify-code-a"), _make_code("verify-code-b")]
        result = normalize(codes)
        assert len(result) == 1
        assert isinstance(result[0], CodeList)
        assert len(result[0].codes) == 2

    def test_legacy_code_dicts(self):
        """Legacy serialized code dicts (no __type__) are reconstructed."""
        normalize = self._get_normalizer()
        dicts = [
            {
                "slug": "legacy-code-slug",
                "name": "Legacy Code",
                "description": "A legacy code.",
                "quotes": [],
            },
        ]
        result = normalize(dicts)
        assert len(result) == 1
        assert isinstance(result[0], CodeList)
        assert result[0].codes[0].slug == "legacy-code-slug"

    def test_legacy_theme_dicts(self):
        """Legacy serialized theme dicts (no __type__) are reconstructed."""
        normalize = self._get_normalizer()
        dicts = [
            {
                "name": "Legacy Theme Name Here",
                "description": "A legacy theme.",
                "code_slugs": ["some-code-ref"],
            },
        ]
        result = normalize(dicts)
        assert len(result) == 1
        assert isinstance(result[0], Themes)

    def test_complete_result_with_codes(self):
        """StruckdownResult containing Code list is extracted."""
        normalize = self._get_normalizer()
        codes = [_make_code("cr-code-aaa")]
        cr = _make_complete({"codes": codes})
        result = normalize(cr)
        assert len(result) == 1
        assert isinstance(result[0], CodeList)

    def test_template_proxy_with_codes(self):
        """_TemplateProxy wrapping StruckdownResult with codes is extracted."""
        normalize = self._get_normalizer()
        codes = [_make_code("proxy-code-a")]
        cr = _make_complete({"codes": codes})
        proxy = _TemplateProxy(cr)
        result = normalize(proxy)
        assert len(result) == 1
        assert isinstance(result[0], CodeList)

    def test_empty_input_returns_empty(self):
        """Empty input returns empty list (no crash)."""
        normalize = self._get_normalizer()
        result = normalize([])
        assert result == []
