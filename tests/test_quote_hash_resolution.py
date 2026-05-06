"""Unit tests for the quote/code hash refactor.

Covers the resolution mechanics that the rest of the system depends on:
- compute_code_hash / Quote.hash() determinism
- build_quote_lookup (duplicate vs collision handling)
- resolve_quote_reference (matched + unresolved fallback)
- post_process_code_quotes (extract / reference / mixed / multi-source)
- post_process_theme_code_refs (hash match + fuzzy fallback)
- Theme._migrate_code_slugs backward compatibility
- code_response_model factory (extract vs reference schemas)
- FrameworkMatrix.from_results hash join
"""

import pytest

from soak.models.base import (
    Code,
    CodeList,
    Quote,
    QuoteProvenanceError,
    QuoteReference,
    Theme,
    TrackedItem,
    code_response_model,
    compute_code_hash,
)
from soak.models.matrix import FrameworkMatrix
from soak.models.utils import (
    build_quote_lookup,
    fuzzy_match_code_slug,
    post_process_code_quotes,
    post_process_theme_code_refs,
    resolve_quote_reference,
)


# --------------------------------------------------------------------------- #
# 1.  Hash determinism + format
# --------------------------------------------------------------------------- #


class TestComputeCodeHash:
    def test_deterministic(self):
        h1 = compute_code_hash("name", "description")
        h2 = compute_code_hash("name", "description")
        assert h1 == h2

    def test_different_inputs_differ(self):
        a = compute_code_hash("name", "description")
        b = compute_code_hash("name", "different description")
        c = compute_code_hash("different name", "description")
        assert len({a, b, c}) == 3

    def test_format_is_lowercase_base32_8char(self):
        h = compute_code_hash("anything", "here")
        assert len(h) == 8
        assert h == h.lower()
        # base32 alphabet is a-z2-7 lowercased
        assert all(c in "abcdefghijklmnopqrstuvwxyz234567" for c in h)


class TestQuoteHash:
    def test_deterministic(self):
        q1 = Quote(text="hello world", source="docA")
        q2 = Quote(text="hello world", source="docA")
        assert q1.hash() == q2.hash()

    def test_source_affects_hash(self):
        a = Quote(text="hello", source="docA").hash()
        b = Quote(text="hello", source="docB").hash()
        assert a != b

    def test_format(self):
        h = Quote(text="x", source="y").hash()
        assert len(h) == 8
        assert h == h.lower()


# --------------------------------------------------------------------------- #
# 2.  build_quote_lookup -- duplicates vs collisions
# --------------------------------------------------------------------------- #


def _code_with_quotes(name: str, *quotes: Quote) -> Code:
    return Code(
        name=name, description=f"description of {name} code", quotes=list(quotes)
    )


_LONG = "long enough description"


class TestBuildQuoteLookup:
    def test_basic(self):
        c = _code_with_quotes("c1", Quote(text="hello", source="d1"))
        lookup = build_quote_lookup([c])
        assert len(lookup) == 1
        h = c.quotes[0].hash()
        assert lookup[h].text == "hello"

    def test_duplicate_quote_in_two_codes_does_not_raise(self):
        """Same text+source quote referenced from multiple codes is fine."""
        q = Quote(text="shared", source="d1")
        c1 = _code_with_quotes("c1", q)
        c2 = _code_with_quotes("c2", Quote(text="shared", source="d1"))
        lookup = build_quote_lookup([c1, c2])
        # only one entry, no exception
        assert len(lookup) == 1

    def test_hash_collision_raises(self, monkeypatch):
        """Different text/source with the same hash -> ValueError."""
        from soak.models import utils as utils_mod

        # force collisions by stubbing Quote.hash
        monkeypatch.setattr(Quote, "hash", lambda self: "samehash")
        c1 = _code_with_quotes("c1", Quote(text="A", source="d1"))
        c2 = _code_with_quotes("c2", Quote(text="B", source="d2"))
        with pytest.raises(ValueError, match="Hash collision"):
            build_quote_lookup([c1, c2])


# --------------------------------------------------------------------------- #
# 3.  resolve_quote_reference -- matched + unresolved fallback
# --------------------------------------------------------------------------- #


class TestResolveQuoteReference:
    def test_matched(self):
        q = Quote(text="hi", source="docA")
        lookup = {q.hash(): q}
        ref = QuoteReference(hash=q.hash())
        assert resolve_quote_reference(ref, lookup).text == "hi"

    def test_unresolved_returns_placeholder_quote(self):
        ref = QuoteReference(hash="zzzzzzzz")
        result = resolve_quote_reference(ref, {})
        assert isinstance(result, Quote)
        assert "unresolved" in result.text.lower()
        assert "zzzzzzzz" in result.text


# --------------------------------------------------------------------------- #
# 4.  post_process_code_quotes
# --------------------------------------------------------------------------- #


class TestPostProcessCodeQuotesExtractMode:
    def test_assigns_source_from_tracked_item(self):
        ti = TrackedItem(content="text", id="docA__chunks__0", sources=["docA"])
        code = Code(
            name="extracted",
            description=_LONG,
            quotes=[Quote(text="something", source="")],
        )
        post_process_code_quotes(code, {"input": ti})
        assert code.quotes[0].source == "docA__chunks__0"
        # resolved_quotes mirrors quotes
        assert code.resolved_quotes[0]["text"] == "something"

    def test_falls_back_to_source_id_in_context(self):
        code = Code(
            name="extracted",
            description=_LONG,
            quotes=[Quote(text="x", source="")],
        )
        post_process_code_quotes(code, {"source_id": "manual_id"})
        assert code.quotes[0].source == "manual_id"

    def test_raises_on_multi_source_tracked_item(self):
        ti = TrackedItem(
            content="combined", id="reduced", sources=["docA", "docB"]
        )
        code = Code(
            name="extracted",
            description=_LONG,
            quotes=[Quote(text="x", source="")],
        )
        with pytest.raises(QuoteProvenanceError):
            post_process_code_quotes(code, {"input": ti})


class TestPostProcessCodeQuotesReferenceMode:
    def _input_codes_context(self):
        """Build a context with input codes containing real quotes."""
        original_quote = Quote(text="original quote text", source="docA")
        upstream_code = Code(
            name="upstream",
            description=_LONG,
            quotes=[original_quote],
        )
        return original_quote, {"codes": CodeList(codes=[upstream_code])}

    def test_resolves_reference_to_original_quote(self):
        original, ctx = self._input_codes_context()

        consolidated = Code(
            name="consolidated",
            description=_LONG,
            quotes=[QuoteReference(hash=original.hash())],
        )
        post_process_code_quotes(consolidated, ctx)

        assert len(consolidated.quotes) == 1
        assert isinstance(consolidated.quotes[0], Quote)
        assert consolidated.quotes[0].text == "original quote text"
        assert consolidated.quotes[0].source == "docA"

    def test_unresolvable_reference_becomes_placeholder(self):
        _, ctx = self._input_codes_context()
        consolidated = Code(
            name="consolidated",
            description=_LONG,
            quotes=[QuoteReference(hash="zzzzzzzz")],
        )
        post_process_code_quotes(consolidated, ctx)
        assert "unresolved" in consolidated.quotes[0].text.lower()

    def test_mixed_quote_and_reference(self):
        original, ctx = self._input_codes_context()
        consolidated = Code(
            name="mixed",
            description=_LONG,
            quotes=[
                QuoteReference(hash=original.hash()),
                Quote(text="literal", source=""),
            ],
        )
        post_process_code_quotes(consolidated, ctx)
        assert consolidated.quotes[0].text == "original quote text"
        assert consolidated.quotes[1].text == "literal"


# --------------------------------------------------------------------------- #
# 5.  post_process_theme_code_refs
# --------------------------------------------------------------------------- #


class TestPostProcessThemeCodeRefs:
    def test_resolves_by_hash(self):
        c = Code(name="alpha", description="alpha description")
        ctx = {"codes": CodeList(codes=[c])}
        theme = Theme(
            name="A theme name long enough",
            description="A theme description",
            code_hashes=[c.hash()],
        )
        post_process_theme_code_refs(theme, ctx)
        assert theme.resolved_code_refs and len(theme.resolved_code_refs) == 1
        assert theme.resolved_code_refs[0]["name"] == "alpha"

    def test_unmatched_hash_dropped(self):
        c = Code(name="alpha", description="alpha description")
        ctx = {"codes": CodeList(codes=[c])}
        theme = Theme(
            name="A theme name long enough",
            description="A theme description",
            code_hashes=["zzzzzzzz"],
        )
        post_process_theme_code_refs(theme, ctx)
        assert theme.resolved_code_refs == []


class TestFuzzyMatchCodeSlug:
    def test_close_match(self):
        c = Code(
            name="alpha", description=_LONG, slug="alpha-code"
        )
        match = fuzzy_match_code_slug("alpha-cod", [c])
        assert match is c

    def test_no_match_below_threshold(self):
        c = Code(
            name="alpha", description=_LONG, slug="alpha-code"
        )
        assert fuzzy_match_code_slug("zeta-omega", [c]) is None


# --------------------------------------------------------------------------- #
# 6.  Theme migration: code_slugs -> code_hashes
# --------------------------------------------------------------------------- #


class TestThemeMigration:
    def test_legacy_code_slugs_dict_input(self):
        t = Theme.model_validate(
            {
                "name": "A theme name",
                "description": "A theme description",
                "code_slugs": ["abcdefgh", "ijklmnop"],
            }
        )
        assert t.code_hashes == ["abcdefgh", "ijklmnop"]
        # alias still readable
        assert t.code_slugs == ["abcdefgh", "ijklmnop"]


# --------------------------------------------------------------------------- #
# 7.  code_response_model factory: extract vs reference schemas
# --------------------------------------------------------------------------- #


class TestCodeResponseModelFactory:
    def test_extract_mode_quote_field_type(self):
        Model = code_response_model(options=None)
        schema = Model.model_json_schema()
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
        # extract mode uses Quote (with 'text'), not QuoteReference (with 'hash')
        assert "Quote" in defs
        quote_props = defs["Quote"].get("properties", {})
        assert "text" in quote_props
        assert "hash" not in quote_props

    def test_reference_mode_quote_field_type(self):
        Model = code_response_model(options=["quotes=reference"])
        schema = Model.model_json_schema()
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
        # reference mode swaps the Quote ref for QuoteReference (with a 'hash' prop)
        assert "QuoteReference" in defs
        ref_props = defs["QuoteReference"].get("properties", {})
        assert "hash" in ref_props

    def test_theme_name_option_adds_field(self):
        Model = code_response_model(options=["theme_name=true"])
        m = Model(
            name="x",
            description=_LONG,
            quotes=[],
            theme_name="My Theme",
        )
        assert m.theme_name == "My Theme"


# --------------------------------------------------------------------------- #
# 8.  FrameworkMatrix.from_results -- hash join
# --------------------------------------------------------------------------- #


class TestFrameworkMatrix:
    def _build_inputs(self):
        # one code with a quote sourced from docA
        code = {
            "name": "alpha code",
            "description": "alpha description",
            "resolved_quotes": [
                {"text": "alpha quote", "source": "docA__chunks__0"}
            ],
        }
        code_hash = compute_code_hash(code["name"], code["description"])
        theme = {
            "name": "Theme One",
            "description": "Theme description",
            "code_hashes": [code_hash],
        }
        return [theme], [code], code_hash

    def test_basic_join(self):
        themes, codes, code_hash = self._build_inputs()
        matrix = FrameworkMatrix.from_results(
            themes=themes,
            codes=codes,
            document_ids=["docA"],
            document_labels={"docA": "Document A"},
        )
        cell = matrix.cells["docA"]["Theme One"]
        assert cell.code_count == 1
        assert cell.codes[0]["code_hash"] == code_hash
        assert cell.quote_count == 1

    def test_empty_cell_when_quote_from_other_doc(self):
        themes, codes, _ = self._build_inputs()
        matrix = FrameworkMatrix.from_results(
            themes=themes,
            codes=codes,
            document_ids=["docB"],
            document_labels={"docB": "Document B"},
        )
        # docA quote shouldn't show up under docB
        assert matrix.cells["docB"]["Theme One"].is_empty

    def test_unmatched_theme_hash_yields_empty_cell(self):
        _, codes, _ = self._build_inputs()
        themes = [
            {
                "name": "Theme One",
                "description": "T",
                "code_hashes": ["zzzzzzzz"],
            }
        ]
        matrix = FrameworkMatrix.from_results(
            themes=themes,
            codes=codes,
            document_ids=["docA"],
            document_labels={"docA": "Document A"},
        )
        assert matrix.cells["docA"]["Theme One"].is_empty

    def test_legacy_code_slugs_field_is_honoured(self):
        themes, codes, code_hash = self._build_inputs()
        # rewrite theme to use legacy code_slugs
        themes = [
            {
                "name": themes[0]["name"],
                "description": themes[0]["description"],
                "code_slugs": [code_hash],
            }
        ]
        matrix = FrameworkMatrix.from_results(
            themes=themes,
            codes=codes,
            document_ids=["docA"],
            document_labels={"docA": "Document A"},
        )
        assert matrix.cells["docA"]["Theme One"].code_count == 1
