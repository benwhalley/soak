"""Framework analysis matrix models.

Constructs a case-by-theme matrix from qualitative analysis results.
Rows = documents/cases, columns = themes. Each cell contains the codes
and quotes that link a particular document to a particular theme.

Works with existing TA run results -- no pipeline changes required.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _root_document(source_id: str) -> str:
    """Extract the root document identifier from a TrackedItem source chain.

    TrackedItem IDs follow the pattern "root__node__index__node__index".
    The root document is the first segment before any "__" separator.

    Examples:
        "interview_p3__chunks__5" -> "interview_p3"
        "doc_0__chunks__2__coded" -> "doc_0"
        "transcript" -> "transcript"
    """
    return source_id.split("__")[0] if "__" in source_id else source_id


class MatrixCell(BaseModel):
    """One cell: what one document says about one theme."""

    codes: List[Dict] = Field(default_factory=list)
    quotes: List[Dict] = Field(default_factory=list)
    summary: Optional[str] = None

    @property
    def code_count(self) -> int:
        return len(self.codes)

    @property
    def quote_count(self) -> int:
        return len(self.quotes)

    @property
    def is_empty(self) -> bool:
        return len(self.codes) == 0


class FrameworkMatrix(BaseModel):
    """The framework analysis matrix. Rows = documents, columns = themes.

    Constructed deterministically from codes, themes, and document metadata.
    Not stored in the database -- computed on-the-fly and cached.
    """

    document_ids: List[str]
    document_labels: Dict[str, str]  # root_id -> display name
    themes: List[Dict]  # theme dicts with name, description, code_hashes
    cells: Dict[str, Dict[str, MatrixCell]]  # root_id -> theme_name -> cell

    @classmethod
    def from_results(
        cls,
        themes: List[Dict],
        codes: List[Dict],
        document_ids: List[str],
        document_labels: Dict[str, str],
    ) -> "FrameworkMatrix":
        """Construct matrix from analysis results by tracing quote provenance.

        Args:
            themes: list of theme dicts, each with 'name' and 'code_hashes'
            codes: list of code dicts, each with 'name', 'description',
                   and 'resolved_quotes' (containing 'source' fields)
            document_ids: ordered list of root document identifiers
            document_labels: mapping from root_id to display name
        """
        # pre-index: code_hash -> code dict
        from soak.models.base import compute_code_hash

        code_by_hash = {}
        for code in codes:
            name = code.get("name", "")
            description = code.get("description", "")
            if name and description:
                h = compute_code_hash(name, description)
                code_by_hash[h] = code
            # also index by slug for backward compatibility
            slug = code.get("slug")
            if slug:
                code_by_hash[slug] = code

        # pre-index: root_doc -> code_hash -> list of quotes from that doc
        doc_code_quotes: Dict[str, Dict[str, List[Dict]]] = {
            doc_id: {} for doc_id in document_ids
        }

        for code in codes:
            name = code.get("name", "")
            description = code.get("description", "")
            if not name:
                continue
            code_hash = compute_code_hash(name, description) if description else name

            # look at resolved_quotes for source provenance
            for quote in code.get("resolved_quotes") or []:
                source = ""
                if isinstance(quote, dict):
                    source = quote.get("source", "")
                if not source:
                    continue
                root = _root_document(source)
                if root in doc_code_quotes:
                    doc_code_quotes[root].setdefault(code_hash, []).append(quote)

        # build cells
        cells: Dict[str, Dict[str, MatrixCell]] = {}
        for doc_id in document_ids:
            cells[doc_id] = {}
            for theme in themes:
                theme_name = theme.get("name", "")
                theme_hashes = set(
                    theme.get("code_hashes", theme.get("code_slugs", []))
                )

                cell_codes = []
                cell_quotes = []

                for ch in theme_hashes:
                    code = code_by_hash.get(ch)
                    if not code:
                        continue
                    # check if this code has quotes from this document
                    code_hash = compute_code_hash(
                        code.get("name", ""), code.get("description", "")
                    )
                    doc_quotes = doc_code_quotes.get(doc_id, {}).get(code_hash, [])
                    if doc_quotes:
                        cell_codes.append(
                            {
                                "name": code.get("name", ""),
                                "description": code.get("description", ""),
                                "code_hash": code_hash,
                            }
                        )
                        cell_quotes.extend(doc_quotes)

                cells[doc_id][theme_name] = MatrixCell(
                    codes=cell_codes,
                    quotes=cell_quotes,
                )

        return cls(
            document_ids=document_ids,
            document_labels=document_labels,
            themes=themes,
            cells=cells,
        )

    @property
    def theme_names(self) -> List[str]:
        """Ordered list of theme names (column headers)."""
        return [t.get("name", "") for t in self.themes]

    @property
    def total_codes(self) -> int:
        """Total number of code appearances across all cells."""
        return sum(
            cell.code_count
            for row in self.cells.values()
            for cell in row.values()
        )

    @property
    def total_quotes(self) -> int:
        """Total number of quotes across all cells."""
        return sum(
            cell.quote_count
            for row in self.cells.values()
            for cell in row.values()
        )

    def cell_stats(self) -> Dict:
        """Compute statistics for density colouring.

        Returns dict with median, p80 for code and quote counts.
        """
        code_counts = []
        quote_counts = []
        for row in self.cells.values():
            for cell in row.values():
                code_counts.append(cell.code_count)
                quote_counts.append(cell.quote_count)

        code_counts.sort()
        quote_counts.sort()

        def percentile(data: List[int], p: float) -> float:
            if not data:
                return 0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        return {
            "code_median": percentile(code_counts, 0.5),
            "code_p80": percentile(code_counts, 0.8),
            "quote_median": percentile(quote_counts, 0.5),
            "quote_p80": percentile(quote_counts, 0.8),
        }

    def row_totals(self) -> Dict[str, Dict[str, int]]:
        """Total codes and quotes per document (row totals)."""
        totals = {}
        for doc_id in self.document_ids:
            row = self.cells.get(doc_id, {})
            totals[doc_id] = {
                "codes": sum(c.code_count for c in row.values()),
                "quotes": sum(c.quote_count for c in row.values()),
            }
        return totals

    def column_totals(self) -> Dict[str, Dict[str, int]]:
        """Total codes and quotes per theme (column totals)."""
        totals = {}
        for theme in self.themes:
            name = theme.get("name", "")
            totals[name] = {"codes": 0, "quotes": 0}
            for doc_id in self.document_ids:
                cell = self.cells.get(doc_id, {}).get(name)
                if cell:
                    totals[name]["codes"] += cell.code_count
                    totals[name]["quotes"] += cell.quote_count
        return totals
