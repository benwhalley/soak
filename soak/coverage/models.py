"""Data models for theme coverage analysis."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentCoverage(BaseModel):
    """Coverage scores for a single document."""

    document_id: str
    filename: str
    group: Optional[str] = None
    theme_scores: Dict[str, float]  # theme_name -> aggregated score
    n_chunks: int


class GroupCoverage(BaseModel):
    """Aggregated coverage statistics for a document group."""

    group_name: str
    n_documents: int
    theme_scores: Dict[str, float]  # mean coverage per theme
    theme_scores_std: Dict[str, float]  # standard deviation per theme


class ChunkExample(BaseModel):
    """A chunk example with its similarity score to a theme."""

    chunk_text: str
    document_id: str
    filename: str
    similarity: float
    chunk_index: int = 0  # index into chunk_similarity_matrix for recalibration


class ChunkInfo(BaseModel):
    """Metadata for a chunk in the chunk-level heatmap."""

    document_id: str
    filename: str
    chunk_index: int  # position within document


class ThemeCoverageResult(BaseModel):
    """Complete result of theme coverage analysis."""

    analysis_name: str
    themes: List[Dict[str, str]]  # [{name, description}]
    documents: List[DocumentCoverage]
    coverage_matrix: List[List[float]]  # [n_docs x n_themes]
    document_ids: List[str]
    theme_names: List[str]
    groups: Optional[List[GroupCoverage]] = None
    top_chunks_per_theme: Dict[str, List[ChunkExample]] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)

    # chunk-level data for detailed heatmap
    chunk_similarity_matrix: Optional[List[List[float]]] = None  # [n_chunks x n_themes]
    chunk_info: Optional[List[ChunkInfo]] = None  # metadata for each chunk row

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert coverage results to a pandas DataFrame.

        Returns:
            DataFrame with documents as rows and themes as columns,
            plus metadata columns (document_id, filename, group, n_chunks).
        """
        import pandas as pd

        rows = []
        for doc in self.documents:
            row = {
                "document_id": doc.document_id,
                "filename": doc.filename,
                "n_chunks": doc.n_chunks,
            }
            if doc.group:
                row["group"] = doc.group

            # add theme scores
            for theme_name, score in doc.theme_scores.items():
                row[theme_name] = score

            rows.append(row)

        df = pd.DataFrame(rows)

        # reorder columns: metadata first, then themes
        meta_cols = ["document_id", "filename"]
        if "group" in df.columns:
            meta_cols.append("group")
        meta_cols.append("n_chunks")

        theme_cols = [c for c in df.columns if c not in meta_cols]
        return df[meta_cols + sorted(theme_cols)]

    def to_csv(self) -> str:
        """Export coverage matrix as CSV string."""
        return self.to_dataframe().to_csv(index=False)
