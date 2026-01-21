"""Core theme coverage analysis logic."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from soak.models.base import QualitativeAnalysis, TrackedItem, get_embedding
from soak.models.nodes.base import Split

from .models import ChunkExample, ChunkInfo, DocumentCoverage, GroupCoverage, ThemeCoverageResult

logger = logging.getLogger(__name__)


class ThemeCoverageAnalyzer:
    """Analyzes how well themes from a qualitative analysis are represented across documents."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        split_unit: Literal["words", "tokens", "chars", "sentences", "paragraphs"] = "words",
        aggregation: Literal["max", "mean", "p95"] = "max",
        embedding_template: str = "{name}: {description}",
        embedding_model: str = "text-embedding-3-large",
        sample_frac: Optional[float] = None,
        sample_n: Optional[int] = None,
        embed_source: Literal["quotes", "themes", "both"] = "quotes",
    ):
        """Initialize the coverage analyzer.

        Args:
            chunk_size: Size of document chunks (in split_unit units)
            overlap: Overlap between chunks (in split_unit units)
            split_unit: Unit for chunking ("words", "tokens", or "chars")
            aggregation: Method to aggregate chunk scores to document ("max", "mean", "p95")
            embedding_template: Template for formatting theme text before embedding
            embedding_model: Model for embeddings (use "local/model-name" for sentence-transformers)
            sample_frac: Sample fraction of chunks (e.g., 0.1 for 10%)
            sample_n: Sample fixed number of chunks
            embed_source: What to embed for themes: "quotes" (default), "themes", or "both"
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_unit = split_unit
        self.aggregation = aggregation
        self.embedding_template = embedding_template
        self.embedding_model = embedding_model
        self.sample_frac = sample_frac
        self.sample_n = sample_n
        self.embed_source = embed_source

    def analyze(
        self,
        analysis: QualitativeAnalysis,
        documents: List[TrackedItem],
        groups: Optional[Dict[str, str]] = None,
    ) -> ThemeCoverageResult:
        """Analyze theme coverage across documents.

        Args:
            analysis: QualitativeAnalysis containing themes
            documents: List of TrackedItem documents
            groups: Optional mapping of filename -> group name

        Returns:
            ThemeCoverageResult with coverage matrix and statistics
        """
        if not analysis.themes:
            raise ValueError("Analysis has no themes to analyze coverage for")

        if not documents:
            raise ValueError("No documents provided for coverage analysis")

        # extract theme info
        theme_names = [t.name for t in analysis.themes]

        logger.info(f"Analyzing coverage for {len(theme_names)} themes across {len(documents)} documents")
        logger.info(f"Embed source: {self.embed_source}")

        # split documents into chunks
        all_chunks, chunk_doc_mapping = self._split_documents(documents)
        total_chunks = len(all_chunks)
        logger.info(f"Split into {total_chunks} chunks")

        # apply sampling if requested
        all_chunks, chunk_doc_mapping = self._sample_chunks(all_chunks, chunk_doc_mapping)
        if len(all_chunks) < total_chunks:
            logger.info(f"Sampled {len(all_chunks)} chunks ({100*len(all_chunks)/total_chunks:.1f}%)")

        # embed chunks
        logger.info("Embedding chunks...")
        chunk_texts = [c.content if isinstance(c, TrackedItem) else c for c in all_chunks]
        # debug: verify chunks have different content
        if len(chunk_texts) >= 3:
            logger.debug(f"Chunk text lengths: {[len(t) for t in chunk_texts[:10]]}...")
            logger.debug(f"First chunk preview: {chunk_texts[0][:100]}...")
            logger.debug(f"Second chunk preview: {chunk_texts[1][:100]}...")
        chunk_embeddings = self._embed_texts(chunk_texts)

        # compute similarity matrix based on embed_source
        chunk_theme_sim = self._compute_theme_similarity(
            analysis, chunk_embeddings, theme_names
        )

        # debug: verify chunk-level matrix has variation
        logger.debug(f"chunk_theme_sim shape: {chunk_theme_sim.shape}")
        logger.debug(f"chunk_theme_sim min: {chunk_theme_sim.min():.3f}, max: {chunk_theme_sim.max():.3f}")
        logger.debug(f"chunk_theme_sim std per theme: {chunk_theme_sim.std(axis=0)}")
        # show first 5 chunks' scores for first theme to verify variation
        if chunk_theme_sim.shape[0] >= 5:
            logger.debug(f"First 5 chunks, theme 0: {chunk_theme_sim[:5, 0]}")

        # aggregate to document level
        logger.info(f"Aggregating to document level using {self.aggregation}...")
        doc_coverages, coverage_matrix = self._aggregate_to_documents(
            all_chunks,
            chunk_doc_mapping,
            chunk_theme_sim,
            theme_names,
            documents,
            groups,
        )

        # compute group statistics if groups provided
        group_coverages = None
        if groups:
            group_coverages = self._compute_group_statistics(doc_coverages, theme_names)

        # extract top chunks per theme
        top_chunks_per_theme = self._extract_top_chunks(
            all_chunks,
            chunk_doc_mapping,
            chunk_theme_sim,
            theme_names,
            documents,
            n_top=50,
        )

        # count quotes if using quotes mode
        n_quotes = 0
        if self.embed_source in ("quotes", "both"):
            for theme in analysis.themes:
                for code in theme.resolved_codes:
                    n_quotes += len(code.all_quotes)

        # build chunk info for chunk-level heatmap
        # create doc_id -> filename mapping
        doc_filenames = {}
        for doc in documents:
            doc_id = doc.root_document if isinstance(doc, TrackedItem) else doc.id
            filename = doc.metadata.get("filename", doc_id) if isinstance(doc, TrackedItem) else doc_id
            doc_filenames[doc_id] = filename

        chunk_info = []
        for chunk_idx, chunk in enumerate(all_chunks):
            doc_id = chunk_doc_mapping[chunk_idx]
            chunk_index = chunk.metadata.get("split_index", 0) if isinstance(chunk, TrackedItem) else 0
            chunk_info.append(
                ChunkInfo(
                    document_id=doc_id,
                    filename=doc_filenames.get(doc_id, doc_id),
                    chunk_index=chunk_index,
                )
            )

        return ThemeCoverageResult(
            analysis_name=analysis.name or "unnamed",
            themes=[{"name": t.name, "description": t.description} for t in analysis.themes],
            documents=doc_coverages,
            coverage_matrix=coverage_matrix,
            document_ids=[d.document_id for d in doc_coverages],
            theme_names=theme_names,
            groups=group_coverages,
            top_chunks_per_theme=top_chunks_per_theme,
            chunk_similarity_matrix=chunk_theme_sim.tolist(),
            chunk_info=chunk_info,
            config={
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "split_unit": self.split_unit,
                "aggregation": self.aggregation,
                "embedding_template": self.embedding_template,
                "embedding_model": self.embedding_model,
                "embed_source": self.embed_source,
                "sample_frac": self.sample_frac,
                "sample_n": self.sample_n,
                "n_documents": len(documents),
                "n_chunks_total": total_chunks,
                "n_chunks_sampled": len(all_chunks),
                "n_chunks": len(all_chunks),
                "n_themes": len(theme_names),
                "n_quotes": n_quotes,
            },
        )

    def _compute_theme_similarity(
        self,
        analysis: QualitativeAnalysis,
        chunk_embeddings: np.ndarray,
        theme_names: List[str],
    ) -> np.ndarray:
        """Compute chunk-theme similarity matrix based on embed_source setting.

        Returns:
            np.ndarray of shape [n_chunks x n_themes]
        """
        n_chunks = len(chunk_embeddings)
        n_themes = len(theme_names)

        if self.embed_source == "themes":
            # original behavior: embed theme name + description
            logger.info("Embedding theme descriptions...")
            theme_texts = [
                self.embedding_template.format(name=t.name, description=t.description)
                for t in analysis.themes
            ]
            theme_embeddings = self._embed_texts(theme_texts)
            return cosine_similarity(chunk_embeddings, theme_embeddings)

        elif self.embed_source == "quotes":
            # embed quotes for each theme, take max similarity across quotes
            return self._compute_quote_similarity(analysis, chunk_embeddings, theme_names)

        elif self.embed_source == "both":
            # compute both and take element-wise max
            logger.info("Computing similarity using both themes and quotes...")

            # theme-based similarity
            theme_texts = [
                self.embedding_template.format(name=t.name, description=t.description)
                for t in analysis.themes
            ]
            theme_embeddings = self._embed_texts(theme_texts)
            theme_sim = cosine_similarity(chunk_embeddings, theme_embeddings)

            # quote-based similarity
            quote_sim = self._compute_quote_similarity(analysis, chunk_embeddings, theme_names)

            # element-wise max
            return np.maximum(theme_sim, quote_sim)

        raise ValueError(f"Unknown embed_source: {self.embed_source}")

    def _compute_quote_similarity(
        self,
        analysis: QualitativeAnalysis,
        chunk_embeddings: np.ndarray,
        theme_names: List[str],
    ) -> np.ndarray:
        """Compute chunk-theme similarity using quotes.

        For each theme, embeds all quotes from its codes, computes similarity
        between chunks and quotes, then takes the max across quotes.

        Returns:
            np.ndarray of shape [n_chunks x n_themes]
        """
        n_chunks = len(chunk_embeddings)
        n_themes = len(theme_names)

        # collect all quotes per theme
        theme_quotes: Dict[int, List[str]] = {}  # theme_idx -> list of quote texts
        total_quotes = 0

        for theme_idx, theme in enumerate(analysis.themes):
            quotes = []
            for code in theme.resolved_codes:
                for quote in code.all_quotes:
                    if quote.text.strip():
                        quotes.append(quote.text)
            theme_quotes[theme_idx] = quotes
            total_quotes += len(quotes)

        if total_quotes == 0:
            logger.warning("No quotes found in analysis. Falling back to theme descriptions.")
            theme_texts = [
                self.embedding_template.format(name=t.name, description=t.description)
                for t in analysis.themes
            ]
            theme_embeddings = self._embed_texts(theme_texts)
            return cosine_similarity(chunk_embeddings, theme_embeddings)

        logger.info(f"Embedding {total_quotes} quotes across {n_themes} themes...")

        # embed all quotes at once for efficiency
        all_quotes = []
        quote_to_theme: List[int] = []  # maps quote index to theme index

        for theme_idx in range(n_themes):
            for quote_text in theme_quotes.get(theme_idx, []):
                all_quotes.append(quote_text)
                quote_to_theme.append(theme_idx)

        if not all_quotes:
            # no quotes at all, fall back to theme descriptions
            theme_texts = [
                self.embedding_template.format(name=t.name, description=t.description)
                for t in analysis.themes
            ]
            theme_embeddings = self._embed_texts(theme_texts)
            return cosine_similarity(chunk_embeddings, theme_embeddings)

        quote_embeddings = self._embed_texts(all_quotes)

        # compute chunk-quote similarity [n_chunks x n_quotes]
        chunk_quote_sim = cosine_similarity(chunk_embeddings, quote_embeddings)

        # aggregate to chunk-theme: for each theme, take max across its quotes
        chunk_theme_sim = np.zeros((n_chunks, n_themes))

        for theme_idx in range(n_themes):
            # find quote indices for this theme
            quote_indices = [i for i, t in enumerate(quote_to_theme) if t == theme_idx]

            if quote_indices:
                # max similarity across quotes for this theme
                theme_quote_sims = chunk_quote_sim[:, quote_indices]
                chunk_theme_sim[:, theme_idx] = np.max(theme_quote_sims, axis=1)
            else:
                # no quotes for this theme, use theme description as fallback
                theme_text = self.embedding_template.format(
                    name=analysis.themes[theme_idx].name,
                    description=analysis.themes[theme_idx].description,
                )
                theme_emb = self._embed_texts([theme_text])
                chunk_theme_sim[:, theme_idx] = cosine_similarity(
                    chunk_embeddings, theme_emb
                ).flatten()

        return chunk_theme_sim

    def _split_documents(
        self, documents: List[TrackedItem]
    ) -> Tuple[List[TrackedItem], Dict[int, str]]:
        """Split documents into chunks and track mapping.

        Uses the Split node's splitting logic for consistency with pipeline splitting.

        Returns:
            Tuple of (list of chunks, dict mapping chunk_index -> root_document_id)
        """
        # create a Split node instance to reuse its splitting logic
        splitter = Split(
            name="coverage",
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            split_unit=self.split_unit,
            min_split=self.chunk_size // 2,  # reasonable default
        )

        all_chunks = []
        chunk_doc_mapping = {}

        for doc in documents:
            if isinstance(doc, TrackedItem):
                chunks = splitter.split_tracked_document(doc)
            else:
                # wrap plain strings in TrackedItem
                temp_tracked = TrackedItem(
                    content=str(doc),
                    id="unknown_doc",
                    sources=["unknown_doc"],
                    metadata={},
                )
                chunks = splitter.split_tracked_document(temp_tracked)

            doc_id = doc.root_document if isinstance(doc, TrackedItem) else "unknown_doc"

            for chunk in chunks:
                chunk_idx = len(all_chunks)
                chunk_doc_mapping[chunk_idx] = doc_id
                all_chunks.append(chunk)

        return all_chunks, chunk_doc_mapping

    def _sample_chunks(
        self,
        chunks: List[TrackedItem],
        chunk_doc_mapping: Dict[int, str],
    ) -> Tuple[List[TrackedItem], Dict[int, str]]:
        """Sample chunks if sampling is configured.

        Sampling is stratified by document to ensure each document is represented.
        """
        import random

        if self.sample_frac is None and self.sample_n is None:
            return chunks, chunk_doc_mapping

        n_chunks = len(chunks)

        # determine target sample size
        if self.sample_n is not None:
            target_n = min(self.sample_n, n_chunks)
        else:
            target_n = max(1, int(n_chunks * self.sample_frac))

        if target_n >= n_chunks:
            return chunks, chunk_doc_mapping

        # stratified sampling: sample proportionally from each document
        doc_chunks: Dict[str, List[int]] = defaultdict(list)
        for idx, doc_id in chunk_doc_mapping.items():
            doc_chunks[doc_id].append(idx)

        # calculate samples per document (at least 1 per doc if possible)
        n_docs = len(doc_chunks)
        base_per_doc = max(1, target_n // n_docs)
        remainder = target_n - (base_per_doc * n_docs)

        sampled_indices = []
        for doc_id, indices in doc_chunks.items():
            # sample from this document
            n_sample = min(len(indices), base_per_doc + (1 if remainder > 0 else 0))
            if remainder > 0:
                remainder -= 1
            sampled = random.sample(indices, n_sample)
            sampled_indices.extend(sampled)

        # rebuild chunks and mapping with new indices
        sampled_indices.sort()
        new_chunks = []
        new_mapping = {}
        for new_idx, old_idx in enumerate(sampled_indices):
            new_chunks.append(chunks[old_idx])
            new_mapping[new_idx] = chunk_doc_mapping[old_idx]

        return new_chunks, new_mapping

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using configured backend and model.

        Local backend uses sentence-transformers with batching.
        API backend uses struckdown/litellm with parallel batch processing.
        """
        embeddings = get_embedding(
            list(texts),
            model=self.embedding_model,
        )
        return np.array(embeddings)

    def _aggregate_to_documents(
        self,
        chunks: List[TrackedItem],
        chunk_doc_mapping: Dict[int, str],
        chunk_theme_sim: np.ndarray,
        theme_names: List[str],
        documents: List[TrackedItem],
        groups: Optional[Dict[str, str]],
    ) -> Tuple[List[DocumentCoverage], List[List[float]]]:
        """Aggregate chunk-level scores to document level.

        Returns:
            Tuple of (list of DocumentCoverage, coverage matrix as nested lists)
        """
        # group chunks by document
        doc_chunks: Dict[str, List[int]] = defaultdict(list)
        for chunk_idx, doc_id in chunk_doc_mapping.items():
            doc_chunks[doc_id].append(chunk_idx)

        # create doc_id -> filename mapping
        doc_filenames = {}
        for doc in documents:
            doc_id = doc.root_document if isinstance(doc, TrackedItem) else doc.id
            filename = doc.metadata.get("filename", doc_id) if isinstance(doc, TrackedItem) else doc_id
            doc_filenames[doc_id] = filename

        doc_coverages = []
        coverage_matrix = []

        # process documents in order
        doc_ids = list(doc_chunks.keys())

        for doc_id in doc_ids:
            chunk_indices = doc_chunks[doc_id]
            chunk_sims = chunk_theme_sim[chunk_indices]  # [n_doc_chunks x n_themes]

            # aggregate per theme
            doc_scores = {}
            doc_scores_list = []

            for theme_idx, theme_name in enumerate(theme_names):
                theme_sims = chunk_sims[:, theme_idx]

                if self.aggregation == "max":
                    score = float(np.max(theme_sims))
                elif self.aggregation == "mean":
                    score = float(np.mean(theme_sims))
                elif self.aggregation == "p95":
                    score = float(np.percentile(theme_sims, 95))
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")

                doc_scores[theme_name] = score
                doc_scores_list.append(score)

            filename = doc_filenames.get(doc_id, doc_id)
            group = None
            if groups:
                # try matching by filename (exact or basename)
                group = groups.get(filename)
                if not group:
                    # try basename
                    import os
                    basename = os.path.basename(filename)
                    group = groups.get(basename)

            doc_coverages.append(
                DocumentCoverage(
                    document_id=doc_id,
                    filename=filename,
                    group=group,
                    theme_scores=doc_scores,
                    n_chunks=len(chunk_indices),
                )
            )
            coverage_matrix.append(doc_scores_list)

        return doc_coverages, coverage_matrix

    def _compute_group_statistics(
        self, doc_coverages: List[DocumentCoverage], theme_names: List[str]
    ) -> List[GroupCoverage]:
        """Compute per-group statistics."""
        # group documents
        group_docs: Dict[str, List[DocumentCoverage]] = defaultdict(list)
        for doc in doc_coverages:
            if doc.group:
                group_docs[doc.group].append(doc)

        group_coverages = []

        for group_name, docs in sorted(group_docs.items()):
            theme_means = {}
            theme_stds = {}

            for theme_name in theme_names:
                scores = [doc.theme_scores[theme_name] for doc in docs]
                theme_means[theme_name] = float(np.mean(scores))
                theme_stds[theme_name] = float(np.std(scores))

            group_coverages.append(
                GroupCoverage(
                    group_name=group_name,
                    n_documents=len(docs),
                    theme_scores=theme_means,
                    theme_scores_std=theme_stds,
                )
            )

        return group_coverages

    def _extract_top_chunks(
        self,
        chunks: List[TrackedItem],
        chunk_doc_mapping: Dict[int, str],
        chunk_theme_sim: np.ndarray,
        theme_names: List[str],
        documents: List[TrackedItem],
        n_top: int = 10,
    ) -> Dict[str, List[ChunkExample]]:
        """Extract top N chunks with highest similarity for each theme."""
        # create doc_id -> filename mapping
        doc_filenames = {}
        for doc in documents:
            doc_id = doc.root_document if isinstance(doc, TrackedItem) else doc.id
            filename = doc.metadata.get("filename", doc_id) if isinstance(doc, TrackedItem) else doc_id
            doc_filenames[doc_id] = filename

        top_chunks = {}

        for theme_idx, theme_name in enumerate(theme_names):
            # get similarities for this theme
            theme_sims = chunk_theme_sim[:, theme_idx]

            # get top N indices
            top_indices = np.argsort(theme_sims)[::-1][:n_top]

            examples = []
            for idx in top_indices:
                chunk = chunks[idx]
                doc_id = chunk_doc_mapping[idx]
                chunk_text = chunk.content if isinstance(chunk, TrackedItem) else str(chunk)

                # truncate chunk text for display
                if len(chunk_text) > 500:
                    chunk_text = chunk_text[:500] + "..."

                examples.append(
                    ChunkExample(
                        chunk_text=chunk_text,
                        document_id=doc_id,
                        filename=doc_filenames.get(doc_id, doc_id),
                        similarity=float(theme_sims[idx]),
                    )
                )

            top_chunks[theme_name] = examples

        return top_chunks


def generate_coverage_heatmap(
    result: ThemeCoverageResult,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
    threshold: Optional[float] = None,
) -> str:
    """Generate a heatmap visualization of theme coverage.

    Args:
        result: ThemeCoverageResult from analysis
        figsize: Figure size as (width, height)
        cmap: Matplotlib colormap name
        threshold: If provided, creates binary thresholded heatmap (above=red, below=grey)

    Returns:
        Base64-encoded PNG image string
    """
    import base64
    import io

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # prepare data
    matrix = np.array(result.coverage_matrix)
    theme_labels = [t[:25] + "..." if len(t) > 25 else t for t in result.theme_names]

    if threshold is not None:
        # binary thresholded heatmap: above threshold = red, below = grey
        mask_below = matrix < threshold
        mask_above = ~mask_below

        # draw red for above-threshold values
        sns.heatmap(
            matrix,
            xticklabels=theme_labels,
            yticklabels=False,
            cmap=ListedColormap(["#d62728"]),  # single red color
            vmin=0,
            vmax=1,
            mask=mask_below,
            annot=False,
            ax=ax,
            cbar=False,
        )
        # draw grey for below-threshold values
        sns.heatmap(
            matrix,
            xticklabels=theme_labels,
            yticklabels=False,
            cmap=ListedColormap(["#e0e0e0"]),  # single grey color
            vmin=0,
            vmax=1,
            mask=mask_above,
            annot=False,
            ax=ax,
            cbar=False,
        )
        title_suffix = f" (threshold={threshold})"
    else:
        # continuous heatmap with dynamic range (30th percentile to max)
        vmin = np.percentile(matrix, 30)
        vmax = np.max(matrix)
        # ensure some range
        if vmax <= vmin:
            vmin = np.min(matrix)

        sns.heatmap(
            matrix,
            xticklabels=theme_labels,
            yticklabels=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=False,
            ax=ax,
            cbar_kws={"label": "Coverage Score"},
        )
        title_suffix = ""

    ax.set_xlabel("Themes")
    ax.set_ylabel(f"Documents (n={len(result.documents)})")
    ax.set_title(f"Theme Coverage: {result.analysis_name}{title_suffix}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def generate_group_heatmap(
    result: ThemeCoverageResult,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "YlOrRd",
) -> Optional[str]:
    """Generate a heatmap of group-level coverage.

    Args:
        result: ThemeCoverageResult with group data
        figsize: Figure size
        cmap: Colormap name

    Returns:
        Base64-encoded PNG or None if no groups
    """
    if not result.groups:
        return None

    import base64
    import io

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    # build matrix
    group_names = [g.group_name for g in result.groups]
    matrix = np.array([
        [g.theme_scores[t] for t in result.theme_names]
        for g in result.groups
    ])

    theme_labels = [t[:25] + "..." if len(t) > 25 else t for t in result.theme_names]

    sns.heatmap(
        matrix,
        xticklabels=theme_labels,
        yticklabels=group_names,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Mean Coverage Score"},
    )

    ax.set_xlabel("Themes")
    ax.set_ylabel("Groups")
    ax.set_title(f"Group-Level Theme Coverage: {result.analysis_name}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def generate_chunk_heatmap(
    result: ThemeCoverageResult,
    window_size: int = 10,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "YlOrRd",
) -> Optional[str]:
    """Generate a chunk-level heatmap showing where theme matches occur in documents.

    Chunks are aggregated in windows (max within window) to reduce visual clutter.
    Documents are shown as horizontal bands with vertical lines between them.

    Args:
        result: ThemeCoverageResult with chunk-level data
        window_size: Number of chunks to aggregate with max (default: 10)
        figsize: Figure size
        cmap: Colormap name

    Returns:
        Base64-encoded PNG or None if no chunk data
    """
    if not result.chunk_similarity_matrix or not result.chunk_info:
        return None

    import base64
    import io
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import seaborn as sns

    # group chunks by document, preserving order
    doc_chunks: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # doc_id -> [(chunk_idx, chunk_position)]
    doc_order = []  # preserve document order

    for idx, info in enumerate(result.chunk_info):
        if info.document_id not in doc_order:
            doc_order.append(info.document_id)
        doc_chunks[info.document_id].append((idx, info.chunk_index))

    # sort chunks within each document by position
    for doc_id in doc_chunks:
        doc_chunks[doc_id].sort(key=lambda x: x[1])

    chunk_sim = np.array(result.chunk_similarity_matrix)

    # build windowed matrix, tracking document boundaries
    windowed_rows = []
    doc_boundaries = []  # row indices where new documents start
    doc_labels = []  # (row_index, label) for y-axis

    current_row = 0
    for doc_id in doc_order:
        chunks = doc_chunks[doc_id]
        if not chunks:
            continue

        doc_boundaries.append(current_row)
        # get filename for label
        filename = result.chunk_info[chunks[0][0]].filename
        short_name = filename[:20] + "..." if len(filename) > 20 else filename
        doc_labels.append((current_row, short_name))

        # aggregate chunks in windows
        n_doc_chunks = len(chunks)
        for win_start in range(0, n_doc_chunks, window_size):
            win_end = min(win_start + window_size, n_doc_chunks)
            win_chunk_indices = [chunks[i][0] for i in range(win_start, win_end)]

            # max similarity within window for each theme
            win_sims = chunk_sim[win_chunk_indices, :]
            win_max = np.max(win_sims, axis=0)
            windowed_rows.append(win_max)
            current_row += 1

    if not windowed_rows:
        return None

    matrix = np.array(windowed_rows)

    # debug: verify windowed matrix has variation
    logger.debug(f"Chunk heatmap: {len(doc_order)} docs, {len(windowed_rows)} windows, {matrix.shape}")
    logger.debug(f"Chunk heatmap matrix std per theme: {matrix.std(axis=0)}")
    logger.debug(f"Chunk heatmap matrix min: {matrix.min():.3f}, max: {matrix.max():.3f}")

    # debug: check within-document vs between-document variation
    doc_row_ranges = []  # track which rows belong to which doc
    row_idx = 0
    for doc_id in doc_order:
        n_windows = (len(doc_chunks[doc_id]) + window_size - 1) // window_size
        if n_windows > 0:
            doc_row_ranges.append((row_idx, row_idx + n_windows))
            row_idx += n_windows

    # compute mean within-doc std
    within_doc_stds = []
    for start, end in doc_row_ranges:
        if end - start > 1:  # need at least 2 windows
            doc_matrix = matrix[start:end, :]
            within_doc_stds.append(doc_matrix.std())
    if within_doc_stds:
        logger.debug(f"Mean within-doc std: {np.mean(within_doc_stds):.4f}, between-doc std: {matrix.std():.4f}")

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    theme_labels = [t[:20] + "..." if len(t) > 20 else t for t in result.theme_names]

    # use full data range for better contrast (not percentile-based like doc heatmap)
    vmin = np.min(matrix)
    vmax = np.max(matrix)

    sns.heatmap(
        matrix,
        xticklabels=theme_labels,
        yticklabels=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        ax=ax,
        cbar_kws={"label": "Coverage Score (max in window)"},
    )

    # draw document boundary lines
    for boundary in doc_boundaries[1:]:  # skip first (top of plot)
        ax.axhline(y=boundary, color="white", linewidth=2)

    # add document labels on y-axis
    if len(doc_labels) <= 30:  # only show labels if not too many documents
        y_positions = [row + 0.5 for row, _ in doc_labels]
        y_labels = [label for _, label in doc_labels]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel("Themes")
    ax.set_ylabel(f"Document Chunks (window={window_size})")
    ax.set_title(f"Chunk-Level Theme Coverage: {result.analysis_name}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _build_chunk_dataframe(result: ThemeCoverageResult) -> "pd.DataFrame":
    """Build long-form DataFrame from coverage result for chunk-level analysis.

    Returns DataFrame with columns: document, document_id, chunk_position, theme, similarity
    """
    import pandas as pd

    chunk_sim = np.array(result.chunk_similarity_matrix)
    chunk_info = result.chunk_info
    theme_names = result.theme_names

    rows = []
    for chunk_idx, info in enumerate(chunk_info):
        for theme_idx, theme_name in enumerate(theme_names):
            rows.append({
                "document": info.filename,
                "document_id": info.document_id,
                "chunk_position": info.chunk_index,
                "theme": theme_name,
                "similarity": chunk_sim[chunk_idx, theme_idx],
            })

    return pd.DataFrame(rows)


def _add_normalized_position(df: "pd.DataFrame", n_bins: int = 20) -> "pd.DataFrame":
    """Add normalized position columns to dataframe.

    Normalizes chunk positions to 0-1 range and bins them for consistent width.
    """
    import pandas as pd

    df = df.copy()
    max_pos = df.groupby("document")["chunk_position"].transform("max")
    df["position_norm"] = df["chunk_position"] / max_pos.where(max_pos > 0, 1)
    df["position_bin"] = pd.cut(
        df["position_norm"],
        bins=n_bins,
        labels=range(n_bins),
        include_lowest=True,
    ).astype(int)
    return df


def _add_absolute_bins(df: "pd.DataFrame", chunks_per_bin: int) -> "pd.DataFrame":
    """Bin by absolute chunk count (not normalized)."""
    df = df.copy()
    df["abs_bin"] = df["chunk_position"] // chunks_per_bin
    return df


def _aggregate_bins(df: "pd.DataFrame", bin_col: str, agg: str = "max") -> "pd.DataFrame":
    """Aggregate similarity within bins."""
    return (
        df.groupby(["document", "document_id", bin_col, "theme"], as_index=False)["similarity"].agg(agg)
    )


def generate_normalized_chunk_heatmap(
    result: ThemeCoverageResult,
    n_bins: int = 20,
    figsize: Tuple[int, int] = (24, 10),
    cmap: str = "YlOrRd",
    z_score: bool = False,
) -> Optional[str]:
    """Generate horizontally-faceted heatmap with normalized document positions.

    Each document is shown with the same width (n_bins), regardless of actual length.
    Themes are arranged horizontally as separate panels.

    Args:
        result: ThemeCoverageResult with chunk-level data
        n_bins: Number of bins per document (default: 20)
        figsize: Figure size
        cmap: Colormap name
        z_score: If True, z-score values and truncate at 0

    Returns:
        Base64-encoded PNG or None if no chunk data
    """
    if not result.chunk_similarity_matrix or not result.chunk_info:
        return None

    import base64
    import io

    import matplotlib.pyplot as plt
    import seaborn as sns

    # build dataframe
    df = _build_chunk_dataframe(result)
    df = _add_normalized_position(df, n_bins=n_bins)
    df_binned = _aggregate_bins(df, "position_bin", agg="max")

    themes = result.theme_names
    n_themes = len(themes)

    fig, axes = plt.subplots(1, n_themes, figsize=figsize, sharey=True)
    if n_themes == 1:
        axes = [axes]

    doc_order = sorted(df_binned["document"].unique())

    # compute global scale
    vmin = df_binned["similarity"].min()
    vmax = df_binned["similarity"].max()

    for idx, (theme, ax) in enumerate(zip(themes, axes)):
        theme_df = df_binned[df_binned["theme"] == theme].copy()

        pivot = theme_df.pivot(index="document", columns="position_bin", values="similarity")
        valid_docs = [d for d in doc_order if d in pivot.index]
        pivot = pivot.reindex(valid_docs)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        plot_data = pivot.values.copy()

        # apply z-score transformation if requested
        if z_score:
            mean_val = np.nanmean(plot_data)
            std_val = np.nanstd(plot_data)
            if std_val > 0:
                plot_data = (plot_data - mean_val) / std_val
                plot_data = np.clip(plot_data, 0, None)  # truncate at 0
            plot_vmin, plot_vmax = 0, np.nanmax(plot_data)
        else:
            plot_vmin, plot_vmax = vmin, vmax

        show_cbar = idx == n_themes - 1

        sns.heatmap(
            plot_data,
            ax=ax,
            cmap=cmap,
            vmin=plot_vmin,
            vmax=plot_vmax,
            cbar=show_cbar,
            xticklabels=False,
            yticklabels=False,
        )

        short_title = theme[:25] + "..." if len(theme) > 25 else theme
        ax.set_title(short_title, fontsize=10)
        ax.set_xlabel("" if idx != n_themes // 2 else "Document Position (0% → 100%)")
        if idx > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Document")

    title_suffix = " (z-scored, truncated at 0)" if z_score else ""
    plt.suptitle(
        f"Theme Coverage -- Normalized ({n_bins} bins){title_suffix}",
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def generate_absolute_chunk_heatmap(
    result: ThemeCoverageResult,
    n_bins_target: int = 20,
    figsize: Tuple[int, int] = (24, 10),
    cmap: str = "YlOrRd",
) -> Optional[str]:
    """Generate horizontally-faceted heatmap with absolute chunk bins.

    Documents have variable widths based on their actual length.
    Chunks are grouped so the average document has approximately n_bins_target groups.

    Args:
        result: ThemeCoverageResult with chunk-level data
        n_bins_target: Target number of bins for average-length document
        figsize: Figure size
        cmap: Colormap name

    Returns:
        Base64-encoded PNG or None if no chunk data
    """
    if not result.chunk_similarity_matrix or not result.chunk_info:
        return None

    import base64
    import io

    import matplotlib.pyplot as plt
    import seaborn as sns

    # build dataframe
    df = _build_chunk_dataframe(result)

    # calculate chunks per bin based on average document length
    doc_chunk_counts = df.groupby("document")["chunk_position"].max() + 1
    mean_chunks = doc_chunk_counts.mean()
    chunks_per_bin = max(1, int(mean_chunks / n_bins_target))

    logger.debug(f"Absolute heatmap: mean_chunks={mean_chunks:.1f}, chunks_per_bin={chunks_per_bin}")

    df = _add_absolute_bins(df, chunks_per_bin=chunks_per_bin)
    df_binned = _aggregate_bins(df, "abs_bin", agg="max")

    themes = result.theme_names
    n_themes = len(themes)

    fig, axes = plt.subplots(1, n_themes, figsize=figsize, sharey=True)
    if n_themes == 1:
        axes = [axes]

    doc_order = sorted(df_binned["document"].unique())
    vmin = df_binned["similarity"].min()
    vmax = df_binned["similarity"].max()

    for idx, (theme, ax) in enumerate(zip(themes, axes)):
        theme_df = df_binned[df_binned["theme"] == theme].copy()

        pivot = theme_df.pivot(index="document", columns="abs_bin", values="similarity")
        valid_docs = [d for d in doc_order if d in pivot.index]
        pivot = pivot.reindex(valid_docs)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        show_cbar = idx == n_themes - 1

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=show_cbar,
            xticklabels=False,
            yticklabels=False,
        )

        short_title = theme[:25] + "..." if len(theme) > 25 else theme
        ax.set_title(short_title, fontsize=10)
        ax.set_xlabel("" if idx != n_themes // 2 else f"Chunk Groups ({chunks_per_bin} chunks each)")
        if idx > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Document")

    plt.suptitle(
        f"Theme Coverage -- Absolute Chunks ({chunks_per_bin} chunks per bin)",
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def compute_within_doc_variation(
    result: ThemeCoverageResult,
    n_bins: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Compute within-document variation statistics.

    For each document, takes max similarity per bin, then computes avg/max/p95.

    Args:
        result: ThemeCoverageResult with chunk-level data
        n_bins: Number of bins per document

    Returns:
        Dict mapping theme name to {"avg": float, "max": float, "p95": float}
    """
    if not result.chunk_similarity_matrix or not result.chunk_info:
        return {}

    df = _build_chunk_dataframe(result)
    df = _add_normalized_position(df, n_bins=n_bins)
    df_binned = _aggregate_bins(df, "position_bin", agg="max")

    # compute stats per document per theme
    doc_stats = (
        df_binned.groupby(["document", "theme"])["similarity"]
        .agg(["mean", "max", lambda x: np.percentile(x, 95)])
        .rename(columns={"<lambda_0>": "p95"})
        .reset_index()
    )

    # average across documents per theme
    theme_stats = {}
    for theme in result.theme_names:
        theme_data = doc_stats[doc_stats["theme"] == theme]
        theme_stats[theme] = {
            "avg": float(theme_data["mean"].mean()),
            "max": float(theme_data["max"].mean()),
            "p95": float(theme_data["p95"].mean()),
        }

    return theme_stats


def generate_theme_trajectories(
    result: ThemeCoverageResult,
    n_bins: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    loess_frac: float = 0.3,
) -> Optional[str]:
    """Generate theme trajectory plot showing similarity across document position.

    Shows LOESS-smoothed average similarity at each normalized position.

    Args:
        result: ThemeCoverageResult with chunk-level data
        n_bins: Number of bins per document
        figsize: Figure size
        loess_frac: LOESS smoothing fraction

    Returns:
        Base64-encoded PNG or None if no chunk data
    """
    if not result.chunk_similarity_matrix or not result.chunk_info:
        return None

    import base64
    import io

    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        has_lowess = True
    except ImportError:
        has_lowess = False
        logger.warning("statsmodels not available, using rolling mean for trajectories")

    df = _build_chunk_dataframe(result)
    df = _add_normalized_position(df, n_bins=n_bins)
    df_binned = _aggregate_bins(df, "position_bin", agg="max")

    themes = result.theme_names
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(themes)))

    for theme, color in zip(themes, colors):
        theme_df = df_binned[df_binned["theme"] == theme]

        # compute mean per position bin (across all documents)
        mean_by_pos = (
            theme_df.groupby("position_bin")["similarity"]
            .mean()
            .reset_index()
            .sort_values("position_bin")
        )

        x = mean_by_pos["position_bin"].values
        y = mean_by_pos["similarity"].values

        # LOESS smoothing
        if has_lowess and len(x) > 5:
            smoothed = lowess(y, x, frac=loess_frac, return_sorted=True)
            x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]
        else:
            x_smooth = x
            y_smooth = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values

        short_label = theme[:30] + "..." if len(theme) > 30 else theme
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, label=short_label)

    ax.set_xlabel("Document Position (normalized bins)")
    ax.set_ylabel("Similarity (max within bin, averaged across docs)")
    ax.set_title("Theme Trajectories Across Document Position")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # x-axis as percentage
    ax.set_xticks([0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
