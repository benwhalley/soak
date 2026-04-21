"""Cluster node for grouping items by semantic similarity."""

import hashlib
import logging
import random
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import numpy as np
from decouple import config as env_config
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field, PrivateAttr

from soak.events import NodeProgress
from soak.models.base import TrackedItem, get_embedding_async, memory
from soak.models.utils import unwrap_complete_items

from .base import DAGNode
from .batch import BatchList

MAX_LLM_CONCURRENCY = env_config("SD_MAX_CONCURRENCY", default=20, cast=int)

logger = logging.getLogger(__name__)


def _hash_embeddings(embeddings: np.ndarray) -> str:
    """Create a stable hash of embeddings array for cache key."""
    return hashlib.sha256(embeddings.tobytes()).hexdigest()[:16]


@memory.cache
def _cached_hdbscan_fit(
    embeddings_hash: str,
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Cached HDBSCAN fit_predict."""
    import hdbscan

    logger.info(f"HDBSCAN cache miss - running fit_predict (hash={embeddings_hash})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


class HDBSCANMethod(BaseModel):
    """HDBSCAN clustering method with recursive splitting for oversized clusters.

    Cluster size control:
        - min_cluster_size_proportion: Goal/target as proportion of items (e.g., 0.25 = 4 clusters)
        - min_cluster_size: Hard floor -- clusters never smaller than this (overrides proportion)
        - max_cluster_size: Hard ceiling -- clusters split if larger than this (overrides proportion)

    Example: With 100 items, proportion=0.25, min=10, max=50:
        - Proportion suggests min_cluster_size=25
        - Floor of 10 doesn't apply (25 > 10)
        - Clusters larger than 50 get split
    """

    name: Literal["hdbscan"] = "hdbscan"
    max_cluster_size: Optional[int] = 100  # hard ceiling, clusters split if larger
    min_cluster_size: int = 2  # hard floor, overrides proportion
    min_cluster_size_proportion: Optional[float] = (
        None  # goal: e.g., 0.25 = ~4 clusters
    )
    min_samples: int = 1

    # runtime stats (populated during cluster()) -- not serialized
    _effective_min_cluster_size: Optional[int] = PrivateAttr(default=None)
    _singleton_count: int = PrivateAttr(default=0)
    _oversized_count: int = PrivateAttr(default=0)

    def cluster(self, embeddings: np.ndarray, seed: int = 42) -> List[List[int]]:
        """Run HDBSCAN and return list of lists of indices.

        Handles:
        - Recursive splitting of oversized clusters (when max_cluster_size is set)
        - Random chunking fallback when HDBSCAN won't split
        - Singleton grouping into low-coherence batches

        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            seed: random seed for deterministic chunking fallback

        Returns:
            List of lists of indices, where each inner list is a cluster
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            return []

        if n_samples == 1:
            return [[0]]

        # calculate effective min_cluster_size
        # use max of: fixed min_cluster_size AND proportion-based size
        effective_min_cluster_size = self.min_cluster_size
        if self.min_cluster_size_proportion is not None:
            proportional_size = max(
                2, int(n_samples * self.min_cluster_size_proportion)
            )
            effective_min_cluster_size = max(
                effective_min_cluster_size, proportional_size
            )

        # store for reporting
        self._effective_min_cluster_size = effective_min_cluster_size

        # run HDBSCAN (cached)
        logger.info(
            f"HDBSCAN: min_cluster_size={effective_min_cluster_size}, "
            f"min_samples={self.min_samples}, n_samples={n_samples}"
        )
        embeddings_hash = _hash_embeddings(embeddings)
        logger.info(
            f"Running HDBSCAN fit_predict on {n_samples} samples (hash={embeddings_hash})..."
        )
        labels = _cached_hdbscan_fit(
            embeddings_hash, embeddings, effective_min_cluster_size, self.min_samples
        )
        logger.info("HDBSCAN fit_predict complete.")

        # group indices by label
        clusters_by_label: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters_by_label.setdefault(label, []).append(idx)

        # process clusters
        final_clusters = []
        singletons = clusters_by_label.pop(-1, [])  # noise points
        self._singleton_count = len(singletons)

        # count oversized clusters before splitting
        oversized_count = 0
        oversized_sizes = []
        for label, indices in clusters_by_label.items():
            if (
                self.max_cluster_size is not None
                and len(indices) > self.max_cluster_size
            ):
                oversized_count += 1
                oversized_sizes.append(len(indices))
        self._oversized_count = oversized_count

        logger.info(
            f"Initial clustering: {len(clusters_by_label)} clusters, "
            f"{self._singleton_count} singletons"
        )
        if oversized_count > 0:
            logger.info(
                f"Splitting {oversized_count} oversized clusters "
                f"(sizes: {sorted(oversized_sizes, reverse=True)}, max_size={self.max_cluster_size})..."
            )

        for label, indices in clusters_by_label.items():
            if self.max_cluster_size is None or len(indices) <= self.max_cluster_size:
                final_clusters.append(indices)
            else:
                # recursively split oversized cluster
                sub_embeddings = embeddings[indices]
                sub_clusters = self._split_oversized(
                    sub_embeddings, indices, effective_min_cluster_size, seed=seed
                )
                final_clusters.extend(sub_clusters)

        # handle singletons: group into batches
        if singletons:
            singleton_batches = self._batch_singletons(singletons)
            final_clusters.extend(singleton_batches)

        return final_clusters

    def _split_oversized(
        self,
        embeddings: np.ndarray,
        original_indices: List[int],
        effective_min_cluster_size: int,
        depth: int = 0,
        max_depth: int = 5,
        seed: int = 42,
    ) -> List[List[int]]:
        """Recursively split an oversized cluster.

        Args:
            embeddings: embeddings for items in this cluster
            original_indices: original indices in the full dataset
            effective_min_cluster_size: the calculated min size to use for splitting
            depth: current recursion depth
            max_depth: maximum recursion depth before falling back to random chunking
            seed: random seed for deterministic chunking fallback

        Returns:
            List of lists of original indices, each <= max_cluster_size
        """
        n = len(embeddings)
        if n <= self.max_cluster_size:
            return [original_indices]

        # check recursion depth limit
        if depth >= max_depth:
            logger.info(
                f"  Max recursion depth ({max_depth}) reached for cluster of {n} items, "
                f"using random chunking"
            )
            return self._random_chunk(
                original_indices, effective_min_cluster_size, seed=seed
            )

        logger.debug(f"  Splitting oversized cluster: {n} items (depth={depth})")

        # try HDBSCAN with smaller min_cluster_size to force splitting
        # use half of effective, but not smaller than 2
        smaller_min = max(2, effective_min_cluster_size // 2)
        embeddings_hash = _hash_embeddings(embeddings)
        labels = _cached_hdbscan_fit(embeddings_hash, embeddings, smaller_min, 1)

        # check if it actually split
        unique_labels = set(labels) - {-1}
        if len(unique_labels) <= 1:
            logger.debug(
                f"  HDBSCAN won't split {n} items further, using random chunking"
            )
            return self._random_chunk(
                original_indices, effective_min_cluster_size, seed=seed
            )

        # group by new labels
        clusters_by_label: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            if label == -1:
                # treat noise as singleton
                clusters_by_label.setdefault(-1, []).append(i)
            else:
                clusters_by_label.setdefault(label, []).append(i)

        # check if we made meaningful progress (largest cluster should be < 80% of input)
        largest_cluster_size = max(
            len(indices) for indices in clusters_by_label.values()
        )
        if largest_cluster_size > n * 0.8:
            logger.debug(
                f"  Insufficient split progress: largest sub-cluster is {largest_cluster_size}/{n} "
                f"({largest_cluster_size/n*100:.0f}%), using random chunking"
            )
            return self._random_chunk(original_indices, effective_min_cluster_size)

        logger.debug(
            f"  Split {n} items into {len(clusters_by_label)} sub-clusters "
            f"(sizes: {sorted([len(v) for v in clusters_by_label.values()], reverse=True)[:5]})"
        )

        result = []
        for label, local_indices in clusters_by_label.items():
            global_indices = [original_indices[i] for i in local_indices]

            if len(global_indices) <= self.max_cluster_size:
                result.append(global_indices)
            else:
                # recurse with incremented depth
                sub_embeddings = embeddings[local_indices]
                sub_result = self._split_oversized(
                    sub_embeddings,
                    global_indices,
                    effective_min_cluster_size,
                    depth=depth + 1,
                    max_depth=max_depth,
                    seed=seed,
                )
                result.extend(sub_result)

        return result

    def _random_chunk(
        self, indices: List[int], effective_min_cluster_size: int, seed: int = 42
    ) -> List[List[int]]:
        """Split indices into random chunks respecting max cluster size.

        Note: min_cluster_size is a goal for HDBSCAN clustering, but when we fall back
        to random chunking, we prioritise the max_cluster_size hard limit. Small final
        chunks are acceptable here since they come from splitting, not initial clustering.

        Uses a seeded RNG for deterministic results across runs.
        """
        rng = random.Random(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        chunks = []
        for i in range(0, len(shuffled), self.max_cluster_size):
            chunks.append(shuffled[i : i + self.max_cluster_size])

        return chunks

    def _batch_singletons(self, singletons: List[int]) -> List[List[int]]:
        """Group singletons into batches of max_cluster_size."""
        if self.max_cluster_size is None:
            # no size limit -- return all singletons as one batch
            return [singletons] if singletons else []

        batches = []
        for i in range(0, len(singletons), self.max_cluster_size):
            batches.append(singletons[i : i + self.max_cluster_size])

        return batches


# discriminated union for clustering methods
# future methods (KMeans, Agglomerative, etc.) can be added here
ClusterMethod = Annotated[Union[HDBSCANMethod], Field(discriminator="name")]


class Cluster(DAGNode):
    """Node that groups items by semantic similarity using configurable clustering.

    Groups items into clusters based on embedding similarity. Uses HDBSCAN by default,
    which automatically determines the number of clusters.

    Attributes:
        method: Clustering method configuration (defaults to HDBSCAN)
        items_field: How to extract items from container types:
            - None (default): Use items as-is
            - "codes": Extract from CodeList.codes (use after Map producing CodeLists)
            - "results.codes.output.codes": Dotted path for nested extraction
            For StruckdownResult inputs, extraction happens from segment.output
        text_field: How to extract text for embedding from each item:
            - "content" (default): Uses TrackedItem.content or str(item)
            - "metadata.field_name": Extracts from item.metadata["field_name"]
            - "{{name}}: {{description}}": Jinja2 template for custom text

    Example YAML:
        # Clustering Code objects or TrackedItems directly (default)
        - name: grouped_codes
          type: Cluster
          inputs: [codes]

        # Extract codes from CodeList containers (after Map producing CodeLists)
        - name: grouped_codes
          type: Cluster
          inputs: [coded_chunks]
          items_field: codes

        # With custom method config
        - name: grouped
          type: Cluster
          inputs: [coded_chunks]
          method:
            name: hdbscan
            max_cluster_size: 50
            min_cluster_size: 3
    """

    type: Literal["Cluster"] = "Cluster"
    method: ClusterMethod = Field(default_factory=HDBSCANMethod)
    items_field: Optional[str] = None
    text_field: str = "content"
    skip_below: int = Field(
        default=20,
        description="If input item count is below this threshold, "
        "skip clustering and return all items as a single cluster.",
    )
    if_skipped_bypass_to: Optional[str] = Field(
        default=None,
        description="When skip_below triggers, skip intermediate nodes up to this target. "
        "The target node will receive the original input directly, bypassing any "
        "intermediate processing nodes.",
    )

    async def run(self) -> BatchList:
        """Execute clustering on input items.

        Returns:
            BatchList with items grouped by cluster similarity.
            Each item's metadata is annotated with cluster_id and cluster_coherence.
        """
        await super().run()
        logger.info(f"Cluster '{self.name}': Starting clustering...")

        # gather inputs using helper method
        items, texts = self._gather_inputs()

        if not items:
            logger.warning(f"Cluster '{self.name}': No items to cluster")
            self.output = BatchList(batches=[], group_field="cluster", group_keys=[])
            return self.output

        logger.info(f"Cluster '{self.name}': {len(items)} items to cluster")

        # check skip_below threshold
        if self.skip_below is not None and len(items) < self.skip_below:
            # Mark this node as skipped since we're not actually clustering
            self._skipped = True
            if self.if_skipped_bypass_to:
                # bypass mode: skip intermediate nodes and pass items directly to target
                target = self.if_skipped_bypass_to
                intermediate = self._find_path_to(target)
                for node_name in intermediate:
                    if node_name not in self.dag.config.skip_nodes:
                        self.dag.config.skip_nodes.append(node_name)
                        logger.info(f"Bypass: skipping '{node_name}'")

                # passthrough raw items (not wrapped in cluster) for target node
                self.output = items
                logger.info(
                    f"Cluster '{self.name}': {len(items)} items < skip_below={self.skip_below}, "
                    f"bypassing to '{target}'"
                )
                return self.output
            else:
                logger.warning(
                    f"Cluster '{self.name}': {len(items)} items below skip_below={self.skip_below}, "
                    "skipping clustering and returning single cluster"
                )
                self.output = self._make_single_cluster(items, texts)
                return self.output

        # deduplicate texts for embedding efficiency
        unique_texts = list(set(texts))
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}

        logger.info(
            f"Cluster '{self.name}': {len(texts)} items, {len(unique_texts)} unique texts to embed"
        )

        # get embeddings with progress bar
        logger.info(f"Computing embeddings for {len(unique_texts)} texts...")

        # signal start
        self.dag.emit(NodeProgress(self.name, 0, len(unique_texts), 0.0))

        embeddings_list = await get_embedding_async(
            unique_texts,
            model=self.dag.config.embedding_model,
            credentials=self.dag.config.get_embedding_credentials(),
        )
        unique_embeddings = np.array(embeddings_list)
        logger.info("Embeddings computed.")

        # signal completion
        self.dag.emit(NodeProgress(self.name, len(unique_texts), len(unique_texts), 0.0))

        # map back to full set
        logger.debug(
            f"Mapping {len(unique_texts)} unique embeddings back to {len(texts)} items..."
        )
        full_embeddings = np.array([unique_embeddings[text_to_idx[t]] for t in texts])

        # run clustering
        logger.info(
            f"Running HDBSCAN clustering on {len(full_embeddings)} embeddings..."
        )
        cluster_indices = self.method.cluster(
            full_embeddings, seed=self.dag.config.seed
        )
        logger.info(f"Clustering complete: {len(cluster_indices)} clusters found.")

        # build output: one TrackedItem per cluster
        # this allows Map to process each cluster as a single item
        logger.info(f"Building {len(cluster_indices)} cluster output items...")
        cluster_items_output = []

        for cluster_idx, indices in enumerate(cluster_indices):
            cluster_key = f"cluster_{cluster_idx}"
            cluster_items = [items[idx] for idx in indices]

            # stringify the cluster items for content
            content_parts = []
            for item in cluster_items:
                if hasattr(item, "model_dump"):
                    # Pydantic model (Code, Theme, etc.) - use str representation
                    content_parts.append(str(item))
                elif isinstance(item, TrackedItem):
                    content_parts.append(item.content)
                else:
                    content_parts.append(str(item))

            cluster_content = "\n\n---\n\n".join(content_parts)

            # create TrackedItem for this cluster
            # store original items in metadata for template access
            cluster_tracked = TrackedItem(
                content=cluster_content,
                id=f"{self.name}__{cluster_key}",
                sources=[
                    (
                        item.id
                        if isinstance(item, TrackedItem)
                        else (item.slug if hasattr(item, "slug") else str(idx))
                    )
                    for idx, item in zip(indices, cluster_items)
                ],
                metadata={
                    "cluster_id": cluster_key,
                    "cluster_index": cluster_idx,
                    "cluster_size": len(cluster_items),
                    "items": cluster_items,  # original items for template access
                },
            )
            cluster_items_output.append(cluster_tracked)

        self.output = cluster_items_output

        # log detailed cluster statistics
        sizes = [c.metadata["cluster_size"] for c in cluster_items_output]
        if sizes:
            logger.info(
                f"Cluster '{self.name}': {len(cluster_items_output)} clusters from {len(items)} items"
            )
            logger.info(
                f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
                f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}"
            )

            # show top clusters by size
            sorted_clusters = sorted(
                cluster_items_output,
                key=lambda x: x.metadata["cluster_size"],
                reverse=True,
            )
            top_n = min(5, len(sorted_clusters))
            logger.info(f"  Top {top_n} clusters by size:")
            for cluster in sorted_clusters[:top_n]:
                cluster_items = cluster.metadata["items"]
                size = cluster.metadata["cluster_size"]
                cluster_id = cluster.metadata["cluster_id"]

                # get sample item names/descriptions
                sample_items = cluster_items[:3]
                sample_strs = []
                for item in sample_items:
                    if hasattr(item, "name"):
                        sample_strs.append(item.name[:30])
                    elif hasattr(item, "slug"):
                        sample_strs.append(item.slug)
                    elif isinstance(item, TrackedItem):
                        sample_strs.append(item.id[:30])
                    else:
                        sample_strs.append(str(item)[:30])
                samples = ", ".join(sample_strs)
                if size > 3:
                    samples += f", ... (+{size-3} more)"
                logger.info(f"    {cluster_id}: {size} items [{samples}]")

        return self.output

    def _gather_inputs(self) -> tuple[List[Any], List[str]]:
        """Gather and process input items for clustering.

        Returns:
            Tuple of (all_items, all_texts) where all_items are the unwrapped items
            and all_texts are the extracted text strings for embedding.
        """
        # get input data directly from the upstream node's raw output.
        # self.context[...] wraps StruckdownResults in _TemplateProxy for template
        # rendering, which hides individual items from unwrap_complete_items.
        if self.inputs:
            input_name = self.inputs[0]
            if input_name == "documents":
                input_data = self.dag.config.load_documents()
            else:
                input_node = self.dag.nodes_dict.get(input_name)
                input_data = input_node.output if input_node else None
        else:
            input_data = self.dag.config.load_documents()

        # flatten if BatchList
        if isinstance(input_data, BatchList):
            raw_items = input_data.flatten_all()
            logger.debug(f"Flattened BatchList to {len(raw_items)} items")
        elif isinstance(input_data, list):
            raw_items = input_data
        else:
            raw_items = [input_data]

        # unwrap items from containers (e.g., CodeList -> Code)
        if self.items_field:
            items = unwrap_complete_items(raw_items, self.items_field)
        else:
            items = raw_items

        if not items:
            return [], []

        logger.debug(
            f"Cluster '{self.name}': {len(raw_items)} raw items -> {len(items)} items after unwrap"
        )

        # extract texts for embedding
        texts = [self._extract_text(item) for item in items]

        return items, texts

    def _find_path_to(self, target: str) -> List[str]:
        """Find intermediate node names between self and target (exclusive of both).

        Uses BFS to find all nodes in the dependency path from this node to the target.

        Args:
            target: Name of the target node to bypass to

        Returns:
            List of intermediate node names to skip

        Raises:
            ValueError: If target node doesn't exist in the DAG
        """
        if target not in self.dag.nodes_dict:
            raise ValueError(
                f"Bypass target '{target}' not found in DAG. "
                f"Available nodes: {list(self.dag.nodes_dict.keys())}"
            )

        visited = set()
        queue = [self.name]
        path_nodes = []

        while queue:
            current = queue.pop(0)
            if current == target:
                break
            if current in visited:
                continue
            visited.add(current)

            # find nodes that depend on current (i.e., have current in their inputs)
            for node in self.dag.nodes:
                if current in (node.inputs or []) and node.name not in visited:
                    if node.name != target:
                        path_nodes.append(node.name)
                    queue.append(node.name)

        return path_nodes

    def _make_single_cluster(
        self, all_items: List[Any], all_texts: List[str]
    ) -> List[TrackedItem]:
        """Create a single cluster containing all items (used for skip scenarios).

        Args:
            all_items: List of items to include in the cluster
            all_texts: List of text representations of items

        Returns:
            List containing a single TrackedItem representing the cluster
        """
        # stringify the cluster items for content
        content_parts = []
        for item in all_items:
            if hasattr(item, "model_dump"):
                content_parts.append(str(item))
            elif isinstance(item, TrackedItem):
                content_parts.append(item.content)
            else:
                content_parts.append(str(item))

        cluster_content = "\n\n---\n\n".join(content_parts)

        single_cluster = TrackedItem(
            content=cluster_content,
            id=f"{self.name}__cluster_0",
            sources=[
                (
                    item.id
                    if isinstance(item, TrackedItem)
                    else (item.slug if hasattr(item, "slug") else str(i))
                )
                for i, item in enumerate(all_items)
            ],
            metadata={
                "cluster_id": "cluster_0",
                "cluster_index": 0,
                "cluster_size": len(all_items),
                "items": all_items,
                "skipped_clustering": True,
            },
        )
        return [single_cluster]

    async def skip(self) -> List[TrackedItem]:
        """When skipped, return all inputs as a single cluster.

        This provides passthrough behavior: downstream nodes still receive
        a cluster TrackedItem, but no actual clustering is performed.

        Returns:
            List containing a single TrackedItem with all inputs as one cluster.
        """
        logger.info(
            f"Cluster '{self.name}': skipped, returning single cluster passthrough"
        )

        # gather inputs using the same logic as run()
        items, texts = self._gather_inputs()

        if not items:
            logger.warning(f"Cluster '{self.name}': No items to cluster (skipped)")
            self.output = []
            return self.output

        self.output = self._make_single_cluster(items, texts)
        return self.output

    def _extract_text(self, item: Any) -> str:
        """Extract text from item based on text_field configuration.

        Args:
            item: Input item (TrackedItem, dict, or string)

        Returns:
            Text string for embedding
        """
        # handle Jinja2 template
        if "{{" in self.text_field:
            return self._render_template(item)

        # handle metadata.field_name
        if self.text_field.startswith("metadata."):
            field_name = self.text_field[9:]  # strip "metadata."
            if isinstance(item, TrackedItem) and item.metadata:
                return str(item.metadata.get(field_name, ""))
            elif isinstance(item, dict) and "metadata" in item:
                return str(item["metadata"].get(field_name, ""))
            return ""

        # default: use content
        if isinstance(item, TrackedItem):
            return item.content
        elif isinstance(item, dict):
            return str(item.get("content", item))
        else:
            return str(item)

    def _render_template(self, item: Any) -> str:
        """Render Jinja2 template with item data.

        Args:
            item: Input item to build context from

        Returns:
            Rendered template string
        """
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(self.text_field)

        # build context
        if isinstance(item, TrackedItem):
            context = {"content": item.content, **(item.metadata or {})}
        elif isinstance(item, dict):
            context = item.copy()
        elif hasattr(item, "model_dump"):
            # Pydantic model (e.g., Code, Theme)
            context = item.model_dump()
        elif hasattr(item, "__dict__"):
            # regular object with attributes
            context = vars(item).copy()
        else:
            context = {"content": str(item)}

        return template.render(**context)

    def result(self) -> Dict[str, Any]:
        """Extract results from this node."""
        result = super().result()

        if self.output:
            result["metadata"]["num_clusters"] = len(self.output)
            # only compute cluster stats when output is TrackedItems with cluster metadata
            first = self.output[0] if isinstance(self.output, list) and self.output else None
            if isinstance(first, TrackedItem) and "cluster_size" in first.metadata:
                sizes = [c.metadata["cluster_size"] for c in self.output]
                result["metadata"]["total_items"] = sum(sizes)
                if sizes:
                    result["metadata"]["cluster_sizes"] = {
                        "min": min(sizes),
                        "max": max(sizes),
                        "mean": float(np.mean(sizes)),
                    }

        return result

    def export(self, folder, unique_id: str = ""):
        """Export cluster details to folder."""
        from pathlib import Path

        super().export(folder, unique_id=unique_id)
        folder = Path(folder)

        if not self.output:
            return

        # when skip_below triggers in bypass mode, output is raw items (not TrackedItems)
        # -- nothing meaningful to export in that case
        first = self.output[0] if isinstance(self.output, list) and self.output else self.output
        if not isinstance(first, TrackedItem) or "cluster_size" not in getattr(first, "metadata", {}):
            return

        # write cluster summary
        sizes = [c.metadata["cluster_size"] for c in self.output]
        total_items = sum(sizes)
        summary = f"""Cluster Summary
===============
Total items: {total_items}
Number of clusters: {len(self.output)}
Cluster size min: {min(sizes) if sizes else 0}
Cluster size max: {max(sizes) if sizes else 0}
Cluster size mean: {(np.mean(sizes) if sizes else 0.0):.1f}

Method: {self.method.name}
"""
        if isinstance(self.method, HDBSCANMethod):
            summary += f"""max_cluster_size: {self.method.max_cluster_size}
min_cluster_size: {self.method.min_cluster_size}
min_cluster_size_proportion: {self.method.min_cluster_size_proportion}
min_samples: {self.method.min_samples}
effective_min_cluster_size: {self.method._effective_min_cluster_size}

Processing stats:
  Singletons (noise points): {self.method._singleton_count}
  Oversized clusters split: {self.method._oversized_count}

Per-cluster sizes:
"""
            # sort clusters by size descending, but keep original cluster_id
            sorted_clusters = sorted(
                self.output, key=lambda c: c.metadata["cluster_size"], reverse=True
            )
            for cluster in sorted_clusters:
                cluster_id = cluster.metadata["cluster_id"]
                size = cluster.metadata["cluster_size"]
                summary += f"  {cluster_id}: {size}\n"

        (folder / "cluster_summary.txt").write_text(summary)

        # export each cluster
        outputs_folder = folder / "outputs"
        outputs_folder.mkdir(exist_ok=True)

        for cluster in self.output:
            cluster_id = cluster.metadata["cluster_id"]
            cluster_items = cluster.metadata["items"]

            # write cluster content to top level (e.g., cluster_0_content.txt)
            (folder / f"{cluster_id}_content.txt").write_text(cluster.content)

            # individual items go in subfolder
            cluster_folder = outputs_folder / cluster_id
            cluster_folder.mkdir(exist_ok=True)

            for item_idx, item in enumerate(cluster_items):
                if isinstance(item, TrackedItem):
                    (cluster_folder / f"{item_idx:04d}_{item.safe_id}.txt").write_text(
                        item.content
                    )
                elif hasattr(item, "slug"):
                    # Code or similar with slug
                    (cluster_folder / f"{item_idx:04d}_{item.slug}.txt").write_text(
                        str(item)
                    )
                else:
                    (cluster_folder / f"{item_idx:04d}_item.txt").write_text(str(item))
