"""Base node classes for DAG execution."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import anyio
import numpy as np
import pandas as pd
import tiktoken
import nltk
from box import Box
from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field
from struckdown import LLM, ChatterResult, chatter_async
from struckdown.parsing import parse_syntax

from ..dag import DAG, OutputUnion, render_strict_template
from ..base import TrackedItem, extract_content, get_action_lookup

logger = logging.getLogger(__name__)


class DAGNode(BaseModel):
    # this used for reserialization
    model_config = {"discriminator": "type"}
    type: str = Field(default_factory=lambda self: type(self).__name__, exclude=False)

    dag: Optional["DAG"] = Field(default=None, exclude=True)

    name: str
    inputs: Optional[List[str]] = []
    template_text: Optional[str] = None
    output: Optional[OutputUnion] = Field(default=None)

    def get_model(self):
        model_name = getattr(self, 'model_name', None)
        if model_name:
            m = LLM(model_name=model_name)
            return m
        return self.dag.config.get_model()

    def validate_template(self):
        try:
            Environment().parse(self.template_text)
            return True
        except TemplateSyntaxError as e:
            raise e
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self, items: List[Any] = None) -> List[Any]:
        logger.debug(f"\n\nRunning `{self.name}` ({self.__class__.__name__})\n\n")

    @property
    def context(self) -> Dict[str, Any]:
        ctx = self.dag.default_context.copy()

        # merge in extra_context from config (includes persona, research_question, etc.)
        ctx.update(self.dag.config.extra_context)

        # if there are no inputs, assume we'll be using input documents
        if not self.inputs:
            self.inputs = ["documents"]

        if "documents" in self.inputs:
            ctx["documents"] = self.dag.config.load_documents()

        prev_nodes = {k: self.dag.nodes_dict.get(k) for k in self.inputs}
        prev_output = {
            k: v.output for k, v in prev_nodes.items() if v and v.output is not None
        }
        ctx.update(prev_output)

        return ctx

    @property
    def template(self) -> str:
        return self.template_text

    def get_input_items(self) -> Optional[List[Any]]:
        """Get the input items that were processed by this node."""
        if not self.inputs or not self.dag:
            return None

        # Get the first input (most common case)
        input_name = self.inputs[0]
        if input_name == "documents":
            return self.dag.config.documents
        elif input_name in self.dag.nodes_dict:
            input_node = self.dag.nodes_dict[input_name]
            return input_node.output if input_node.output else None

        return None

    def result(self) -> Dict[str, Any]:
        """Extract results from this node. Override in subclasses for node-specific results."""
        metadata = {
            "name": self.name,
            "type": self.type,
            "inputs": self.inputs,
        }

        # Add template text if available
        if hasattr(self, "template_text") and self.template_text:
            metadata["template_text"] = self.template_text

        # Add model info if available
        if hasattr(self, "model_name") and self.model_name:
            metadata["model_name"] = self.model_name
        if hasattr(self, "temperature") and self.temperature is not None:
            metadata["temperature"] = self.temperature

        return {
            "metadata": metadata,
        }

    def export(self, folder: Path, unique_id: str = ""):
        """Export node execution details to a folder. Override in subclasses.

        Args:
            folder: Folder to export to
            unique_id: Optional unique identifier to append to file names
        """
        folder.mkdir(parents=True, exist_ok=True)

        # Write comprehensive node config including all fields and defaults
        config_data = self.model_dump(
            exclude={
                "output",
                "context",
                "dag",
                "_processed_items",
                "_model_results",
                "_agreement_stats",
            },
            exclude_none=False,  # Include fields with None to show defaults
        )

        # Add type info for clarity
        config_text = f"""Node Configuration
==================
Type: {self.type}
Name: {self.name}

Parameters:
"""
        for key, value in sorted(config_data.items()):
            if key not in ["name", "type"]:
                config_text += f"  {key}: {value}\n"

        (folder / "meta.txt").write_text(config_text)

        # Export input items for traceability
        input_items = self.get_input_items()
        if input_items and isinstance(input_items, list):
            inputs_folder = folder / "inputs"
            inputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(input_items):
                if isinstance(item, TrackedItem):
                    # Export with source_id in filename
                    (inputs_folder / f"{idx:04d}_{item.safe_id}.txt").write_text(
                        item.content
                    )

                    # Export metadata
                    if item.metadata:
                        (
                            inputs_folder / f"{idx:04d}_{item.safe_id}_metadata.json"
                        ).write_text(json.dumps(item.metadata, indent=2, default=str))
                elif isinstance(item, str):
                    # Backward compatibility
                    (inputs_folder / f"{idx:04d}_input.txt").write_text(item)
                else:
                    # Try to convert to string
                    try:
                        (inputs_folder / f"{idx:04d}_input.txt").write_text(str(item))
                    except Exception as e:
                        logger.warning(f"Failed to export input {idx}: {e}")


class CompletionDAGNode(DAGNode):
    model_name: Optional[str] = None
    temperature: float = 1
    max_tokens: Optional[int] = None

    async def run(self, items: List[Any] = None) -> List[Any]:
        await super().run()

    def output_keys(self) -> List[str]:
        """Return the list of output keys provided by this node."""
        try:
            sections = parse_syntax(self.template)
            keys = []
            for section in sections:
                keys.extend(section.keys())
            return keys or [self.name]
        except Exception as e:
            logger.warning(f"Failed to parse template for output keys: {e}")
            return ["input"]


class Split(DAGNode):
    type: Literal["Split"] = "Split"

    name: str = "chunks"
    template_text: str = "{{input}}"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = (
        "tokens"
    )
    encoding_name: str = "cl100k_base"

    @property
    def template(self):
        return None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.inputs) > 1:
            raise ValueError("Split node can only have one input")

        if self.chunk_size < self.min_split:
            logger.warning(
                f"Chunk size must be larger than than min_split. Setting min_split to chunk_size // 2 = {self.chunk_size // 2}"
            )
            self.min_split = self.chunk_size // 2

    async def run(self) -> List[Union[str, "TrackedItem"]]:
        import numpy as np

        await super().run()
        input_docs = self.context[self.inputs[0]]

        # Split each document and track provenance
        all_chunks = []
        for doc in input_docs:
            if isinstance(doc, TrackedItem):
                chunks = self.split_tracked_document(doc)
            else:
                # Backward compatibility: wrap plain strings
                temp_tracked = TrackedItem(
                    content=doc, source_id="unknown_doc", metadata={}
                )
                chunks = self.split_tracked_document(temp_tracked)
            all_chunks.extend(chunks)

        self.output = all_chunks

        # Calculate stats on content
        lens = [
            len(
                self.tokenize(
                    chunk.content if isinstance(chunk, TrackedItem) else chunk,
                    method=self.split_unit,
                )
            )
            for chunk in self.output
        ]
        logger.debug(
            f"CREATED {len(self.output)} chunks; average length ({self.split_unit}): {np.mean(lens).round(1)}, max: {max(lens)}, min: {min(lens)}."
        )

        return self.output

    def split_tracked_document(self, doc: TrackedItem) -> List[TrackedItem]:
        """Split a TrackedItem into multiple TrackedItems with nested source_ids.

        The node name is included in the source_id to track which Split node created the chunk.
        Format: parent_source_id__nodename__index
        Example: doc_A__sentences__0 (first chunk from 'sentences' Split node)
        """
        text_chunks = self.split_document(doc.content)

        tracked_chunks = []
        for idx, chunk in enumerate(text_chunks):
            # Include node name in source_id for tracking which split created this
            new_source_id = f"{doc.source_id}__{self.name}__{idx}"

            tracked_chunks.append(
                TrackedItem(
                    content=chunk,
                    source_id=new_source_id,
                    metadata={
                        **doc.metadata,
                        "split_index": idx,
                        "split_node": self.name,
                        "parent_source_id": doc.source_id,
                    },
                )
            )

        return tracked_chunks

    def split_document(self, doc: str) -> List[str]:
        """Split a document string into chunks (backward compatibility)."""
        if self.split_unit == "chars":
            return self._split_by_length(doc, len_fn=len, overlap=self.overlap)

        tokens = self.tokenize(doc, method=self.split_unit)
        spans = self._compute_spans(
            len(tokens), self.chunk_size, self.min_split, self.overlap
        )
        return [
            self._format_chunk(tokens[start:end]) for start, end in spans if end > start
        ]

    def tokenize(self, doc: str, method: str) -> List[Union[int, str]]:
        if method == "tokens":
            return self.token_encoder.encode(doc)

        elif method == "sentences":
            from pysbd import Segmenter
            seg = Segmenter(language="en", clean=True)
            return seg.segment(doc)
        elif method == "words":
            import nltk

            return nltk.word_tokenize(doc)
        elif method == "paragraphs":
            from nltk.tokenize import BlanklineTokenizer

            return BlanklineTokenizer().tokenize(doc)
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")

    def _compute_spans(
        self, n: int, chunk_size: int, min_split: int, overlap: int
    ) -> List[Tuple[int, int]]:
        if n <= chunk_size:
            return [(0, n)]

        n_chunks = max(1, math.ceil(n / chunk_size))
        target = max(min_split, math.ceil(n / n_chunks))
        spans = []
        start = 0
        for _ in range(n_chunks - 1):
            end = min(n, start + target)
            spans.append((start, end))
            start = max(0, end - overlap)
        spans.append((start, n))
        return spans

    def _format_chunk(self, chunk_tokens: List[Union[int, str]]) -> str:
        if not chunk_tokens:
            return ""
        if self.split_unit == "tokens":
            return self.token_encoder.decode(chunk_tokens).strip()
        elif self.split_unit in {"words", "sentences"}:
            return " ".join(chunk_tokens).strip()
        elif self.split_unit == "paragraphs":
            return "\n\n".join(chunk_tokens).strip()
        else:
            raise ValueError(
                f"Unexpected split_unit in format_chunk: {self.split_unit}"
            )

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and DataFrame of chunks with length statistics."""
        # Get base metadata from parent
        result = super().result()

        if not self.output:
            logger.warning(f"Split node '{self.name}' has no output to create statistics from")
            result["chunks_df"] = pd.DataFrame()
            result["summary_stats"] = {}
            result["metadata"]["split_unit"] = self.split_unit
            result["metadata"]["chunk_size"] = self.chunk_size
            result["metadata"]["num_chunks"] = 0
            return result

        # Debug: log what types we're seeing
        type_counts = {}
        for chunk in self.output:
            type_name = type(chunk).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        logger.debug(f"Split node '{self.name}' output types: {type_counts}")

        # Build DataFrame of chunks with comprehensive statistics
        rows = []
        for idx, chunk in enumerate(self.output):
            # Handle both TrackedItem objects and deserialized dicts
            if isinstance(chunk, dict):
                # Deserialized from JSON
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {}) or {}
                source_id = chunk.get("source_id", "")
            elif isinstance(chunk, TrackedItem):
                # Live object
                content = chunk.content
                metadata = chunk.metadata or {}
                source_id = chunk.source_id
            else:
                # Try extract_content as fallback, but this likely won't work for wrong types
                content = extract_content(chunk)
                if not content:
                    logger.debug(f"Skipping chunk {idx} - wrong type: {type(chunk).__name__}")
                    continue
                metadata = getattr(chunk, "metadata", {}) or {}
                source_id = getattr(chunk, "source_id", "")

            if not content:
                logger.debug(f"Skipping chunk {idx} - no content")
                continue

            # Extract filename from metadata or source_id
            filename = metadata.get("filename", "")
            if not filename:
                # Try to extract filename from source_id (format: filename__node__index)
                if source_id and "__" in source_id:
                    filename = source_id.split("__")[0]
                else:
                    filename = source_id or "unknown"

            # Get chunk index from metadata
            chunk_index = metadata.get("split_index", idx)

            # Compute various length statistics
            length_chars = len(content)
            length_words = len(content.split())

            # Tokenize to get token count using tiktoken (OpenAI tokenizer)
            try:
                length_tokens = len(self.token_encoder.encode(content))
            except Exception as e:
                logger.debug(f"Failed to tokenize chunk {idx}: {e}")
                length_tokens = 0

            # Count sentences using nltk
            try:
                from pysbd import Segmenter
                seg = Segmenter(language="en", clean=True)
                length_sentences = len(seg.segment(content))
            except Exception as e:
                logger.debug(f"Failed to count sentences in chunk {idx}: {e}")
                length_sentences = 0

            # Count paragraphs (split on blank lines)
            length_paragraphs = len([p for p in content.split("\n\n") if p.strip()])

            rows.append(
                {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "length_chars": length_chars,
                    "length_words": length_words,
                    "length_sentences": length_sentences,
                    "length_paragraphs": length_paragraphs,
                    "length_tokens": length_tokens,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            logger.warning(f"Split node '{self.name}' created empty DataFrame from {len(self.output)} chunks - check if output is the correct type")
        else:
            logger.debug(f"Split node '{self.name}' created DataFrame with {len(df)} rows")

        # Compute summary statistics
        summary_stats = {}
        if not df.empty:
            for col in ["length_chars", "length_words", "length_sentences", "length_paragraphs", "length_tokens"]:
                if col in df.columns:
                    summary_stats[col] = {
                        "min": int(df[col].min()),
                        "max": int(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                    }

        # Add Split-specific data
        result["chunks_df"] = df
        result["summary_stats"] = summary_stats
        result["metadata"]["split_unit"] = self.split_unit
        result["metadata"]["chunk_size"] = self.chunk_size
        result["metadata"]["num_chunks"] = len(self.output)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Split node details."""
        super().export(folder, unique_id=unique_id)

        if self.output:
            import numpy as np

            # Extract content from TrackedItems for statistics
            lens = []
            for doc in self.output:
                # Handle TrackedItem, string, or dict (from JSON deserialization)
                if isinstance(doc, TrackedItem):
                    content = doc.content
                elif isinstance(doc, str):
                    content = doc
                elif isinstance(doc, dict) and "content" in doc:
                    content = doc["content"]
                else:
                    # Skip items that can't be processed (e.g., ChatterResult from wrong node type)
                    logger.debug(
                        f"Skipping non-text output in Split export: {type(doc)}"
                    )
                    continue

                if content and isinstance(content, str):
                    lens.append(len(self.tokenize(content, method=self.split_unit)))

            if lens:
                summary = f"""Split Summary
==============
Chunks created: {len(self.output)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
Average length: {np.mean(lens).round(1)}
Max length: {max(lens)}
Min length: {min(lens)}
"""
            else:
                summary = f"""Split Summary
==============
Chunks created: {len(self.output)}
Split unit: {self.split_unit}
Chunk size: {self.chunk_size}
(Statistics unavailable - output not in expected format)
"""
            (folder / "split_summary.txt").write_text(summary)

            # Export output chunks with source_id naming
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, chunk in enumerate(self.output):
                if isinstance(chunk, TrackedItem):
                    # Use source_id in filename
                    (outputs_folder / f"{idx:04d}_{chunk.safe_id}.txt").write_text(
                        chunk.content
                    )

                    # Export metadata if present
                    if chunk.metadata:
                        (
                            outputs_folder / f"{idx:04d}_{chunk.safe_id}_metadata.json"
                        ).write_text(json.dumps(chunk.metadata, indent=2, default=str))
                else:
                    # Backward compatibility: plain strings
                    (outputs_folder / f"{idx:04d}_chunk.txt").write_text(str(chunk))


class ItemsNode(DAGNode):
    """Any note which applies to multiple items at once"""

    async def get_items(self) -> List[Dict[str, Any]]:
        """Resolve all inputs, then zip together, combining multiple inputs.

        Handles TrackedItem transparently: extracts content for templates while
        preserving source_id and metadata for provenance tracking.
        """
        # resolve futures now (it's lazy to this point)
        input_data = {k: self.context[k] for k in self.inputs}

        lengths = {
            k: len(v) if isinstance(v, list) else 1 for k, v in input_data.items()
        }
        max_len = max(lengths.values())

        for k, v in input_data.items():
            if isinstance(v, list):
                if len(v) != max_len:
                    raise ValueError(
                        f"Length mismatch for input '{k}': expected {max_len}, got {len(v)}"
                    )
            else:
                input_data[k] = [v] * max_len

        zipped = list(zip(*[input_data[k] for k in self.inputs]))

        # make the first input available as {{input}} in any template
        items = []
        for idx, values in enumerate(zipped):
            item_dict = {}

            for key, val in zip(self.inputs, values):
                if isinstance(val, TrackedItem):
                    # Extract content for template, preserve metadata
                    item_dict[key] = val.content
                    item_dict[f"__{key}__source_id"] = val.source_id
                    item_dict[f"__{key}__metadata"] = val.metadata
                    item_dict[f"__{key}__tracked_item"] = val  # Keep reference
                else:
                    item_dict[key] = val

            # Make first input available as {{input}}
            if self.inputs:
                first_val = values[0]
                if isinstance(first_val, TrackedItem):
                    item_dict["input"] = first_val.content
                    item_dict["source_id"] = first_val.source_id
                    item_dict["metadata"] = first_val.metadata
                    item_dict["tracked_item"] = first_val
                else:
                    item_dict["input"] = first_val

            items.append(Box(item_dict))

        return items


async def default_map_task(template, context, model, credentials, **kwargs):
    """Default map task renders the Step template for each input item and calls the LLM."""

    rt = render_strict_template(template, context)

    # call chatter as async function within the main event loop
    result = await chatter_async(
        multipart_prompt=rt,
        context=context,
        model=model,
        credentials=credentials,
        action_lookup=get_action_lookup(),
        extra_kwargs=kwargs,
    )
    return result
