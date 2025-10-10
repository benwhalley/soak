"""TransformReduce node for hierarchical reduction with LLM."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import anyio
from pydantic import Field
from struckdown import chatter_async

from ..base import extract_prompt, get_action_lookup, safe_json_dump, semaphore
from ..dag import OutputUnion, render_strict_template
from .base import CompletionDAGNode

logger = logging.getLogger(__name__)


class TransformReduce(CompletionDAGNode):
    """
    Recursively reduce a list into a single item by transforming it with an LLM template.

    If inputs are ChatterResult with specific keys, then the TransformReduce must produce a ChatterResult with the same keys.

    """

    type: Literal["TransformReduce"] = "TransformReduce"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences", "paragraphs"] = (
        "tokens"
    )
    encoding_name: str = "cl100k_base"
    reduce_template: str = "{{input}}\n\n"
    template_text: str = (
        "<text>\n{{input}}\n</text>\n\n-----\nSummarise the text: [[output]]"
    )
    max_levels: int = 5

    reduction_tree: List[List[OutputUnion]] = Field(default_factory=list, exclude=False)

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    def tokenize(self, doc: str) -> List[Union[int, str]]:
        if self.split_unit == "tokens":
            return self.token_encoder.encode(doc)
        elif self.split_unit == "sentences":
            import nltk

            return nltk.sent_tokenize(doc)
        elif self.split_unit == "words":
            import nltk

            return nltk.word_tokenize(doc)
        elif self.split_unit == "paragraphs":
            from nltk.tokenize import BlanklineTokenizer

            return BlanklineTokenizer().tokenize(doc)
        else:
            raise ValueError(f"Unsupported tokenization method: {self.split_unit}")

    def _compute_spans(self, n: int) -> List[Tuple[int, int]]:
        if n <= self.chunk_size:
            return [(0, n)]
        n_chunks = max(1, math.ceil(n / self.chunk_size))
        target = max(self.min_split, math.ceil(n / n_chunks))
        spans = []
        start = 0
        for _ in range(n_chunks - 1):
            end = min(n, start + target)
            spans.append((start, end))
            start = max(0, end - self.overlap)
        spans.append((start, n))
        return spans

    def chunk_document(self, doc: str) -> List[str]:
        if self.split_unit == "chars":
            if len(doc) <= self.chunk_size:
                return [doc.strip()]
            spans = self._compute_spans(len(doc))
            return [doc[start:end].strip() for start, end in spans]
        tokens = self.tokenize(doc)
        spans = self._compute_spans(len(tokens))

        if self.split_unit == "tokens":
            return [
                self.token_encoder.decode(tokens[start:end]).strip()
                for start, end in spans
            ]
        else:
            return [" ".join(tokens[start:end]).strip() for start, end in spans]

    def chunk_items(self, items: List[str]) -> List[str]:
        # import pdb; pdb.set_trace()
        joined = "\n".join(
            [
                render_strict_template(
                    self.reduce_template, {**self.context, **i.outputs, "input": i}
                )
                for i in items
            ]
        )
        return self.chunk_document(joined)

    def batch_by_units(self, items: List[str]) -> List[List[str]]:
        batches = []
        current_batch = []
        current_units = 0

        for item in items:
            if self.split_unit == "chars":
                item_len = len(item)
            elif self.split_unit == "tokens":
                item_len = len(self.token_encoder.encode(item))
            elif self.split_unit == "words":
                item_len = len(item.split())
            elif self.split_unit == "sentences":
                import nltk

                item_len = len(nltk.sent_tokenize(item))
            elif self.split_unit == "paragraphs":
                from nltk.tokenize import BlanklineTokenizer

                item_len = len(BlanklineTokenizer().tokenize(item))
            else:
                raise ValueError(f"Unknown split_unit: {self.split_unit}")

            # If adding this item would exceed batch size, flush
            if current_units + item_len > self.chunk_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_units = 0

            current_batch.append(item)
            current_units += item_len

        if current_batch:
            batches.append(current_batch)

        return batches

    async def run(self) -> str:
        await super().run()

        raw_items = self.context[self.inputs[0]]

        if isinstance(raw_items, str):
            raw_items = [raw_items]
        elif not isinstance(raw_items, list):
            raise ValueError("Input to TransformReduce must be a list of strings")

        # Initial chunking
        current = self.chunk_items(raw_items)

        self.reduction_tree = [current]

        level = 0
        nbatches = len(current)
        logger.info(f"Starting with {nbatches} batches")

        while len(current) > 1:
            level += 1
            logger.warning(f"TransformReduce level: {level}")
            if level > self.max_levels:
                raise RuntimeError("Exceeded max cascade depth")

            # Chunk input into batches
            batches = self.batch_by_units(current)
            if len(batches) > (nbatches and nbatches or 1e10):
                import pdb

                pdb.set_trace()
                raise Exception(
                    f"Batches {len(batches)} equals or exceeded original batch size {nbatches} so likely won't ever converge to 1 item. Try increasing the chunk_size or rewriting the prompt to be more concise."
                )
            nbatches = len(batches)

            logger.info(f"Cascade Reducing {len(batches)} batches")
            results: List[Any] = [None] * len(batches)

            async with anyio.create_task_group() as tg:
                for idx, batch in enumerate(batches):

                    async def run_and_store(index=idx, batch=batch):
                        prompt = render_strict_template(
                            self.template_text, {**self.context, "input": batch}
                        )
                        # Collect extra kwargs for LLM
                        extra_kwargs = {}
                        if self.max_tokens is not None:
                            extra_kwargs["max_tokens"] = self.max_tokens
                        extra_kwargs["temperature"] = self.temperature
                        extra_kwargs["seed"] = self.dag.config.seed

                        async with semaphore:
                            results[index] = await chatter_async(
                                multipart_prompt=prompt,
                                model=self.get_model(),
                                credentials=self.dag.config.llm_credentials,
                                action_lookup=get_action_lookup(),
                                extra_kwargs=extra_kwargs,
                            )

                    tg.start_soon(run_and_store)

            # Extract string results and re-chunk for next level
            current = self.chunk_items(results)
            self.reduction_tree.append(current)

        final_prompt = render_strict_template(
            self.template_text, {"input": current[0], **self.context}
        )

        # Collect extra kwargs for LLM
        extra_kwargs = {}
        if self.max_tokens is not None:
            extra_kwargs["max_tokens"] = self.max_tokens
        extra_kwargs["temperature"] = self.temperature
        extra_kwargs["seed"] = self.dag.config.seed

        final_response = await chatter_async(
            multipart_prompt=final_prompt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs=extra_kwargs,
        )

        self.output = final_response
        return final_response

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata, reduction tree structure and final result."""
        # Get base metadata from parent
        result = super().result()

        # Build tree structure with DataFrames per level
        tree_dfs = []

        for level_idx, level_items in enumerate(self.reduction_tree):
            level_rows = []
            for item_idx, item in enumerate(level_items):
                level_rows.append(
                    {
                        "level": level_idx,
                        "item_idx": item_idx,
                        "text": str(item),
                        "response_obj": (
                            item.response if hasattr(item, "response") else None
                        ),
                        "chatter_result": item if hasattr(item, "results") else None,
                    }
                )
            tree_dfs.append(pd.DataFrame(level_rows))

        # Add TransformReduce-specific data
        result["tree_levels"] = tree_dfs
        result["final_prompt"] = extract_prompt(self.output)
        result["final_response"] = (
            self.output.response if hasattr(self.output, "response") else None
        )
        result["final_chatter_result"] = self.output
        result["metadata"]["num_levels"] = len(tree_dfs)
        result["metadata"]["batch_size"] = self.batch_size

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export TransformReduce node with multi-level structure."""
        super().export(folder, unique_id=unique_id)

        # Write templates
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)
        if self.reduce_template:
            (folder / "reduce_template.md").write_text(self.reduce_template)

        # Export each level of the reduction tree
        if self.reduction_tree:
            for level_idx, level_items in enumerate(self.reduction_tree):
                level_folder = folder / f"level_{level_idx}"
                level_folder.mkdir(exist_ok=True)

                for item_idx, item in enumerate(level_items, 1):
                    try:
                        # Write the item as text
                        item_text = str(item)
                        (level_folder / f"{item_idx:03d}_item.txt").write_text(
                            item_text
                        )

                        # If it's a ChatterResult, export it fully
                        if hasattr(item, "results"):
                            (level_folder / f"{item_idx:03d}_result.json").write_text(
                                safe_json_dump(item)
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to export TransformReduce level {level_idx} item {item_idx}: {e}"
                        )

        # Export final result
        if self.output:
            final_folder = folder / "final"
            final_folder.mkdir(exist_ok=True)

            try:
                if hasattr(self.output, "results") and self.output.results:
                    first_seg = next(iter(self.output.results.values()))
                    if hasattr(first_seg, "prompt"):
                        (final_folder / "prompt.md").write_text(first_seg.prompt)

                if hasattr(self.output, "response"):
                    (final_folder / "response.txt").write_text(
                        str(self.output.response)
                    )

                (final_folder / "response.json").write_text(safe_json_dump(self.output))
            except Exception as e:
                logger.warning(f"Failed to export TransformReduce final result: {e}")
