"""Filter node for boolean filtering of items using safe expressions."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anyio
import pandas as pd
from box import Box
from pydantic import Field, PrivateAttr
from simpleeval import (AttributeDoesNotExist, EvalWithCompoundTypes,
                        NameNotDefined)

from soak.error_handlers import managed_llm_call
from soak.events import NodeProgress
from soak.models.base import (TrackedItem, extract_content,
                              get_max_concurrency, get_semaphore,
                              safe_json_dump)

from .base import (CompletionDAGNode, ItemsNode, default_map_task,
                   render_strict_template)

logger = logging.getLogger(__name__)


class Filter(ItemsNode, CompletionDAGNode):
    """
    Filter items based on a boolean expression.

    Supports two modes:
    - llm: Run template through LLM, evaluate expression on extracted fields
    - simple: Evaluate expression directly on item data

    Mode is auto-detected: if template is provided, defaults to 'llm',
    otherwise defaults to 'simple'.

    Examples:
        # LLM mode - combines Map + Filter
        expression: "funny == True"
        template: "Is this funny? [[bool:funny]]"

        # Simple mode - filter on content length
        expression: "len(input) > 100"

        # Simple mode - filter on metadata
        expression: "category == 'relevant'"
    """

    model_config = {
        "discriminator": "type",
    }

    type: Literal["Filter"] = "Filter"
    expression: str = Field(
        ..., description="Boolean expression to filter items (e.g., 'funny == True')"
    )

    omitted_text: str = Field(
        default=" ... ",
        description="Text to insert when items are omitted; defaults to ellipsis (...).",
    )

    template: Optional[str] = Field(
        default=None,
        description="Optional LLM template for extraction before filtering",
    )
    mode: Optional[Literal["llm", "simple"]] = Field(
        default=None,
        description="Filter mode: 'llm' (use template + LLM) or 'simple' (evaluate on item data). Auto-detected if None.",
    )

    # statistics tracking
    _total_items: int = PrivateAttr(default=0)
    _included_count: int = PrivateAttr(default=0)
    _excluded_count: int = PrivateAttr(default=0)
    _all_input_items: List[Any] = PrivateAttr(default_factory=list)
    _included_items: List[Any] = PrivateAttr(default_factory=list)

    @property
    def effective_mode(self) -> str:
        """Determine the effective mode based on mode field and template."""
        if self.mode is not None:
            return self.mode
        return "llm" if self.template is not None else "simple"

    def _create_evaluator(self) -> EvalWithCompoundTypes:
        """Create an evaluator with safe defaults and attribute access enabled."""
        evaluator = EvalWithCompoundTypes()

        # allow safe builtins only
        evaluator.names = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "True": True,
            "False": False,
            "None": None,
        }

        return evaluator

    def _evaluate_expression(self, context: Dict[str, Any]) -> bool:
        """Safely evaluate expression with given context.

        Args:
            context: Dictionary of variables available to the expression

        Returns:
            Boolean result of expression evaluation (False on error)
        """
        evaluator = self._create_evaluator()

        try:
            # add context to evaluator
            evaluator.names.update(context)

            # evaluate expression
            result = evaluator.eval(self.expression)

            # convert to bool
            return bool(result)

        except NameNotDefined as e:
            logger.warning(f"Filter '{self.name}': Undefined name in expression: {e}")
            return False
        except AttributeDoesNotExist as e:
            logger.warning(
                f"Filter '{self.name}': Attribute does not exist in expression: {e}"
            )
            return False
        except Exception as e:
            logger.warning(f"Filter '{self.name}': Error evaluating expression: {e}")
            return False

    async def _process_llm_mode(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[Any]:
        """Process items using LLM template extraction + expression filtering.

        Args:
            items: Flat list of items to process in this batch
            progress_bar: Optional tqdm progress bar to update

        Returns:
            Filtered list of original items where expression evaluated to True
        """
        # convert items to Box items for templates
        boxed_items = []
        for item in items:
            if isinstance(item, TrackedItem):
                boxed_items.append(Box({"input": item.content, "tracked_item": item}))
            elif isinstance(item, dict):
                boxed_items.append(Box(item))
            else:
                boxed_items.append(Box({"input": item}))

        # filter context to remove BatchList objects
        from .batch import BatchList

        filtered_context = {
            k: v for k, v in self.context.items() if not isinstance(v, BatchList)
        }

        # run LLM for each item to extract fields
        llm_results = [None] * len(boxed_items)

        # Use passed-in progress bar if available (for batched execution)
        pbar = progress_bar

        logger.info(
            f"Filter '{self.name}': processing {len(boxed_items)} items "
            f"with concurrency={get_max_concurrency()}"
        )

        try:
            async with anyio.create_task_group() as tg:
                for idx, item in enumerate(boxed_items):

                    async def run_and_store(index=idx, item=item, progress_bar=pbar):
                        async with get_semaphore():
                            try:
                                extra_kwargs = self.get_llm_kwargs()
                                llm_results[index] = await managed_llm_call(
                                    node_name=self.name,
                                    config=self.dag.config,
                                    llm_func=default_map_task,
                                    item_index=index,
                                    template=self.template,
                                    context={**filtered_context, **item},
                                    model=self.get_model(),
                                    credentials=self.dag.config.llm_credentials,
                                    **extra_kwargs,
                                )
                                if llm_results[index] is not None:
                                    self._accumulate_costs(llm_results[index])
                                    self._llm_results.append(llm_results[index])
                                else:
                                    source_id = item.get("source_id", f"item {index}")
                                    logger.warning(
                                        f"Filter '{self.name}': skipped item {index} "
                                        f"(source: {source_id}) due to LLM error"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Filter '{self.name}': Error in LLM call for item {index}: {e}"
                                )
                                raise
                            finally:
                                if progress_bar is not None:
                                    progress_bar.update(
                                        getattr(progress_bar, "slots_per_item", 1)
                                    )

                                # emit progress for real-time updates
                                self.dag.emit(
                                    NodeProgress(
                                        self.name,
                                        len(self._llm_results),
                                        self._input_count,
                                        self._total_cost,
                                    )
                                )

                    tg.start_soon(run_and_store)
        finally:
            # Only close progress bar if we created it locally
            if progress_bar is None and pbar is not None:
                pbar.close()

        # now filter based on expression
        included = []
        for idx, (item, llm_result) in enumerate(zip(items, llm_results)):
            if llm_result is None:
                logger.debug(
                    f"Filter '{self.name}': Item {idx} has no LLM result, excluding"
                )
                self._excluded_count += 1
                continue

            # build evaluation context from LLM result
            # the expression references extracted fields directly (e.g., "funny == True")
            context = {}

            # add extracted fields from StruckdownResult
            if hasattr(llm_result, "response"):
                response = llm_result.response
                if isinstance(response, dict):
                    # Box or dict response - spread fields into context
                    context.update(response)
                elif hasattr(response, "__dict__"):
                    # object with attributes
                    context.update(response.__dict__)

            # add outputs if available
            if hasattr(llm_result, "outputs"):
                context.update(llm_result.outputs)

            # also make the full StruckdownResult available
            context["_complete_result"] = llm_result

            # evaluate expression
            eval_result = self._evaluate_expression(context)
            logger.debug(f"Filter '{self.name}': Item {idx} → {eval_result}")

            if eval_result:
                included.append(item)
                self._included_count += 1
            else:
                # create new TrackedItem to preserve original metadata
                if isinstance(item, TrackedItem):
                    filtered_item = TrackedItem(
                        content=self.omitted_text,
                        id=item.id,
                        sources=item.sources,
                        metadata={**(item.metadata or {}), "filtered": True},
                    )
                    included.append(filtered_item)
                else:
                    # non-TrackedItem, keep as-is
                    included.append(item)
                self._excluded_count += 1

        self._total_items += len(items)
        self._all_input_items.extend(items)
        self._included_items.extend(included)

        logger.debug(
            f"Filter '{self.name}' (LLM mode): {len(included)}/{len(items)} items included"
        )

        return included

    async def _process_simple_mode(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[Any]:
        """Process items using direct expression evaluation on item data.

        Args:
            items: Flat list of items to process in this batch
            progress_bar: Optional tqdm progress bar to update

        Returns:
            Filtered list of items where expression evaluated to True
        """
        included = []

        for idx, item in enumerate(items):
            # build evaluation context from item
            context = {}

            if isinstance(item, TrackedItem):
                # make content available as 'input'
                context["input"] = item.content

                # spread metadata into context
                if item.metadata:
                    context.update(item.metadata)

                # also provide full item
                context["tracked_item"] = item
            elif isinstance(item, dict):
                # dict item - spread into context
                context.update(item)
                # also make available as 'input' if there's a content-like field
                if "content" in item:
                    context["input"] = item["content"]
            else:
                # plain value - make available as 'input'
                context["input"] = item

            # evaluate expression
            eval_result = self._evaluate_expression(context)
            logger.debug(f"Filter '{self.name}': Item {idx} → {eval_result}")

            if eval_result:
                included.append(item)
                self._included_count += 1
            else:
                # create new TrackedItem to preserve original metadata
                if isinstance(item, TrackedItem):
                    filtered_item = TrackedItem(
                        content=self.omitted_text,
                        id=item.id,
                        sources=item.sources,
                        metadata={**(item.metadata or {}), "filtered": True},
                    )
                    included.append(filtered_item)
                else:
                    # non-TrackedItem, keep as-is
                    included.append(item)
                self._excluded_count += 1

            # Update progress bar after processing each item
            if progress_bar is not None:
                progress_bar.update(getattr(progress_bar, "slots_per_item", 1))

        self._total_items += len(items)
        self._all_input_items.extend(items)
        self._included_items.extend(included)

        logger.debug(
            f"Filter '{self.name}' (simple mode): {len(included)}/{len(items)} items included"
        )

        return included

    async def process_items(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[Any]:
        """Filter items based on mode and expression.

        Args:
            items: Flat list of items to process in this batch
            progress_bar: Optional tqdm progress bar to update

        Returns:
            Filtered list of items where expression evaluated to True
        """
        # Track input count for progress reporting
        self._input_count = len(items)

        mode = self.effective_mode

        if mode == "llm":
            return await self._process_llm_mode(items, progress_bar)
        else:
            return await self._process_simple_mode(items, progress_bar)

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and filter statistics."""
        result = super().result()

        result["metadata"]["mode"] = self.effective_mode
        result["metadata"]["expression"] = self.expression
        result["metadata"]["total_items"] = self._total_items
        result["metadata"]["included_count"] = self._included_count
        result["metadata"]["excluded_count"] = self._excluded_count

        # build DataFrame of included items for HTML display
        included_rows = []
        for idx, item in enumerate(self._included_items):
            if isinstance(item, TrackedItem):
                content = item.content
                source_id = item.id
            elif isinstance(item, dict) and "content" in item:
                content = item["content"]
                source_id = item.get("id", "N/A")
            else:
                content = str(item)
                source_id = "N/A"

            # truncate content for display
            display_content = content[:100] + "..." if len(content) > 100 else content

            included_rows.append(
                {
                    "index": idx,
                    "source_id": source_id,
                    "content": display_content,
                }
            )

        result["included_items_df"] = (
            pd.DataFrame(included_rows) if included_rows else pd.DataFrame()
        )

        # stats DataFrame for display
        stats_data = {
            "mode": [self.effective_mode],
            "expression": [self.expression],
            "total_items": [self._total_items],
            "included_count": [self._included_count],
            "excluded_count": [self._excluded_count],
        }
        result["stats_df"] = pd.DataFrame(stats_data)

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Filter node with statistics and filtered items."""
        super().export(folder, unique_id=unique_id)

        # write mode and expression
        (folder / "mode.txt").write_text(self.effective_mode)
        (folder / "expression.txt").write_text(self.expression)

        # write template if in LLM mode
        if self.template:
            (folder / "template.sd").write_text(self.template)

        # write statistics
        stats = {
            "mode": self.effective_mode,
            "expression": self.expression,
            "total_items": self._total_items,
            "included_count": self._included_count,
            "excluded_count": self._excluded_count,
        }

        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(folder / "filter_stats.csv", index=False)

        # export included items to outputs/ folder
        if self._included_items:
            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(self._included_items):
                # get item ID for filename
                if isinstance(item, TrackedItem):
                    item_id = item.safe_id
                    content = item.content
                else:
                    item_id = f"item_{idx}"
                    content = str(item)

                output_file = outputs_folder / f"{idx:04d}_{item_id}.txt"
                output_file.write_text(content)

                # export metadata if TrackedItem
                if isinstance(item, TrackedItem) and item.metadata:
                    metadata_file = (
                        outputs_folder / f"{idx:04d}_{item_id}_metadata.json"
                    )
                    metadata_file.write_text(item.to_json())
