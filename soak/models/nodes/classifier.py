"""Classifier node for multi-model classification with agreement metrics."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import anyio
import pandas as pd
from pydantic import Field, PrivateAttr
from struckdown import LLM, LLMError, chatter_async
from struckdown.parsing import parse_syntax

from soak.error_handlers import managed_llm_call
from soak.models.base import TrackedItem, get_action_lookup, semaphore
from soak.models.utils import extract_output_dict

from .base import CompletionDAGNode, ItemsNode

logger = logging.getLogger(__name__)


class Classifier(ItemsNode, CompletionDAGNode):
    """
    Apply a classification prompt to each input item and extract structured outputs.

    Returns a list of dictionaries where each dict contains the classification results
    for one input item. Fields are extracted from [[name]] and [[pick:name|...]] syntax
    in the template.

    Example output: [{"diagnosis": "cfs", "severity": "high"}, ...]

    If model_names is provided with multiple models, runs classification with each model
    and calculates inter-rater agreement statistics.
    """

    type: Literal["Classifier"] = "Classifier"
    temperature: float = 0.5
    template: str = None
    model_names: Optional[List[str]] = None  # Multiple models for agreement analysis
    agreement_fields: Optional[List[str]] = None  # Fields to calculate agreement on
    ground_truths: Optional[Dict[str, Dict[str, Any]]] = (
        None  # Ground truth comparison config
    )

    # Private attributes - automatically excluded from serialization
    _processed_items: Optional[List[Any]] = PrivateAttr(default=None)
    _model_results: Optional[Dict[str, List[Any]]] = PrivateAttr(default=None)
    _agreement_stats: Optional[Dict[str, Dict[str, float]]] = PrivateAttr(default=None)

    def validate_template(self):
        try:
            parse_syntax(self.template)
            return True
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Process each item through the classification template.

        Returns:
            If single model: List of ChatterResults
            If multiple models: Dict mapping model_name -> List of ChatterResults
        """
        # Import here to avoid circular import
        from .batch import BatchList

        input_data = self.context[self.inputs[0]] if self.inputs else None
        if not input_data:
            raise Exception("Classifier node must have input data")

        if isinstance(input_data, BatchList):
            raise Exception("Classifier node does not support batch input")

        items = await self.get_items()
        filtered_context = self.context

        # Track input count for progress reporting
        self._input_count = len(items)

        # Store items for export to access TrackedItem metadata
        self._processed_items = items

        # Normalize model_names to always be a list
        if not self.model_names:
            self.model_names = (
                [self.model_name] if self.model_name else [self.dag.config.model_name]
            )

        # Initialize result storage
        self._model_results = {}

        # Run classification for each model
        for model_name in self.model_names:
            logger.debug(f"Running classification with model: {model_name}")

            results = [None] * len(items)

            # Use progress bar context manager - customize description for multi-model case
            with self.progress_bar(
                items, node_type=f"Classifier [{model_name}]"
            ) as pbar:
                async with anyio.create_task_group() as tg:
                    for idx, item in enumerate(items):

                        async def run_and_store(
                            index=idx,
                            item=item,
                            current_model=model_name,
                            progress_bar=pbar,
                        ):
                            async with semaphore:
                                # Create model instance for this specific model
                                model = LLM(model_name=current_model)

                                # Get LLM kwargs using helper method
                                extra_kwargs = self.get_llm_kwargs()

                                try:
                                    chatter_result = await managed_llm_call(
                                        node_name=self.name,
                                        config=self.dag.config,
                                        llm_func=chatter_async,
                                        item_index=index,
                                        multipart_prompt=self.template,
                                        context={**filtered_context, **item},
                                        model=model,
                                        credentials=self.dag.config.llm_credentials,
                                        extra_kwargs=extra_kwargs,
                                    )
                                    results[index] = chatter_result
                                except Exception as e:
                                    # catch-all for any non-struckdown errors
                                    logger.error(
                                        f"Unexpected error in node '{self.name}' for item {index}: {e}"
                                    )
                                    # default to skip + continue for unknown errors
                                    results[index] = None
                                finally:
                                    # Update progress bar on completion (success or failure)
                                    if progress_bar is not None:
                                        progress_bar.update(
                                            getattr(progress_bar, "slots_per_item", 1)
                                        )

                        tg.start_soon(run_and_store)

            # accumulate costs and track for cache statistics
            for result in results:
                if result is not None:
                    self._accumulate_costs(result)
                    self._llm_results.append(result)

                    # update progress bar with per-node cost if using CostProgressBar
                    from soak.models.progress import CostProgressBar

                    if isinstance(pbar, CostProgressBar):
                        pbar.update_cost(
                            result.fresh_cost,
                            result.prompt_tokens + result.completion_tokens,
                        )

                    # call progress callback for web UI if available
                    if self.dag.config.progress_callback:
                        try:
                            self.dag.config.progress_callback(
                                self.name,
                                len(self._llm_results),
                                self._input_count,
                                self._total_cost,
                            )
                        except Exception:
                            pass  # don't let callback errors affect execution

            self._model_results[model_name] = results

        # Auto-detect agreement fields if multiple models (before result() is called)
        if len(self.model_names) >= 2:
            self._auto_detect_agreement_fields()

        # Set output: single model returns list, multiple models return dict
        if len(self.model_names) == 1:
            self.output = self._model_results[self.model_names[0]]
        else:
            self.output = self._model_results

        return self.output

    def _build_dataframes_from_results(self):
        """Build DataFrames from in-memory results for agreement calculation."""
        model_dfs = {}
        for model_name, results in self._model_results.items():
            # Ensure results is a list, not a generator
            if not isinstance(results, list):
                results = (
                    list(results)
                    if hasattr(results, "__iter__")
                    and not isinstance(results, (str, dict))
                    else [results]
                )

            rows = []
            for idx, output_item in enumerate(results):
                # Get metadata using TrackedItem helper
                if self._processed_items and idx < len(self._processed_items):
                    row = TrackedItem.extract_export_metadata(
                        self._processed_items[idx], idx
                    )
                else:
                    row = {"item_id": f"item_{idx}", "index": idx}

                # Add classification outputs
                output_dict = extract_output_dict(
                    output_item, serialize_iterables=False
                )
                if not output_dict:
                    continue

                # Convert non-hashable types (lists, BoxList) to strings for agreement calculation
                for k, v in output_dict.items():
                    if (
                        isinstance(v, (list, tuple))
                        or hasattr(v, "__iter__")
                        and not isinstance(v, str)
                    ):
                        row[k] = str(v)  # Convert to string for CSV/agreement
                    else:
                        row[k] = v

                rows.append(row)

            if rows:
                from ..base import order_export_columns

                df = pd.DataFrame(rows)
                # Get output keys from ALL sections of template
                output_keys = []
                if self.template:
                    sections = parse_syntax(self.template)
                    for section in sections:
                        output_keys.extend(section.keys())
                # Extract ground truth column names if configured
                ground_truth_columns = None
                if self.ground_truths:
                    ground_truth_columns = [
                        config.get("existing")
                        for config in self.ground_truths.values()
                        if config.get("existing")
                    ]
                df = order_export_columns(
                    df,
                    output_keys=output_keys,
                    template=self.template,
                    ground_truth_columns=ground_truth_columns,
                )
                model_dfs[model_name] = df

        return model_dfs if model_dfs else None

    def _auto_detect_agreement_fields(self) -> None:
        """Auto-detect agreement fields from template output keys.

        Only includes categorical fields suitable for nominal agreement metrics:
        - [[pick:name|options]] - single-choice categorical
        - [[bool:name]] - boolean fields

        Excludes:
        - Multi-select (pick*)
        - Continuous data (int, float)
        - Free text fields
        """
        if not self.agreement_fields and self.template:
            CATEGORICAL_TYPES = {"pick", "bool"}

            output_keys = []
            sections = parse_syntax(self.template)

            for section in sections:
                for key, action in section.items():
                    # Get action type (might be 'action_type' or 'type')
                    action_type = getattr(action, "action_type", None) or getattr(
                        action, "type", None
                    )

                    # Check for quantifier (pick* is multi-select)
                    quantifier = getattr(action, "quantifier", None)
                    is_multi_select = quantifier in ("*", "+", "?")

                    # Only include single-select categorical fields
                    if action_type in CATEGORICAL_TYPES and not is_multi_select:
                        output_keys.append(key)

            if output_keys:
                self.agreement_fields = output_keys
                logger.info(f"Auto-detected agreement fields: {output_keys}")
            else:
                self.agreement_fields = None
                logger.warning(
                    "No categorical fields (pick/bool) detected for agreement calculation. "
                    "Agreement metrics require categorical variables. "
                    "Use agreement_fields parameter to specify fields manually."
                )

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and classification results per model."""
        # Get base metadata from parent
        result = super().result()

        # Reconstruct _model_results if needed
        if not hasattr(self, "_model_results") or not self._model_results:
            if isinstance(self.output, dict):
                self._model_results = self.output
            else:
                model_name = (
                    self.model_names[0]
                    if hasattr(self, "model_names") and self.model_names
                    else (self.model_name if hasattr(self, "model_name") else None)
                    or "default"
                )
                self._model_results = {model_name: self.output}

        # Use the same method as export() to build DataFrames
        model_dfs = self._build_dataframes_from_results()

        # Calculate agreement statistics if multiple models
        agreement_stats = None
        agreement_stats_df = None
        if model_dfs and len(model_dfs) >= 2:
            try:
                from ...agreement import calculate_agreement_from_dataframes

                # Agreement fields already auto-detected in run() if needed
                if not self.agreement_fields:
                    logger.warning(
                        "No output fields detected for agreement calculation - skipping"
                    )
                    agreement_stats = None
                else:
                    agreement_stats = calculate_agreement_from_dataframes(
                        model_dfs, self.agreement_fields
                    )

                # Convert to DataFrame for easier HTML rendering
                if agreement_stats:
                    agreement_stats_df = pd.DataFrame(agreement_stats).T
                    agreement_stats_df.index.name = "field"
            except Exception as e:
                logger.warning(f"Could not calculate agreement statistics: {e}")
                agreement_stats = None
                agreement_stats_df = None

        # Calculate ground truth metrics if configured
        ground_truth_stats = None
        ground_truth_stats_df = None
        if self.ground_truths and model_dfs and self._processed_items:
            try:
                from ...ground_truth_metrics import \
                    calculate_ground_truth_metrics

                ground_truth_stats = calculate_ground_truth_metrics(
                    model_dfs=model_dfs,
                    processed_items=self._processed_items,
                    ground_truth_config=self.ground_truths,
                )

                # Flatten for DataFrame display
                if ground_truth_stats:
                    rows = []
                    for field_name, model_metrics in ground_truth_stats.items():
                        for model_name, metric_dict in model_metrics.items():
                            row = {"field": field_name, "model": model_name}
                            # Add scalar metrics only
                            for key, value in metric_dict.items():
                                if key not in [
                                    "confusion_matrix",
                                    "classification_report",
                                    "labels",
                                    "field",
                                    "model",
                                ]:
                                    row[key] = value
                            rows.append(row)

                    if rows:
                        ground_truth_stats_df = pd.DataFrame(rows)

            except Exception as e:
                logger.warning(f"Could not calculate ground truth metrics: {e}")
                ground_truth_stats = None
                ground_truth_stats_df = None

        # Add Classifier-specific data
        result["model_dfs"] = model_dfs if model_dfs else {}
        result["agreement_stats"] = agreement_stats  # Dict format
        result["agreement_stats_df"] = agreement_stats_df  # DataFrame format
        result["ground_truth_stats"] = ground_truth_stats  # Dict format
        result["ground_truth_stats_df"] = ground_truth_stats_df  # DataFrame format
        result["metadata"]["num_models"] = len(model_dfs) if model_dfs else 0
        result["metadata"]["model_names"] = list(model_dfs.keys()) if model_dfs else []
        result["metadata"]["agreement_fields"] = self.agreement_fields or []
        result["metadata"]["ground_truth_fields"] = (
            list(self.ground_truths.keys()) if self.ground_truths else []
        )

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Classifier node with CSV output and individual responses."""
        from ...agreement import export_agreement_stats
        from ...agreement_scripts import (collect_field_categories,
                                          generate_human_rater_template,
                                          write_agreement_scripts)
        from ...export_utils import (export_to_csv, export_to_html,
                                     export_to_json)
        from ...helpers import build_combined_long_form_dataset

        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template:
            (folder / "prompt_template.sd").write_text(self.template)

        # Reconstruct _model_results if needed (e.g., from JSON deserialization)
        if not self._model_results:
            if isinstance(self.output, dict):
                self._model_results = self.output
            else:
                # Single model case - wrap in dict
                model_name = (
                    self.model_names[0]
                    if self.model_names
                    else (self.model_name or "default")
                )
                self._model_results = {model_name: self.output}

        # Early return if no results to export (e.g., node failed)
        if not self._model_results or all(
            v is None for v in self._model_results.values()
        ):
            logger.warning(f"No results to export for {self.name}")
            return

        # Export prompts for each model (prompts can differ when models build on previous completions)
        for model_name, results in self._model_results.items():
            if not results:
                logger.warning(f"No results to export for model {model_name}")
                continue

            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            suffix = f"_{safe_model_name}" if len(self._model_results) > 1 else ""
            prompts_folder = folder / f"prompts{suffix}"
            prompts_folder.mkdir(parents=True, exist_ok=True)

            for i, result_item in enumerate(results):
                item = (
                    self._processed_items[i]
                    if self._processed_items and i < len(self._processed_items)
                    else None
                )
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))

                # Export prompts and responses
                if hasattr(result_item, "outputs"):
                    (prompts_folder / f"{i:04d}_{safe_id}_response.json").write_text(
                        result_item.outputs.to_json()
                    )
                    if hasattr(result_item, "results"):
                        for k, v in result_item.results.items():
                            (
                                prompts_folder / f"{i:04d}_{safe_id}_{k}_prompt.txt"
                            ).write_text(v.prompt)
                else:
                    # Plain dict from JSON deserialization
                    (prompts_folder / f"{i:04d}_{safe_id}_response.json").write_text(
                        json.dumps(result_item, indent=2, default=str)
                    )

        # Export responses for each model
        responses_folder = folder / "responses"
        responses_folder.mkdir(parents=True, exist_ok=True)

        for model_name, results in self._model_results.items():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")

            for i, result_item in enumerate(results):
                item = (
                    self._processed_items[i]
                    if self._processed_items and i < len(self._processed_items)
                    else None
                )
                safe_id = TrackedItem.make_safe_id(TrackedItem.extract_source_id(item))
                file_prefix = f"{i:04d}_{safe_id}_{safe_model_name}"

                # Export response for this model
                if hasattr(result_item, "outputs"):
                    (responses_folder / f"{file_prefix}_response.json").write_text(
                        result_item.outputs.to_json()
                    )
                else:
                    # Plain dict from JSON deserialization
                    (responses_folder / f"{file_prefix}_response.json").write_text(
                        json.dumps(result_item, indent=2, default=str)
                    )

        # Build DataFrames for each model (reuse logic from _build_dataframes_from_results)
        model_dfs = self._build_dataframes_from_results()

        if not model_dfs:
            logger.warning(f"No valid classification results to export")
            return

        # Export CSV for each model using pre-built DataFrames
        for model_name, df in model_dfs.items():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            suffix = f"_{safe_model_name}" if len(self._model_results) > 1 else ""

            # Add unique_id to filename if provided
            uid_suffix = f"_{unique_id}" if unique_id else ""
            export_to_csv(
                df, folder / f"classifications_{self.name}{suffix}{uid_suffix}.csv"
            )
            export_to_html(
                df, folder / f"classifications_{self.name}{suffix}{uid_suffix}.html"
            )
            # Convert DataFrame back to list of dicts for JSON export
            export_to_json(
                df.to_dict("records"),
                folder / f"classifications_{self.name}{suffix}{uid_suffix}.json",
            )

        # Export combined long-form dataset for multi-model comparison
        if len(self._model_results) >= 2:
            combined_df = build_combined_long_form_dataset(
                self._model_results, self._processed_items
            )

            if not combined_df.empty:
                uid_suffix = f"_{unique_id}" if unique_id else ""
                export_to_csv(
                    combined_df,
                    folder
                    / f"classifications_{self.name}_combined_long{uid_suffix}.csv",
                )
                export_to_html(
                    combined_df,
                    folder
                    / f"classifications_{self.name}_combined_long{uid_suffix}.html",
                )
                logger.info(
                    f"Exported combined long-form dataset with {len(combined_df)} rows"
                )

        # Calculate and export agreement statistics if multiple models
        if len(self._model_results) >= 2 and model_dfs:
            from ...agreement import calculate_agreement_from_dataframes

            # Agreement fields already auto-detected in run() if needed
            if not self.agreement_fields:
                logger.warning(
                    "No output fields detected for agreement calculation - skipping"
                )
            else:
                stats = calculate_agreement_from_dataframes(
                    model_dfs, self.agreement_fields
                )

            if stats:
                stats_prefix = str(folder / "agreement_stats")
                export_agreement_stats(stats, stats_prefix)
                logger.info(
                    f"Exported agreement statistics to {stats_prefix}.csv and {stats_prefix}.json"
                )

            # Generate agreement calculation script and template
            self._generate_agreement_script(folder, model_dfs)

        # Calculate and export ground truth metrics if configured
        if self.ground_truths and self._processed_items and model_dfs:
            from ...ground_truth_metrics import (
                calculate_ground_truth_metrics, export_confusion_matrices,
                export_ground_truth_metrics)

            try:
                ground_truth_stats = calculate_ground_truth_metrics(
                    model_dfs=model_dfs,
                    processed_items=self._processed_items,
                    ground_truth_config=self.ground_truths,
                )

                if ground_truth_stats:
                    # Export summary metrics with config info
                    stats_prefix = str(folder / "ground_truth_metrics")
                    export_ground_truth_metrics(
                        ground_truth_stats, stats_prefix, self.ground_truths
                    )
                    logger.info(
                        f"Exported ground truth metrics to {stats_prefix}.csv and {stats_prefix}.json"
                    )

                    # Export confusion matrices with config for metadata
                    export_confusion_matrices(
                        ground_truth_stats, folder, self.ground_truths
                    )

            except Exception as e:
                logger.error(
                    f"Error exporting ground truth metrics: {e}", exc_info=True
                )

    def _generate_agreement_script(
        self, folder: Path, model_dfs: Dict[str, pd.DataFrame]
    ):
        """Generate a standalone Python script for agreement calculation.

        Args:
            folder: Output folder for scripts
            model_dfs: Pre-built DataFrames for each model from _build_dataframes_from_results()
        """
        from ...agreement_scripts import (collect_field_categories,
                                          generate_human_rater_template,
                                          write_agreement_scripts)

        if not self._model_results or not self.agreement_fields or not model_dfs:
            return

        # Get list of CSV files
        csv_files = []
        for model_name in self._model_results.keys():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            csv_files.append(f"classifications_{self.name}_{safe_model_name}.csv")

        # Collect valid categories using extracted function
        field_categories = collect_field_categories(
            self._model_results, self.agreement_fields
        )

        # Use the first model's pre-built DataFrame for template generation
        first_model_name = next(iter(model_dfs.keys()))
        df = model_dfs[first_model_name]

        generate_human_rater_template(
            folder, self.name, first_model_name, df, field_categories
        )

        # Generate scripts using extracted function
        write_agreement_scripts(
            folder, self.name, self.agreement_fields, csv_files, field_categories
        )
