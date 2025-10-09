"""Classifier node for multi-model classification with agreement metrics."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import anyio
import pandas as pd
from pydantic import Field, PrivateAttr
from struckdown import LLM, chatter_async
from struckdown.parsing import parse_syntax

from ..base import TrackedItem, get_action_lookup, semaphore
from .base import ItemsNode, CompletionDAGNode

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
    template_text: str = None
    model_names: Optional[List[str]] = None  # Multiple models for agreement analysis
    agreement_fields: Optional[List[str]] = None  # Fields to calculate agreement on

    # Private attributes - automatically excluded from serialization
    _processed_items: Optional[List[Any]] = PrivateAttr(default=None)
    _model_results: Optional[Dict[str, List[Any]]] = PrivateAttr(default=None)
    _agreement_stats: Optional[Dict[str, Dict[str, float]]] = PrivateAttr(default=None)

    @property
    def template(self) -> str:
        return self.template_text

    def validate_template(self):
        try:
            parse_syntax(self.template_text)
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

            async with anyio.create_task_group() as tg:
                for idx, item in enumerate(items):

                    async def run_and_store(
                        index=idx, item=item, current_model=model_name
                    ):
                        async with semaphore:
                            # Create model instance for this specific model
                            model = LLM(model_name=current_model)

                            # Run the classification template
                            # Collect extra kwargs for LLM
                            extra_kwargs = {}
                            if self.max_tokens is not None:
                                extra_kwargs['max_tokens'] = self.max_tokens
                            extra_kwargs['temperature'] = self.temperature
                            extra_kwargs['seed'] = self.dag.config.seed

                            chatter_result = await chatter_async(
                                multipart_prompt=self.template,
                                context={**filtered_context, **item},
                                model=model,
                                credentials=self.dag.config.llm_credentials,
                                action_lookup=get_action_lookup(),
                                extra_kwargs=extra_kwargs,
                            )

                            # Extract structured outputs from ChatterResult
                            results[index] = chatter_result

                    tg.start_soon(run_and_store)

            self._model_results[model_name] = results

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
                if hasattr(output_item, "outputs"):
                    output_dict = output_item.outputs
                elif isinstance(output_item, dict):
                    output_dict = {
                        k: v for k, v in output_item.items() if not k.startswith("__")
                    }
                else:
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
                model_dfs[model_name] = pd.DataFrame(rows)

        return model_dfs if model_dfs else None

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

                # Auto-detect agreement fields if not set
                if not self.agreement_fields:
                    metadata_cols = {"item_id", "document", "filename", "index"}
                    all_fields = {c for df in model_dfs.values() for c in df.columns}
                    self.agreement_fields = sorted(
                        f
                        for f in all_fields
                        if f not in metadata_cols and not f.endswith("__evidence")
                    )

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

        # Add Classifier-specific data
        result["model_dfs"] = model_dfs if model_dfs else {}
        result["agreement_stats"] = agreement_stats  # Dict format
        result["agreement_stats_df"] = agreement_stats_df  # DataFrame format
        result["metadata"]["num_models"] = len(model_dfs) if model_dfs else 0
        result["metadata"]["model_names"] = list(model_dfs.keys()) if model_dfs else []
        result["metadata"]["agreement_fields"] = self.agreement_fields or []

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Classifier node with CSV output and individual responses."""
        from ...agreement import export_agreement_stats
        from ...agreement_scripts import (
            collect_field_categories,
            generate_human_rater_template,
            write_agreement_scripts,
        )
        from ...export_utils import export_to_csv, export_to_html, export_to_json
        from ...helpers import build_combined_long_form_dataset

        super().export(folder, unique_id=unique_id)

        # Write template
        if self.template_text:
            (folder / "prompt_template.sd.md").write_text(self.template_text)

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

        # Export prompts once (same for all models)
        prompts_folder = folder / "prompts"
        prompts_folder.mkdir(parents=True, exist_ok=True)

        first_results = next(iter(self._model_results.values()))
        if not first_results:
            logger.warning(f"No results to export for {self.name}")
            return

        for i, result_item in enumerate(first_results):
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

        # Export CSV for each model
        for model_name, results in self._model_results.items():
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            suffix = f"_{safe_model_name}" if len(self._model_results) > 1 else ""

            # Build rows with source_id tracking
            rows = []
            for idx, output_item in enumerate(results):
                # Get metadata using TrackedItem helper
                if self._processed_items and idx < len(self._processed_items):
                    row = TrackedItem.extract_export_metadata(
                        self._processed_items[idx], idx
                    )
                else:
                    row = {
                        "item_id": f"item_{idx}",
                        "document": f"item_{idx}",
                        "index": idx,
                    }

                # Add classification outputs
                if hasattr(output_item, "outputs"):
                    output_dict = output_item.outputs
                elif isinstance(output_item, dict):
                    output_dict = {
                        k: v for k, v in output_item.items() if not k.startswith("__")
                    }
                else:
                    continue

                # Convert non-hashable types to strings
                for k, v in output_dict.items():
                    if (
                        isinstance(v, (list, tuple))
                        or hasattr(v, "__iter__")
                        and not isinstance(v, str)
                    ):
                        row[k] = str(v)
                    else:
                        row[k] = v

                rows.append(row)

            if not rows:
                logger.warning(
                    f"No valid classification results to export for {model_name}"
                )
                continue

            # Export CSV, HTML, JSON using utility functions
            df = pd.DataFrame(rows)
            # Add unique_id to filename if provided
            uid_suffix = f"_{unique_id}" if unique_id else ""
            export_to_csv(
                df, folder / f"classifications_{self.name}{suffix}{uid_suffix}.csv"
            )
            export_to_html(
                df, folder / f"classifications_{self.name}{suffix}{uid_suffix}.html"
            )
            export_to_json(
                rows, folder / f"classifications_{self.name}{suffix}{uid_suffix}.json"
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
                    folder / f"classifications_{self.name}_combined_long{uid_suffix}.csv",
                )
                export_to_html(
                    combined_df,
                    folder / f"classifications_{self.name}_combined_long{uid_suffix}.html",
                )
                logger.info(
                    f"Exported combined long-form dataset with {len(combined_df)} rows"
                )

        # Calculate and export agreement statistics if multiple models
        if len(self._model_results) >= 2:
            from ...agreement import calculate_agreement_from_dataframes

            model_dfs = self._build_dataframes_from_results()
            if model_dfs:
                stats = calculate_agreement_from_dataframes(
                    model_dfs, self.agreement_fields
                )

                # Auto-detect agreement fields if not set (same logic as in calculate_agreement_from_dataframes)
                if not self.agreement_fields:
                    metadata_cols = {"item_id", "document", "filename", "index"}
                    all_fields = {c for df in model_dfs.values() for c in df.columns}
                    self.agreement_fields = sorted(
                        f
                        for f in all_fields
                        if f not in metadata_cols and not f.endswith("__evidence")
                    )

                if stats:
                    stats_prefix = str(folder / "agreement_stats")
                    export_agreement_stats(stats, stats_prefix)
                    logger.info(
                        f"Exported agreement statistics to {stats_prefix}.csv and {stats_prefix}.json"
                    )

                # Generate agreement calculation script and template
                self._generate_agreement_script(folder)

    def _generate_agreement_script(self, folder: Path):
        """Generate a standalone Python script for agreement calculation."""
        from ...agreement_scripts import (
            collect_field_categories,
            generate_human_rater_template,
            write_agreement_scripts,
        )

        if not self._model_results or not self.agreement_fields:
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

        # Generate template CSV using extracted function
        first_results = next(iter(self._model_results.values()))
        first_model_name = next(iter(self._model_results.keys()))

        # Build DataFrame for template generation
        rows = []
        for idx, output_item in enumerate(first_results):
            # Get metadata using TrackedItem helper
            if self._processed_items and idx < len(self._processed_items):
                row = TrackedItem.extract_export_metadata(
                    self._processed_items[idx], idx
                )
            else:
                row = {
                    "item_id": f"item_{idx}",
                    "document": f"item_{idx}",
                    "index": idx,
                }

            # Add classification outputs
            if hasattr(output_item, "outputs"):
                output_dict = output_item.outputs
            elif isinstance(output_item, dict):
                output_dict = {
                    k: v for k, v in output_item.items() if not k.startswith("__")
                }
            else:
                continue

            for k, v in output_dict.items():
                if (
                    isinstance(v, (list, tuple))
                    or hasattr(v, "__iter__")
                    and not isinstance(v, str)
                ):
                    row[k] = str(v)
                else:
                    row[k] = v

            rows.append(row)

        df = pd.DataFrame(rows)
        generate_human_rater_template(
            folder, self.name, first_model_name, df, field_categories
        )

        # Generate scripts using extracted function
        write_agreement_scripts(
            folder, self.name, self.agreement_fields, csv_files, field_categories
        )
