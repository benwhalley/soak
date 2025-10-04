"""Utility functions for generating agreement analysis scripts."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

logger = logging.getLogger(__name__)


def collect_field_categories(
    model_results: Dict[str, List[Any]],
    agreement_fields: List[str]
) -> Dict[str, Set[str]]:
    """Collect all unique categories for each agreement field across all models.

    Args:
        model_results: Dict mapping model_name -> list of results
        agreement_fields: List of field names to collect categories for

    Returns:
        Dict mapping field_name -> set of unique category values
    """
    field_categories = {field: set() for field in agreement_fields}

    for model_name, results in model_results.items():
        for result in results:
            if not result:
                continue

            # Handle different result types
            output_dict = {}
            if hasattr(result, 'results'):
                # ChatterResult
                output_dict = result.results
            elif isinstance(result, dict):
                output_dict = result
            else:
                continue

            # Collect categories
            for field in agreement_fields:
                if field in output_dict:
                    value = output_dict[field]
                    # Convert to string and add to set
                    if value is not None:
                        if isinstance(value, (list, tuple)):
                            field_categories[field].update(str(v) for v in value)
                        else:
                            field_categories[field].add(str(value))

    return field_categories


def generate_human_rater_template(
    output_folder: Path,
    node_name: str,
    model_name: str,
    df: pd.DataFrame,
    field_categories: Dict[str, Set[str]]
) -> Path:
    """Generate a template CSV for human raters with example categories.

    Args:
        output_folder: Folder to write template to
        node_name: Name of the classifier node
        model_name: Name of the model (for filename)
        df: DataFrame with model results
        field_categories: Dict mapping field -> set of valid categories

    Returns:
        Path to the generated template file
    """
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    template_path = output_folder / f"human_rater_template_{node_name}_{safe_model_name}.csv"

    # Create template with example row showing valid categories
    template_rows = []

    # Add example row showing valid categories
    example_row = {}
    for col in df.columns:
        if col in field_categories and field_categories[col]:
            # Show first few valid categories as examples
            examples = sorted(field_categories[col])[:3]
            example_row[col] = f"e.g., {' or '.join(examples)}"
        else:
            example_row[col] = ""
    template_rows.append(example_row)

    # Add actual rows with empty classification fields
    for _, row in df.iterrows():
        template_row = row.to_dict()
        # Clear classification fields for human to fill in
        for field in field_categories.keys():
            template_row[field] = ""
        template_rows.append(template_row)

    template_df = pd.DataFrame(template_rows)
    template_df.to_csv(template_path, index=False)
    logger.info(f"Generated human rater template: {template_path}")

    return template_path


def generate_agreement_script_content(
    node_name: str,
    agreement_fields: List[str],
    model_csvs: List[str],
    field_categories: Dict[str, Set[str]]
) -> str:
    """Generate Python script content for recalculating agreement with human raters.

    Args:
        node_name: Name of the classifier node
        agreement_fields: List of field names to calculate agreement for
        model_csvs: List of CSV filenames for models
        field_categories: Dict mapping field -> set of valid categories

    Returns:
        Python script content as a string
    """
    # Convert sets to sorted lists for JSON serialization
    valid_categories = {
        field: sorted(categories)
        for field, categories in field_categories.items()
    }

    script_content = f'''#!/usr/bin/env python3
"""
Recalculate agreement statistics for {node_name} with human raters.

Usage:
    1. Fill in human_rater_template_*.csv files with your classifications
    2. Run this script:
       python calculate_agreement.py human_rater1.csv human_rater2.csv ...

The script will calculate agreement between:
- All raters (models + humans)
- Just the human raters (if 2+)
- Each human rater vs each model (pairwise)
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Configuration
AGREEMENT_FIELDS = {agreement_fields!r}
MODEL_CSVS = {model_csvs!r}
VALID_CATEGORIES = {valid_categories!r}


def validate_human_csv(csv_path, agreement_fields, valid_categories):
    """Validate human rater CSV and warn about new categories."""
    df = pd.read_csv(csv_path)

    # Check required columns
    missing = [f for f in agreement_fields if f not in df.columns]
    if missing:
        raise ValueError(f"{{csv_path}} missing required columns: {{missing}}")

    # Warn about new categories
    for field in agreement_fields:
        if field not in valid_categories:
            continue

        valid = set(valid_categories[field])
        actual = set(df[field].dropna().astype(str).unique())
        new_cats = actual - valid

        if new_cats:
            print(f"⚠️  Warning: {{csv_path}} has new categories for '{{field}}': {{new_cats}}")
            print(f"   Valid categories were: {{valid}}")

    return df


def main():
    """Main function."""
    # Change to script directory so paths work
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)

    # Try to import soak's agreement module
    try:
        from soak.agreement import calculate_agreement_from_dataframes
        print("✓ Using soak.agreement module")
    except ImportError:
        print("⚠️  soak package not found, using standalone implementation")
        print("   Install with: pip install soak-llm")
        print()

        # Standalone implementation
        from irrCAC.raw import CAC

        def calculate_agreement_from_dataframes(model_dfs, agreement_fields=None, item_id_col="item_id"):
            if not agreement_fields:
                metadata_cols = {{'item_id', 'document', 'filename', 'index'}}
                all_fields = {{c for df in model_dfs.values() for c in df.columns}}
                agreement_fields = sorted(
                    f for f in all_fields if f not in metadata_cols and not f.endswith('__evidence')
                )

            stats = {{}}
            for field in agreement_fields:
                ratings_data = {{
                    model_name: df.set_index(item_id_col)[field]
                    for model_name, df in model_dfs.items()
                    if field in df.columns
                }}

                if len(ratings_data) < 2:
                    continue

                ratings = pd.DataFrame(ratings_data).dropna()
                if len(ratings) == 0:
                    continue

                try:
                    cac = CAC(ratings)
                    stats[field] = {{
                        "AC1": round(float(cac.gwet()["est"]["coefficient_value"]), 3),
                        "Kripp_alpha": round(float(cac.krippendorff()["est"]["coefficient_value"]), 3),
                        "Percent_Agreement": round((ratings.nunique(axis=1) == 1).mean(), 3),
                        "n_items": len(ratings),
                        "n_raters": len(ratings_data)
                    }}
                except Exception as e:
                    print(f"Error calculating agreement for {{field}}: {{e}}")

            return stats if stats else None

    # Get human CSV paths
    if len(sys.argv) > 1:
        human_csvs = sys.argv[1:]
    else:
        # Interactive mode (for double-clicking)
        print("Agreement Analysis for {node_name}")
        print("=" * 60)
        print()
        print("Enter paths to human rater CSV files (one per line).")
        print("Press Enter on empty line when done.")
        print()

        human_csvs = []
        while True:
            path = input(f"Human rater {{len(human_csvs) + 1}} CSV (or Enter to finish): ").strip()
            if not path:
                break
            if Path(path).exists():
                human_csvs.append(path)
            else:
                print(f"⚠️  File not found: {{path}}")

        if not human_csvs:
            print("No human rater files provided. Exiting.")
            return

    # Validate human CSVs
    print(f"\\nValidating {{len(human_csvs)}} human rater CSV(s)...")
    human_dfs = {{}}
    for csv_path in human_csvs:
        try:
            df = validate_human_csv(csv_path, AGREEMENT_FIELDS, VALID_CATEGORIES)
            rater_name = Path(csv_path).stem
            human_dfs[rater_name] = df
            print(f"✓ {{rater_name}}: {{len(df)}} items")
        except Exception as e:
            print(f"✗ Error loading {{csv_path}}: {{e}}")
            return

    # Load model CSVs
    print(f"\\nLoading {{len(MODEL_CSVS)}} model CSV(s)...")
    model_dfs = {{}}
    for csv_file in MODEL_CSVS:
        try:
            df = pd.read_csv(csv_file)
            model_name = Path(csv_file).stem.replace(f"{{node_name}}_", "")
            model_dfs[model_name] = df
            print(f"✓ {{model_name}}: {{len(df)}} items")
        except Exception as e:
            print(f"⚠️  Could not load {{csv_file}}: {{e}}")

    if not model_dfs:
        print("⚠️  No model CSVs found. Continuing with human raters only.")

    # Calculate overall agreement (all raters)
    print("\\n" + "=" * 60)
    print("OVERALL AGREEMENT (All Raters)")
    print("=" * 60)
    all_dfs = {{**model_dfs, **human_dfs}}
    overall_stats = calculate_agreement_from_dataframes(all_dfs, AGREEMENT_FIELDS)

    if overall_stats:
        df_overall = pd.DataFrame(overall_stats).T
        print(df_overall.to_string())
        df_overall.to_csv("agreement_all_raters.csv")
        print("\\n✓ Saved to agreement_all_raters.csv")
    else:
        print("⚠️  Could not calculate overall agreement")

    # Calculate inter-human reliability (if 2+ humans)
    if len(human_dfs) >= 2:
        print("\\n" + "=" * 60)
        print("INTER-HUMAN RATER RELIABILITY")
        print("=" * 60)
        human_stats = calculate_agreement_from_dataframes(human_dfs, AGREEMENT_FIELDS)

        if human_stats:
            df_human = pd.DataFrame(human_stats).T
            print(df_human.to_string())
            df_human.to_csv("agreement_human_raters.csv")
            print("\\n✓ Saved to agreement_human_raters.csv")
        else:
            print("⚠️  Could not calculate inter-human agreement")

    # Calculate pairwise: each human vs each model
    if human_dfs and model_dfs:
        print("\\n" + "=" * 60)
        print("HUMAN vs MODEL COMPARISONS (Pairwise)")
        print("=" * 60)

        for human_name, human_df in human_dfs.items():
            print(f"\\n{{human_name}}:")

            for field in AGREEMENT_FIELDS:
                # Build comparison table for this field
                comparison_data = {{}}

                for model_name, model_df in model_dfs.items():
                    pair_dfs = {{human_name: human_df, model_name: model_df}}
                    pair_stats = calculate_agreement_from_dataframes(pair_dfs, [field])

                    if pair_stats and field in pair_stats:
                        comparison_data[model_name] = pair_stats[field]

                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data).T
                    print(f"\\n  {{field}}:")
                    print("  " + df_comparison.to_string().replace("\\n", "\\n  "))

            # Save CSV for this human
            human_model_stats = {{}}
            for model_name, model_df in model_dfs.items():
                pair_dfs = {{human_name: human_df, model_name: model_df}}
                pair_stats = calculate_agreement_from_dataframes(pair_dfs, AGREEMENT_FIELDS)
                if pair_stats:
                    for field, stats in pair_stats.items():
                        if field not in human_model_stats:
                            human_model_stats[field] = {{}}
                        human_model_stats[field][model_name] = stats

            if human_model_stats:
                csv_path = f"agreement_{{human_name}}_vs_models.csv"
                # Flatten nested dict for CSV export
                rows = []
                for field, models in human_model_stats.items():
                    for model, stats in models.items():
                        row = {{"field": field, "model": model, **stats}}
                        rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"\\n✓ Saved {{human_name}} comparisons to {{csv_path}}")

    print("\\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''

    return script_content


def generate_agreement_command_script(node_name: str) -> str:
    """Generate macOS .command file for double-click execution.

    Args:
        node_name: Name of the classifier node

    Returns:
        Bash script content as a string
    """
    return f'''#!/bin/bash
# macOS launcher for agreement analysis
cd "$(dirname "$0")"
python3 calculate_agreement.py "$@"
'''


def write_agreement_scripts(
    output_folder: Path,
    node_name: str,
    agreement_fields: List[str],
    model_csvs: List[str],
    field_categories: Dict[str, Set[str]]
) -> None:
    """Write agreement analysis scripts to output folder.

    Args:
        output_folder: Folder to write scripts to
        node_name: Name of the classifier node
        agreement_fields: List of field names to calculate agreement for
        model_csvs: List of CSV filenames for models
        field_categories: Dict mapping field -> set of valid categories
    """
    # Generate Python script
    script_content = generate_agreement_script_content(
        node_name, agreement_fields, model_csvs, field_categories
    )
    script_path = output_folder / "calculate_agreement.py"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    logger.info(f"Generated agreement script: {script_path}")

    # Generate macOS .command file
    command_content = generate_agreement_command_script(node_name)
    command_path = output_folder / "calculate_agreement.command"
    command_path.write_text(command_content)
    command_path.chmod(0o755)
    logger.info(f"Generated macOS launcher: {command_path}")
