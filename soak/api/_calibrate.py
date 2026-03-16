"""Calibrate implementation for soak API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class CalibrateError(Exception):
    """Error during calibration."""

    pass


@dataclass
class CalibrationResult:
    """Result of a calibration operation."""

    model: dict
    """The fitted calibration model."""

    method: str
    """Calibration method used ('scam' or 'gam')."""

    output_folder: Path
    """Path to the output folder containing all files."""

    pkl_path: Path
    """Path to the saved model pickle file."""

    yaml_path: Path
    """Path to the saved metadata YAML file."""

    png_path: Path
    """Path to the calibration plot."""

    csv_path: Path
    """Path to the paraphrases CSV."""

    validation_stats: Optional[dict]
    """Validation statistics if holdout was used."""

    category_stats: dict
    """Statistics for each category."""

    paraphrases_df: pd.DataFrame
    """DataFrame with paraphrases and similarities."""

    def calibrate(self, values: np.ndarray) -> np.ndarray:
        """Apply the calibration to raw similarity values.

        Args:
            values: Array of raw angular similarity values

        Returns:
            Array of calibrated values (0-1 scale)
        """
        values = np.asarray(values)
        if self.method == "scam":
            return np.clip(
                np.interp(values, self.model["x_lookup"], self.model["y_lookup"]), 0, 1
            )
        else:  # gam
            return np.clip(self.model.predict(values.reshape(-1, 1)), 0, 1)


def calibrate(
    *,
    # input mode 1: generate paraphrases
    input_csv: Optional[Path] = None,
    column: Optional[str] = None,
    prompt: Optional[Path] = None,
    llm_model: str = "gpt-4.1-mini",
    max_concurrent: int = 20,
    # input mode 2: use existing paraphrases
    paraphrases_csv: Optional[Path] = None,
    # common options
    config: Optional[Path] = None,
    embedding_model: str = "local/intfloat/e5-base",
    embedding_template: str = "{text}",
    method: str = "scam",
    spline_df: int = 5,
    n_anchors: int = 0,
    holdout: float = 0.2,
    seed: int = 42,
    sample: Optional[int] = None,
    head: Optional[int] = None,
    group_col: Optional[list[str]] = None,
    output: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> CalibrationResult:
    """Fit a calibration model to map similarity scores to a meaningful scale.

    Two modes for input:

    1. Generate paraphrases from sentences:
       result = api.calibrate(input_csv="sentences.csv", prompt="paraphrases.sd")

    2. Use existing paraphrases:
       result = api.calibrate(paraphrases_csv="existing.csv")

    The calibration maps raw angular similarity to a 0.1-0.9 scale where:
    - 0.9 = same meaning (paraphrase quality)
    - 0.75 = close meaning
    - 0.5 = diverging (partial overlap)
    - 0.3 = distant (weak relation)
    - 0.1 = unrelated

    Args:
        input_csv: CSV file with sentences to paraphrase
        column: Column name to paraphrase (default: 'original' or first column)
        prompt: Struckdown prompt file for generating paraphrases
        llm_model: LLM model for paraphrase generation
        max_concurrent: Max concurrent LLM requests
        paraphrases_csv: CSV with pre-generated paraphrases
        config: YAML config mapping category names to target values
        embedding_model: Embedding model (use 'local/model-name' for local)
        embedding_template: Template for embedding text
        method: Calibration method ('scam' or 'gam')
        spline_df: Degrees of freedom for spline
        n_anchors: Anchor points at each tail (GAM only)
        holdout: Fraction of data for validation (0 to disable)
        seed: Random seed
        sample: Randomly sample N rows
        head: Take first N rows
        group_col: Column(s) for random effects and grouped holdout
        output: Output folder name
        cwd: Working directory

    Returns:
        CalibrationResult with model, paths, and statistics

    Raises:
        CalibrateError: If calibration fails
    """
    import yaml

    from ..calibration import (DEFAULT_TARGETS, compute_similarities,
                               fit_calibration_gam, generate_paraphrases,
                               plot_calibration_curve, save_calibration)

    cwd = cwd or Path.cwd()

    # validate method
    if method not in ("scam", "gam"):
        raise CalibrateError(f"Invalid method '{method}'. Use 'scam' or 'gam'")

    # check R availability for scam
    if method == "scam":
        from ..calibration import _check_r_available

        if not _check_r_available():
            raise CalibrateError(
                "R and scam package required for scam calibration. "
                "Install with: pip install 'soaking[calibration]' "
                "Then in R: install.packages('scam')"
            )

    # validate inputs
    if input_csv is None and paraphrases_csv is None:
        raise CalibrateError("Must provide either input_csv or paraphrases_csv")

    if input_csv is not None and prompt is None:
        raise CalibrateError("prompt required when providing input_csv")

    if input_csv is not None and paraphrases_csv is not None:
        raise CalibrateError("Cannot use both input_csv and paraphrases_csv")

    if head is not None and sample is not None:
        raise CalibrateError("Cannot use both head and sample")

    # load config
    if config:
        cfg = yaml.safe_load(config.read_text())
        targets = cfg.get("targets", DEFAULT_TARGETS)
        columns_cfg = cfg.get("columns", {})
    else:
        targets = DEFAULT_TARGETS
        columns_cfg = {}

    # STEP 1: Load/generate paraphrases
    if input_csv is not None:
        original_col = "original"
        text_col = "text"
        category_col = "category"

        df_input = pd.read_csv(input_csv)

        # determine which column to paraphrase
        if column:
            if column not in df_input.columns:
                raise CalibrateError(
                    f"Column '{column}' not found. Available: {', '.join(df_input.columns)}"
                )
            input_text_col = column
        elif "original" in df_input.columns:
            input_text_col = "original"
        else:
            input_text_col = df_input.columns[0]

        # apply head/sample
        if head is not None:
            df_input = df_input.head(head)
        elif sample is not None:
            if sample <= len(df_input):
                df_input = df_input.sample(n=sample, random_state=seed)

        sentences = df_input[input_text_col].tolist()

        df = generate_paraphrases(
            sentences, prompt, llm_model, max_concurrent, list(targets.keys())
        )

        # join group columns
        if group_col:
            for col in group_col:
                if col not in df_input.columns:
                    raise CalibrateError(
                        f"Group column '{col}' not found. Available: {', '.join(df_input.columns)}"
                    )

            if input_text_col in group_col:
                group_col = [c for c in group_col if c != input_text_col]

            if group_col:
                cols_to_join = [input_text_col] + list(group_col)
                df_groups = df_input[cols_to_join].drop_duplicates()
                df_groups = df_groups.rename(columns={input_text_col: "original"})
                df = df.merge(df_groups, on="original", how="left")
    else:
        original_col = columns_cfg.get("original", "original")
        text_col = columns_cfg.get("text", "text")
        category_col = columns_cfg.get("category", "category")

        df = pd.read_csv(paraphrases_csv)

        if head is not None:
            df = df.head(head)
        elif sample is not None:
            if sample <= len(df):
                df = df.sample(n=sample, random_state=seed)

    # validate columns
    for col in [original_col, text_col, category_col]:
        if col not in df.columns:
            raise CalibrateError(
                f"Missing column '{col}'. Available: {', '.join(df.columns)}"
            )

    # validate categories
    categories = df[category_col].unique()
    missing = set(categories) - set(targets.keys())
    if missing:
        raise CalibrateError(
            f"Categories {missing} not in config targets: {list(targets.keys())}"
        )

    # STEP 2: Compute embeddings and similarities
    similarities = compute_similarities(
        df, embedding_model, embedding_template, original_col, text_col
    )
    df["similarity"] = similarities

    # determine output folder
    if output is None:
        model_slug = embedding_model.replace("/", "-").replace("local-", "")
        hash_parts = [
            str(prompt) if prompt else "",
            str(input_csv) if input_csv else "",
            llm_model,
            embedding_model,
            embedding_template,
            str(spline_df),
            str(n_anchors),
            str(sample) if sample else "",
            str(head) if head else "",
            str(holdout),
            ",".join(group_col) if group_col else "",
            str(seed),
            method,
            str(config) if config else "",
        ]
        args_hash = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:6]
        output_folder = cwd / f"calibration-{model_slug}-{args_hash}"
    else:
        output_folder = cwd / output

    output_folder.mkdir(parents=True, exist_ok=True)

    # save paraphrases
    csv_path = output_folder / "paraphrases.csv"
    df.to_csv(csv_path, index=False)

    # prepare groups
    groups = None
    if group_col:
        for col in group_col:
            if col not in df.columns:
                raise CalibrateError(
                    f"Group column '{col}' not found. Available: {', '.join(df.columns)}"
                )
        groups = df[group_col[0]].values

    # STEP 3: Fit calibration model
    if method == "scam":
        from ..calibration import fit_calibration_scam

        model, validation_stats = fit_calibration_scam(
            similarities,
            df[category_col].tolist(),
            targets,
            df=spline_df,
            holdout_fraction=holdout,
            random_seed=seed,
            groups=groups,
            n_anchors=n_anchors,
        )
    else:
        model, validation_stats = fit_calibration_gam(
            similarities,
            df[category_col].tolist(),
            targets,
            spline_df,
            holdout_fraction=holdout,
            random_seed=seed,
            n_anchors=n_anchors,
            groups=groups,
        )

    # compute category stats
    category_stats = (
        df.groupby(category_col)["similarity"]
        .agg(["mean", "std", "min", "max"])
        .to_dict("index")
    )

    # STEP 4: Save calibration outputs
    output_path = output_folder / "calibration"

    import sys

    cli_info = {
        "command": "api.calibrate()",
        "timestamp": datetime.now().isoformat(),
        "options": {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "embedding_template": embedding_template,
            "df": spline_df,
            "n_anchors": n_anchors,
            "holdout": holdout,
            "seed": seed,
            "method": method,
            "head": head,
            "sample": sample,
        },
    }
    if input_csv is not None:
        cli_info["options"]["input_csv"] = str(input_csv)
        cli_info["options"]["prompt"] = str(prompt)
    else:
        cli_info["options"]["paraphrases_csv"] = str(paraphrases_csv)

    pkl_path, yaml_path = save_calibration(
        model,
        output_path,
        embedding_model,
        embedding_template,
        category_stats,
        targets,
        validation_stats,
        method=method,
        cli_info=cli_info,
        group_columns=group_col if group_col else None,
    )

    # STEP 5: Generate visualisation
    png_path = plot_calibration_curve(
        model,
        similarities,
        df[category_col].tolist(),
        targets,
        output_path,
        category_stats,
        method=method,
    )

    return CalibrationResult(
        model=model,
        method=method,
        output_folder=output_folder,
        pkl_path=pkl_path,
        yaml_path=yaml_path,
        png_path=png_path,
        csv_path=csv_path,
        validation_stats=validation_stats,
        category_stats=category_stats,
        paraphrases_df=df,
    )
