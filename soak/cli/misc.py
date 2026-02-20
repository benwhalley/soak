"""Miscellaneous CLI commands: test, calibrate."""

import hashlib
import logging
from pathlib import Path

import typer

from ._common import check_and_prompt_credentials

logger = logging.getLogger(__name__)


def test():
    """Test LLM and embedding connections with current settings.

    Passes through to struckdown's `sd test` command.

    Examples:
        soak test
    """
    from struckdown.sd_cli import test as sd_test

    # check and prompt for credentials first (soak-specific)
    check_and_prompt_credentials(Path.cwd())

    # delegate to struckdown's test command directly
    sd_test()


def calibrate(
    input: Path = typer.Argument(
        None,
        help="CSV file with sentences/themes to paraphrase",
    ),
    column: str = typer.Option(
        None,
        "--column",
        "-C",
        help="Column name to paraphrase (default: 'original' if exists, else first column)",
    ),
    prompt: Path = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Struckdown prompt file for generating paraphrases",
    ),
    llm_model: str = typer.Option(
        "gpt-4.1-mini",
        "--llm-model",
        "-m",
        envvar="SOAK_MODEL",
        help="LLM model for paraphrase generation",
    ),
    max_concurrent: int = typer.Option(
        20,
        "--max-concurrent",
        help="Max concurrent LLM requests for paraphrase generation",
    ),
    paraphrases: Path = typer.Option(
        None,
        "--paraphrases",
        help="CSV with pre-generated paraphrases (columns: original, text, category)",
    ),
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML config mapping category names to target values (0-1)",
    ),
    embedding_model: str = typer.Option(
        "local/intfloat/e5-base",
        "--embedding-model",
        envvar="SOAK_EMBEDDING_MODEL",
        help="Embedding model (use 'local/model-name' for sentence-transformers)",
    ),
    embedding_template: str = typer.Option(
        "{text}",
        "--embedding-template",
        envvar="SOAK_EMBEDDING_TEMPLATE",
        help="Template for embedding text (use {text} placeholder)",
    ),
    spline_df: int = typer.Option(
        5,
        "--df",
        help="Degrees of freedom for spline (more = more flexible). Default 5.",
    ),
    n_anchors: int = typer.Option(
        0,
        "--n-anchors",
        help="Anchor points at each tail to stabilise curve (GAM only, usually not needed). Default 0.",
    ),
    sample: int = typer.Option(
        None,
        "--sample",
        "-S",
        help="Randomly sample N rows from paraphrases (mutually exclusive with --head)",
    ),
    head: int = typer.Option(
        None,
        "--head",
        "-H",
        help="Take first N rows from paraphrases (mutually exclusive with --sample)",
    ),
    holdout: float = typer.Option(
        0.2,
        "--holdout",
        help="Fraction of data to hold out for validation (0 to disable, default 0.2)",
    ),
    group_col: list[str] = typer.Option(
        None,
        "--group-col",
        "-g",
        help="Column(s) for random effects and grouped holdout. Can specify multiple: -g paper -g theme",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible train/test split and sampling",
    ),
    method: str = typer.Option(
        "scam",
        "--method",
        help="Calibration method: 'scam' (monotonic, requires R), 'gam' (not monotonic)",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output folder name. Creates folder with .pkl, .yaml, .png, and .csv files",
    ),
):
    """Fit a calibration model to map similarity scores to a meaningful scale.

    Two modes for input:

    1. Generate paraphrases from sentences:
       soak calibrate sentences.csv --prompt paraphrases.sd

    2. Use existing paraphrases:
       soak calibrate --paraphrases existing.csv

    Two calibration methods:

    1. SCAM (default): Monotonic spline via R, supports random effects
       soak calibrate --paraphrases data.csv

    2. GAM: Monotonic spline via Python, no random effects
       soak calibrate --paraphrases data.csv --method gam

    The calibration maps raw angular similarity to a 0.1-0.9 scale where:
    - 0.9 = same meaning (paraphrase quality)
    - 0.75 = close meaning
    - 0.5 = diverging (partial overlap)
    - 0.3 = distant (weak relation)
    - 0.1 = unrelated

    The resulting calibration files can be used with `soak compare --calibration`.
    """
    import yaml

    from ..calibration import (DEFAULT_TARGETS, compute_similarities,
                               fit_calibration_gam, generate_paraphrases,
                               plot_calibration_curve, save_calibration)

    # validate method
    if method not in ("scam", "gam"):
        logger.error(f"Invalid method '{method}'. Use 'scam' or 'gam'")
        raise typer.Exit(1)

    # check R availability for scam
    if method == "scam":
        from ..calibration import _check_r_available

        if not _check_r_available():
            logger.error("R and scam package required for scam calibration.")
            logger.error("Install with: pip install 'soaking[calibration]'")
            logger.error("Then in R: install.packages('scam')")
            raise typer.Exit(1)

    # validate inputs
    if input is None and paraphrases is None:
        logger.error("Must provide either input sentences or --paraphrases")
        raise typer.Exit(1)

    if input is not None and prompt is None:
        logger.error("--prompt required when providing input sentences")
        raise typer.Exit(1)

    if input is not None and paraphrases is not None:
        logger.error("Cannot use both input sentences and --paraphrases")
        raise typer.Exit(1)

    # load config
    if config:
        cfg = yaml.safe_load(config.read_text())
        targets = cfg.get("targets", DEFAULT_TARGETS)
        columns = cfg.get("columns", {})
    else:
        targets = DEFAULT_TARGETS
        columns = {}

    import pandas as pd

    # STEP 1: Load/generate paraphrases
    print("\n[1/5] Loading paraphrases...")
    # validate head/sample mutual exclusivity early
    if head is not None and sample is not None:
        logger.error("Cannot use both --head and --sample")
        raise typer.Exit(1)

    if input is not None:
        # when generating, we use standard column names
        original_col = "original"
        text_col = "text"
        category_col = "category"

        # check credentials for LLM
        if not embedding_model.startswith("local/"):
            check_and_prompt_credentials(Path.cwd())

        # load and optionally filter input sentences BEFORE generating
        df_input = pd.read_csv(input)

        # determine which column to paraphrase
        if column:
            if column not in df_input.columns:
                logger.error(f"Column '{column}' not found in input file")
                logger.info(f"Available columns: {', '.join(df_input.columns)}")
                raise typer.Exit(1)
            input_text_col = column
        elif "original" in df_input.columns:
            input_text_col = "original"
        else:
            input_text_col = df_input.columns[0]
            logger.info(
                f"Using first column '{input_text_col}' for paraphrasing (use --column to specify)"
            )

        n_total = len(df_input)

        # apply head/sample to input sentences
        if head is not None:
            df_input = df_input.head(head)
            print(f"  Input: {input} (using first {head} of {n_total} sentences)")
        elif sample is not None:
            if sample > n_total:
                logger.warning(
                    f"Sample size {sample} exceeds data size {n_total}, using all data"
                )
            else:
                df_input = df_input.sample(n=sample, random_state=seed)
                print(
                    f"  Input: {input} (sampled {sample} of {n_total} sentences, seed={seed})"
                )
        else:
            print(f"  Input: {input} ({n_total} sentences)")

        sentences = df_input[input_text_col].tolist()
        print(f"  Column: {input_text_col}")
        print(f"  LLM: {llm_model}")
        print(f"  Prompt: {prompt}")
        print(f"  Categories: {', '.join(targets.keys())}")

        df = generate_paraphrases(
            sentences, prompt, llm_model, max_concurrent, list(targets.keys())
        )

        # join group columns from input data to generated paraphrases
        if group_col and len(group_col) > 0:
            # validate group columns exist in input
            for col in group_col:
                if col not in df_input.columns:
                    logger.error(f"Group column '{col}' not found in input file")
                    logger.info(f"Available columns: {', '.join(df_input.columns)}")
                    raise typer.Exit(1)

            # exclude the paraphrase column from group cols to avoid duplicates
            if input_text_col in group_col:
                logger.warning(
                    f"Ignoring '{input_text_col}' as group column since it's the paraphrase source column"
                )
                # update group_col for downstream code
                group_col = [c for c in group_col if c != input_text_col]

            # create mapping from original text to group values
            if group_col:
                cols_to_join = [input_text_col] + list(group_col)
                df_groups = df_input[cols_to_join].drop_duplicates()
                df_groups = df_groups.rename(columns={input_text_col: "original"})

                # merge group columns into paraphrases
                df = df.merge(df_groups, on="original", how="left")
    else:
        # when loading existing, use config column names (or defaults)
        original_col = columns.get("original", "original")
        text_col = columns.get("text", "text")
        category_col = columns.get("category", "category")

        print(f"  Loading from {paraphrases}")
        df = pd.read_csv(paraphrases)

        # apply head/sample to loaded paraphrases
        if head is not None:
            df = df.head(head)
            print(f"  Using first {head} rows")
        elif sample is not None:
            if sample > len(df):
                logger.warning(
                    f"Sample size {sample} exceeds data size {len(df)}, using all data"
                )
            else:
                df = df.sample(n=sample, random_state=seed)
                print(f"  Sampled {sample} rows (seed={seed})")

    # validate required columns
    for col in [original_col, text_col, category_col]:
        if col not in df.columns:
            logger.error(f"Missing column '{col}' in paraphrases data")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            raise typer.Exit(1)

    # validate categories match targets
    categories = df[category_col].unique()
    missing = set(categories) - set(targets.keys())
    if missing:
        logger.error(
            f"Categories {missing} not in config targets: {list(targets.keys())}"
        )
        raise typer.Exit(1)

    # show data summary
    n_unique_originals = df[original_col].nunique()
    print(
        f"  Loaded {len(df)} paraphrase pairs from {n_unique_originals} unique originals"
    )
    print(f"  Categories: {', '.join(sorted(categories))}")

    # STEP 2: Compute embeddings and similarities
    print(f"\n[2/5] Computing embeddings and similarities...")
    print(f"  Model: {embedding_model}")
    print(f"  Template: {embedding_template}")
    similarities = compute_similarities(
        df, embedding_model, embedding_template, original_col, text_col
    )
    df["similarity"] = similarities
    print(
        f"  Done. Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]"
    )

    # determine output folder early so we can save intermediate results
    if output is None:
        model_slug = embedding_model.replace("/", "-").replace("local-", "")
        # hash key args to make folder name unique for different configurations
        hash_parts = [
            str(prompt) if prompt else "",
            str(input) if input else "",
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
        output_folder = Path(f"calibration-{model_slug}-{args_hash}")
    else:
        output_folder = Path(output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # save paraphrases early (before model fitting which can be slow/aborted)
    csv_path = output_folder / "paraphrases.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved paraphrases: {csv_path}")

    # prepare groups for holdout/mixed model
    groups = None
    if group_col and len(group_col) > 0:
        # validate all group columns exist
        for col in group_col:
            if col not in df.columns:
                logger.error(f"Group column '{col}' not found in data")
                logger.info(f"Available columns: {', '.join(df.columns)}")
                raise typer.Exit(1)

        # for grouped holdout, use first column (most coarse grouping)
        groups = df[group_col[0]].values
        print(
            f"  Using grouped holdout by '{group_col[0]}' ({len(df[group_col[0]].unique())} groups)"
        )

    # STEP 3: Fit calibration model
    print(f"\n[3/5] Fitting calibration model...")
    if method == "scam":
        from ..calibration import fit_calibration_scam

        re_info = f" + RE ({group_col[0]})" if groups is not None else ""
        print(f"  Method: Monotonic scam (R){re_info}")
        print(f"  df: {spline_df}, Anchors: {n_anchors}, Holdout: {holdout:.0%}")
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
    else:  # gam
        print(f"  Method: Monotonic GAM (Python)")
        print(f"  df: {spline_df}, Anchors: {n_anchors}, Holdout: {holdout:.0%}")
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
    print("  Model fitted successfully")

    # compute category stats for metadata
    category_stats = (
        df.groupby(category_col)["similarity"]
        .agg(["mean", "std", "min", "max"])
        .to_dict("index")
    )

    # STEP 4: Save calibration outputs
    print(f"\n[4/5] Saving calibration outputs...")
    print(f"  Output folder: {output_folder}/")

    # all files go inside the folder with consistent names
    output_path = output_folder / "calibration"

    # build CLI info for metadata
    import sys
    from datetime import datetime

    cli_info = {
        "command": " ".join(sys.argv),
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
    if input is not None:
        cli_info["options"]["input"] = str(input)
        cli_info["options"]["prompt"] = str(prompt)
    else:
        cli_info["options"]["paraphrases"] = str(paraphrases)

    # save calibration model
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
    print(f"  Saved model: {pkl_path.name}")
    print(f"  Saved metadata: {yaml_path.name}")

    # paraphrases already saved earlier as paraphrases.csv

    # STEP 5: Generate visualisation
    print(f"\n[5/5] Generating calibration plot...")
    png_path = plot_calibration_curve(
        model,
        similarities,
        df[category_col].tolist(),
        targets,
        output_path,
        category_stats,
        method=method,
    )
    print(f"  Saved plot: {png_path.name}")

    # print summary
    print("=" * 50)
    print("Calibration Complete")
    print("=" * 50)
    print(f"\nMethod: {method}")
    print(f"Embedding model: {embedding_model}")
    print(f"Paraphrases: {len(df)}")
    print(f"Categories: {list(targets.keys())}")
    print("\nCategory Statistics (raw angular similarity):")
    for cat, stats in category_stats.items():
        print(f"  {cat}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # display validation results if holdout was used
    if validation_stats:
        print("\nHoldout Validation:")
        print(
            f"  Train/Test: {validation_stats['n_train']}/{validation_stats['n_test']} ({holdout:.0%} holdout)"
        )
        print(f"  MAE: {validation_stats['mae']:.3f}")
        print(f"  Category Accuracy: {validation_stats['category_accuracy']:.1%}")

    # test calibration with example values
    import numpy as np

    test_values = np.array([0.78, 0.83, 0.87, 0.89, 0.92, 0.95])
    print("\nExample transformations:")

    if method == "scam":
        # scam uses lookup table
        calibrated = np.clip(
            np.interp(test_values, model["x_lookup"], model["y_lookup"]), 0, 1
        )
    else:  # gam
        calibrated = np.clip(model.predict(test_values.reshape(-1, 1)), 0, 1)

    for raw, cal in zip(test_values, calibrated):
        print(f"  {raw:.2f} -> {cal:.2f}")

    # output files summary
    print(f"\nOutput: {output_folder}/")
    print(f"  {pkl_path.name}, {yaml_path.name}, {png_path.name}, {csv_path.name}")
    print(f"\nUsage: soak compare ... --calibration {output_folder}")
