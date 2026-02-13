"""Calibration model fitting and application for similarity scores."""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .models.base import get_embedding

logger = logging.getLogger(__name__)


def _check_r_available():
    """Check if R and required packages are available."""
    try:
        from rpy2 import robjects
        from rpy2.robjects.packages import importr

        importr("scam")
        return True
    except Exception as e:
        logger.warning(f"R/scam not available: {e}")
        return False


def fit_calibration_scam(
    similarities: np.ndarray,
    categories: list[str],
    targets: dict[str, float],
    df: int = 10,
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[dict, dict | None]:
    """Fit monotonic scam model via R for calibration.

    Uses R's scam package with bs="mpi" for monotone increasing P-splines.
    Optionally includes random effects if groups are provided.

    Args:
        similarities: Array of raw angular similarity scores
        categories: List of category labels for each similarity
        targets: Dict mapping category names to target values (0-1)
        df: Degrees of freedom for the spline (default 10)
        holdout_fraction: Fraction of data to hold out for validation (0-1)
        random_seed: Random seed for reproducible train/test split
        groups: Optional grouping variable for random effects (e.g., doi)

    Returns:
        Tuple of (lookup_dict, validation_stats or None)
        lookup_dict contains x_lookup and y_lookup arrays for interpolation
    """
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    scam = importr("scam")
    stats = importr("stats")
    base = importr("base")

    # set R seed for reproducibility
    base.set_seed(random_seed)

    # helper to convert pandas df to R
    def to_r_df(df):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            return robjects.conversion.py2rpy(df)

    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    # map categories to target values
    y = np.array([targets[cat] for cat in categories])
    X = similarities
    categories_arr = np.array(categories)

    # create dataframe for R
    df_data = pd.DataFrame(
        {
            "similarity": X,
            "target": y,
        }
    )
    if groups is not None:
        df_data["group_factor"] = pd.Categorical(groups)

    # split into train/test if holdout requested
    validation_stats = None
    if holdout_fraction > 0:
        if groups is not None:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=holdout_fraction, random_state=random_seed
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=holdout_fraction,
                random_state=random_seed,
                stratify=categories_arr,
            )

        df_train = df_data.iloc[train_idx].copy()
        df_test = df_data.iloc[test_idx].copy()
        cats_test = categories_arr[test_idx]
    else:
        df_train = df_data.copy()

    # convert to R dataframe
    r_df_train = to_r_df(df_train)

    # build formula
    if groups is not None:
        formula_str = (
            f"target ~ s(similarity, k={df}, bs='mpi') + s(group_factor, bs='re')"
        )
    else:
        formula_str = f"target ~ s(similarity, k={df}, bs='mpi')"

    logger.info(f"Fitting scam model: {formula_str}")

    # fit model
    model = scam.scam(robjects.Formula(formula_str), data=r_df_train)

    # compute validation stats if holdout was used
    if holdout_fraction > 0:
        r_df_test = to_r_df(df_test)

        if groups is not None:
            # predict excluding random effect for marginal prediction
            y_pred = np.array(
                stats.predict(
                    model, newdata=r_df_test, type="response", exclude="s(group_factor)"
                )
            )
        else:
            y_pred = np.array(stats.predict(model, newdata=r_df_test, type="response"))

        y_pred = np.clip(y_pred, 0, 1)
        y_test = df_test["target"].values

        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))

        # category-level accuracy
        category_values = np.array(list(targets.values()))
        category_names = np.array(list(targets.keys()))

        def predict_category(calibrated_value):
            idx = np.argmin(np.abs(category_values - calibrated_value))
            return category_names[idx]

        predicted_cats = np.array([predict_category(v) for v in y_pred])
        accuracy = np.mean(predicted_cats == cats_test)

        # per-category accuracy
        per_category_accuracy = {}
        for cat in targets.keys():
            mask = cats_test == cat
            if mask.sum() > 0:
                per_category_accuracy[cat] = float(np.mean(predicted_cats[mask] == cat))

        validation_stats = {
            "holdout_fraction": holdout_fraction,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "mse": float(mse),
            "mae": float(mae),
            "category_accuracy": float(accuracy),
            "per_category_accuracy": per_category_accuracy,
        }
        if groups is not None:
            validation_stats["n_groups_train"] = len(np.unique(groups[train_idx]))
            validation_stats["n_groups_test"] = len(np.unique(groups[test_idx]))

    # refit on full data for final model
    if holdout_fraction > 0:
        logger.info("Refitting on full data...")
        r_df_full = to_r_df(df_data)
        model = scam.scam(robjects.Formula(formula_str), data=r_df_full)

    # create lookup table for later interpolation
    x_lookup = np.linspace(0.5, 1.0, 500)
    df_lookup = pd.DataFrame({"similarity": x_lookup})
    if groups is not None:
        # use first group level as placeholder (will be excluded)
        df_lookup["group_factor"] = df_data["group_factor"].cat.categories[0]

    r_df_lookup = to_r_df(df_lookup)

    if groups is not None:
        y_lookup = np.array(
            stats.predict(
                model, newdata=r_df_lookup, type="response", exclude="s(group_factor)"
            )
        )
    else:
        y_lookup = np.array(stats.predict(model, newdata=r_df_lookup, type="response"))

    y_lookup = np.clip(y_lookup, 0, 1)

    # get standard errors for confidence intervals
    try:
        if groups is not None:
            se = np.array(
                stats.predict(
                    model,
                    newdata=r_df_lookup,
                    type="response",
                    se_fit=True,
                    exclude="s(group_factor)",
                ).rx2("se.fit")
            )
        else:
            se = np.array(
                stats.predict(
                    model, newdata=r_df_lookup, type="response", se_fit=True
                ).rx2("se.fit")
            )
        y_lookup_lower = np.clip(y_lookup - 1.96 * se, 0, 1)
        y_lookup_upper = np.clip(y_lookup + 1.96 * se, 0, 1)
    except Exception:
        # if SE extraction fails, just use the point estimates
        y_lookup_lower = y_lookup
        y_lookup_upper = y_lookup

    lookup_dict = {
        "x_lookup": x_lookup,
        "y_lookup": y_lookup,
        "y_lookup_lower": y_lookup_lower,
        "y_lookup_upper": y_lookup_upper,
    }

    return lookup_dict, validation_stats


DEFAULT_TARGETS = {
    "same": 0.9,
    "close": 0.8,
    "diverging": 0.6,
    "distant": 0.4,
    "unrelated": 0.1,
}


def angular_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute angular similarity between embeddings.

    Angular similarity maps cosine similarity to a 0-1 scale that satisfies
    the triangle inequality, making it suitable for distance-based methods.

    Args:
        emb_a: First embedding vector
        emb_b: Second embedding vector

    Returns:
        Angular similarity in [0, 1] where 1 = identical direction
    """
    cos_sim = cosine_similarity(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0, 0]
    angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    return 1 - angle / 180.0


def compute_similarities(
    df: pd.DataFrame,
    embedding_model: str,
    embedding_template: str,
    original_col: str = "original",
    text_col: str = "text",
    show_progress: bool = True,
) -> np.ndarray:
    """Compute angular similarities between originals and paraphrases.

    Args:
        df: DataFrame with original and paraphrase text columns
        embedding_model: Model to use for embeddings (e.g., 'local/intfloat/e5-base')
        embedding_template: Template string with {text} placeholder
        original_col: Column name for original text
        text_col: Column name for paraphrase text
        show_progress: Whether to show progress bars

    Returns:
        Array of angular similarity scores
    """
    # get unique texts for embedding (avoid redundant computation)
    all_texts = list(set(df[original_col].tolist() + df[text_col].tolist()))

    # apply template
    texts_for_embedding = [embedding_template.replace("{text}", t) for t in all_texts]

    # compute embeddings with progress
    pbar = tqdm(
        total=len(all_texts), desc="Computing embeddings", disable=not show_progress
    )

    def progress_callback(n):
        pbar.update(n - pbar.n)

    embeddings = get_embedding(
        texts_for_embedding,
        model=embedding_model,
        progress_callback=progress_callback,
    )
    pbar.close()

    text_to_emb = {t: np.array(e) for t, e in zip(all_texts, embeddings)}

    # compute similarities with progress
    sims = []
    rows = list(df.iterrows())
    for _, row in tqdm(rows, desc="Computing similarities", disable=not show_progress):
        orig_emb = text_to_emb[row[original_col]]
        para_emb = text_to_emb[row[text_col]]
        sims.append(angular_similarity(orig_emb, para_emb))

    return np.array(sims)


def fit_calibration_gam(
    similarities: np.ndarray,
    categories: list[str],
    targets: dict[str, float],
    n_splines: int = 7,
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
    n_anchors: int = 0,
    groups: np.ndarray | None = None,
):
    """Fit monotonic GAM for calibration with optional holdout validation.

    The GAM maps raw angular similarity scores to calibrated values based on
    known category labels. Anchor points at 0 and 1 stabilise the tails.

    Args:
        similarities: Array of raw angular similarity scores
        categories: List of category labels for each similarity
        targets: Dict mapping category names to target values (0-1)
        n_splines: Number of splines (more = more flexible, risk of overfitting)
        holdout_fraction: Fraction of data to hold out for validation (0-1, default 0.2)
        random_seed: Random seed for reproducible train/test split
        n_anchors: Number of anchor points at each tail (0 to disable)
        groups: Optional grouping variable for clustered holdout (e.g., theme_id).
                If provided, entire groups are held out together.

    Returns:
        Tuple of (fitted GAM, validation_stats or None if holdout_fraction=0)
    """
    from pygam import LinearGAM, s
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    # map categories to target values
    y = np.array([targets[cat] for cat in categories])
    X = similarities.reshape(-1, 1)
    categories_arr = np.array(categories)

    # split into train/test if holdout requested
    validation_stats = None
    if holdout_fraction > 0:
        if groups is not None:
            # grouped holdout: hold out entire groups (themes) together
            gss = GroupShuffleSplit(
                n_splits=1, test_size=holdout_fraction, random_state=random_seed
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            cats_train, cats_test = categories_arr[train_idx], categories_arr[test_idx]
            n_groups_train = len(np.unique(groups[train_idx]))
            n_groups_test = len(np.unique(groups[test_idx]))
            logger.info(
                f"Grouped holdout: {n_groups_train} groups in train, {n_groups_test} groups in test"
            )
        else:
            # standard stratified split
            X_train, X_test, y_train, y_test, cats_train, cats_test = train_test_split(
                X,
                y,
                categories_arr,
                test_size=holdout_fraction,
                random_state=random_seed,
                stratify=categories_arr,
            )
    else:
        X_train, y_train = X, y

    # add anchor points to stabilise tails (data-driven bounds)
    # place anchors slightly beyond data range, not at fixed 0/1
    if n_anchors > 0:
        data_min = X_train.min()
        data_max = X_train.max()
        data_range = data_max - data_min
        # anchors at 10% beyond data range on each side
        anchor_low = max(0, data_min - 0.1 * data_range)
        anchor_high = min(1, data_max + 0.1 * data_range)

        anchor_X = np.vstack(
            [
                np.full((n_anchors, 1), anchor_low),
                np.full((n_anchors, 1), anchor_high),
            ]
        )
        anchor_y = np.concatenate(
            [
                np.zeros(n_anchors),
                np.ones(n_anchors),
            ]
        )
        X_full = np.vstack([X_train, anchor_X])
        y_full = np.concatenate([y_train, anchor_y])
    else:
        X_full = X_train
        y_full = y_train

    # fit with monotonic constraint
    gam = LinearGAM(s(0, n_splines=n_splines, constraints="monotonic_inc"))
    gam.fit(X_full, y_full)

    # compute validation stats if holdout was used
    if holdout_fraction > 0:
        y_pred = np.clip(gam.predict(X_test), 0, 1)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))

        # category-level accuracy (predict closest category)
        category_values = np.array(list(targets.values()))
        category_names = np.array(list(targets.keys()))

        def predict_category(calibrated_value):
            idx = np.argmin(np.abs(category_values - calibrated_value))
            return category_names[idx]

        predicted_cats = np.array([predict_category(v) for v in y_pred])
        accuracy = np.mean(predicted_cats == cats_test)

        # per-category accuracy
        per_category_accuracy = {}
        for cat in targets.keys():
            mask = cats_test == cat
            if mask.sum() > 0:
                per_category_accuracy[cat] = np.mean(predicted_cats[mask] == cat)

        validation_stats = {
            "holdout_fraction": holdout_fraction,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mse": float(mse),
            "mae": float(mae),
            "category_accuracy": float(accuracy),
            "per_category_accuracy": {
                k: float(v) for k, v in per_category_accuracy.items()
            },
        }

    # refit on full data for final model
    if holdout_fraction > 0:
        if n_anchors > 0:
            data_min = X.min()
            data_max = X.max()
            data_range = data_max - data_min
            anchor_low = max(0, data_min - 0.1 * data_range)
            anchor_high = min(1, data_max + 0.1 * data_range)
            anchor_X_full = np.vstack(
                [
                    np.full((n_anchors, 1), anchor_low),
                    np.full((n_anchors, 1), anchor_high),
                ]
            )
            anchor_y_full = np.concatenate(
                [
                    np.zeros(n_anchors),
                    np.ones(n_anchors),
                ]
            )
            X_full = np.vstack([X, anchor_X_full])
            y_full = np.concatenate([y, anchor_y_full])
        else:
            X_full = X
            y_full = y
        gam = LinearGAM(s(0, n_splines=n_splines, constraints="monotonic_inc"))
        gam.fit(X_full, y_full)

    return gam, validation_stats


def save_calibration(
    model,
    output_path: Path,
    embedding_model: str,
    embedding_template: str,
    category_stats: dict,
    targets: dict[str, float],
    validation_stats: dict | None = None,
    method: str = "gam",
    cli_info: dict | None = None,
    group_columns: list[str] | None = None,
) -> tuple[Path, Path]:
    """Save fitted calibration model and metadata.

    Creates two files:
    - .pkl: Pickled model (GAM or lookup dict for scam)
    - .yaml: Metadata for validation and documentation

    Args:
        model: Fitted model (LinearGAM or scam lookup dict)
        output_path: Base path (without extension)
        embedding_model: Model used for embeddings
        embedding_template: Template used for embedding text
        category_stats: Statistics per category (mean, std, etc.)
        targets: Target values for each category
        validation_stats: Optional holdout validation statistics
        method: Calibration method ("scam" or "gam")
        cli_info: Optional CLI invocation info (command, options, timestamp)
        group_columns: Column names for random effects (scam only)

    Returns:
        Tuple of (pkl_path, yaml_path)
    """
    output_path = Path(output_path)

    # save model
    pkl_path = output_path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)  # scam is a lookup dict, gam is the LinearGAM model

    # save metadata as YAML
    metadata = {
        "version": "1.0",
        "method": method,
        "embedding_model": embedding_model,
        "embedding_template": embedding_template,
        "calibration_file": pkl_path.name,
        "targets": targets,
        "category_stats": category_stats,
    }
    if group_columns:
        metadata["group_columns"] = group_columns
    if validation_stats:
        metadata["validation"] = validation_stats
    if cli_info:
        metadata["cli"] = cli_info

    import yaml

    yaml_path = output_path.with_suffix(".yaml")
    yaml_path.write_text(yaml.dump(metadata, default_flow_style=False, sort_keys=False))

    return pkl_path, yaml_path


def plot_calibration_curve(
    model,
    similarities: np.ndarray,
    categories: list[str],
    targets: dict[str, float],
    output_path: Path,
    category_stats: dict | None = None,
    method: str = "gam",
) -> Path:
    """Generate and save a calibration curve plot.

    Shows:
    - The fitted calibration curve (with uncertainty for scam)
    - Scatter points coloured by category
    - Category means with error bars
    - Target values as horizontal reference lines

    Args:
        model: Fitted calibration model (LinearGAM or scam lookup dict)
        similarities: Raw angular similarity values
        categories: Category labels for each similarity
        targets: Target values for each category
        output_path: Base path for output (will add .png suffix)
        category_stats: Optional pre-computed stats per category
        method: Calibration method ("scam" or "gam")

    Returns:
        Path to saved PNG file
    """
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # generate smooth curve
    x_smooth = np.linspace(0.5, 1.0, 200)

    if method == "scam" and isinstance(model, dict) and "x_lookup" in model:
        # scam with precomputed lookup table
        y_smooth = np.interp(x_smooth, model["x_lookup"], model["y_lookup"])
        y_lower = np.interp(x_smooth, model["x_lookup"], model["y_lookup_lower"])
        y_upper = np.interp(x_smooth, model["x_lookup"], model["y_lookup_upper"])

        ax.fill_between(
            x_smooth, y_lower, y_upper, alpha=0.2, color="blue", label="95% CI"
        )
        ax.plot(
            x_smooth,
            y_smooth,
            "b-",
            linewidth=2.5,
            label="Calibration curve (monotonic)",
            zorder=10,
        )
    else:
        # GAM: simple point prediction
        y_smooth = np.clip(model.predict(x_smooth.reshape(-1, 1)), 0, 1)
        ax.plot(
            x_smooth,
            y_smooth,
            "b-",
            linewidth=2.5,
            label="Calibration curve",
            zorder=10,
        )

    # colour map for categories
    colours = {
        "same": "#2ecc71",  # green
        "close": "#3498db",  # blue
        "diverging": "#f39c12",  # orange
        "distant": "#e74c3c",  # red
        "unrelated": "#9b59b6",  # purple
    }

    # scatter plot of training data (with jitter on y for visibility)
    categories_arr = np.array(categories)
    for cat in targets.keys():
        mask = categories_arr == cat
        if mask.sum() > 0:
            cat_sims = similarities[mask]
            cat_targets = np.full(mask.sum(), targets[cat])
            # add small jitter to y for visibility
            jitter = np.random.normal(0, 0.02, mask.sum())
            colour = colours.get(cat, "#95a5a6")
            ax.scatter(
                cat_sims,
                cat_targets + jitter,
                c=colour,
                alpha=0.3,
                s=20,
                label=f"{cat} (n={mask.sum()})",
                zorder=5,
            )

    # plot category means with error bars
    if category_stats:
        for cat, stats in category_stats.items():
            mean_sim = stats["mean"]
            std_sim = stats["std"]
            target_val = targets.get(cat, 0.5)
            colour = colours.get(cat, "#95a5a6")
            ax.errorbar(
                mean_sim,
                target_val,
                xerr=std_sim,
                fmt="o",
                markersize=10,
                capsize=5,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=1.5,
                linewidth=2,
                zorder=15,
            )

    # add target reference lines
    for cat, target_val in targets.items():
        ax.axhline(
            y=target_val,
            color=colours.get(cat, "#95a5a6"),
            linestyle="--",
            alpha=0.4,
            linewidth=1,
        )
        ax.text(
            0.705, target_val + 0.01, cat, fontsize=9, color=colours.get(cat, "#95a5a6")
        )

    # labels and formatting
    ax.set_xlabel("Raw Angular Similarity", fontsize=12)
    ax.set_ylabel("Calibrated Score", fontsize=12)
    ax.set_title("Calibration Curve: Raw Similarity → Calibrated Score", fontsize=14)
    ax.set_xlim(0.50, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # save
    png_path = output_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return png_path


def load_calibration(
    path: Path,
    embedding_model: Optional[str] = None,
) -> tuple:
    """Load calibration model and metadata.

    Args:
        path: Path to .yaml, .json, or .pkl calibration file
        embedding_model: If provided, warn if different from trained model

    Returns:
        For GAM: Tuple of (model, metadata)
        For bambi: Tuple of (lookup_dict, metadata) where lookup_dict contains x/y arrays
    """
    import yaml

    path = Path(path)

    # find metadata file (prefer yaml, fall back to json for backwards compatibility)
    if path.suffix == ".pkl":
        yaml_path = path.with_suffix(".yaml")
        json_path = path.with_suffix(".json")
    elif path.suffix == ".yaml":
        yaml_path = path
        json_path = path.with_suffix(".json")
    else:
        yaml_path = path.with_suffix(".yaml")
        json_path = path

    if yaml_path.exists():
        metadata = yaml.safe_load(yaml_path.read_text())
        meta_path = yaml_path
    elif json_path.exists():
        metadata = json.loads(json_path.read_text())
        meta_path = json_path
    else:
        raise FileNotFoundError(
            f"Calibration metadata not found: {yaml_path} or {json_path}"
        )

    if embedding_model and metadata["embedding_model"] != embedding_model:
        warnings.warn(
            f"Calibration trained on {metadata['embedding_model']} "
            f"but using {embedding_model}. Results may be inaccurate."
        )

    pkl_path = meta_path.parent / metadata["calibration_file"]
    if not pkl_path.exists():
        raise FileNotFoundError(f"Calibration model not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    return loaded, metadata


def calibrate(similarity: np.ndarray, model, method: str = "scam") -> np.ndarray:
    """Apply calibration to similarity scores.

    Args:
        similarity: Array of raw angular similarity scores
        model: Fitted calibration model (LinearGAM) or lookup dict (scam)
        method: Calibration method ("scam" or "gam")

    Returns:
        Calibrated scores clipped to [0, 1]
    """
    shape = similarity.shape
    flat_sim = similarity.reshape(-1)

    if method == "scam":
        # scam: interpolate from precomputed lookup table
        x_lookup = model["x_lookup"]
        y_lookup = model["y_lookup"]
        pred = np.interp(flat_sim, x_lookup, y_lookup)
    else:
        # GAM: direct prediction
        pred = model.predict(flat_sim.reshape(-1, 1))

    return np.clip(pred, 0.0, 1.0).reshape(shape)


def generate_paraphrases(
    sentences: list[str],
    prompt_path: Path,
    llm_model: str,
    max_concurrent: int,
    categories: list[str],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Generate paraphrases using struckdown prompt.

    Args:
        sentences: List of sentences to paraphrase
        prompt_path: Struckdown prompt file
        llm_model: LLM model for generation
        max_concurrent: Max concurrent LLM requests
        categories: List of category names to extract from results
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with columns: original, text, category
    """
    from struckdown import LLM, chatter

    # load prompt
    prompt = prompt_path.read_text()

    # generate paraphrases with progress bar
    contexts = [{"theme": s} for s in sentences]
    pbar = tqdm(
        total=len(contexts), desc="Generating paraphrases", disable=not show_progress
    )

    def on_complete(index, result):
        pbar.update(1)

    results = chatter(
        prompt,
        contexts,
        model=LLM(model=llm_model),
        max_concurrent=max_concurrent,
        on_complete=on_complete,
    )
    pbar.close()

    # parse results into records
    records = []
    for sentence, result in zip(sentences, results):
        if isinstance(result, Exception):
            logger.warning(f"Failed to generate paraphrases for: {sentence[:50]}...")
            continue

        for key, seg_result in result.results.items():
            # find which category this slot belongs to
            for cat in categories:
                if cat in key:
                    output = seg_result.output
                    if isinstance(output, list):
                        for text in output:
                            records.append(
                                {
                                    "original": sentence,
                                    "text": str(text),
                                    "category": cat,
                                }
                            )
                    else:
                        records.append(
                            {
                                "original": sentence,
                                "text": str(output),
                                "category": cat,
                            }
                        )
                    break

    logger.info(f"Generated {len(records)} paraphrase pairs")
    return pd.DataFrame(records)
