"""Similarity matrix rescaling methods."""

from typing import Literal

import numpy as np

# rescaling method type
RescaleMethod = Literal["off", "clip", "adaptive", "sigmoid", "temperature", "rank"]


def rescale_similarity(
    sim_matrix: np.ndarray,
    method: RescaleMethod = "clip",
    # clip method parameters
    rescale_min: float = 0.5,
    rescale_max: float = 0.9,
    # adaptive method parameters
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    # sigmoid method parameters
    sigmoid_center: float = 0.7,
    sigmoid_steepness: float = 10.0,
    # temperature method parameters
    temperature: float = 0.5,
) -> np.ndarray:
    """Rescale similarity matrix using various methods.

    Methods:
        - "off": No rescaling, return original matrix
        - "clip": Hard clip to [rescale_min, rescale_max] and stretch to [0, 1]
        - "adaptive": Use percentile-based bounds from the data itself
        - "sigmoid": Soft clipping using sigmoid function (preserves tails)
        - "temperature": Power transformation (preserves ordering, changes spacing)
        - "rank": Convert to percentile ranks (most robust, uniform output)

    Args:
        sim_matrix: Similarity matrix with values nominally in [0, 1]
        method: Rescaling method to use

        # clip method parameters
        rescale_min: Floor value for clip method (default: 0.5)
        rescale_max: Ceiling value for clip method (default: 0.9)

        # adaptive method parameters
        lower_percentile: Lower percentile for adaptive bounds (default: 5.0)
        upper_percentile: Upper percentile for adaptive bounds (default: 95.0)

        # sigmoid method parameters
        sigmoid_center: Center point for sigmoid (default: 0.7)
        sigmoid_steepness: Steepness of sigmoid curve (default: 10.0)
            Higher = sharper transition, lower = gentler

        # temperature method parameters
        temperature: Temperature for power transformation (default: 0.5)
            < 1.0 = sharpens (spreads high values), > 1.0 = flattens

    Returns:
        Rescaled similarity matrix with values in [0, 1]
    """
    from scipy.stats import rankdata

    if method == "off":
        return sim_matrix.copy()

    elif method == "clip":
        # original hard clipping method
        clipped = np.clip(sim_matrix, rescale_min, rescale_max)
        rescaled = (clipped - rescale_min) / (rescale_max - rescale_min)
        return rescaled

    elif method == "adaptive":
        # percentile-based bounds from actual data
        p_lo = np.percentile(sim_matrix, lower_percentile)
        p_hi = np.percentile(sim_matrix, upper_percentile)
        if p_hi <= p_lo:
            # degenerate case: all values same
            return np.ones_like(sim_matrix) * 0.5
        clipped = np.clip(sim_matrix, p_lo, p_hi)
        rescaled = (clipped - p_lo) / (p_hi - p_lo)
        return rescaled

    elif method == "sigmoid":
        # soft clipping using sigmoid - preserves structure in tails
        # sigmoid(x) = 1 / (1 + exp(-steepness * (x - center)))
        z = sigmoid_steepness * (sim_matrix - sigmoid_center)
        # clip z to avoid overflow
        z = np.clip(z, -500, 500)
        rescaled = 1.0 / (1.0 + np.exp(-z))
        return rescaled

    elif method == "temperature":
        # power transformation: sim^(1/temperature)
        # temperature < 1 sharpens (spreads out high values)
        # temperature > 1 flattens (compresses high values)
        # ensure non-negative
        sim_clipped = np.clip(sim_matrix, 0, 1)
        rescaled = np.power(sim_clipped, 1.0 / temperature)
        return rescaled

    elif method == "rank":
        # convert to percentile ranks - most robust, preserves all ordering
        # output is uniformly distributed in [0, 1]
        flat = sim_matrix.flatten()
        ranks = rankdata(flat, method="average")
        # normalize to [0, 1]
        rescaled_flat = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(ranks)
        rescaled = rescaled_flat.reshape(sim_matrix.shape)
        return rescaled

    else:
        raise ValueError(f"Unknown rescale method: {method}")
