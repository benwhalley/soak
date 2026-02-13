"""Similarity matrix calibration."""

from pathlib import Path
from typing import Optional

import numpy as np


def apply_calibration(
    sim_matrix: np.ndarray,
    calibration_path: Optional[Path] = None,
) -> np.ndarray:
    """Apply calibration to a similarity matrix if a calibration path is provided.

    Args:
        sim_matrix: Similarity matrix with values nominally in [0, 1]
        calibration_path: Path to calibration .yaml/.json file. If None, returns a copy unchanged.

    Returns:
        Calibrated similarity matrix with values in [0, 1]
    """
    if calibration_path is None:
        return sim_matrix.copy()

    from soak.calibration import calibrate as _calibrate
    from soak.calibration import load_calibration

    model, metadata = load_calibration(calibration_path)
    method = metadata.get("method", "gam")
    return _calibrate(sim_matrix, model, method=method)
