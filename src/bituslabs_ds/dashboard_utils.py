"""
Lightweight utils for the dashboard. No scipy/sklearn to keep the dashboard image lean.
"""

import logging
from pprint import pformat
from typing import Any, Dict, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_file}:")
    logger.info(f"config: \n {pformat(config)} \n")
    return config


def bootstrap_worker(arr_np: np.ndarray, n_boot: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Top-level function required for ProcessPoolExecutor pickling.
    Calculates bootstrap CI for a single array.
    """
    if len(arr_np) == 0:
        return float("nan"), float("nan")
    if np.min(arr_np) == np.max(arr_np):
        return float(arr_np[0]), float(arr_np[0])

    resamples = np.random.choice(arr_np, size=(n_boot, len(arr_np)), replace=True)
    boot_means = np.mean(resamples, axis=1)
    lower = float(np.percentile(boot_means, (1 - ci_level) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci_level) / 2 * 100))
    return lower, upper
