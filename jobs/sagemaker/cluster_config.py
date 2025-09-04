"""
Lightweight configuration loader for cluster analysis.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# Load configuration from YAML
_config_path = Path(__file__).parent / "cluster_config.yaml"
with open(_config_path, "r") as f:
    _config: Dict[str, Any] = yaml.safe_load(f)

# Extract commonly used values
ENVIRONMENT = _config["environment"]["current"]
USE_CNY = _config["analysis"]["use_cny"]
WORK_DIR = Path(__file__).parent / _config["directories"]["work_dir"]
OUTPUT_PATH = Path(__file__).parent / _config["directories"]["output_path"]
DEFAULT_N_CLUSTERS = _config["analysis"]["default_n_clusters"]
DEFAULT_TOP_FEATURES = _config["analysis"]["default_top_features"]
CORRELATION_THRESHOLD = _config["feature_selection"]["correlation_threshold"]
VARIANCE_THRESHOLD = _config["feature_selection"]["variance_threshold"]
KMEANS_RANDOM_STATE = _config["kmeans"]["random_state"]
KMEANS_N_INIT = _config["kmeans"]["n_init"]
ELBOW_K_RANGE = _config["elbow_method"]["k_range"]
