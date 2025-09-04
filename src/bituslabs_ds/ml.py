"""
WIP
"""

from typing import Union

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import silhouette_score


class ClusterAnalysis(object):

    def __init__(
        self,
    ):

        pass


def calculate_inertia(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the inertia (within-cluster sum of squared distances) for a given clustering.
    This is useful when we use some pacakge that does not report inertia directly (i.e. kmean_equal)

    Args:
        X (np.ndarray): The feature data, with shape (n_samples, n_features).
        y (np.ndarray): The cluster labels for each data point, with shape (n_samples,).

    Returns:
        float: The total inertia score.
    """

    unique_clusters = np.unique(y)
    total_inertia = 0.0

    for cluster_id in unique_clusters:
        cluster_points = x[y == cluster_id]

        if len(cluster_points) > 0:
            cluster_centroid = np.mean(cluster_points, axis=0)
            squared_distances = np.sum((cluster_points - cluster_centroid) ** 2, axis=1)
            total_inertia += np.sum(squared_distances)

    return total_inertia


def calculate_silhouette_score(x: Union[np.ndarray, pd.DataFrame], labels: np.ndarray) -> float:

    if len(np.unique(labels)) < 2:
        return float("nan")

    try:
        with parallel_backend("loky"):
            # a small sample size may lead to small cluster totally omitted!
            score = silhouette_score(x, labels, sample_size=min(5000, x.shape[0]), random_state=42)
    except ValueError:
        score = float("nan")
    return score
