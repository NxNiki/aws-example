import numpy as np
import pandas as pd
import pytest

from bituslabs_ds.utils import remove_outliers


# -------------------------
# Helper function to create data
# -------------------------
def generate_data_with_outliers(n_samples=50, n_features=3, outlier_indices=None, outlier_value=100):
    """
    Generate synthetic data with controlled outliers.

    :param n_samples: total number of rows
    :param n_features: number of columns
    :param outlier_indices: list of row indices to inject outliers
    :param outlier_value: value of outliers
    :return: data (numpy array), expected_mask
    """
    np.random.seed(42)  # reproducible
    data = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))
    expected_mask = np.ones(n_samples, dtype=bool)

    if outlier_indices is not None:
        for idx in outlier_indices:
            col = idx % n_features  # choose column to inject outlier
            data[idx, col] = outlier_value
            expected_mask[idx] = False

    return data, expected_mask


# -------------------------
# Tests
# -------------------------
def test_numpy_array_outlier_removal():
    arr, expected_mask = generate_data_with_outliers(n_samples=50, n_features=3, outlier_indices=[10, 20, 30])
    filtered, mask = remove_outliers(arr, z_thresh=3)
    expected_filtered = arr[expected_mask]

    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(filtered, expected_filtered)


def test_dataframe_outlier_removal():
    arr, expected_mask = generate_data_with_outliers(n_samples=50, n_features=3, outlier_indices=[5, 15, 25])
    df = pd.DataFrame(arr, columns=["A", "B", "C"])
    filtered, mask = remove_outliers(df, z_thresh=3)
    expected_filtered = df.loc[expected_mask]

    pd.testing.assert_frame_equal(filtered, expected_filtered)
    assert np.array_equal(mask, expected_mask)


def test_multiple_outliers():
    arr, expected_mask = generate_data_with_outliers(n_samples=50, n_features=4, outlier_indices=[1, 2, 10, 20])
    df = pd.DataFrame(arr, columns=["A", "B", "C", "D"])
    filtered, mask = remove_outliers(df, z_thresh=3)
    expected_filtered = df.loc[expected_mask]

    pd.testing.assert_frame_equal(filtered, expected_filtered)
    assert np.array_equal(mask, expected_mask)


def test_no_outliers():
    arr = np.random.normal(0, 1, size=(50, 3))
    expected_mask = np.ones(50, dtype=bool)

    filtered, mask = remove_outliers(arr, z_thresh=5)
    expected_filtered = arr

    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(filtered, expected_filtered)


def test_1d_array_removal():
    arr = np.random.normal(0, 1, size=50)
    arr[25] = 10  # inject outlier
    expected_mask = np.ones(50, dtype=bool)
    expected_mask[25] = False

    filtered, mask = remove_outliers(arr, z_thresh=3)
    expected_filtered = arr[expected_mask]

    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(filtered, expected_filtered)
