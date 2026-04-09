import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from functools import lru_cache
from pprint import pformat
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from numpy.core.defchararray import upper
from pandas.api.types import is_numeric_dtype
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import power_transform

from bituslabs_ds.config import DEFAULT_MAX_JOBS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_data_from_url(url: str, output_filename: str, extraction_dir: str, overwrite=False):
    """
    Downloads a zip file from a given URL and unzips its contents locally.

    Args:
        url (str): The URL of the zip file to download.
        output_filename (str): The name to save the downloaded zip file as.
        extraction_dir (str): The directory to extract the contents into.
        overwrite (bool): If True, delete the extraction directory
                                   and the zip file before downloading and unzipping.
                                   If False, and the extraction_dir already exists
                                   and is not empty, the download/unzip is skipped.
    Returns:
        bool: True if the download and extraction were successful or skipped, False otherwise.
    """

    if not overwrite:
        if os.path.exists(extraction_dir) and os.path.isdir(extraction_dir):
            if os.listdir(extraction_dir):
                print(
                    f"Directory '{extraction_dir}' already exists and is not empty. Skipping download and extraction."
                )
                return True
            else:
                print(f"Directory '{extraction_dir}' exists but is empty. Proceeding with download.")
        elif os.path.exists(output_filename):
            print(
                f"Zip file '{output_filename}' exists. Consider running with overwrite_existing=True if you want to re-extract or delete it."
            )
            # We'll still try to unzip if the dir is empty or doesn't exist,
            # but won't re-download the zip if it's already there and not overwriting.

    if overwrite:
        print(f"'{overwrite}' is True. Cleaning up '{extraction_dir}' and '{output_filename}' before proceeding...")
        if os.path.exists(extraction_dir) and os.path.isdir(extraction_dir):
            try:
                shutil.rmtree(extraction_dir)
                print(f"Removed existing directory: {extraction_dir}")
            except OSError as e:
                print(f"Error removing directory {extraction_dir}: {e}")
                return False
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                print(f"Removed existing file: {output_filename}")
            except OSError as e:
                print(f"Error removing file {output_filename}: {e}")
                return False

    # 1. Download the file using wget
    # Only download if the output_filename doesn't exist OR if overwrite is True
    if not os.path.exists(output_filename) or overwrite:
        try:
            print(f"Downloading {url}...")
            subprocess.run(["wget", url, "-O", output_filename], check=True)
            print(f"Downloaded {output_filename}")
        except FileNotFoundError:
            print("Error: 'wget' command not found. Please ensure wget is installed and in your system's PATH.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Error downloading the file: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            return False
    else:
        print(f"Skipping download: '{output_filename}' already exists (overwrite_existing is False).")

    # 2. Unzip the file
    try:
        print(f"Unzipping {output_filename}...")
        # Ensure extraction_dir exists or will be created by unzip
        # If the directory exists but is empty, unzip will populate it.
        # If we are not overwriting and the directory is already populated, we would have skipped earlier.
        # So at this point, if extraction_dir exists, it's either empty or we're overwriting it.
        subprocess.run(["unzip", output_filename, "-d", extraction_dir], check=True)
        print(f"Unzipped {output_filename} into {extraction_dir}")
        return True
    except FileNotFoundError:
        print("Error: 'unzip' command not found. Please ensure unzip is installed and in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error unzipping the file: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during unzipping: {e}")
        return False


def save_list(items: List[Any], filepath: str, output_format: Literal["json", "python"] = "json") -> None:
    """
    Saves a list of strings to a file in either JSON or Python list format.

    Parameters:
        items (List[BasicType]): The list of strings, ints, floats, or bools to save.
        filepath (str): The output file path.
        output_format (str): Format to save: "json" or "python". Default is "json".
    """
    if output_format == "json":
        with open(filepath, "w") as f:
            json.dump(items, f, indent=4)
    elif output_format == "python":
        with open(filepath, "w") as f:
            f.write("my_list = [\n")
            for item in items:
                f.write(f"    {repr(item)},\n")
            f.write("]\n")
    else:
        message = f"Unsupported format: '{output_format}' in save_string_list. Expected 'json' or 'python'."
        logger.error(message)
        raise ValueError(message)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_file}:")
    logger.info(f"config: \n {pformat(config)} \n")
    return config


def remove_outliers(
    data: Union[np.ndarray, pd.DataFrame], z_thresh: float = 3
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """
    Remove samples (rows) of data that has zscore above a certain threshold.
    It is suggested to remove outliers before power transform and scaling.
    :param data:
    :param z_thresh:
    :return:
    """
    mask = np.ones(data.shape[0], dtype=bool)

    if z_thresh == 0:
        logger.info("skip remove outlier with 0 threshold")
        return data, mask

    if isinstance(data, pd.DataFrame):
        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        arr = np.asarray(data[numeric_columns])
    else:
        arr = np.asarray(data)

    # Ensure 2D
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Compute mean and std per column
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    std[std == 0] = 1.0  # avoid division by zero

    # Compute boolean mask without creating z_scores array
    for col in range(arr.shape[1]):
        col_mask = np.abs(arr[:, col] - mean[col]) < z_thresh * std[col]
        mask &= col_mask  # remove rows where any column exceeds threshold

    logger.info(
        f"remove {len(mask)-sum(mask)} outliers out of {len(mask)} samples, ratio: {1 - sum(mask)/len(mask):.3f}"
    )
    if isinstance(data, pd.DataFrame):
        return data.loc[mask], mask
    else:
        # If input was 1D, return 1D
        if mask.shape[0] == arr.shape[0] and arr.shape[1] == 1:
            return arr[mask, 0], mask
        return arr[mask], mask


def clip_outliers(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], List[float], List[float]]:
    """
    Clips the extreme positive and negative values in each column independently,
    preserving the central mass of zeros.

    Args:
        data (pd.DataFrame, pd.Series, or np.ndarray): The input data (features).
        lower_quantile (float): The quantile used for the negative tail clipping
                                (e.g., 0.01 will clip the bottom 1% of negative values).
        upper_quantile (float): The quantile used for the positive tail clipping
                                (e.g., 0.99 will clip the top 1% of positive values).

    Returns:
        pd.DataFrame, pd.Series, or np.ndarray: The data with non-zero tails clipped,
                                                matching the type and dimension of the input.
    """

    if (lower_quantile == 0 or lower_quantile == 1) and (upper_quantile == 0 or upper_quantile == 1):
        logger.info(f"skip clip outliers")
        return data, [], []

    if isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        df = pd.DataFrame({"col_0": data})
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or a numpy array.")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    upper_bounds = []
    lower_bounds = []
    for col_name in numeric_columns:
        positives = df.loc[df[col_name] > 0, col_name]
        negatives = df.loc[df[col_name] < 0, col_name]

        if not positives.empty:
            upper_bound = positives.quantile(upper_quantile)
            upper_bounds.append(upper_bound)
            df.loc[df[col_name] > upper_bound, col_name] = upper_bound
            logger.info(f"apply clip to positive values in {col_name}, upper bound: {upper_bound}")
        else:
            upper_bounds.append(np.nan)

        if not negatives.empty:
            lower_bound = negatives.quantile(lower_quantile)
            lower_bounds.append(lower_bound)
            df.loc[df[col_name] < lower_bound, col_name] = lower_bound
            logger.info(f"apply clip to negative values in {col_name}, lower bound: {lower_bound}")
        else:
            lower_bounds.append(np.nan)

    if isinstance(data, pd.Series):
        return df.iloc[:, 0], upper_bounds, lower_bounds
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        return df.iloc[:, 0].values, upper_bounds, lower_bounds
    elif isinstance(data, pd.DataFrame):
        return df, upper_bounds, lower_bounds
    else:
        return df.to_numpy(), upper_bounds, lower_bounds


def keep_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes non-numeric columns from a pandas DataFrame and logs a warning.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing only numeric columns.
    """
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

    if non_numeric_cols:
        message = f"Non-numeric columns removed from DataFrame: {non_numeric_cols}"
        logger.warning(message)
        df = df.drop(columns=non_numeric_cols)

    return df


def _transform_column(data_col):
    col_name = data_col.name
    # Convert all pandas extension types to their underlying numpy dtype for correct type checking
    # For pandas extension dtypes (e.g., Int64), use pandas.api.types to check if numeric
    dtype = data_col.dtype
    if is_numeric_dtype(dtype):
        data_col = data_col.astype(float).copy()
        data_col = power_transform(data_col.to_frame())
        return (col_name, data_col)
    else:
        logger.warning(f"skip power transform column: {col_name}.")
        return None


def df_power_transform(
    data: pd.DataFrame,
    col_names: Optional[Union[List[str], str]] = None,
    suffix="",
) -> pd.DataFrame:
    """
    Apply log/exp transformation to selected numeric columns of a DataFrame.
    Handles negative values by preserving their sign: sign(x) * log(1 + abs(x))
    For details, see: https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/preprocessing/_data.py#L3187
    0 will be preserved.

    :param data: Input DataFrame
    :param col_names: List of column names to transform
    :param suffix: Suffix to add to column names, default is "":
    :return: Transformed DataFrame
    """
    data_transformed = data.copy()

    if col_names is None:
        columns_to_transform: List[str] = data.columns.tolist()
    elif isinstance(col_names, str):
        columns_to_transform = [col_names]
    else:
        columns_to_transform = col_names

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=DEFAULT_MAX_JOBS) as executor:
        futures = {executor.submit(_transform_column, data_transformed[col]): col for col in columns_to_transform}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    for col_name, col_data in results:
        transform_col_name = f"{col_name}{suffix}"
        logger.info(f"power transform column: {col_name}, to {transform_col_name}")
        data_transformed[transform_col_name] = col_data

    return data_transformed


def check_consecutive_event(
    df: pd.DataFrame,
    time_col: str,
    threshold: float,
    col_name: str = "is_consecutive",
    check_gap: bool = False,
    direction: Literal["before", "after", "both"] = "both",
) -> pd.DataFrame:
    """
    Calculates whether the time difference between the current timestamp and *either* the previous or the next timestamp
    is less or larger (when check_gap is True) than a given threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        time_col (str): The name of the timestamp column. Defaults to "BILL_TIME".
        threshold (float): The time difference threshold in seconds. Defaults to 40.
        col_name (str, optional): The name of the new boolean column to be created.
                                   Defaults to "is_consecutive".
        check_gap (bool, optional): Whether to check the gap in time difference between the current timestamp
        direction (enumerate): check the event before or after the current one. Defaults to "both".

    Returns:
        pd.DataFrame: The DataFrame with the new 'is_consecutive' column.
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    if direction == "before" or direction == "both":
        # Calculate the difference from the previous timestamp in seconds
        df["_diff_from_prev_seconds"] = df[time_col].diff().dt.total_seconds()

    if direction == "after" or direction == "both":
        # Calculate the difference to the next timestamp in seconds
        # shift(-1) brings the next row's timestamp into the current row
        df["_diff_to_next_seconds"] = (df[time_col].shift(-1) - df[time_col]).dt.total_seconds()

    # Determine if either the difference from the previous OR to the next is less than the threshold
    # Note: NaN values (from the first and last rows' diff calculations) will evaluate to False when compared
    if direction == "both":
        if check_gap:
            df[col_name] = (df["_diff_from_prev_seconds"] > threshold) | (df["_diff_to_next_seconds"] > threshold)
        else:
            df[col_name] = (df["_diff_from_prev_seconds"] <= threshold) | (df["_diff_to_next_seconds"] <= threshold)
        df.drop(columns=["_diff_from_prev_seconds", "_diff_to_next_seconds"], inplace=True)
    elif direction == "before":
        if check_gap:
            df[col_name] = df["_diff_from_prev_seconds"] > threshold
        else:
            df[col_name] = df["_diff_from_prev_seconds"] <= threshold
        df.drop(columns=["_diff_from_prev_seconds"], inplace=True)
    elif direction == "after":
        if check_gap:
            df[col_name] = df["_diff_to_next_seconds"] > threshold
        else:
            df[col_name] = df["_diff_to_next_seconds"] <= threshold
        df.drop(columns=["_diff_from_prev_seconds"], inplace=True)

    return df


def add_event_group_by_gap(
    df: pd.DataFrame, time_col: str, threshold: float, col_name: str = "gap_group"
) -> pd.DataFrame:

    df = check_consecutive_event(df, time_col, threshold, col_name="_is_gap", check_gap=True, direction="before")
    df[col_name] = df["_is_gap"].cumsum()
    df.drop(columns=["_is_gap"], inplace=True)

    return df


def column_iterator(
    data: pd.DataFrame, ordered_column_names: List[str], n_columns: Optional[Union[List[int], int]] = None
) -> Iterator[Tuple[pd.DataFrame, int]]:
    """
    iterator to select first n columns of dataframe with order defined by ordered_column_names.
    This is typcially used in elbow method to select top n features.

    :param data:
    :param ordered_column_names: order of columns to select first n columns of dataframe
    :param n_columns: first n columns of dataframe to select
    :return:
    """

    max_n = len(ordered_column_names)
    if n_columns is None:
        n_columns = [max_n]
    elif isinstance(n_columns, int):
        n_columns = [n_columns]
    elif isinstance(n_columns, list):
        n_columns.sort()

    for n in n_columns:
        if n > max_n:
            logger.warning(
                f"selected columns ({n}) larger than max length ({max_n}) of the specified ordered columns \n"
                f"will only select {max_n} columns."
            )
            yield data.loc[:, ordered_column_names], max_n
            break
        else:
            yield data.loc[:, ordered_column_names[:n]], n


def group_iterator(
    data: pd.DataFrame, group_col: Optional[str] = None, count_thresh: int = 1, preserve_index: bool = False
) -> Iterator[Tuple[pd.DataFrame, Any, int]]:
    """
    iterator to select rows of data according to value in group_col
    :param data:
    :param group_col:
    :param count_thresh: ignore group data less than count_thresh
    :param preserve_index: yield data with all index preserved (fill NA for other groups)
    :return:
    """

    if group_col is None or group_col not in data.columns:
        yield data.copy(), None, 0
        logger.warning(f"group_col: {group_col} not in data to iterate over.")
        return

    group_vals = sorted(pd.Series(data[group_col].unique()).dropna())
    index = 0
    for group in group_vals:
        mask = data[group_col] == group
        res: pd.DataFrame = data.loc[mask].copy()
        if len(res) < count_thresh:
            continue

        if preserve_index:
            res = data.copy()
            res.loc[res[group_col] != group, :] = pd.NA

        yield res, group, index
        index += 1


def batch_iterator(data: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
    """
    iterator to select elements of data by chunk_size.
    :param data:
    :param chunk_size:
    :return:
    """
    i = 1
    num_chunks = (len(data) - 1) // chunk_size + 1
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size], i, num_chunks
        i += 1


def convert_to_list(arg: Any) -> List[Any]:
    """
    Convert str, int, float, bool, or iterables (including numpy arrays) to a list. This is used to coerce input arg to
    a type of List[Any].

    :param arg: The input argument of any type.
    :return: A list representation of the input argument.
    """
    if not arg:
        logger.warning(f"empty arg: {arg}")
        return []
    elif isinstance(arg, (str, int, float, bool)):
        return [arg]
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, Iterable):
        return list(arg)
    elif isinstance(arg, list):
        return arg
    else:
        raise TypeError(f"cannot convert {type(arg)}")


def convert_numpy_types(obj):
    """Recursively convert NumPy types, Decimals, and dict keys for JSON."""

    # Handle Dictionaries
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert the KEY to string if it's a Decimal or non-standard type
            if isinstance(k, Decimal) or (
                isinstance(k, np.generic)
                and (np.issubdtype(type(k), np.integer) or np.issubdtype(type(k), np.floating))
            ):
                new_key = str(k)
            else:
                new_key = k

            # Recursively convert the VALUE
            new_dict[new_key] = convert_numpy_types(v)
        return new_dict

    # Handle Lists/Tuples
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # Handle NumPy Arrays
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())

    # Handle Counters
    elif isinstance(obj, Counter):
        return convert_numpy_types(dict(obj))

    # Handle Individual Scalars
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.generic) and np.issubdtype(type(obj), np.integer):
        return int(obj)
    elif isinstance(obj, np.generic) and np.issubdtype(type(obj), np.floating):
        return float(obj)
    elif isinstance(obj, Decimal):
        return float(obj)

    else:
        return obj


def bootstrap_worker(arr_np: np.ndarray, n_boot: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Top-level function required for ProcessPoolExecutor pickling.
    Calculates bootstrap CI for a single array.
    """
    if len(arr_np) == 0:
        return float("nan"), float("nan")
    if np.min(arr_np) == np.max(arr_np):
        return float(arr_np[0]), float(arr_np[0])

    # Vectorized sampling: (n_boot, len(arr))
    resamples = np.random.choice(arr_np, size=(n_boot, len(arr_np)), replace=True)
    boot_means = np.mean(resamples, axis=1)

    lower = float(np.percentile(boot_means, (1 - ci_level) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci_level) / 2 * 100))
    return lower, upper


@lru_cache(maxsize=None)
def _resolve_ip_location(ip: str, timeout: float, lang: str) -> str:
    """HTTP lookup; cached by :func:`functools.lru_cache` on ``(ip, timeout, lang)``."""
    url = f"http://ip-api.com/json/{ip}?lang={lang}"
    location = "Timeout/Failed"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            if data.get("status") == "success":
                location = f"{data.get('country')} {data.get('regionName')} {data.get('city')}"
    except Exception:
        pass
    return location


def get_ip_location(ip: str, *, timeout: float = 5.0, lang: str = "zh-CN") -> str:
    """
    Look up a rough geographic location for an IP via ip-api.com (Chinese labels when lang=zh-CN).

    Results are memoized with :func:`functools.lru_cache` on ``(ip, timeout, lang)`` inside this module,
    so all importers share one cache per process. Empty IP returns Timeout/Failed without caching.

    To inspect or reset the cache: ``_resolve_ip_location.cache_info()`` / ``cache_clear()``.
    """
    key = (ip or "").strip()
    if not key:
        return "Timeout/Failed"
    return _resolve_ip_location(key, timeout, lang)
