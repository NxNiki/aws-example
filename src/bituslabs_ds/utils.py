import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
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
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = data

    # Ensure 2D
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Compute mean and std per column
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    std[std == 0] = 1.0  # avoid division by zero

    # Compute boolean mask without creating z_scores array
    mask = np.ones(arr.shape[0], dtype=bool)
    for col in range(arr.shape[1]):
        col_mask = np.abs(arr[:, col] - mean[col]) < z_thresh * std[col]
        mask &= col_mask  # remove rows where any column exceeds threshold

    # Return filtered data
    if isinstance(data, pd.DataFrame):
        return data.loc[mask], mask
    else:
        # If input was 1D, return 1D
        if mask.shape[0] == arr.shape[0] and arr.shape[1] == 1:
            return arr[mask, 0], mask
        return arr[mask], mask


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
        col_names = data.columns.tolist()

    if isinstance(col_names, str):
        col_names = [col_names]

    def transform_column(col):
        if col in data_transformed.columns and np.issubdtype(data_transformed[col].dtype, np.number):
            col_data = data_transformed[col].astype(float).copy()
            col_data = power_transform(col_data.to_frame())
            logger.info(f"Column: {col} transformed to: {col}{suffix}.")
            return (f"{col}{suffix}", col_data)
        else:
            logger.warning(f"Column: {col} not transformed.")
            return None

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=DEFAULT_MAX_JOBS) as executor:
        futures = {executor.submit(transform_column, col): col for col in col_names}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    for col_name, col_data in results:
        data_transformed[col_name] = col_data

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
            yield data[ordered_column_names], max_n
            break
        else:
            yield data[ordered_column_names[:n]], n


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
        res = data[data[group_col] == group].copy()
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
