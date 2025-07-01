import json
import logging
import warnings
from typing import Any, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    remove samples (rows) of data that has zscore above a certain threshold.
    :param data:
    :param z_thresh:
    :return:
    """
    z_scores = np.abs(zscore(data))
    filtered_indices = (z_scores < z_thresh).all(axis=1)
    data_filtered = data[filtered_indices]
    return data_filtered, filtered_indices


def count_missing_columns(df: pd.DataFrame, verbose: bool = True) -> int:
    """
    Count the number of columns in a DataFrame that contain missing (NaN) values.

    :param df: Input DataFrame
    :param verbose: If True, print columns with their missing counts
    :return: Number of columns with missing values
    """
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if verbose:
        logger.info(f"Columns with missing values: \n{cols_with_missing}")

    return len(cols_with_missing)


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
        if not logger.hasHandlers():
            warnings.warn(message)
        else:
            logger.warning(message)
        df = df.drop(columns=non_numeric_cols)

    return df


def log_transform(
    data: pd.DataFrame, col_names: Optional[Union[List[str], str]] = None, base: int = 10
) -> pd.DataFrame:
    """
    Apply log transformation to selected numeric columns of a DataFrame.
    Handles negative values by preserving their sign: log(abs(x)) * sign(x)
    0 will be preserved.

    :param data: Input DataFrame
    :param col_names: List of column names to transform
    :param base: Log base, default is 10
    :return: Transformed DataFrame
    """
    data_transformed = data.copy()

    if col_names is None:
        col_names = data.columns.tolist()

    if isinstance(col_names, str):
        col_names = [col_names]

    for col in col_names:
        if col in data_transformed.columns and np.issubdtype(data_transformed[col].dtype, np.number):
            col_data = data_transformed[col].astype(float).copy()
            mask_nonzero = col_data != 0
            with np.errstate(divide="ignore", invalid="ignore"):
                col_data[mask_nonzero] = (
                    np.sign(col_data[mask_nonzero]) * np.log(np.abs(col_data[mask_nonzero])) / np.log(base)
                )
            data_transformed[col] = col_data

    return data_transformed


def column_iterator(
    data: pd.DataFrame, ordered_column_names: List[str], n_columns: Optional[Union[List[int], int]] = None
) -> Iterator[Tuple[pd.DataFrame, int]]:
    """
    iterator to select first n columns of dataframe with order defined by ordered_column_names.
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
