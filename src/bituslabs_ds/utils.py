import json
import logging
import warnings
from typing import Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

BasicType = Union[str, int, float, bool, None]


def save_list(items: List[BasicType], filepath: str, format: Literal["json", "python"] = "json") -> None:
    """
    Saves a list of strings to a file in either JSON or Python list format.

    Parameters:
        items (List[BasicType]): The list of strings, ints, floats, or bools to save.
        filepath (str): The output file path.
        format (str): Format to save: "json" or "python". Default is "json".
    """
    if format == "json":
        with open(filepath, "w") as f:
            json.dump(items, f, indent=4)
    elif format == "python":
        with open(filepath, "w") as f:
            f.write("my_list = [\n")
            for item in items:
                f.write(f"    {repr(item)},\n")
            f.write("]\n")
    else:
        message = f"Unsupported format: '{format}' in save_string_list. Expected 'json' or 'python'."
        logger.error(message)
        raise ValueError(message)


def remove_outliers(
    data: np.ndarray | pd.DataFrame, z_thresh: float = 3
) -> Tuple[np.ndarray | pd.DataFrame, np.ndarray]:
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


def column_iterator(
    data: pd.DataFrame, ordered_column_names: List[str], n_columns: Optional[List[int] | int] = None
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
