import json
import logging
import warnings
from typing import List, Literal, Union

import pandas as pd

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
