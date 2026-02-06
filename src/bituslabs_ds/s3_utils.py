from __future__ import annotations

import gc
import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from decimal import Decimal
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import boto3
import pandas as pd
import polars as pl
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from pyarrow import fs
from pyarrow.dataset import Dataset, dataset
from pyspark.sql import DataFrame as SparkDataFrame

from bituslabs_ds.config import DEFAULT_MAX_JOBS, REGION


@lru_cache(maxsize=None)
def _get_s3_client_for_pid(pid: int):
    # Lazily create and cache one client per process (fork-safe)
    cfg = Config(max_pool_connections=50, retries={"max_attempts": 10, "mode": "adaptive"})
    return boto3.client("s3", config=cfg)


def get_s3_client():
    # Return the cached client for the current process
    return _get_s3_client_for_pid(os.getpid())


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import


def parse_bucket_name(bucket: str) -> str:
    if bucket.endswith("/"):
        logger.info(f"remove '/' from {bucket}")
        bucket = bucket[:-1]
    if bucket.startswith("s3a://"):
        logger.info(f"remove 's3a://' from {bucket}")
        bucket = bucket[len("s3a://") :]
    if bucket.startswith("s3://"):
        logger.info(f"remove 's3://' from {bucket}")
        bucket = bucket[len("s3://") :]

    return bucket


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse an S3 URI (s3://, s3a://, s3n://) into (bucket, key).

    :param s3_path: S3 URI like s3://bucket/key
    :return: Tuple of (bucket, key)
    """
    if not s3_path.startswith(("s3://", "s3a://", "s3n://")):
        raise ValueError(f"Unsupported S3 URI scheme: {s3_path}")

    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError(f"Missing bucket in S3 URI: {s3_path}")

    return bucket, key


def read_to_pandas_df(
    bucket: str,
    key: str,
    columns: Optional[List[str]] = None,
    row_filters: Optional[Dict] = None,
    data_types: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Read a CSV or Parquet file from S3 and return it as a Pandas DataFrame.

    :param bucket: S3 bucket name
    :param key: S3 object key
    :param columns: Optional list of columns to read
    :return: Pandas DataFrame
    """
    bucket = parse_bucket_name(bucket)

    logger.info(f"read file from s3://{bucket}/{key}")
    _, ext = os.path.splitext(key.lower())
    if ext == ".csv":
        response = get_s3_client().get_object(Bucket=bucket, Key=key)
        data = pd.read_csv(response["Body"], usecols=columns, low_memory=True, dtype=data_types)
    elif ext == ".parquet":
        s3 = fs.S3FileSystem(region=REGION)
        s3_path = f"{bucket}/{key}"
        with s3.open_input_file(s3_path) as f:
            data = pd.read_parquet(f, columns=columns)
            if data_types:
                data = data.astype(data_types)
    else:
        raise ValueError(f"Unsupported file extension for S3 object: {key}")

    if row_filters:
        for column_name, allowed in row_filters.items():
            if column_name not in data.columns:
                logger.warning(f"Row filter column '{column_name}' not in data; skipping this filter")
                continue
            if isinstance(allowed, (list, set, tuple)):
                data = data[data[column_name].isin(list(allowed))]
                logger.info(
                    f"Applied filter on '{column_name}' with {len(list(allowed))} allowed values; remaining {len(data)} rows"
                )
            else:
                data = data[data[column_name] == allowed]
                logger.info(f"Applied filter on '{column_name}' == {allowed!r}; remaining {len(data)} rows")

    return data


def write_df_to_s3(data: Union[pd.DataFrame, SparkDataFrame], bucket: str, key: str) -> None:
    if isinstance(data, pd.DataFrame):
        return write_pandas_to_s3(data, bucket, key)
    elif isinstance(data, SparkDataFrame):
        return write_spark_to_s3(data, bucket, key)
    else:
        raise TypeError("Unsupported DataFrame type.")


def write_pandas_to_s3(data: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a Pandas DataFrame to a CSV file in S3."""
    bucket = parse_bucket_name(bucket)
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    get_s3_client().put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

    logger.info(f"Writing {key} to {bucket}")


def write_spark_to_s3(data: SparkDataFrame, bucket: str, key: str, file_format: str = "csv") -> None:
    """Write a PySpark DataFrame to S3 in CSV or Parquet format."""
    bucket = parse_bucket_name(bucket)
    s3_path = f"s3://{bucket}/{key}"

    if file_format not in ("csv", "parquet"):
        raise ValueError("PySpark supports only 'csv' or 'parquet' formats.")

    try:
        write_options = {"header": "true", "compression": "gzip"} if file_format == "csv" else {}
        data.write.mode("overwrite").options(**write_options).format(file_format).save(s3_path)
        logger.info(f"Successfully wrote PySpark DataFrame to {s3_path}")
    except Exception as e:
        logger.error(f"Failed to write PySpark DataFrame to S3: {e}")
        raise


def upload_file_to_s3(local_path: Union[str, Path], s3_bucket: str, s3_key: str) -> Optional[str]:
    """
    Uploads a local file to an S3 bucket. Add content type so we can open uploaded files directly on aws.

    :param local_path: Path to the local Python file.
    :param s3_bucket: Name of the S3 bucket.
    :param s3_key: S3 object key (e.g., 'scripts/red_violations.py').
    :return: Full S3 URI of the uploaded script.
    """

    filename = os.path.basename(local_path).lower()
    content_type_map = {
        # ".csv": "text/csv",
        ".csv": "text/plain",  # make it plain so that we can view small csv file online
        ".json": "application/json",
        ".log": "text/plain",
        ".txt": "text/plain",
        ".parquet": "application/x-parquet",
        ".html": "text/html",
    }
    extra_args = {"ContentType": "application/octet-stream"}
    for ext, content_type in content_type_map.items():
        if filename.endswith(ext + ".gz"):
            extra_args = {"ContentType": content_type, "ContentEncoding": "gzip"}
        if filename.endswith(ext):
            extra_args = {"ContentType": content_type}

    s3_bucket = parse_bucket_name(s3_bucket)

    try:
        logger.info(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
        get_s3_client().upload_file(local_path, s3_bucket, s3_key, ExtraArgs=extra_args)
        return f"s3://{s3_bucket}/{s3_key}"
    except FileNotFoundError:
        logger.error("Error: The specified file was not found.")
    except NoCredentialsError:
        logger.error("Error: AWS credentials not available.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return None


def upload_folder_to_s3(
    local_folder_path: Union[str, Path],
    s3_bucket: str,
    s3_prefix: str = "",
    max_workers: int = DEFAULT_MAX_JOBS,
    ignore_hidden: bool = True,
) -> Tuple[str, List[str]]:
    """
    Uploads a local folder to an S3 bucket in parallel.

    :param local_folder_path: Path to the local folder.
    :param s3_bucket: Name of the S3 bucket.
    :param s3_prefix: S3 object key prefix (e.g., 'my_data/').
    :param max_workers: Maximum number of concurrent file uploads.
    :param ignore_hidden: If True, ignore hidden files and directories (starting with '.').
    :return: List of S3 URIs for successfully uploaded files.
    """
    local_folder_path = Path(local_folder_path).resolve()
    if not local_folder_path.is_dir():
        logger.error(f"'{local_folder_path}' is not a valid directory.")
        return f"s3://{s3_bucket}/{s3_prefix}", []

    s3_bucket = parse_bucket_name(s3_bucket)
    uploaded_uris = []
    futures = []

    def is_hidden(path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(local_folder_path):
            if ignore_hidden:
                # Remove hidden directories from traversal
                dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if ignore_hidden and file.startswith("."):
                    continue
                local_file_path = Path(root) / file
                # If ignore_hidden, also check if any parent is hidden
                if ignore_hidden and is_hidden(local_file_path.relative_to(local_folder_path)):
                    continue
                relative_path = local_file_path.relative_to(local_folder_path)
                s3_key = str(Path(s3_prefix) / relative_path).replace("\\", "/")
                future = executor.submit(upload_file_to_s3, local_file_path, s3_bucket, s3_key)
                futures.append(future)

        # Wait for all futures to complete and collect results
        for future in as_completed(futures):
            result = future.result()
            if result:
                uploaded_uris.append(result)

    return f"s3://{s3_bucket}/{s3_prefix}", uploaded_uris


def list_s3_files(bucket: str, prefix: str, pattern: Optional[str] = None) -> List[str]:
    """
    List all S3 files in bucket starting from the prefix, filtering with an optional regex pattern.

    :param bucket: S3 bucket name
    :param prefix: S3 key prefix (acts like a folder)
    :param pattern: Optional regex pattern to match the key
        if no pattern is provided, all files with the prefix (folder) will be listed.
    :return: List of matching s3 keys (not including bucket)
    """
    logger.info(f"Listing S3 files in: {bucket}/{prefix}, with pattern: {pattern}")
    paginator = get_s3_client().get_paginator("list_objects_v2")
    matching_keys = []

    if bucket:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if pattern is None or re.search(pattern, key):
                    s3_uri = f"s3://{bucket}/{key}"
                    matching_keys.append(s3_uri)
                    logging.info(f"Found {s3_uri}")

        logger.info(f"Found {len(matching_keys)} S3 keys")
    return matching_keys


def _read_file(
    file: str, columns: Optional[List[str]], row_filters: Optional[Dict], data_types: Optional[Dict]
) -> pd.DataFrame:
    logger.info(f"Reading {file}")
    bucket, file = parse_s3_path(file)
    return read_to_pandas_df(bucket, file, columns, row_filters, data_types)


def read_local_cache(
    local_cache_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    data_types: Optional[Dict[str, Any]] = None,
    lazy_load: bool = False,
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """
    Reads data from a local CSV or Parquet file with options for lazy loading.

    Args:
        local_cache_path: Path to the file.
        columns: List of columns to read.
        data_types: Dictionary of column types.
        lazy_load: If True, returns a Polars LazyFrame for query optimization.
                   If False, returns a standard Pandas DataFrame.

    Returns:
        pd.DataFrame or pl.LazyFrame
    """
    path = Path(local_cache_path)

    if not path.exists():
        logger.warning(f"File not found: {path}. Returning empty structure.")
        # Return appropriate empty type based on requested mode
        return pl.DataFrame().lazy() if lazy_load else pd.DataFrame()

    logger.info(f"Reading data from {path} (Lazy: {lazy_load})")

    try:
        if lazy_load:
            return _read_lazy(path, columns, data_types)
        else:
            return _read_eager(path, columns, data_types)

    except Exception as e:
        logger.error(f"Failed to read cache {path}: {e}", exc_info=True)
        raise


def _read_lazy(path: Path, columns: Optional[List[str]], data_types: Optional[Dict]) -> pl.LazyFrame:
    """Internal handler for Polars Lazy loading."""
    # 1. Scan the file (Metadata only)
    if path.suffix == ".csv":
        lf = pl.scan_csv(path)
    elif path.suffix == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        raise ValueError(f"Unsupported file type for lazy load: {path.suffix}")

    # 2. Pushdown Predicates (Filter columns early)
    if columns:
        lf = lf.select(columns)

    # 3. Handle Types (Cast Decimals and User Types)
    # Note: Polars handles Decimals natively, but if you strictly need Float:
    if data_types:
        # Convert Python types to Polars DataType if needed, or rely on string names
        lf = lf.cast(data_types)

    return lf


def _read_eager(path: Path, columns: Optional[List[str]], data_types: Optional[Dict]) -> pd.DataFrame:
    """Internal handler for Pandas Eager loading."""
    if path.suffix == ".csv":
        data = pd.read_csv(path, usecols=columns, dtype=data_types)
    elif path.suffix == ".parquet":
        data = pd.read_parquet(path, columns=columns)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # OPTIMIZATION: Vectorized Decimal conversion
    # The original .map(lambda...) is very slow.
    # We check object columns specifically to see if they contain Decimals.
    if path.suffix == ".parquet":
        for col in data.select_dtypes(include=["object", "float"]).columns:
            # Check a sample to see if it's actually Decimal objects
            if len(data) > 0 and isinstance(data[col].iloc[0], Decimal):
                data[col] = data[col].astype(float)

    if data_types:
        data = data.astype(data_types)

    logger.info("Read data finished.")
    logger.debug(f"Head:\n{data.head(5)}")
    return data


def _read_s3_lazy(s3_uri: str, columns: Optional[List[str]], data_types: Optional[Dict]) -> pl.LazyFrame:
    """Internal handler for Polars Lazy loading from S3."""
    _, ext = os.path.splitext(s3_uri.lower())
    if ext == ".csv":
        lf = pl.scan_csv(s3_uri)
    elif ext == ".parquet":
        lf = pl.scan_parquet(s3_uri)
    else:
        raise ValueError(f"Unsupported file type for S3 lazy load: {ext}")
    if columns:
        lf = lf.select(columns)
    if data_types:
        lf = lf.cast(data_types)
    return lf


def _read_s3_eager(s3_uri: str, columns: Optional[List[str]], data_types: Optional[Dict]) -> pd.DataFrame:
    """Internal handler for eager loading from S3."""
    bucket, key = parse_s3_path(s3_uri)
    return read_to_pandas_df(bucket, key, columns, row_filters=None, data_types=data_types)


def _is_s3_path(path: str) -> bool:
    """Return True if path is an S3 URI."""
    return path.startswith(("s3://", "s3a://", "s3n://"))


# --------------- Output directory helpers (local Path vs S3 str) ---------------
# Path does not handle s3:// well; use these for ETL output roots and joining.

OutputDir = Union[str, Path]


def is_s3_path(path: OutputDir) -> bool:
    """Return True if path is an S3 URI (e.g. s3://bucket/prefix). Accepts str or Path."""
    return _is_s3_path(str(path).strip())


def normalize_storage_root(storage_root: OutputDir) -> Tuple[bool, OutputDir]:
    """
    Normalize a storage root for use in ETL: local paths become Path (and dir is created),
    S3 paths stay as str. Returns (is_s3, normalized_root) for use in branch logic.
    """
    if is_s3_path(storage_root):
        return (True, str(storage_root).rstrip("/"))
    path_root = Path(storage_root)
    path_root.mkdir(parents=True, exist_ok=True)
    return (False, path_root)


def join_output_path(root: OutputDir, *parts: str) -> OutputDir:
    """
    Join root with path parts. Use this instead of Path / or string concat so that
    S3 paths remain str and local paths remain Path (mypy-friendly and correct at runtime).
    """
    if isinstance(root, Path):
        return root.joinpath(*parts) if parts else root
    base = str(root).rstrip("/")
    for part in parts:
        base = f"{base}/{part.lstrip('/')}"
    return base


def output_path_as_str(path: OutputDir) -> str:
    """
    Return the path as a string suitable for S3 or local APIs (e.g. wr.s3, df.to_parquet).
    """
    return str(path) if isinstance(path, Path) else path


# --------------- Path expansion and file reading ---------------

_SUPPORTED_EXTENSIONS = (".parquet", ".csv")


def expand_paths_to_files(paths: List[str]) -> List[str]:
    """
    Expand S3 prefixes to individual file URIs. Local paths are returned as-is.
    For S3 paths, list objects under the prefix and return only .parquet/.csv files.
    If listing returns empty, the prefix is skipped (and a warning is logged).
    """
    result: List[str] = []
    for p in paths:
        if not p or not isinstance(p, str):
            continue
        if _is_s3_path(p):
            bucket, prefix = parse_s3_path(p)
            all_objects = list_s3_files(bucket, prefix)
            # Keep only files with supported extensions (exclude directory markers, etc.)
            files = [uri for uri in all_objects if uri.lower().endswith(_SUPPORTED_EXTENSIONS)]
            if files:
                result.extend(files)
            else:
                if not all_objects:
                    logger.warning(
                        f"No objects found under S3 prefix {p}. Skipping. " "Check bucket/prefix and credentials."
                    )
                else:
                    logger.warning(
                        f"No .parquet or .csv files under S3 prefix {p} "
                        f"(found {len(all_objects)} object(s)). Skipping."
                    )
        else:
            result.append(p)
    return result


def read_single_file(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    data_types: Optional[Dict[str, Any]] = None,
    lazy_load: bool = False,
) -> Union[pd.DataFrame, pl.LazyFrame]:
    """
    Read a single file from local path or S3, with optional lazy loading.

    Args:
        path: Local file path or s3:// URI.
        columns: List of columns to read.
        data_types: Dictionary of column types.
        lazy_load: If True, returns a Polars LazyFrame; else returns a Pandas DataFrame.

    Returns:
        pd.DataFrame or pl.LazyFrame
    """
    path_str = str(path)
    if _is_s3_path(path_str):
        logger.info(f"Reading S3 data from {path_str} (Lazy: {lazy_load})")
        try:
            if lazy_load:
                return _read_s3_lazy(path_str, columns, data_types)
            else:
                return _read_s3_eager(path_str, columns, data_types)
        except Exception as e:
            logger.error(f"Failed to read S3 file {path_str}: {e}", exc_info=True)
            raise
    else:
        return read_local_cache(path, columns, data_types, lazy_load)


def align_lazyframe_schemas(lfs: List[pl.LazyFrame], context: str = "") -> List[pl.LazyFrame]:
    """
    Align schemas of multiple LazyFrames for safe vertical concatenation.

    Handles:
    - Different column types (casts to compatible types)
    - Missing columns (adds as null)
    - Logs warnings about schema mismatches

    Args:
        lfs: List of LazyFrames to align
        context: Context string for logging (e.g., "date data: day")

    Returns:
        List of LazyFrames with aligned schemas
    """
    if len(lfs) <= 1:
        return lfs

    schemas = [lf.collect_schema() for lf in lfs]
    all_columns = set()
    for schema in schemas:
        all_columns.update(schema.names())

    column_types: Dict[str, List[pl.DataType]] = {}
    for schema in schemas:
        for col_name in schema.names():
            if col_name not in column_types:
                column_types[col_name] = []
            column_types[col_name].append(schema[col_name])

    target_types: Dict[str, pl.DataType] = {}
    mismatches: List[str] = []

    def get_type_class(dtype: pl.DataType) -> str:
        type_str = str(dtype)
        if "(" in type_str:
            return type_str.split("(")[0]
        return type_str

    for col_name, types in column_types.items():
        type_strs = [str(t) for t in types]
        unique_type_strs = list(set(type_strs))

        if len(unique_type_strs) > 1:
            mismatches.append(f"{col_name}: {unique_type_strs}")
            type_classes = [get_type_class(t) for t in types]
            has_string = any(cls == "String" for cls in type_classes)
            has_float = any(cls in ("Float32", "Float64") for cls in type_classes)
            has_int = any(
                cls in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64")
                for cls in type_classes
            )
            has_datetime = any(cls in ("Datetime", "Date") for cls in type_classes)

            if has_datetime:
                target_types[col_name] = pl.Datetime("ns")
            elif has_string:
                target_types[col_name] = cast(pl.DataType, pl.String)
            elif has_float:
                target_types[col_name] = cast(pl.DataType, pl.Float64)
            elif has_int:
                target_types[col_name] = cast(pl.DataType, pl.Int64)
            else:
                target_types[col_name] = types[0]
        else:
            target_types[col_name] = types[0]

    missing_cols_by_file: List[List[str]] = []
    for schema in schemas:
        missing = sorted(all_columns - set(schema.names()))
        if missing:
            missing_cols_by_file.append(missing)

    if mismatches:
        logger.warning(
            f"Schema mismatches detected {context}: {len(mismatches)} columns have different types. "
            f"Will coerce to compatible types. Mismatches: {', '.join(mismatches[:5])}"
            + (f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else "")
        )
    if missing_cols_by_file:
        total_missing = sum(len(m) for m in missing_cols_by_file)
        logger.warning(
            f"Missing columns detected {context}: Some files are missing {total_missing} column(s). "
            f"Missing columns will be filled with null values."
        )

    col_order = sorted(all_columns)
    aligned_lfs = []

    for i, lf in enumerate(lfs):
        schema = schemas[i]
        selects = []
        for col_name in col_order:
            if col_name in schema.names():
                current_type = schema[col_name]
                target_type = target_types[col_name]
                if str(current_type) != str(target_type):
                    selects.append(pl.col(col_name).cast(target_type).alias(col_name))
                else:
                    selects.append(pl.col(col_name))
            else:
                target_type = target_types[col_name]
                selects.append(pl.lit(None).cast(target_type).alias(col_name))
        lf_aligned = lf.select(selects)
        aligned_lfs.append(lf_aligned)

    return aligned_lfs


def align_dataframe_columns(dfs: List[pd.DataFrame], context: str = "") -> List[pd.DataFrame]:
    """
    Align columns of multiple Pandas DataFrames for safe vertical concatenation.

    Ensures all DataFrames have the same columns in the same order; missing columns
    are filled with NaN.

    Args:
        dfs: List of DataFrames to align
        context: Context string for logging

    Returns:
        List of DataFrames with aligned columns
    """
    if len(dfs) <= 1:
        return dfs

    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)

    col_order = sorted(all_columns)
    missing_by_file = [sorted(all_columns - set(df.columns)) for df in dfs]
    total_missing = sum(len(m) for m in missing_by_file if m)

    if total_missing > 0:
        logger.warning(
            f"Missing columns detected {context}: Some files are missing {total_missing} column(s). "
            f"Missing columns will be filled with NaN."
        )

    aligned = []
    for df in dfs:
        reordered = df.reindex(columns=col_order)
        aligned.append(reordered)

    return aligned


def save_local_cache(data: pd.DataFrame, local_cache_path: str, append: bool = False):

    os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)

    if local_cache_path.endswith(".csv"):
        mode = "a" if append else "w"
        data.to_csv(local_cache_path, index=False, mode=mode, header=not append)
    elif local_cache_path.endswith(".parquet"):
        if append:
            data.to_parquet(local_cache_path, index=False, engine="fastparquet", append=append)
        else:
            data.to_parquet(local_cache_path, index=False, engine="auto")


def read_files(
    files: Union[List[str], str],
    local_cache_path: Optional[str] = None,
    columns: Optional[List[str]] = None,
    row_filters: Optional[Dict] = None,
    data_types: Optional[Dict] = None,
    max_workers: int = DEFAULT_MAX_JOBS,
    parallel_mode: Literal["thread", "process", "none"] = "thread",
    reload: bool = False,
    add_file_source: bool = False,
    lazy_load: bool = False,
    expand_s3_prefixes: bool = True,
    return_as_list: bool = False,
) -> Union[pd.DataFrame, pl.LazyFrame, List[pd.DataFrame], List[pl.LazyFrame]]:
    """
    Read CSV/Parquet files from local paths and/or S3 using optional parallelization.

    :param files: List of file paths (local or s3://). Supports CSV and Parquet.
    :param local_cache_path: Local cache path; if present and not reload, read from cache.
    :param columns: List of column names to read from each file.
    :param row_filters: Optional row filters to apply (eager mode, S3 only).
    :param data_types: Optional column data types.
    :param max_workers: Number of workers to use in parallel execution.
    :param parallel_mode: Parallel execution strategy: 'thread', 'process', or 'none'.
    :param reload: Whether to reload files from source or not.
    :param add_file_source: Whether to add file paths to result.
    :param lazy_load: If True, return Polars LazyFrame(s); else Pandas DataFrame(s).
    :param expand_s3_prefixes: If True, expand S3 prefixes to individual file URIs before reading.
    :param return_as_list: If True with lazy_load, return List[pl.LazyFrame] instead of concatenated LazyFrame.
    :return: Concatenated DataFrame/LazyFrame, or list when return_as_list=True.
    """

    if local_cache_path is not None and os.path.exists(local_cache_path) and not reload:
        logger.info(f"Found local cache at {local_cache_path}")
        data = read_local_cache(local_cache_path, columns, data_types, lazy_load=lazy_load)
        if return_as_list and lazy_load:
            return [cast(pl.LazyFrame, data)]
        if not lazy_load:
            logger.info("first 5 rows of dataframe: \n%s", cast(pd.DataFrame, data).head(5).to_markdown())
        return data

    logger.info(f"read data with data types spec: {data_types}")
    if isinstance(files, str):
        files = [files]
    valid_files = [f for f in files if f]
    if not valid_files:
        return (
            [pl.DataFrame().lazy()]
            if (lazy_load and return_as_list)
            else (pl.DataFrame().lazy() if lazy_load else pd.DataFrame())
        )

    if expand_s3_prefixes:
        file_paths = expand_paths_to_files(valid_files)
    else:
        file_paths = valid_files

    has_local = any(not _is_s3_path(p) for p in file_paths)
    use_single_file = lazy_load or has_local
    if use_single_file:
        if row_filters:
            logger.warning("row_filters are ignored when lazy_load=True or when reading local paths")
        read_func = partial(read_single_file, columns=columns, data_types=data_types, lazy_load=lazy_load)
    else:
        read_func = partial(_read_file, columns=columns, row_filters=row_filters, data_types=data_types)

    if parallel_mode == "none" or max_workers <= 1 or len(file_paths) == 1:
        dfs = [read_func(f) for f in file_paths]
    else:
        logger.info(f"read files using {max_workers} workers")
        executor_cls: Callable = ThreadPoolExecutor if parallel_mode == "thread" else ProcessPoolExecutor
        dfs = []
        with executor_cls(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(read_func, f): f for f in file_paths}
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    dfs.append(future.result())
                except Exception as e:
                    logger.error(f"Failed to read {file}: {e}")

    if local_cache_path is not None and not lazy_load:
        # Write incrementally to cache to avoid memory issues with large concatenation
        logger.info(f"save data to local cache incrementally: {local_cache_path}")
        os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)

        for i in range(len(dfs)):
            append = i != 0
            df = cast(pd.DataFrame, dfs[i])
            if add_file_source:
                df["source_file"] = os.path.basename(file_paths[i])

            save_local_cache(df, local_cache_path, append)
            logger.info(f"Written {len(df)} rows to cache (file {i+1}/{len(dfs)})")
            dfs[i] = None
            gc.collect()

        # Read the final cached file and return
        data = cast(pd.DataFrame, read_local_cache(local_cache_path, lazy_load=False))
        logger.info("first 5 rows of dataframe: \n%s", data.head(5).to_markdown())
        return data
    else:
        if lazy_load:
            valid_lfs = [lf for lf in dfs if isinstance(lf, pl.LazyFrame) and lf.collect_schema().len() > 0]
            if not valid_lfs:
                return [pl.DataFrame().lazy()] if return_as_list else pl.DataFrame().lazy()
            aligned_lfs = align_lazyframe_schemas(valid_lfs, context="read_files")
            if return_as_list:
                return aligned_lfs
            return pl.concat(aligned_lfs)
        else:
            pdfs = cast(List[pd.DataFrame], dfs)
            aligned_dfs = align_dataframe_columns(pdfs, context="read_files")
            if add_file_source:
                data = pd.concat(
                    aligned_dfs,
                    keys=[os.path.basename(f) for f in file_paths],
                )
                data = data.reset_index(level=0).rename(columns={"level_0": "source_file"})
            else:
                data = pd.concat(aligned_dfs, ignore_index=True)
            logger.info("first 5 rows of dataframe: \n%s", data.head(5).to_markdown())
            return data


def read_dataset(file_path: str, region: str = REGION, data_format: str = "parquet") -> Dataset:
    file_path = parse_bucket_name(file_path)
    logger.info(f"Read data from: {file_path}")
    s3 = fs.S3FileSystem(region=region)
    ds = dataset(file_path, format=data_format, partitioning="hive", filesystem=s3)  # recognizes year=, month=, etc.
    logger.info("Finished reading dataset")

    return ds
