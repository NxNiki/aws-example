import logging
import re
from typing import List, Optional

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import (
    avg as Favg,
    col,
    input_file_name,
    max as Fmax,
    min as Fmin,
    percentile_approx,
    regexp_extract,
    size,
    split,
    sum as Fsum,
    when,
)
from pyspark.sql.types import IntegerType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def read_data_with_partition(
    spark: SparkSession,
    path_pattern: str,
    regex_pattern: str,
    format: str = "csv",  # or "parquet"
    read_opts: Optional[dict] = None,
) -> DataFrame:
    """
    Load CSV or Parquet files and extract partition info (e.g., year/month) from the file path using regex.

    Args:
        spark: SparkSession
        path_pattern: File path pattern (e.g., "s3://bucket/data/*/*.csv")
        regex_pattern: Regex with named capture groups (?P<year>...), (?P<month>...), etc.
        format: "csv" or "parquet"
        read_opts: Dictionary of read options for Spark (e.g., {"header": "true"})

    Returns:
        DataFrame with extracted partition columns added.
    """

    read_opts = read_opts or {}
    df = spark.read.format(format).options(**read_opts).load(path_pattern)
    df = df.withColumn("_filepath", input_file_name())

    compiled = re.compile(regex_pattern)
    group_index_map = {name: idx for idx, name in enumerate(compiled.groupindex, start=1)}

    # Convert named groups to unnamed groups for Spark regex engine
    # remove all occurrences of ?P<name>
    spark_compatible_pattern = re.sub(r"\?P<\w+>", "", regex_pattern)
    # Extract each group by its index, assign column name as original group name
    for group_name, group_idx in group_index_map.items():
        df = df.withColumn(group_name, regexp_extract(col("_filepath"), spark_compatible_pattern, group_idx))

    return df.drop("_filepath")


def read_files_to_spark(
    spark: SparkSession,
    s3_files: List[str],
    column_names: Optional[List[str]] = None,
    keep_columns: Optional[List[str]] = None,
) -> DataFrame:
    """
    read files to spark dataframe
    :param spark: spark session
    :param s3_files:
    :param column_names:
    :param keep_columns: a subset of columns to keep.
    :return:
    """

    df = spark.read.option("header", "false").csv(s3_files)
    logger.info("read files to spark dataframe done!")

    if column_names:
        for i, col_name in enumerate(column_names):
            df = df.withColumnRenamed(f"_c{i}", col_name)

    df = df.select(*keep_columns)
    display_df_rows(df, "data loaded:")
    return df


def display_df_rows(df: DataFrame, msg: str = "dataframe rows:", n_rows: int = 5) -> None:
    logger.info(msg)
    rows = df.take(n_rows)
    for row in rows:
        logger.info(row)


def encode_label(df: DataFrame, label: str, index_label: str, data_type: str = "int") -> DataFrame:
    """
    create index label for the selected column.
    :param df:
    :param label:
    :param index_label:
    :param data_type:
    :return:
    """

    indexer = StringIndexer(inputCol=label, outputCol=index_label)
    currency_model = indexer.fit(df)
    df = currency_model.transform(df)
    df = df.withColumn(index_label, col(index_label).cast(data_type))

    return df


def create_stat_aggregations(column_name: str, rename: Optional[str] = None, avg_type: str = "float") -> List[Column]:

    if rename is None:
        rename = column_name

    return [
        Fmin(column_name).alias(f"{rename}_min"),
        Fmax(column_name).alias(f"{rename}_max"),
        Favg(column_name).cast(avg_type).alias(f"{rename}_mean"),
        percentile_approx(column_name, 0.25).alias(f"{rename}_p25"),
        percentile_approx(column_name, 0.5).alias(f"{rename}_median"),
        percentile_approx(column_name, 0.75).alias(f"{rename}_p75"),
    ]


def estimate_num_partitions(sdf: DataFrame, target_file_size_mb: int = 128) -> int:
    """
    Estimate the number of partitions based on DataFrame size and desired file size.

    Args:
        sdf: Spark DataFrame
        target_file_size_mb: Target file size per partition in megabytes

    Returns:
        Estimated number of partitions
    """
    target_size_bytes = target_file_size_mb * 1024 * 1024
    sample_size = sdf.sample(False, 0.01).rdd.map(lambda r: len(str(r))).mean()
    estimated_size = sample_size * sdf.count()
    num_partitions = max(1, int(estimated_size / target_size_bytes))

    return num_partitions


def split_column(
    df: DataFrame,
    col_name: str,
    sep: str = ",",
    max_items: Optional[int] = None,
    drop_original: bool = True,
    cast_type=IntegerType(),
) -> DataFrame:
    """
    Split a string column into multiple columns by a separator.

    Args:
        df (DataFrame): Input Spark DataFrame.
        col_name (str): Column name to split.
        sep (str): Separator (default is comma).
        max_items (int, optional): Max number of items to split. If None, compute from data.
        drop_original (bool): If True, drop the original column after split.
        cast_type (DataType): Spark type to cast split values to (e.g., IntegerType()).

    Returns:
        DataFrame: Updated DataFrame with new columns.
    """

    split_col = split(col(col_name), sep)

    if max_items is None:
        max_items = df.select(size(split_col).alias("len")).agg({"len": "max"}).collect()[0][0]

    for i in range(max_items):
        new_col = f"{col_name}_pos{i + 1}"
        df = df.withColumn(new_col, split_col.getItem(i).cast(cast_type))

    if drop_original:
        df = df.drop(col_name)

    return df
