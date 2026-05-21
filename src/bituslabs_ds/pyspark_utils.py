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
    sum as Fsum,
    when,
)

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

    if keep_columns is not None:
        df = df.select(*keep_columns)
    display_df_rows(df, "data loaded:")
    return df


def display_df_rows(df: DataFrame, msg: str = "dataframe rows:", n_rows: int = 5) -> None:
    logger.info(msg)
    rows = df.take(n_rows)
    for row in rows:
        logger.info(row)


def encode_label(df: DataFrame, label: str, index_label: str) -> DataFrame:
    """
    create index label for the selected column.
    :param df:
    :param label:
    :param index_label:
    :return:
    """

    indexer = StringIndexer(inputCol=label, outputCol=index_label)
    currency_model = indexer.fit(df)
    df = currency_model.transform(df)

    return df


def create_stat_aggregations(column_name: str, rename: Optional[str] = None) -> List[Column]:

    if rename is None:
        rename = column_name

    return [
        Fmin(column_name).alias(f"{rename}_min"),
        Fmax(column_name).alias(f"{rename}_max"),
        Favg(column_name).alias(f"{rename}_mean"),
        percentile_approx(column_name, 0.25).alias(f"{rename}_p25"),
        percentile_approx(column_name, 0.5).alias(f"{rename}_median"),
        percentile_approx(column_name, 0.75).alias(f"{rename}_p75"),
    ]
