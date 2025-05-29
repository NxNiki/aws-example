from typing import List, Optional

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import avg as Favg, col, max as Fmax, min as Fmin, percentile_approx, sum as Fsum, when


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
    print("数据加载完成！")

    if column_names:
        print("重新命名列 ...")
        for i, col_name in enumerate(column_names):
            df = df.withColumnRenamed(f"_c{i}", col_name)
        print("列重命名完成！")

    df = df.select(*keep_columns)

    print("显示前 5 行数据预览：")
    df.show(5)

    return df


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
