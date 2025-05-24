
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def read_files_to_spark(
        spark: SparkSession, s3_files: List[str], column_names: List[str]=None, keep_columns: List[str]=None
) -> DataFrame:
    """
    read files to spark dataframe
    :param spark: spark session
    :param s3_files:
    :param column_names:
    :param keep_columns:
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

