"""
This example script cannot run locally
use submit emr job to submit this script to EMR

process wucaishen data
"""

import logging
import os

from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pyspark_project.s3_utils import list_s3_files

os.makedirs('.log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(".log/spark_job_wucaishen_data_processing.log"),
        logging.StreamHandler()
    ]
)

def read_files_to_spark(spark: SparkSession, s3_files: List[str], column_names: List[str]=None) -> DataFrame:
    """
    read files to spark dataframe
    :param spark: spark session
    :param s3_files:
    :param column_names:
    :return:
    """

    df = spark.read.option("header", "false").csv(s3_files)
    print("数据加载完成！")

    if column_names:
        print("重新命名列 ...")
        for i, col_name in enumerate(column_names):
            df = df.withColumnRenamed(f"_c{i}", col_name)
        print("列重命名完成！")

    print("正在按照 user_id, created_at, creditseq 排序 ...")
    df_sorted = df.orderBy(col("loginname"), col("billtime"))
    print("排序完成！")

    print("显示前 5 行数据预览：")
    df_sorted.show(5)

    return df


if __name__ == "__main__":

    print("🚀 正在初始化 SparkSession ...")
    spark = SparkSession.builder \
        .appName("Aggregateddata") \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "64g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    print("SparkSession 初始化完成！")

    s3_files = list_s3_files("hyber-slot", "wucaishen_oringaldata/", ".csv.gz")
    column_names = ['productid',
                    'loginname',
                    'billno',
                    'billtime',
                    'account',
                    'cus_account',
                    'currency',
                    'slottype',
                    'basepoint',
                    'result',
                    'cur_ip',
                    'flag']

    read_files_to_spark(spark, s3_files, column_names)