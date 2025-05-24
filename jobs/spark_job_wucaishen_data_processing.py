"""
This example script cannot run locally
use submit emr job to submit this script to EMR

process wucaishen data
"""

import logging
import os
import time

from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.column import Column


from pyspark.sql.functions import (
    col, unix_timestamp, to_timestamp, expr, lag, when, split, row_number,hour, dayofweek,stddev,monotonically_increasing_id,
    sum as Fsum, floor, concat_ws, count as Fcount, min as Fmin, max as Fmax, avg as Favg, percentile_approx, coalesce, lit
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import hour, dayofweek
from pyspark.ml.feature import StringIndexer
from pyspark_project.s3_utils import list_s3_files
from pyspark_project.pyspark_utils import read_files_to_spark


@pandas_udf("row_id long, streak int", PandasUDFType.GROUPED_MAP)
def compute_streak_udf(pdf: DataFrame) -> DataFrame:
    """
    ========= 连续投注（streak） =========
    :param pdf:
    :return:
    """
    pdf = pdf.sort_values("billtime")
    streaks = []
    streak = 0
    for dt in pdf["delta_t"]:
        if pd.isna(dt) or dt > 200:
            streak = 0
        else:
            streak += 1
        streaks.append(streak)
    pdf["streak"] = streaks
    return pdf[["row_id", "streak"]]


@pandas_udf("row_id long, win_streak int, lose_streak int", PandasUDFType.GROUPED_MAP)
def compute_win_lose_streak(pdf: DataFrame) -> DataFrame:
    """
    ========= 连续赢钱/输钱 streak =========
    :param pdf:
    :return:
    """
    pdf = pdf.sort_values("billtime")
    win_streaks = []
    lose_streaks = []
    win_streak = 0
    lose_streak = 0
    for profit in pdf["cus_account"]:
        if profit > 0:
            win_streak += 1
            lose_streak = 0
        elif profit < 0:
            lose_streak += 1
            win_streak = 0
        else:
            win_streak = 0
            lose_streak = 0
        win_streaks.append(win_streak)
        lose_streaks.append(lose_streak)
    pdf["win_streak"] = win_streaks
    pdf["lose_streak"] = lose_streaks
    return pdf[["row_id", "win_streak", "lose_streak"]]


def create_aggregations(column_name:str, rename:str=None) -> List[Column]:

    if rename is None:
        rename = column_name

    return [
        Fcount("*").alias("group_num"),
        Fmin(column_name).alias(f"{rename}_min"),
        Fmax(column_name).alias(f"{rename}_max"),
        Favg(column_name).alias(f"{rename}_mean"),
        percentile_approx(column_name, 0.25).alias(f"{rename}_p25"),
        percentile_approx(column_name, 0.5).alias(f"{rename}_median"),
        percentile_approx(column_name, 0.75).alias(f"{rename}_p75"),
    ]


def process_wucaishen_data(df: DataFrame) -> DataFrame:
    start_time = time.time()

    df = df.filter((col("flag") != -8.0) & (col("productid") != "B26"))
    df = df.orderBy(col("loginname"), col("billtime"))

    # 时间转换
    df = df.withColumn("billtime_utc", to_timestamp((col("billtime") / 1e9).cast("long")))
    df = df.withColumn("billtime", expr("from_utc_timestamp(billtime_utc, 'America/New_York')"))

    # 排序窗口
    # 添加唯一标识行 ID
    df = df.withColumn("row_id", monotonically_increasing_id())

    # payout + current_point
    df = df.withColumn("payout", col("cus_account") + col("account"))
    df = df.withColumn("current_point", col("basepoint") + col("cus_account"))

    # 对 currency 做 label encoding
    indexer = StringIndexer(inputCol="currency", outputCol="currency_label")
    currency_model = indexer.fit(df)
    df = currency_model.transform(df)

    # 上一笔记录
    w = Window.partitionBy("loginname").orderBy("billtime")
    df = df.withColumn("prev_time", lag("billtime").over(w))

    df = df.withColumn("prev_account",
                       when(lag("account").over(w).isNotNull(),
                            lag("account").over(w)).otherwise(0)
                       )
    df = df.withColumn("prev_profit",
                       when(lag("cus_account").over(w).isNotNull(),
                            lag("cus_account").over(w)).otherwise(0)
                       )
    df = df.withColumn("prev_payout",
                       when((col("prev_account") + col("prev_profit")).isNotNull(),
                       col("prev_account") + col("prev_profit")).otherwise(0)
                       )
    df = df.withColumn("last_current_point",
                       when(lag("basepoint").over(w).isNotNull(),
                            lag("basepoint").over(w) + lag("cus_account").over(w)).otherwise(0)
                       )

    df = df.withColumn("is_payout_gt0", when(col("payout") > 0, 1).otherwise(0))
    df = df.withColumn("is_profit_gt0", when(col("cus_account") > 0, 1).otherwise(0))
    # 衍生字段：delta_t
    df = df.withColumn("delta_t", when(col("prev_time").isNull(), 0.0).otherwise( (unix_timestamp("billtime") - unix_timestamp("prev_time")).cast("double")))

    # delta_bet
    df = df.withColumn( "delta_bet",
        when(col("prev_account").isNotNull(), col("account") - col("prev_account")).otherwise(0)
    )

    # delta_profit
    df = df.withColumn(
        "delta_profit",
        when(col("prev_profit").isNotNull(), col("cus_account") - col("prev_profit")).otherwise(0)
    )
    # delta_payout
    df = df.withColumn(
        "delta_payout",
        when(col("prev_payout").isNotNull(), col("payout") - col("prev_payout")).otherwise(0)
    )
    # balance_change = 当前 basepoint - 上一笔 current_point
    df = df.withColumn(
        "balance_change",
        when(col("last_current_point").isNotNull(), col("basepoint") - col("last_current_point")).otherwise(0)
    )

    # 识别充值与提现
    df = df.withColumn("deposit", when(col("balance_change") > 0, col("balance_change")).otherwise(0))
    df = df.withColumn("withdrawal", when(col("balance_change") < 0, -col("balance_change")).otherwise(0))

    df = df.withColumn("rtp", when(col("account") != 0, col("payout") / col("account")).otherwise(0))
    # 添加时段分类列
    df = df.withColumn("hour_of_day", hour("billtime"))
    df = df.withColumn("is_morning", when((col("hour_of_day") >= 6) & (col("hour_of_day") < 12), 1).otherwise(0))
    df = df.withColumn("is_afternoon", when((col("hour_of_day") >= 12) & (col("hour_of_day") < 18), 1).otherwise(0))
    df = df.withColumn("is_night", when((col("hour_of_day") >= 18) & (col("hour_of_day") <= 23), 1).otherwise(0))
    df = df.withColumn("is_midnight", when((col("hour_of_day") >= 0) & (col("hour_of_day") < 6), 1).otherwise(0))

    # 节假日（周末）
    df = df.withColumn("is_weekend", when(dayofweek("billtime").isin([1, 7]), 1).otherwise(0))  # 1=Sunday, 7=Saturday
    # 拆分 result 字段为 result_pos1 ~ result_pos15
    df = df.withColumn("result_clean", expr("trim(BOTH ';' FROM result)"))
    split_cols = split(col("result_clean"), ",")
    for i in range(15):
        df = df.withColumn(f"result_pos{i + 1}", split_cols.getItem(i).cast(IntegerType()))

    end_time = time.time()
    print("Total execution time: {:.2f} seconds".format(end_time - start_time))

    return df


if __name__ == "__main__":

    os.makedirs('.log', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(".log/spark_job_wucaishen_data_processing.log"),
            logging.StreamHandler()
        ]
    )

    print("🚀 正在初始化 SparkSession ...")
    spark = SparkSession.builder \
        .appName("Aggregateddata") \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "64g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    print("SparkSession 初始化完成！")

    s3_files = list_s3_files("hyber-slot", "wucaishen_oringaldata/", ".csv.gz")

    column_names = [
        'productid', 'loginname', 'billno', 'billtime',
        'account', 'cus_account', 'currency', 'slottype',
        'basepoint', 'result', 'cur_ip', 'flag'
    ]

    columns_to_keep = [
        'productid', 'loginname', 'billno', 'billtime',
        'account', 'cus_account', 'currency', 'slottype',
        'basepoint', 'result', 'cur_ip'
    ]

    read_files_to_spark(spark, s3_files, column_names, columns_to_keep)