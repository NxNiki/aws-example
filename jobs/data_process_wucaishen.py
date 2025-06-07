"""
This script is designed to run on AWS EMR and cannot be executed locally.
To run, use the EMR job submission script.

Purpose: Process and aggregate Wucaishen gaming data.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from exceptiongroup import catch
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import (
    PandasUDFType,
    avg as Favg,
    coalesce,
    col,
    concat_ws,
    count as Fcount,
    dayofweek,
    expr,
    floor,
    hour,
    lag,
    lit,
    max as Fmax,
    min as Fmin,
    monotonically_increasing_id,
    pandas_udf,
    percentile_approx,
    row_number,
    split,
    stddev,
    sum as Fsum,
    to_timestamp,
    unix_timestamp,
    when,
)
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window, WindowSpec

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.pyspark_utils import create_stat_aggregations, encode_label, read_files_to_spark
from bituslabs_ds.s3_utils import list_s3_files, upload_file_to_s3, write_spark_to_s3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logger.info("start SparkSession ...")
spark = (
    SparkSession.builder.appName("Aggregateddata")
    .config("spark.driver.memory", "32g")  # choose instance >=m5.4xlarge to meet memory demand
    .config("spark.executor.memory", "28g")
    .config("spark.executor.memoryOverhead", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # for better performance with Pandas UDFs
    .getOrCreate()
)
logger.info("SparkSession started！")


@pandas_udf("row_id long, streak int", PandasUDFType.GROUPED_MAP)
def compute_streak_udf(data: pd.DataFrame) -> pd.DataFrame:
    """
    ========= 连续投注（streak） =========
    :param data:
    :return:
    """
    data = data.sort_values("billtime")
    streaks = []
    streak = 0
    for dt in data["delta_t"]:
        if pd.isna(dt) or dt > 200:
            streak = 0
        else:
            streak += 1
        streaks.append(streak)
    data["streak"] = streaks
    return data[["row_id", "streak"]]


@pandas_udf("row_id long, win_streak int, lose_streak int", PandasUDFType.GROUPED_MAP)
def compute_win_lose_streak(data: pd.DataFrame) -> pd.DataFrame:
    """
    ========= 连续赢钱/输钱 streak =========
    :param data:
    :return:
    """
    data = data.sort_values("billtime")
    win_streaks = []
    lose_streaks = []
    win_streak = 0
    lose_streak = 0
    for profit in data["cus_account"]:
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
    data["win_streak"] = win_streaks
    data["lose_streak"] = lose_streaks
    return data[["row_id", "win_streak", "lose_streak"]]


def create_aggregations() -> List[Column]:

    agg_expressions = [
        Fcount("*").alias("group_num"),
        Favg("rtp").alias("rtp_mean"),
    ]

    agg_expressions.extend(create_stat_aggregations("account", "bet"))
    agg_expressions.extend(create_stat_aggregations("basepoint"))
    agg_expressions.extend(create_stat_aggregations("payout"))
    agg_expressions.extend(create_stat_aggregations("cus_account", "profit"))
    agg_expressions.extend(create_stat_aggregations("delta_t"))
    agg_expressions.extend(create_stat_aggregations("delta_bet"))
    agg_expressions.extend(create_stat_aggregations("delta_profit"))
    agg_expressions.extend(create_stat_aggregations("streak"))
    agg_expressions.extend(create_stat_aggregations("win_streak"))
    agg_expressions.extend(create_stat_aggregations("lose_streak"))
    agg_expressions.extend(create_stat_aggregations("deposit"))
    agg_expressions.extend(create_stat_aggregations("withdrawal"))

    agg_expressions.extend(
        [
            Fsum(when(col("slottype") == 2.0, 1).otherwise(0)).alias("slottype_2_count"),
            # 获奖率：is_payout_gt0 的和 / group_num
            (Fsum("is_payout_gt0") / Fcount("*")).alias("payout_rate"),
            # 盈利率：is_profit_gt0 的和 / group_num
            (Fsum("is_profit_gt0") / Fcount("*")).alias("profit_rate"),
            # 盈利波动率：利润的标准差
            coalesce(stddev("cus_account"), lit(0)).alias("profit_stddev"),
            # 投注波动率：投注额的标准差
            coalesce(stddev("account"), lit(0)).alias("account_stddev"),
            # start_time（切片开始时间）
            Fmin("billtime").alias("start_time"),
            # end_time（切片结束时间）
            Fmax("billtime").alias("end_time"),
            # 时间段
            Fsum("is_morning").alias("morning_count"),
            Fsum("is_afternoon").alias("afternoon_count"),
            Fsum("is_night").alias("night_count"),
            Fsum("is_midnight").alias("midnight_count"),
            Fsum("is_weekend").alias("weekend_count"),
            # 持续时间（秒）
            (unix_timestamp(Fmax("billtime")) - unix_timestamp(Fmin("billtime"))).alias("duration_seconds"),
            # 每注平均耗时（秒/注）
            ((unix_timestamp(Fmax("billtime")) - unix_timestamp(Fmin("billtime"))) / Fcount("*")).alias(
                "avg_time_per_bet"
            ),
        ]
    )

    return agg_expressions


def get_currency_count_by_group(df: DataFrame) -> DataFrame:
    """
    统计每个 group_id（loginname + group_index + sub_index）里统计 currency 出现次数
    并选择每组中使用次数最多的currency
    :param df:
    :param column_name:
    :return:
    """
    currency_count = df.groupBy("loginname", "group_index", "sub_index", "group_id", "currency_label", "currency").agg(
        Fcount("*").alias("currency_count")
    )
    # 取每个 group_id 出现最多的 currency
    currency_window = Window.partitionBy("group_id").orderBy(col("currency_count").desc())

    currency_count = currency_count.withColumn("row_number", row_number().over(currency_window)).filter(
        col("row_number") == 1
    )  # 只保留每组里出现次数最多的那一行

    return currency_count


def get_previous_value(
    df: DataFrame, col_name: str, prev_name: str, window: WindowSpec, check_null: bool = True
) -> DataFrame:
    """
    Get the previous value of a column and put into a new name
    :param df:
    :param col_name:
    :param prev_name:
    :param window:
    :param check_null:
    :return:
    """

    if check_null:
        df = df.withColumn(
            prev_name,
            when(lag(col_name).over(window).isNotNull(), lag(col_name).over(window)).otherwise(0),
        )
    else:
        df = df.withColumn(prev_name, lag(col_name).over(window))

    return df


def add_date_columns(df: DataFrame, time_column: str) -> DataFrame:
    # 添加时段分类列
    df = df.withColumn("hour_of_day", hour(time_column))
    df = df.withColumn(
        "is_morning",
        when((col("hour_of_day") >= 6) & (col("hour_of_day") < 12), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_afternoon",
        when((col("hour_of_day") >= 12) & (col("hour_of_day") < 18), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_night",
        when((col("hour_of_day") >= 18) & (col("hour_of_day") <= 23), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_midnight",
        when((col("hour_of_day") >= 0) & (col("hour_of_day") < 6), 1).otherwise(0),
    )

    # 节假日（周末）
    df = df.withColumn("is_weekend", when(dayofweek(time_column).isin([1, 7]), 1).otherwise(0))  # 1=Sunday, 7=Saturday

    return df


def process_wucaishen_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    logger.info("process_wucaishen_data start...")
    start_time = time.time()

    df = df.filter((col("flag") != -8.0) & (col("productid") != "B26"))
    df = df.orderBy(col("loginname"), col("billtime"))

    # 时间转换
    df = df.withColumn("billtime_utc", to_timestamp((col("billtime") / 1e9).cast("long")))
    df = df.withColumn("billtime", expr("from_utc_timestamp(billtime_utc, 'America/New_York')"))

    # 排序窗口
    df = df.withColumn("row_id", monotonically_increasing_id())

    # payout + current_point
    df = df.withColumn("payout", col("cus_account") + col("account"))
    df = df.withColumn("current_point", col("basepoint") + col("cus_account"))

    df = encode_label(df, "currency", "currency_label")

    # 上一笔记录
    w = Window.partitionBy("loginname").orderBy("billtime")
    df = get_previous_value(df, "billtime", "prev_time", w, check_null=False)
    df = get_previous_value(df, "account", "prev_account", w)
    df = get_previous_value(df, "cus_account", "prev_profit", w)

    df = df.withColumn(
        "prev_payout",
        when(
            (col("prev_account") + col("prev_profit")).isNotNull(),
            col("prev_account") + col("prev_profit"),
        ).otherwise(0),
    )
    df = df.withColumn(
        "last_current_point",
        when(
            lag("basepoint").over(w).isNotNull(),
            lag("basepoint").over(w) + lag("cus_account").over(w),
        ).otherwise(0),
    )

    df = df.withColumn("is_payout_gt0", when(col("payout") > 0, 1).otherwise(0))
    df = df.withColumn("is_profit_gt0", when(col("cus_account") > 0, 1).otherwise(0))

    # 衍生字段：delta_t
    df = df.withColumn(
        "delta_t",
        when(col("prev_time").isNull(), 0.0).otherwise(
            (unix_timestamp("billtime") - unix_timestamp("prev_time")).cast("double")
        ),
    )

    # delta_bet
    df = df.withColumn(
        "delta_bet",
        when(col("prev_account").isNotNull(), col("account") - col("prev_account")).otherwise(0),
    )

    # delta_profit
    df = df.withColumn(
        "delta_profit",
        when(col("prev_profit").isNotNull(), col("cus_account") - col("prev_profit")).otherwise(0),
    )
    # delta_payout
    df = df.withColumn(
        "delta_payout",
        when(col("prev_payout").isNotNull(), col("payout") - col("prev_payout")).otherwise(0),
    )
    # balance_change = 当前 basepoint - 上一笔 current_point
    df = df.withColumn(
        "balance_change",
        when(
            col("last_current_point").isNotNull(),
            col("basepoint") - col("last_current_point"),
        ).otherwise(0),
    )

    # 识别充值与提现
    df = df.withColumn("deposit", when(col("balance_change") > 0, col("balance_change")).otherwise(0))
    df = df.withColumn(
        "withdrawal",
        when(col("balance_change") < 0, -col("balance_change")).otherwise(0),
    )

    df = df.withColumn("rtp", when(col("account") != 0, col("payout") / col("account")).otherwise(0))

    # 拆分 result 字段为 result_pos1 ~ result_pos15
    df = df.withColumn("result_clean", expr("trim(BOTH ';' FROM result)"))
    split_cols = split(col("result_clean"), ",")
    for i in range(15):
        df = df.withColumn(f"result_pos{i + 1}", split_cols.getItem(i).cast(IntegerType()))

    streak_df = df.select("row_id", "loginname", "billtime", "delta_t").groupby("loginname").apply(compute_streak_udf)
    df = df.join(streak_df, on=["row_id"], how="left")

    streak_df = (
        df.select("row_id", "loginname", "billtime", "cus_account").groupby("loginname").apply(compute_win_lose_streak)
    )
    df = df.join(streak_df, on=["row_id"], how="left")

    df = add_date_columns(df, "billtime")

    agg_expressions = create_aggregations()
    currency_count = get_currency_count_by_group(df)

    df_grouped = df.groupBy("loginname", "group_index", "sub_index", "group_id").agg(*agg_expressions)
    # 把 currency_label 和 currency 也 join 回来
    df_grouped = df_grouped.join(
        currency_count.select("group_id", "currency_label", "currency"), on="group_id", how="left"
    )

    df = df.orderBy("loginname", "billtime")

    end_time = time.time()
    logger.info("Total execution time: {:.2f} seconds".format(end_time - start_time))

    return df, df_grouped


if __name__ == "__main__":

    os.makedirs(".log", exist_ok=True)
    execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f".log/data_process_wucaishen{execution_time}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        s3_files = list_s3_files("hyber-slot", "wucaishen_oringaldata/2401", ".csv.gz")

        column_names = [
            "productid",
            "loginname",
            "billno",
            "billtime",
            "account",
            "cus_account",
            "currency",
            "slottype",
            "basepoint",
            "result",
            "cur_ip",
            "flag",
        ]

        columns_to_keep = [
            "productid",
            "loginname",
            "billno",
            "billtime",
            "account",
            "cus_account",
            "currency",
            "slottype",
            "basepoint",
            "result",
            "cur_ip",
        ]

        spark_df = read_files_to_spark(spark, s3_files, column_names, columns_to_keep)
        sdf_enriched, sdf_grouped = process_wucaishen_data(spark_df)
        write_spark_to_s3(sdf_enriched, S3_BUCKET, "wucaishen_processed_enriched")
        write_spark_to_s3(sdf_grouped, S3_BUCKET, "wucaishen_processed_grouped")

    except Exception as e:
        logger.error(e)
    finally:
        upload_file_to_s3(log_file_path, S3_BUCKET, f"emr-logs/data_process_wucaishen_{execution_time}.log")
