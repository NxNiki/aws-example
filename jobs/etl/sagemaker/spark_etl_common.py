"""Shared helpers for the SageMaker PySpark cold-data ETL jobs.

Runs ON the Spark container (Python 3.9, no bituslabs_ds): job scripts under
``slot_machine/`` and ``fish_hunter/`` import this as a plain module, and the
submit/pipeline scripts ship it alongside the job via ``submit_py_files``.
Keep it dependency-free beyond pyspark.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from pyspark.sql import SparkSession, functions as F

# Inlined from bituslabs_ds.config (these jobs run without the package).
EXCLUDED_OP_CODES = "('B26', 'TST', 'TSB', 'TSO')"
AI_GROUP_ID = "jojpin-9mokha-rexQug"
AB_TEST_GROUP_A = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"
AB_TEST_GROUP_B = "4a04df21-c749-4808-8e55-3a0b74c084d2"
# Asia/Shanghai has no DST, so a fixed offset equals CONVERT_TIMEZONE.
BJ_UTC_OFFSET_HOURS = 8

# The cold data stores partition_ab as binary JSON (b'["<group-id>"]'), not a
# parquet list, so the first element is extracted via get_json_object.
PARTITION_AB_FIRST = "get_json_object(CAST(t.partition_ab AS STRING), '$[0]')"


def build_spark_session(app_name: str, input_root: str, input_region: str) -> SparkSession:
    """The jobs' common session: UTC, AQE, dynamic partition overwrite, ANSI
    off (Redshift-parity NULL/array semantics), and the cross-region endpoint
    for the source bucket (the game warehouses live in ap-southeast-1)."""
    builder = (
        SparkSession.builder.appName(app_name)  # type: ignore[attr-defined]
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )
    if input_root.startswith("s3"):
        bucket = input_root.split("/")[2]
        builder = builder.config(f"spark.hadoop.fs.s3a.bucket.{bucket}.endpoint.region", input_region)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def check_schema(df, required_columns, table_name: str = "cold data") -> None:
    """Fail fast with the full column inventory when the source moved."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{table_name} table is missing required columns: {missing}; " f"available columns: {sorted(df.columns)}"
        )


def prune_partition_days(df, start: date, end: date, margin_days: int = 0):
    """Path-level pruning on the warehouse's ``year=/month=/day=`` layout,
    widened by ``margin_days`` on both sides (day dirs follow created_at's UTC
    date; boundary rows can sit one partition over). No-op when the frame has
    no partition columns."""
    if not {"year", "month", "day"} <= set(df.columns):
        print("no year/month/day partition columns; relying on timestamp filters only")
        return df
    partition_date = F.make_date("year", "month", "day")
    return df.filter(
        partition_date.between(F.lit(start - timedelta(days=margin_days)), F.lit(end + timedelta(days=margin_days)))
    )


def beijing_today() -> date:
    return (datetime.now(timezone.utc) + timedelta(hours=BJ_UTC_OFFSET_HOURS)).date()


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def month_start(d: date) -> date:
    return d.replace(day=1)


def prev_month_start(d: date) -> date:
    return (d.replace(day=1) - timedelta(days=1)).replace(day=1)
