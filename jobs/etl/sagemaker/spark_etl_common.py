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

# AB grouping policy cutover (announced 2026-08-03, effective 08-04 Beijing
# time): from this date groups are assigned by the LAST DIGIT of user_id —
# 0-3 Default, 4-5 AB_TEST_A, 6-7 AB_TEST_B, 8-9 AI. Before it, by the
# partition_ab ids above (which STOPPED reflecting assignment at the cutover).
AB_GROUP_DIGIT_POLICY_START = "2026-08-04"


def ab_group_sql(ab_label_expr: str, user_id_expr: str, bj_date_expr: str, include_ab_tests: bool = True) -> str:
    """CASE expression labeling a bet's AB group under the date-gated policy.

    ``ab_label_expr`` is the first partition_ab id (e.g. PARTITION_AB_FIRST),
    ``user_id_expr`` the numeric user id, ``bj_date_expr`` a DATE in Beijing
    time. Games without AB test groups (include_ab_tests=False) collapse the
    test digits / legacy test ids into Default, keeping their historical
    two-group vocabulary."""
    digit = f"CAST({user_id_expr} AS BIGINT) % 10"
    if include_ab_tests:
        digit_case = f"""CASE
                WHEN {digit} >= 8 THEN 'AI'
                WHEN {digit} >= 6 THEN 'AB_TEST_B'
                WHEN {digit} >= 4 THEN 'AB_TEST_A'
                ELSE 'Default'
            END"""
        legacy_tests = f"""
                WHEN {ab_label_expr} = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
                WHEN {ab_label_expr} = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'"""
    else:
        digit_case = f"CASE WHEN {digit} >= 8 THEN 'AI' ELSE 'Default' END"
        legacy_tests = ""
    return f"""CASE
            WHEN {bj_date_expr} >= DATE '{AB_GROUP_DIGIT_POLICY_START}' THEN {digit_case}
            ELSE CASE
                WHEN {ab_label_expr} = '{AI_GROUP_ID}' THEN 'AI'{legacy_tests}
                ELSE 'Default'
            END
        END"""


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
