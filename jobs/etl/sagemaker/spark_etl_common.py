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
# Asia/Shanghai has no DST, so a fixed offset equals CONVERT_TIMEZONE.
BJ_UTC_OFFSET_HOURS = 8

# bet_session gap for session-start day attribution; the session vocabulary
# and rationale live in docs/ab_group_policy.md ("Day basis" + table).
SESSION_DAY_GAP_SECONDS = 1800


def session_day_ctes(source_cte: str, tiebreak_col: str, gap_seconds: int = SESSION_DAY_GAP_SECONDS) -> str:
    """CTE pair sessionizing ``source_cte`` (full user stream, portable window
    SQL): a new session starts after a gap > gap_seconds; every row carries
    its session's start timestamp (``session_start_ts``) for day attribution.
    ``tiebreak_col`` orders simultaneous rows deterministically."""
    order = f"ORDER BY t.created_at, t.{tiebreak_col}"
    return f"""sessionized AS (
            SELECT
                t.*,
                SUM(CASE
                        WHEN t.prev_created_at IS NULL THEN 1
                        WHEN CAST(t.created_at AS DOUBLE) - CAST(t.prev_created_at AS DOUBLE) > {gap_seconds} THEN 1
                        ELSE 0
                    END) OVER (
                    PARTITION BY t.user_id {order}
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS session_id
            FROM (
                SELECT
                    t.*,
                    LAG(t.created_at) OVER (PARTITION BY t.user_id {order}) AS prev_created_at
                FROM {source_cte} AS t
            ) AS t
        ),

        session_days AS (
            SELECT
                t.*,
                MIN(t.created_at) OVER (PARTITION BY t.user_id, t.session_id) AS session_start_ts
            FROM sessionized AS t
        )"""


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


def warn_on_schema_drift(spark, out_path: str, df) -> None:
    """Log a loud warning when the frame about to be written adds or drops
    columns versus the dataset already at ``out_path``. An incremental run
    then leaves the dataset split-schema (readers tolerate it, but the new
    column stays invisible) until a full-history rewrite — the warning makes
    that state explicit in the job logs instead of silent."""
    try:
        existing = set(spark.read.parquet(out_path).columns)
    except Exception:
        return  # first write — nothing to compare against
    new = set(df.columns)
    added, dropped = sorted(new - existing), sorted(existing - new)
    if added or dropped:
        print(
            f"WARNING: SCHEMA DRIFT vs {out_path}: added={added} dropped={dropped} — "
            "this run's periods will differ from the rest of the dataset; "
            "plan a full-history rewrite to make the schema uniform"
        )


def prune_period_days(df, start: date, end: date, margin_days: int = 0):
    """Path-level pruning on the ``period=YYYY-MM-DD`` layout the derived
    datasets use (e.g. slot_orders_ab_group), widened by ``margin_days``
    (period follows the BEIJING activity date; boundary rows can sit one
    partition over from a UTC scan bound). No-op without the column."""
    if "period" not in df.columns:
        print("no period partition column; relying on timestamp filters only")
        return df
    lo = str(start - timedelta(days=margin_days))
    hi = str(end + timedelta(days=margin_days))
    return df.filter(F.col("period").between(lo, hi))


def beijing_today() -> date:
    return (datetime.now(timezone.utc) + timedelta(hours=BJ_UTC_OFFSET_HOURS)).date()


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def month_start(d: date) -> date:
    return d.replace(day=1)


def prev_month_start(d: date) -> date:
    return (d.replace(day=1) - timedelta(days=1)).replace(day=1)
