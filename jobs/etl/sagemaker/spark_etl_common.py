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

# AB grouping policy cutover (announced 2026-08-03 LA time): from this moment
# groups are assigned by the LAST DIGIT of user_id — 0-3 Default, 4-5
# AB_TEST_A, 6-7 AB_TEST_B, 8-9 AI. Before it, by the partition_ab ids above
# (which STOPPED reflecting assignment at the cutover). The timestamp is
# empirical: SS03 mathtable serving for old-vs-digit-group users flips from
# 100% old-config to ~99% digit-config across the 05:00-06:00 Beijing hour of
# 2026-08-04 (bet volume also collapses — the deploy window); 05:30 splits
# that mixed hour (~600 bets, mostly group-agnostic kakutei bonus tables).
AB_GROUP_DIGIT_POLICY_START_BJ = "2026-08-04 05:30:00"

# fish_hunter (FM01) group_tag policy: the personalized-retention (个性化挽留)
# experiment went live 2026-07-30 16:00 PST = 2026-07-31 00:00 UTC, assigning
# users with id last digit 0/1. Only the retention branch is gated on this
# moment; the strategy-name branches apply to all history (risk-control
# strategies keep their label wherever they occur).
FISH_RETENTION_POLICY_START_UTC = "2026-07-31 00:00:00"

# Branch order of fish_group_tag_sql, which doubles as the user-day collapse
# priority in fish_group_tag_day_case ('default' is the implicit last tier).
# boost_pool outranks everything for legacy data; the strategy no longer
# exists in new data.
FISH_GROUP_TAG_PRIORITY = ("boost_pool", "dynamic_rtp", "risk_control", "retention")


def ab_group_sql(ab_label_expr: str, user_id_expr: str, bj_ts_expr: str, include_ab_tests: bool = True) -> str:
    """CASE expression labeling a bet's AB group under the timestamp-gated
    policy.

    ``ab_label_expr`` is the first partition_ab id (e.g. PARTITION_AB_FIRST),
    ``user_id_expr`` the numeric user id, ``bj_ts_expr`` a TIMESTAMP in
    Beijing time. Games without AB test groups (include_ab_tests=False)
    collapse the test digits / legacy test ids into Default, keeping their
    historical two-group vocabulary."""
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
            WHEN {bj_ts_expr} >= TIMESTAMP '{AB_GROUP_DIGIT_POLICY_START_BJ}' THEN {digit_case}
            ELSE CASE
                WHEN {ab_label_expr} = '{AI_GROUP_ID}' THEN 'AI'{legacy_tests}
                ELSE 'Default'
            END
        END"""


# Day-attribution session gap, the platform's bet_session convention: a
# session breaks after 30 minutes without a bet. Every game-stats dataset
# derives activity_date from the SESSION-START Beijing date so cross-midnight
# play stays on the day it started (the daily-stats midnight edge fix; see
# the session vocabulary table in docs/ab_group_policy.md).
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


# SS03 dashboard grouping uses the game team's ANNOUNCED digit-policy start
# (2026-08-03 16:00 PDT) instead of the empirically located
# AB_GROUP_DIGIT_POLICY_START_BJ the shared orders dataset keeps: bets in the
# ~1.5h between the serving flip and the announced time carry stale
# partition_ab labels by design (product decision).
SS03_AB_GROUP_ANNOUNCED_START_UTC = "2026-08-03 23:00:00"


def ss03_bet_ab_group_sql(ab_label_expr: str, user_id_expr: str, utc_ts_expr: str) -> str:
    """Row-level SS03 dashboard AB label under the ANNOUNCED cutover.

    ``utc_ts_expr`` must be a UTC TIMESTAMP. New era: user-id last digit
    (0-3 Default, 4-5 A, 6-7 B, 8-9 AI). Old era: partition_ab[0] id mapping
    (4f1a... = A serves the shi-family tables that digit-4/5 users continue,
    4a04... = B the BGadj_v3 family). The caller collapses each user's
    session-day to ONE label with priority AI > AB_TEST_A > AB_TEST_B >
    Default — old-era assignment was per-bet, so a single AI bet claims the
    whole user-day; new-era digit labels are user-stable and collapse as a
    no-op."""
    digit = f"CAST({user_id_expr} AS BIGINT) % 10"
    return f"""CASE
            WHEN {utc_ts_expr} >= TIMESTAMP '{SS03_AB_GROUP_ANNOUNCED_START_UTC}' THEN CASE
                WHEN {digit} >= 8 THEN 'AI'
                WHEN {digit} >= 6 THEN 'AB_TEST_B'
                WHEN {digit} >= 4 THEN 'AB_TEST_A'
                ELSE 'Default'
            END
            WHEN {ab_label_expr} = '{AI_GROUP_ID}' THEN 'AI'
            WHEN {ab_label_expr} = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
            WHEN {ab_label_expr} = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'
            ELSE 'Default'
        END"""


def fish_group_tag_sql(strategy_expr: str, user_id_expr: str, utc_ts_expr: str) -> str:
    """CASE expression labeling a single bullet's ``group_tag``.

    ``utc_ts_expr`` must be a UTC TIMESTAMP (the retention gate is specified
    in UTC). The ``substr`` tests are the escaped ``LIKE '.._FISHING\\_%'``
    (literal underscore); ``'%RISK_CONTROL%'`` also matches the legacy
    ``RISK_CONTROLLED`` strategy, and the whole DYNAMIC_RTP family shares
    the ``dynamic_rtp`` label (V1/V2 are legacy date ranges of the same
    program). CR_FISHING_* is the personalized-retention treatment itself
    (served exclusively to digit-0/1 users, including a small canary in the
    hour before the official launch), so it labels ``retention`` regardless
    of the gate; otherwise rows before the launch can never label
    ``retention``, and applying this to full history preserves the
    pre-launch tagging unchanged."""
    return f"""CASE
            WHEN {strategy_expr} = 'BOOST_POOL' THEN 'boost_pool'
            WHEN {strategy_expr} IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3') THEN 'dynamic_rtp'
            WHEN substr({strategy_expr}, 1, 11) = 'RC_FISHING_' THEN 'risk_control'
            WHEN {strategy_expr} LIKE '%RISK_CONTROL%' THEN 'risk_control'
            WHEN substr({strategy_expr}, 1, 11) = 'CR_FISHING_' THEN 'retention'
            WHEN {utc_ts_expr} >= TIMESTAMP '{FISH_RETENTION_POLICY_START_UTC}'
                AND CAST({user_id_expr} AS BIGINT) % 10 IN (0, 1) THEN 'retention'
            ELSE 'default'
        END"""


def fish_group_tag_day_case(tag_expr: str) -> str:
    """Collapse a user-day's row-level tags to ONE label (use inside a
    GROUP BY user, day aggregate). A single bullet in a higher tier claims
    the whole user-day; priority is the row CASE's branch order."""
    branches = "\n            ".join(
        f"WHEN MAX(CASE WHEN {tag_expr} = '{tag}' THEN 1 ELSE 0 END) > 0 THEN '{tag}'"
        for tag in FISH_GROUP_TAG_PRIORITY
    )
    return f"""CASE
            {branches}
            ELSE 'default'
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
