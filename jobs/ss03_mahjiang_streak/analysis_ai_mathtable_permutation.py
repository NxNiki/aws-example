"""SS03 AI-group mathtable-permutation analysis: retention, bet volume, RTP.

One-off analysis (not wired into ``run_scheduled_etl_jobs.py``), Jira AIP-575:
for the AI ab_group only, compare two windows of Beijing bet dates —
range 1: 2026-06-19..2026-08-17, range 2: 2026-08-19..2026-09-02 — on
day-1 retention, per-user-day total bet (CNY), number of BASE bets,
per-user-day RTP, and pooled RTP across all users per calendar day.
Also ranks the top ordered mathtable permutations (e.g. ``A-B-C``) per
user-day by day-1 retention / total bet / num bets, and compares the AI
group against the fixed-table groups (Default/AB_TEST_A/AB_TEST_B), each
labeled by the single math table it played — group names are not shown.

Definitions:
- **Block**: a maximal run of consecutive bets (time order: created_at,
  spin_id) on the same math table within a (user, Beijing bet_date).
- **Cleaning**: blocks with fewer than 30 BASE bets are dropped from the
  PERMUTATION LABEL (block-level, not per-table daily total: A(20) B(50)
  A(20) drops both A blocks from the label). Metrics — total bet, num bets,
  payout/RTP — always cover the whole day's bets, in every table; cleaning
  only decides the label, and user-days with no surviving block drop out.
  The range summary, daily RTP and AI-vs-fixed-table comparison are unaffected
  by cleaning entirely.
- **Permutation**: the surviving blocks' math tables in time order, with
  adjacent duplicates collapsed (duplicates can appear after small blocks
  are removed), joined with ``-``. Order matters: A-B != B-A.
- **bet_date**: Beijing calendar date of the bet's SESSION start (30-min bet
  gap sessions, repo ``bet_session`` convention) — play crossing midnight
  stays on the day it started, matching analysis_ab_mathtable_retention.py.
- **user_rtp**: per user-day, sum(actual_payout) / sum(BASE bet_amount),
  both CNY-converted. **daily rtp**: the same ratio pooled over all AI
  users on a calendar day.
- **Retention**: cohort = cleaned AI user-days; retained if the user has
  ANY SS03 bet (any group, uncleaned) on D+1. D+1 beyond ``PRESENCE_END``
  (last complete Beijing day) reports null. The AB policy cutover days
  2026-08-03/04 are excluded from cohorts/metrics (group membership
  ambiguous), matching ``analysis_ab_mathtable_retention.py``.
- **Currency**: CNY bets ONLY (the MAB policy applies to CNY users only);
  other currencies are excluded everywhere, including the retention target.

Source: Athena ``bituslabs_ds.slot_orders_ab_group`` (S3 cold data with the
policy-correct ``ab_group``; status/op-code junk pre-filtered upstream).

Usage:
    python jobs/ss03_mahjiang_streak/analysis_ai_mathtable_permutation.py --extract
    python jobs/ss03_mahjiang_streak/analysis_ai_mathtable_permutation.py --analyze
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import awswrangler as wr
import boto3
import polars as pl

from bituslabs_ds.config import DEFAULT_ATHENA_OUTPUT, REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RANGES = {
    "range1_pre": (date(2026, 6, 19), date(2026, 8, 17)),
    "range2_post": (date(2026, 8, 19), date(2026, 9, 2)),
}
SCAN_PERIOD_START = "2026-06-19"
SCAN_PERIOD_END = "2026-09-02"
# Sessions: 30-min bet gap (repo ``bet_session`` convention); every bet in a
# session belongs to the session START's Beijing date, so play crossing
# midnight stays on the day it started. Scan 2 extra days before the window
# so sessions straddling the window start keep their true start date.
SESSION_GAP_SECONDS = 1800
SESSION_SCAN_START = "2026-06-17"
# AB grouping policy flipped partition_ab -> user-id digits across these
# Beijing days; AI membership is ambiguous, so they are dropped from cohorts.
EXCLUDED_COHORT_DATES = (date(2026, 8, 3), date(2026, 8, 4))
# Last COMPLETE Beijing day on S3 at analysis time (2026-09-02 is partial);
# D+1 presence past it would understate retention, so those cells report null.
PRESENCE_END = date(2026, 9, 1)

MIN_BLOCK_BETS = 30
# Permutations ranked only over cells with at least this many user-days,
# so a one-off sequence with 100% retention can't top the table.
MIN_PERMUTATION_USER_DAYS = 30

FX_AS_OF = "2026-08-20"
FX_TO_CNY = {
    "CNY": 1.0,
    "USD": 6.7460,
    "INR": 0.070504,
    "IDR": 0.000378,
    "KRW": 0.0047890,
    "MYR": 1.6690,
    "PHP": 0.108900,
    "THB": 0.204844,
    "VND": 0.000258,
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output_ss03_mahjiang_streak" / "ai_mathtable_permutation"
BLOCKS_DIR = OUTPUT_DIR / "blocks"
PRESENCE_DIR = OUTPUT_DIR / "presence"


def _rate_case() -> str:
    whens = "\n            ".join(f"WHEN '{c}' THEN {r!r}" for c, r in FX_TO_CNY.items())
    return f"CASE currency_type\n            {whens}\n        END"


def blocks_sql() -> str:
    """One row per consecutive same-mathtable block of bets in a user-day.

    Covers all four ab_groups; the AI permutation analysis filters to AI and
    the single-table comparison uses Default/AB_TEST_A/AB_TEST_B labeled by
    the math table they actually played (their group names are never shown).
    """
    return f"""
WITH bets AS (
    SELECT
        user_id,
        spin_id,
        created_at,
        COALESCE(NULLIF(math_table_id, ''), '(none)') AS mathtable,
        bet_type,
        bet_amount,
        actual_payout,
        currency_type,
        ab_group
    FROM slot_orders_ab_group
    WHERE game_id = 'SS03'
      AND ab_group IN ('AI', 'AB_TEST_A', 'AB_TEST_B', 'Default')
      AND currency_type = 'CNY'
      AND period BETWEEN '{SESSION_SCAN_START}' AND '{SCAN_PERIOD_END}'
),

sess AS (
    SELECT
        t.*,
        SUM(
            CASE
                WHEN prev_ts IS NULL OR date_diff('second', prev_ts, created_at) > {SESSION_GAP_SECONDS} THEN 1
                ELSE 0
            END
        ) OVER (PARTITION BY user_id ORDER BY created_at, spin_id ROWS UNBOUNDED PRECEDING) AS session_seq
    FROM (
        SELECT
            *,
            LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at, spin_id) AS prev_ts
        FROM bets
    ) AS t
),

labeled AS (
    SELECT
        *,
        CAST(MIN(created_at) OVER (PARTITION BY user_id, session_seq) + INTERVAL '8' HOUR AS DATE) AS bet_date
    FROM sess
),

seq AS (
    SELECT
        t.*,
        LAG(mathtable) OVER (PARTITION BY user_id, bet_date ORDER BY created_at, spin_id) AS prev_mt
    FROM labeled AS t
),

blocks AS (
    SELECT
        *,
        SUM(CASE WHEN prev_mt IS NULL OR prev_mt <> mathtable THEN 1 ELSE 0 END)
            OVER (PARTITION BY user_id, bet_date ORDER BY created_at, spin_id ROWS UNBOUNDED PRECEDING)
            AS block_seq
    FROM seq
    WHERE bet_date >= DATE '{SCAN_PERIOD_START}'
)

SELECT
    bet_date,
    user_id,
    block_seq,
    MIN(mathtable) AS mathtable,
    MIN(ab_group) AS ab_group,
    COUNT(CASE WHEN bet_type = 'BASE' THEN 1 END) AS n_base,
    COUNT(*) AS n_rows,
    SUM(CASE WHEN bet_type = 'BASE' THEN bet_amount * {_rate_case()} END) AS bet_cny,
    SUM(actual_payout * {_rate_case()}) AS payout_cny,
    COUNT(CASE WHEN currency_type NOT IN ({", ".join(f"'{c}'" for c in FX_TO_CNY)}) THEN 1 END)
        AS n_unknown_currency
FROM blocks
GROUP BY bet_date, user_id, block_seq
"""


def presence_sql() -> str:
    """Distinct (user, session-start Beijing day) over ALL SS03 groups, CNY bets.

    The retention target — same session-start dating as the blocks query so a
    D+1 session crossing midnight counts on the day it started.
    """
    return f"""
WITH bets AS (
    SELECT user_id, spin_id, created_at
    FROM slot_orders_ab_group
    WHERE game_id = 'SS03'
      AND currency_type = 'CNY'
      AND period BETWEEN '{SESSION_SCAN_START}' AND '{SCAN_PERIOD_END}'
),

sess AS (
    SELECT
        user_id,
        created_at,
        SUM(
            CASE
                WHEN prev_ts IS NULL OR date_diff('second', prev_ts, created_at) > {SESSION_GAP_SECONDS} THEN 1
                ELSE 0
            END
        ) OVER (PARTITION BY user_id ORDER BY created_at, spin_id ROWS UNBOUNDED PRECEDING) AS session_seq
    FROM (
        SELECT
            *,
            LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at, spin_id) AS prev_ts
        FROM bets
    ) AS t
)

SELECT DISTINCT
    CAST(MIN(created_at) OVER (PARTITION BY user_id, session_seq) + INTERVAL '8' HOUR AS DATE) AS bet_date,
    user_id
FROM sess
"""


def _run_and_save(sql: str, out_dir: Path, label: str) -> None:
    session = boto3.Session(region_name=REGION)
    logger.info("running Athena extraction for %s", label)
    df = wr.athena.read_sql_query(
        sql,
        database="bituslabs_ds",
        s3_output=DEFAULT_ATHENA_OUTPUT,
        ctas_approach=False,
        boto3_session=session,
    )
    logger.info("fetched %d %s rows", len(df), label)
    pdf = pl.from_pandas(df).with_columns(pl.col("bet_date").cast(pl.Date))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf.write_parquet(out_dir / "data.parquet")
    logger.info("saved %s", out_dir / "data.parquet")


def extract() -> None:
    _run_and_save(blocks_sql(), BLOCKS_DIR, "ai-mathtable-blocks")
    _run_and_save(presence_sql(), PRESENCE_DIR, "presence")


def _load(dir_: Path, label: str) -> pl.DataFrame:
    f = dir_ / "data.parquet"
    if not f.exists():
        raise SystemExit(f"no extracted {label} data at {f}; run with --extract first")
    return pl.read_parquet(f)


def range_expr() -> pl.Expr:
    expr: pl.Expr = pl.lit(None, dtype=pl.String)
    for name, (start, end) in RANGES.items():
        expr = pl.when(pl.col("bet_date").is_between(start, end)).then(pl.lit(name)).otherwise(expr)
    return expr.alias("range")


def build_user_day(blocks: pl.DataFrame, min_block_bets: int = 0) -> pl.DataFrame:
    """Per-user-day table with permutation + winner-take-all labels and CNY totals.

    ``min_block_bets`` > 0 applies the block cleaning to the PERMUTATION LABEL
    only — total bet / num bets / payout (and thus RTP) always cover the WHOLE
    day's blocks, so a dropped short block still counts toward the metrics.
    User-days with no surviving block drop out entirely (they can't be labeled).
    """
    if blocks["n_unknown_currency"].sum() > 0:
        raise SystemExit("rows with un-mapped currency present — extend FX_TO_CNY and re-extract")

    blocks = blocks.sort(["user_id", "bet_date", "block_seq"])
    kept = blocks.filter(pl.col("n_base") >= min_block_bets)
    if min_block_bets:
        logger.info(
            "cleaning (permutation label only): kept %d/%d blocks (>= %d BASE bets), %d/%d user-days",
            len(kept),
            len(blocks),
            min_block_bets,
            kept.select("user_id", "bet_date").n_unique(),
            blocks.select("user_id", "bet_date").n_unique(),
        )

    # Collapse adjacent duplicate tables that appear once small blocks between
    # them are removed, then join the sequence into the permutation label.
    perm = (
        kept.with_columns(
            (pl.col("mathtable") != pl.col("mathtable").shift(1).over("user_id", "bet_date"))
            .fill_null(True)
            .alias("is_new")
        )
        .filter(pl.col("is_new"))
        .group_by("user_id", "bet_date", maintain_order=True)
        .agg(pl.col("mathtable").str.join("-").alias("permutation"))
    )

    # normal_kakuteiB/C are single-spin bonus sub-tables triggered from a main
    # table (median 1 BASE spin) — a sub-mode of the day's table, not a table
    # choice, so they never label a day and don't count toward n_real_tables.
    is_aux = pl.col("mathtable").str.contains("(?i)kakutei")
    winner = (
        blocks.filter(~is_aux)
        .group_by("user_id", "bet_date", "mathtable")
        .agg(pl.col("n_base").sum(), pl.col("block_seq").min())
        .sort(["n_base", "block_seq"], descending=[True, False])
        .group_by("user_id", "bet_date", maintain_order=False)
        .agg(pl.col("mathtable").first().alias("mt_winner"))
    )

    group_rank = {"AI": 0, "AB_TEST_A": 1, "AB_TEST_B": 2, "Default": 3}
    ud = (
        blocks.group_by("user_id", "bet_date")
        .agg(
            pl.col("n_base").sum().alias("user_num_bets"),
            pl.col("bet_cny").sum().alias("user_total_bet"),
            pl.col("payout_cny").sum().alias("payout_cny"),
            pl.len().alias("n_blocks"),
            pl.col("mathtable").filter(~is_aux).n_unique().alias("n_real_tables"),
            pl.col("ab_group")
            .replace_strict(group_rank)
            .min()
            .replace_strict({v: k for k, v in group_rank.items()})
            .alias("ab_group"),
        )
        .join(perm, on=["user_id", "bet_date"], how="inner")
        .join(winner, on=["user_id", "bet_date"], how="left")
        .with_columns(
            pl.when(pl.col("user_total_bet") > 0)
            .then(pl.col("payout_cny") / pl.col("user_total_bet"))
            .alias("user_rtp"),
            range_expr(),
        )
        .filter(pl.col("range").is_not_null() & ~pl.col("bet_date").is_in(list(EXCLUDED_COHORT_DATES)))
    )
    return ud


def add_retention(ud: pl.DataFrame, presence: pl.DataFrame) -> pl.DataFrame:
    target = ud.select("user_id", (pl.col("bet_date") + timedelta(days=1)).alias("target_date"))
    hit = target.join(
        presence.select("user_id", pl.col("bet_date").alias("target_date")).unique(),
        on=["user_id", "target_date"],
        how="left",
        coalesce=False,
    ).select(pl.col("user_id_right").is_not_null().alias("retained_d1"))
    return ud.with_columns(hit["retained_d1"]).with_columns(
        pl.when(pl.col("bet_date") + timedelta(days=1) > PRESENCE_END)
        .then(None)
        .otherwise(pl.col("retained_d1"))
        .alias("retained_d1")
    )


# Winsorization level for the user_total_bet / user_num_bets MEANS: values
# above the 99th percentile are CLIPPED to it (kept, not dropped) — whales are
# legitimate revenue, but one user-day shouldn't own a cell's mean. Thresholds
# are estimated per (range, metric) on the FULL uncleaned user-day population
# (a quantile re-estimated inside a 40-row permutation cell would be noise)
# and applied to every cell. Medians, retention and RTP stay unclipped.
CLIP_PCT = 0.99
CLIP_COLS = ("user_total_bet", "user_num_bets")


def clip_thresholds(ud_raw_all: pl.DataFrame) -> pl.DataFrame:
    return ud_raw_all.group_by("range").agg(
        pl.col(c).quantile(CLIP_PCT, interpolation="linear").alias(f"_{c}_hi") for c in CLIP_COLS
    )


def with_clipped(df: pl.DataFrame, thresholds: pl.DataFrame) -> pl.DataFrame:
    return (
        df.join(thresholds, on="range", how="left")
        .with_columns(pl.min_horizontal(c, f"_{c}_hi").alias(f"{c}_clip") for c in CLIP_COLS)
        .drop(f"_{c}_hi" for c in CLIP_COLS)
    )


AGGS = [
    pl.len().alias("user_days"),
    pl.col("user_id").n_unique().alias("n_users"),
    pl.col("retained_d1").mean().alias("retention_d1"),
    pl.col("retained_d1").count().alias("d1_cohort"),
    pl.col("user_total_bet_clip").mean().alias("avg_user_total_bet"),
    pl.col("user_total_bet").median().alias("med_user_total_bet"),
    pl.col("user_num_bets_clip").mean().alias("avg_user_num_bets"),
    pl.col("user_num_bets").median().alias("med_user_num_bets"),
    pl.col("user_rtp").mean().alias("avg_user_rtp"),
    pl.col("user_rtp").median().alias("med_user_rtp"),
    (pl.col("payout_cny").sum() / pl.col("user_total_bet").sum()).alias("pooled_rtp"),
]


def analyze() -> None:
    blocks = _load(BLOCKS_DIR, "blocks")
    presence = _load(PRESENCE_DIR, "presence")
    # Block cleaning applies ONLY to the permutation ranking; the range
    # summary, daily RTP, and AI-vs-fixed-table comparison use all bets.
    ud_raw_all = add_retention(build_user_day(blocks), presence)
    thresholds = clip_thresholds(ud_raw_all)
    logger.info("p%d clip thresholds:\n%s", int(CLIP_PCT * 100), thresholds.sort("range"))
    ud_raw_all = with_clipped(ud_raw_all, thresholds)
    ud_raw = ud_raw_all.filter(pl.col("ab_group") == "AI")
    ud = with_clipped(
        add_retention(build_user_day(blocks, MIN_BLOCK_BETS), presence).filter(pl.col("ab_group") == "AI"),
        thresholds,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = ud_raw.group_by("range").agg(AGGS).sort("range")
    summary.write_csv(OUTPUT_DIR / "summary_range.csv")

    # Single-mathtable comparison (uncleaned): Default/AB_TEST_A/AB_TEST_B
    # user-days labeled by the ONE table played that day — group names
    # intentionally not shown; a fixed-table group is identified by the table
    # it played within a range. User-days touching 2+ REAL tables (a group's
    # table-switch day) are DROPPED: splitting a day across tables would
    # understate that day's bets/total under any single-table label. The
    # kakutei single-spin bonus sub-tables don't count as a second table.
    fixed = ud_raw_all.filter(pl.col("ab_group") != "AI")
    single = fixed.filter(pl.col("n_real_tables") == 1)
    logger.info("single-table comparison: kept %d/%d non-AI user-days (single-table)", len(single), len(fixed))
    comparison = pl.concat(
        [
            single.group_by("range", pl.col("mt_winner").alias("mathtable")).agg(AGGS),
            ud_raw.group_by("range").agg(AGGS).with_columns(pl.lit("AI(全部排列)").alias("mathtable")),
        ],
        how="diagonal",
    ).sort(["range", "user_days"], descending=[False, True])
    comparison.write_csv(OUTPUT_DIR / "summary_range_by_table.csv")

    daily = (
        ud_raw.group_by("bet_date")
        .agg(
            pl.len().alias("n_users"),
            pl.col("user_total_bet").sum().alias("total_bet_cny"),
            pl.col("payout_cny").sum().alias("payout_cny"),
            (pl.col("payout_cny").sum() / pl.col("user_total_bet").sum()).alias("daily_rtp"),
            pl.col("retained_d1").mean().alias("retention_d1"),
            pl.col("user_num_bets").sum().alias("num_bets"),
        )
        .sort("bet_date")
        .with_columns(range_expr())
    )
    daily.write_csv(OUTPUT_DIR / "daily_all_users.csv")

    perm = ud.group_by("range", "permutation").agg(AGGS).sort(["range", "user_days"], descending=[False, True])
    perm.write_csv(OUTPUT_DIR / "summary_permutation.csv")

    ranked = perm.filter(pl.col("user_days") >= MIN_PERMUTATION_USER_DAYS)
    tops = []
    for metric in ("retention_d1", "avg_user_total_bet", "avg_user_num_bets"):
        for rng in RANGES:
            top = (
                ranked.filter(pl.col("range") == rng)
                .sort(metric, descending=True, nulls_last=True)
                .head(5)
                .with_columns(pl.lit(metric).alias("ranked_by"), pl.int_range(1, pl.len() + 1).alias("rank"))
            )
            tops.append(top)
    top5 = pl.concat(tops).select(["ranked_by", "rank"] + [c for c in perm.columns])
    top5.write_csv(OUTPUT_DIR / "top5_permutations.csv")

    for name in (
        "summary_range",
        "summary_range_by_table",
        "daily_all_users",
        "summary_permutation",
        "top5_permutations",
    ):
        logger.info("wrote %s", OUTPUT_DIR / f"{name}.csv")

    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=4, tbl_width_chars=220):
        print("\n=== Range summary (AI group, uncleaned) ===")
        print(summary)
        print("\n=== Top-5 permutations per metric/range (>= %d user-days) ===" % MIN_PERMUTATION_USER_DAYS)
        print(top5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="run the Athena extraction and cache locally")
    parser.add_argument("--analyze", action="store_true", help="compute metric tables and write CSVs")
    args = parser.parse_args()
    if args.extract:
        extract()
    if args.analyze:
        analyze()
    if not (args.extract or args.analyze):
        parser.print_help()


if __name__ == "__main__":
    main()
