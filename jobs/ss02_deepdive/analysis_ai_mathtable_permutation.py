"""SS02 AI-group mathtable-permutation analysis: retention, bet volume, RTP.

One-off analysis (not wired into ``run_scheduled_etl_jobs.py``): SS02
("Deep Dive"), single window 2026-04-08..2026-09-02 (Beijing bet dates),
CNY only. Reports day-1 retention, per-user-day total bet, number of BASE
bets, per-user-day RTP, and pooled RTP across all users per calendar day.
Also ranks the top ordered mathtable permutations (e.g. ``A|B|C``) per
user-day by day-1 retention / total bet / num bets, and compares the AI
group against Default user-days labeled by the single math table played.

Grouping: SS02 has only AI and Default, derived from the RAW
``partition_ab_label`` for the whole window — the 2026-08-04 digit-policy
cutover applies to SS03 only, and the dataset's global ``ab_group`` column
mislabels SS02 after that date (verified: partition-AI keeps the MAB table
rotation post-08-04 while digit-"AI" converges to Default's medium3 mix).

Definitions:
- **Block**: a maximal run of consecutive bets (time order: created_at,
  spin_id) on the same math table within a (user, Beijing bet_date).
- **Cleaning**: remainder segments with fewer than 20 BASE bets are dropped
  from the PERMUTATION LABEL (segment-level, not per-table daily total). Metrics — total bet, num bets,
  payout/RTP — always cover the whole day's bets, in every table; cleaning
  only decides the label, and user-days with no surviving block drop out.
  The range summary, daily RTP and AI-vs-fixed-table comparison are unaffected
  by cleaning entirely.
- **Permutation**: consecutive same-table runs are split into 30-BASE-bet
  segments (SS02's MAB rotation cadence); the label is the kept segments'
  tables in time order joined with ``|`` (adjacent repeats kept: 60 bets of
  fast1 -> "fast1|fast1"). Label length tracks play volume in ~30-bet units.
  Order matters: A|B != B|A.
- **bet_date**: Beijing calendar date of the bet's SESSION start (30-min bet
  gap sessions, repo ``bet_session`` convention) — play crossing midnight
  stays on the day it started, matching the SS03 analyses.
- **user_rtp**: per user-day, sum(actual_payout) / sum(BASE bet_amount),
  both CNY-converted. **daily rtp**: the same ratio pooled over all AI
  users on a calendar day.
- **Retention**: cohort = cleaned AI user-days; retained if the user has
  ANY SS02 CNY bet (any group, uncleaned) on D+1. D+1 beyond ``PRESENCE_END``
  (last complete Beijing day) reports null.
- **Currency**: CNY bets ONLY (the MAB policy applies to CNY users only);
  other currencies are excluded everywhere, including the retention target.

Source: Athena ``bituslabs_ds.slot_orders_ab_group`` (S3 cold data;
status/op-code junk pre-filtered upstream; grouping re-derived from
``partition_ab_label`` as above).

Usage:
    python jobs/ss02_deepdive/analysis_ai_mathtable_permutation.py --extract
    python jobs/ss02_deepdive/analysis_ai_mathtable_permutation.py --analyze
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
    "full": (date(2026, 4, 8), date(2026, 9, 2)),
}
SCAN_PERIOD_START = "2026-04-08"
SCAN_PERIOD_END = "2026-09-02"
# Sessions: 30-min bet gap (repo ``bet_session`` convention); every bet in a
# session belongs to the session START's Beijing date, so play crossing
# midnight stays on the day it started. Scan 2 extra days before the window
# so sessions straddling the window start keep their true start date.
SESSION_GAP_SECONDS = 1800
# A "bet streak" = consecutive BASE bets each within this many seconds of the
# previous one; user_max_streak is the day's longest such run.
STREAK_GAP_SECONDS = 200
SESSION_SCAN_START = "2026-04-06"
# SS02 grouping is partition_ab-based for the whole window (the 08-04 digit
# cutover is SS03-only), so no ambiguous cohort dates need excluding.
EXCLUDED_COHORT_DATES: tuple[date, ...] = ()
# Last COMPLETE Beijing day on S3 at analysis time (2026-09-02 is partial);
# D+1 presence past it would understate retention, so those cells report null.
PRESENCE_END = date(2026, 9, 1)

MIN_BLOCK_BETS = 20
# Consecutive same-table runs are split into 30-BASE-bet segments for the
# permutation label (60 bets of fast1 -> "fast1|fast1", length 2): SS02's MAB
# rotates every ~30 bets (run-length histogram peaks at 30-34; SS03 uses 50),
# so one segment ~= one rotation decision. MIN_BLOCK_BETS applies to the
# remainder segment.
SEGMENT_BETS = 30
# Daily total-bet (CNY) tiers for the AI-vs-single-table comparison. SS02
# players bet ~2-3x less than SS03 (user-day median 49 vs 110 CNY), so the
# edges sit at ~p18 / median / ~p91 instead of SS03's 10/100/1000.
BET_BUCKET_EDGES = (10, 50, 500)
# Permutations ranked only over cells with at least this many user-days,
# so a one-off sequence with 100% retention can't top the table.
MIN_PERMUTATION_USER_DAYS = 20

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

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output_ss02_deepdive" / "ai_mathtable_permutation"
BLOCKS_DIR = OUTPUT_DIR / "blocks"
PRESENCE_DIR = OUTPUT_DIR / "presence"
SEQ_DIR = OUTPUT_DIR / "seq"


def _rate_case() -> str:
    whens = "\n            ".join(f"WHEN '{c}' THEN {r!r}" for c, r in FX_TO_CNY.items())
    return f"CASE currency_type\n            {whens}\n        END"


def blocks_sql() -> str:
    """One row per consecutive same-mathtable block of bets in a user-day.

    Covers both SS02 groups (AI + Default via partition_ab_label); the AI
    permutation analysis filters to AI and the single-table comparison uses
    Default user-days labeled by the math table actually played.
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
        -- SS02 grouping comes from the RAW partition_ab label for the whole
        -- window: the 2026-08-04 digit-policy cutover applies to SS03 only,
        -- so the dataset's global ab_group column mislabels SS02 after it
        -- (real AI users scatter into digit groups; verified by table mix).
        CASE partition_ab_label
            WHEN 'jojpin-9mokha-rexQug' THEN 'AI'
            ELSE 'Default'
        END AS ab_group
    FROM slot_orders_ab_group
    WHERE game_id = 'SS02'
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
    """Distinct (user, session-start Beijing day) over ALL SS02 groups, CNY bets.

    The retention target — same session-start dating as the blocks query so a
    D+1 session crossing midnight counts on the day it started.
    """
    return f"""
WITH bets AS (
    SELECT user_id, spin_id, created_at
    FROM slot_orders_ab_group
    WHERE game_id = 'SS02'
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


def seq_sql() -> str:
    """Per user-day betting rhythm: median/avg within-session BASE inter-bet
    interval (seconds) and the longest streak of consecutive BASE bets with
    gaps under {STREAK_GAP_SECONDS}s. Same sessionization + session-start dating as
    blocks_sql so the join keys line up."""
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
        -- SS02 grouping comes from the RAW partition_ab label for the whole
        -- window: the 2026-08-04 digit-policy cutover applies to SS03 only,
        -- so the dataset's global ab_group column mislabels SS02 after it
        -- (real AI users scatter into digit groups; verified by table mix).
        CASE partition_ab_label
            WHEN 'jojpin-9mokha-rexQug' THEN 'AI'
            ELSE 'Default'
        END AS ab_group
    FROM slot_orders_ab_group
    WHERE game_id = 'SS02'
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

flows AS (
    SELECT
        *,
        SUM(CASE WHEN bet_type <> 'BASE' THEN 1 ELSE 0 END)
            OVER (PARTITION BY user_id, session_seq ORDER BY created_at, spin_id ROWS UNBOUNDED PRECEDING)
            AS cnt_free
    FROM labeled
),

-- n_free_between > 0 means FREE-game spins played out between the previous
-- BASE bet and this one: that gap is FG animation, not player pacing, so it
-- is excluded from interval stats and does NOT break a bet streak.
base AS (
    SELECT
        user_id,
        bet_date,
        session_seq,
        created_at,
        spin_id,
        date_diff(
            'second',
            LAG(created_at) OVER (PARTITION BY user_id, session_seq ORDER BY created_at, spin_id),
            created_at
        ) AS delta,
        cnt_free - LAG(cnt_free) OVER (PARTITION BY user_id, session_seq ORDER BY created_at, spin_id)
            AS n_free_between
    FROM flows
    WHERE bet_type = 'BASE' AND bet_date >= DATE '{SCAN_PERIOD_START}'
),

streaked AS (
    SELECT
        *,
        SUM(
            CASE
                WHEN delta IS NULL THEN 1
                WHEN n_free_between > 0 THEN 0
                WHEN delta >= {STREAK_GAP_SECONDS} THEN 1
                ELSE 0
            END
        )
            OVER (PARTITION BY user_id, bet_date ORDER BY created_at, spin_id ROWS UNBOUNDED PRECEDING)
            AS streak_id
    FROM base
),

streak_len AS (
    SELECT user_id, bet_date, streak_id, COUNT(*) AS len
    FROM streaked
    GROUP BY user_id, bet_date, streak_id
),

intervals AS (
    SELECT
        user_id,
        bet_date,
        approx_percentile(CASE WHEN n_free_between > 0 THEN NULL ELSE delta END, 0.5) AS user_med_interval,
        AVG(CASE WHEN n_free_between > 0 THEN NULL ELSE delta END) AS user_avg_interval
    FROM base
    GROUP BY user_id, bet_date
),

streaks AS (
    SELECT user_id, bet_date, MAX(len) AS user_max_streak
    FROM streak_len
    GROUP BY user_id, bet_date
)

SELECT s.user_id, s.bet_date, i.user_med_interval, i.user_avg_interval, s.user_max_streak
FROM streaks AS s
LEFT JOIN intervals AS i ON s.user_id = i.user_id AND s.bet_date = i.bet_date
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
    _run_and_save(seq_sql(), SEQ_DIR, "user-day-seq")


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
    segments = (
        blocks.with_columns(
            (pl.col("n_base") // SEGMENT_BETS).alias("_full"),
            (pl.col("n_base") % SEGMENT_BETS).alias("_rem"),
        )
        .with_columns((pl.col("_full") + (pl.col("_rem") > 0).cast(pl.Int64)).alias("_nseg"))
        .with_columns(pl.int_ranges(0, pl.col("_nseg")).alias("_seg"))
        .explode("_seg")
        .drop_nulls("_seg")
        .with_columns(
            pl.when(pl.col("_seg") < pl.col("_full"))
            .then(pl.lit(SEGMENT_BETS))
            .otherwise(pl.col("_rem"))
            .alias("_seg_n")
        )
    )
    kept = segments.filter(pl.col("_seg_n") >= min_block_bets) if min_block_bets else segments
    if min_block_bets:
        logger.info(
            "label cleaning: kept %d/%d %d-bet segments (remainder >= %d BASE bets), %d/%d user-days",
            len(kept),
            len(segments),
            SEGMENT_BETS,
            min_block_bets,
            kept.select("user_id", "bet_date").n_unique(),
            blocks.select("user_id", "bet_date").n_unique(),
        )

    # Label = kept segments in time order, adjacent repeats NOT collapsed —
    # "shi-shi" (a 100-bet run) is distinct from "shi" (a 50-bet run).
    perm = (
        kept.sort(["user_id", "bet_date", "block_seq", "_seg"])
        .group_by("user_id", "bet_date", maintain_order=True)
        .agg(pl.col("mathtable").str.join("|").alias("permutation"))
    )

    # FourScatter (SS02) and kakutei (SS03) rows are feature/bonus spins
    # inside a main table's play — a sub-mode, not a table choice, so they
    # never label a day and don't count toward n_real_tables.
    is_aux = pl.col("mathtable").str.contains("(?i)kakutei|scatter")
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
            pl.when(pl.col("user_num_bets") > 0)
            .then(pl.col("user_total_bet") / pl.col("user_num_bets"))
            .alias("user_avg_bet_amount"),
            range_expr(),
        )
        .filter(
            pl.col("range").is_not_null()
            & (~pl.col("bet_date").is_in(list(EXCLUDED_COHORT_DATES)) if EXCLUDED_COHORT_DATES else pl.lit(True))
        )
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


def bucket_expr() -> pl.Expr:
    """Daily-spend tier label from the RAW (unclipped) user-day total bet."""
    edges = BET_BUCKET_EDGES
    labels = [f"<{edges[0]}"] + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)] + [f"{edges[-1]}+"]
    expr: pl.Expr = pl.lit(labels[-1])
    for edge, label in reversed(list(zip(edges, labels))):
        expr = pl.when(pl.col("user_total_bet") < edge).then(pl.lit(label)).otherwise(expr)
    return expr.alias("bet_bucket")


def with_retention_ci(df: pl.DataFrame, z: float = 1.96) -> pl.DataFrame:
    """95% Wilson interval for retention_d1 over the d1_cohort sample."""
    p, n = pl.col("retention_d1"), pl.col("d1_cohort")
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)).sqrt()) / denom
    return df.with_columns(
        ((center - half).clip(0, 1)).alias("retention_ci_lo"),
        ((center + half).clip(0, 1)).alias("retention_ci_hi"),
    )


# Winsorization level for the user_total_bet / user_num_bets MEANS: values
# above the 99th percentile are CLIPPED to it (kept, not dropped) — whales are
# legitimate revenue, but one user-day shouldn't own a cell's mean. Thresholds
# are estimated per (range, metric) on the FULL uncleaned user-day population
# (a quantile re-estimated inside a 40-row permutation cell would be noise)
# and applied to every cell. Medians, retention and RTP stay unclipped.
CLIP_PCT = 0.99
CLIP_COLS = ("user_total_bet", "user_num_bets", "user_avg_bet_amount", "user_med_interval", "user_max_streak")


def clip_thresholds(ud_raw_all: pl.DataFrame) -> pl.DataFrame:
    return ud_raw_all.group_by("range").agg(
        pl.col(c).quantile(CLIP_PCT, interpolation="linear").alias(f"_{c}_hi") for c in CLIP_COLS
    )


def with_clipped(df: pl.DataFrame, thresholds: pl.DataFrame) -> pl.DataFrame:
    return (
        df.join(thresholds, on="range", how="left")
        .with_columns(
            pl.when(pl.col(c).is_null()).then(None).otherwise(pl.min_horizontal(c, f"_{c}_hi")).alias(f"{c}_clip")
            for c in CLIP_COLS
        )
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
    pl.col("user_avg_bet_amount_clip").mean().alias("avg_user_avg_bet"),
    pl.col("user_avg_bet_amount").median().alias("med_user_avg_bet"),
    pl.col("user_med_interval_clip").mean().alias("avg_bet_interval"),
    pl.col("user_med_interval").median().alias("med_bet_interval"),
    pl.col("user_max_streak_clip").mean().alias("avg_max_streak"),
    pl.col("user_max_streak").median().alias("med_max_streak"),
    pl.col("user_total_bet_clip").std().alias("std_user_total_bet"),
    pl.col("user_num_bets_clip").std().alias("std_user_num_bets"),
    pl.col("user_avg_bet_amount_clip").std().alias("std_user_avg_bet"),
    pl.col("user_med_interval_clip").std().alias("std_bet_interval"),
    pl.col("user_med_interval").count().alias("n_bet_interval"),
    pl.col("user_max_streak_clip").std().alias("std_max_streak"),
    pl.col("user_max_streak").count().alias("n_max_streak"),
    pl.col("user_rtp").mean().alias("avg_user_rtp"),
    pl.col("user_rtp").median().alias("med_user_rtp"),
    (pl.col("payout_cny").sum() / pl.col("user_total_bet").sum()).alias("pooled_rtp"),
]


def ggr_thresholds(ud: pl.DataFrame) -> pl.DataFrame:
    """p99 clip threshold for USER-level GGR, per range, on the full population."""
    per_user = (
        ud.with_columns((pl.col("user_total_bet") - pl.col("payout_cny")).alias("ggr_day"))
        .group_by("range", "user_id")
        .agg(pl.col("ggr_day").sum().alias("user_ggr"))
    )
    # GGR is signed (negative = the user won), so the mean is winsorized on
    # BOTH tails (p1/p99) — a one-sided cap would bias it downward.
    return per_user.group_by("range").agg(
        pl.col("user_ggr").quantile(0.99, interpolation="linear").alias("_ggr_hi"),
        pl.col("user_ggr").quantile(0.01, interpolation="linear").alias("_ggr_lo"),
    )


def ggr_stats(ud: pl.DataFrame, label_cols: list[str], thresholds: pl.DataFrame) -> pl.DataFrame:
    """Per-cell stats of ONE lifetime GGR value per user (bet - payout summed
    over ALL the user's user-days in this frame's range/condition; negative =
    the user won). A user whose days span several labels is assigned to their
    MOST COMMON label; ties go to the label with the larger overall population
    (user-days) in the range. Mean is clipped at the range's p99 user-GGR
    thresholds (BOTH tails, p1/p99 — GGR is signed); median raw.
    """
    day = ud.with_columns((pl.col("user_total_bet") - pl.col("payout_cny")).alias("ggr_day"))
    tot = day.group_by("range", "user_id").agg(pl.col("ggr_day").sum().alias("user_ggr"))
    if label_cols:
        cell_sizes = day.group_by(["range"] + label_cols).agg(pl.len().alias("_cell_days"))
        assigned = (
            day.group_by(["range", "user_id"] + label_cols)
            .agg(pl.len().alias("_days"))
            .join(cell_sizes, on=["range"] + label_cols)
            .sort(["_days", "_cell_days"], descending=[True, True])
            .group_by("range", "user_id", maintain_order=False)
            .agg(pl.col(c).first() for c in label_cols)
        )
        users = assigned.join(tot, on=["range", "user_id"])
    else:
        users = tot
    users = users.join(thresholds, on="range", how="left").with_columns(
        pl.max_horizontal(pl.min_horizontal("user_ggr", "_ggr_hi"), "_ggr_lo").alias("user_ggr_clip")
    )
    return users.group_by(["range"] + label_cols).agg(
        pl.col("user_id").n_unique().alias("n_users_ggr"),
        pl.col("user_ggr_clip").mean().alias("avg_user_ggr"),
        pl.col("user_ggr").median().alias("med_user_ggr"),
        pl.col("user_ggr_clip").std().alias("std_user_ggr"),
    )


def corr_rtp_activity(ud_raw: pl.DataFrame) -> pl.DataFrame:
    """Analysis 4: per-user-day correlation of user_rtp with betting activity.

    user_total_bet / user_num_bets are heavy-tailed (log-normal-ish), so the
    Pearson column uses log10(x) on both activity and (rtp + 0.01); Spearman
    (rank) is transform-invariant and reported as the primary statistic.
    """
    rows = []
    base = ud_raw.filter(pl.col("user_rtp").is_not_null())
    for rng, grp in base.partition_by("range", as_dict=True).items():
        for col in ("user_total_bet", "user_num_bets"):
            g = grp.select(
                pl.col("user_rtp").alias("r"),
                pl.col(col).alias("x"),
                (pl.col("user_rtp") + 0.01).log(10).alias("lr"),
                pl.col(col).log(10).alias("lx"),
                pl.col("user_rtp").rank().alias("rr"),
                pl.col(col).rank().alias("rx"),
            )
            rows.append(
                {
                    "range": rng[0],
                    "metric": col,
                    "n": len(g),
                    "spearman": g.select(pl.corr("rr", "rx")).item(),
                    "pearson_raw": g.select(pl.corr("r", "x")).item(),
                    "pearson_loglog": g.select(pl.corr("lr", "lx")).item(),
                }
            )
    out = pl.DataFrame(rows).sort(["range", "metric"])
    out.write_csv(OUTPUT_DIR / "corr_rtp_activity.csv")
    ud_raw.select("range", "user_rtp", "user_total_bet", "user_num_bets").write_csv(OUTPUT_DIR / "user_day_ai.csv")
    logger.info("wrote %s", OUTPUT_DIR / "corr_rtp_activity.csv")
    return out


def analyze() -> None:
    blocks = _load(BLOCKS_DIR, "blocks")
    presence = _load(PRESENCE_DIR, "presence")
    # Block cleaning applies ONLY to the permutation ranking; the range
    # summary, daily RTP, and AI-vs-fixed-table comparison use all bets.
    seq = _load(SEQ_DIR, "user-day-seq")
    ud_raw_all = add_retention(build_user_day(blocks), presence).join(seq, on=["user_id", "bet_date"], how="left")
    thresholds = clip_thresholds(ud_raw_all)
    logger.info("p%d clip thresholds:\n%s", int(CLIP_PCT * 100), thresholds.sort("range"))
    ud_raw_all = with_clipped(ud_raw_all, thresholds)
    ud_raw = ud_raw_all.filter(pl.col("ab_group") == "AI")
    ud = with_clipped(
        add_retention(build_user_day(blocks, MIN_BLOCK_BETS), presence)
        .filter(pl.col("ab_group") == "AI")
        .join(seq, on=["user_id", "bet_date"], how="left"),
        thresholds,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gthr = ggr_thresholds(ud_raw_all)
    logger.info("p99 user-GGR clip thresholds:\n%s", gthr.sort("range"))

    summary = ud_raw.group_by("range").agg(AGGS).join(ggr_stats(ud_raw, [], gthr), on="range", how="left").sort("range")
    summary.write_csv(OUTPUT_DIR / "summary_range.csv")
    corr_rtp_activity(ud_raw)

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
    cmp_frame = pl.concat(
        [
            single.with_columns(pl.col("mt_winner").alias("mathtable")),
            ud_raw.with_columns(pl.lit("AI(全部排列)").alias("mathtable")),
        ],
        how="diagonal",
    )
    comparison = (
        pl.concat(
            [
                single.group_by("range", pl.col("mt_winner").alias("mathtable")).agg(AGGS),
                ud_raw.group_by("range").agg(AGGS).with_columns(pl.lit("AI(全部排列)").alias("mathtable")),
            ],
            how="diagonal",
        )
        .join(ggr_stats(cmp_frame, ["mathtable"], gthr), on=["range", "mathtable"], how="left")
        .sort(["range", "user_days"], descending=[False, True])
    )
    comparison.write_csv(OUTPUT_DIR / "summary_range_by_table.csv")

    # Same comparison split by daily total-bet tier (bucket on the raw day
    # total), with Wilson CIs on retention for the summary barplot.
    single_b = single.with_columns(bucket_expr())
    ai_b = ud_raw.with_columns(bucket_expr())
    bkt_frame = pl.concat(
        [
            single_b.with_columns(pl.col("mt_winner").alias("mathtable")),
            ai_b.with_columns(pl.lit("AI(全部排列)").alias("mathtable")),
        ],
        how="diagonal",
    )
    comparison_bucket = (
        with_retention_ci(
            pl.concat(
                [
                    single_b.group_by("range", pl.col("mt_winner").alias("mathtable"), "bet_bucket").agg(AGGS),
                    ai_b.group_by("range", "bet_bucket")
                    .agg(AGGS)
                    .with_columns(pl.lit("AI(全部排列)").alias("mathtable")),
                ],
                how="diagonal",
            )
        )
        .join(
            ggr_stats(bkt_frame, ["mathtable", "bet_bucket"], gthr), on=["range", "mathtable", "bet_bucket"], how="left"
        )
        .sort(["range", "mathtable", "bet_bucket"])
    )
    comparison_bucket.write_csv(OUTPUT_DIR / "summary_range_by_table_bucket.csv")

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

    perm = with_retention_ci(
        ud.group_by("range", "permutation")
        .agg(AGGS)
        .join(ggr_stats(ud, ["permutation"], gthr), on=["range", "permutation"], how="left")
    ).sort(["range", "user_days"], descending=[False, True])
    perm.write_csv(OUTPUT_DIR / "summary_permutation.csv")

    # Rank WITHIN permutation length (elements in the label, repeats counted:
    # A-B-A is length 3): engaged users play more bets and therefore pass
    # through more tables, so comparing permutations of different lengths
    # reflects engagement (reverse causality), not table effects.
    perm = perm.with_columns((pl.col("permutation").str.count_matches("|", literal=True) + 1).alias("perm_len"))
    ranked = perm.filter((pl.col("user_days") >= MIN_PERMUTATION_USER_DAYS) & pl.col("perm_len").is_between(1, 4))
    tops = []
    for metric in ("retention_d1", "avg_user_total_bet", "avg_user_ggr", "avg_max_streak"):
        for rng in RANGES:
            for length in (1, 2, 3, 4):
                top = (
                    ranked.filter((pl.col("range") == rng) & (pl.col("perm_len") == length))
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
        "summary_range_by_table_bucket",
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
