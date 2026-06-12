import argparse
import os
from textwrap import dedent
from typing import Optional

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    ETL_EXCLUDED_OP_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend

# Query is slow on full history; backfill of earlier data will be added later.
DEFAULT_DATE_START = "2026-05-01"

GAME_ID = "FM01"
CURRENCY_TYPE = "CNY"
SESSION_GAP_SECONDS = 1800  # 30 min gap splits bet sessions
SCAN_WINDOW_HOURS = 12  # max hours after first play scanned for the first session


def generate_query(start_date: str = DEFAULT_DATE_START, end_date: Optional[str] = None):
    """Bullet-level detail of each new user's first play session (FTUE).

    ETL job: fish_hunter FTUE first-session stats. One row per bullet in the
    first bet session of each new FM01 (CNY) user, joined with the user's
    first-IP info from ``stg_first_ip_per_user``. Output dataset
    ``output_fish_hunter/ftue_first_session`` on S3, hive-partitioned by
    year/month of ``first_play_timestamp``.

    Behavior: cohort = users whose first play is after ``start_date``
    (internal/test op codes excluded via dim_user_latest). Bullets within
    SCAN_WINDOW_HOURS of first play are sessionized with a
    SESSION_GAP_SECONDS gap rule; only session 0 (the first session) is kept.
    A first session running past the scan window is truncated.

    Inputs: ``end_date`` caps the cohort (first play <= end_date) so a slow
    backfill can run in short chunks that finish before the SSH tunnel's idle
    timeout drops the connection. The bullet scan extends 1 day past it so
    boundary users' sessions are not cut off; the next chunk's 3-day watermark
    lookback re-fetches boundary users and dedup keeps the complete session.
    """
    cohort_end_filter = f"AND t.first_play_timestamp <= '{end_date}'" if end_date else ""
    bullet_end_filter = f"AND created_at < DATEADD(DAY, 1, CAST('{end_date}' AS TIMESTAMP))" if end_date else ""

    query = dedent(
        f"""
        WITH bullets_in_range AS (
            -- date-filter the fact table first, before any join or window function;
            -- the literal lets Redshift prune blocks at scan time
            SELECT
                user_id,
                game_id,
                bullet_id,
                event_timestamp,
                bet,
                bullet_level,
                payout,
                fish_value,
                multiplier,
                room_id,
                scene_id,
                prev_balance,
                device_type,
                ip,
                strategy_name
            FROM public.bullet
            WHERE
                created_at >= '{start_date}'
                {bullet_end_filter}
                AND game_id = '{GAME_ID}'
                AND currency_type = '{CURRENCY_TYPE}'
        ),

        cohort AS (
            -- restrict to the {GAME_ID} cohort first so every downstream step stays small;
            -- exclude internal/test operators via dim_user_latest.op_code
            SELECT
                t.user_id,
                t.game_id,
                t.first_play_timestamp
            FROM public.dim_user_first_shot AS t
            INNER JOIN public.dim_user_latest AS ul
                ON t.user_id = ul.user_id
            WHERE
                t.game_id = '{GAME_ID}'
                AND t.first_play_timestamp > '{start_date}'
                {cohort_end_filter}
                AND ul.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
        ),

        first_ip AS (
            -- one row per user (confirmed); pre-filter to the cohort so the final
            -- join doesn't touch the full table
            SELECT
                s.user_id,
                s.account_created,
                s.first_bullet_time,
                s.first_ip,
                s.gap_seconds
            FROM public.stg_first_ip_per_user AS s
            INNER JOIN cohort AS c ON s.user_id = c.user_id
        ),

        session_bullets AS (
            -- bullets within the scan window after first play (hard cap so the bullet
            -- scan stays bounded; a first session running past the cap is truncated),
            -- with the gap to each user's previous bullet
            SELECT
                t.user_id,
                t.game_id,
                t.first_play_timestamp,
                t3.bullet_id,
                t3.event_timestamp,
                t3.bet,
                t3.bullet_level,
                t3.payout,
                t3.fish_value,
                t3.multiplier,
                t3.room_id,
                t3.scene_id,
                t3.prev_balance,
                t3.device_type,
                t3.ip,
                t3.strategy_name,
                DATEDIFF(
                    SECOND,
                    LAG(t3.event_timestamp) OVER (
                        PARTITION BY t3.user_id
                        ORDER BY t3.event_timestamp, t3.bullet_id
                    ),
                    t3.event_timestamp
                ) AS gap_seconds_prev
            FROM cohort AS t
            INNER JOIN bullets_in_range AS t3
                ON
                    t.user_id = t3.user_id
                    AND t.first_play_timestamp <= t3.event_timestamp
                    AND t3.event_timestamp < DATEADD(HOUR, {SCAN_WINDOW_HOURS}, t.first_play_timestamp)
        ),

        sessionized AS (
            -- session breaks when the gap to the previous bullet reaches the threshold;
            -- running count of breaks = session number, first session is 0
            SELECT
                user_id,
                game_id,
                first_play_timestamp,
                bullet_id,
                event_timestamp,
                bet,
                bullet_level,
                payout,
                fish_value,
                multiplier,
                room_id,
                scene_id,
                prev_balance,
                device_type,
                ip,
                strategy_name,
                gap_seconds_prev,
                SUM(CASE WHEN gap_seconds_prev >= {SESSION_GAP_SECONDS} THEN 1 ELSE 0 END) OVER (
                    PARTITION BY user_id
                    ORDER BY event_timestamp, bullet_id
                    ROWS UNBOUNDED PRECEDING
                ) AS session_number
            FROM session_bullets
        )

        SELECT
            sb.user_id,
            sb.game_id,
            sb.first_play_timestamp,
            t2.account_created,
            t2.first_bullet_time,
            t2.first_ip,
            t2.gap_seconds,
            sb.bullet_id,
            sb.event_timestamp,
            sb.bet,
            sb.bullet_level,
            sb.payout,
            sb.fish_value,
            sb.multiplier,
            sb.room_id,
            sb.scene_id,
            sb.prev_balance,
            sb.device_type,
            sb.ip,
            sb.strategy_name
        FROM sessionized AS sb
        INNER JOIN first_ip AS t2
            ON sb.user_id = t2.user_id
        WHERE sb.session_number = 0
        ;

        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL FTUE first-session bullets for FM01")
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=DEFAULT_BASTION_IP,
        help=f"Bastion IP address for Redshift tunnel (default: {DEFAULT_BASTION_IP})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing S3/local output (full reload from default start date)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Cap the cohort at this date (YYYY-MM-DD) to backfill in short chunks; omit for normal runs",
    )
    args = parser.parse_args()

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="transform-agfish-game",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=args.bastion_ip,
        )
    )

    scheduler = ETLScheduler(
        redshift_loader,
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_fish_hunter",
        lookback_days=3,
        overwrite=args.overwrite,
    )
    # Initial/full loads start at the FTUE backfill boundary, not the scheduler's
    # 2025-01-01 default; earlier history will be backfilled separately later.
    scheduler.default_start_date = DEFAULT_DATE_START

    scheduler.run_incremental_job(
        job_name="ftue_first_session",
        query_func=lambda start_date: generate_query(start_date, end_date=args.end_date),
        key_cols=["user_id", "bullet_id"],
        date_col="first_play_timestamp",
        partition_level="month",
        lookback=3,
    )

    redshift_loader.close()
