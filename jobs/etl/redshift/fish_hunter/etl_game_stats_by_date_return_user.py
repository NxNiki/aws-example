import argparse
import os
from textwrap import dedent

from bituslabs_ds.config import (
    DATE_START_HOUR,
    DEFAULT_BASTION_IP,
    DEFAULT_ETL_OUTPUT,
    ETL_CURRENCY_CODES,
    ETL_EXCLUDED_OP_CODES,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    TIMEZONE_SHANGHAI,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend

DATE_START = "2025-12-31"

RETURN_USER_DAYS = 30
RETENTION_DAYS = 3
STREAK_SESSION_THRESH = 600
STREAK_KILL_THRESH = 3  # nearly 10% of all killing intervals.


def generate_query(start_date: str):

    query = dedent(
        f"""
        -- Check game stats between retention users and non-retention users.

        -- 1. FETCH RAW DATA (Keep strictly RAW columns to enable Index Scans)
        WITH base_data AS (
            SELECT
                b.user_id,
                b.room_id,
                b.event_timestamp AS bet_time,
                b.payout,
                b.bet,
                b.fish_value,
                CASE
                    WHEN b.fish_value <= 10 THEN 'low'
                    WHEN b.fish_value <= 130 THEN 'medium'
                    WHEN b.fish_value <= 200 THEN 'high'
                    ELSE 'ultra'
                END as fish_type,
                b.killed,
                b.profit,
                LAG(b.event_timestamp) OVER (PARTITION BY b.user_id ORDER BY b.event_timestamp) AS prev_bet_time,
                DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', b.created_at))) AS activity_date
            FROM public.bullet b
            WHERE
                b.currency_type IN {ETL_CURRENCY_CODES}
                AND b.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
                -- ---------------------------------------------------------
                -- FAST FILTERING: Transform the INPUTS, not the COLUMN
                -- ---------------------------------------------------------
                -- Logic: We want events where (EventTime + UserDays) >= Start
                -- So: EventTime >= Start - UserDays
                AND b.created_at >= CONVERT_TIMEZONE('{TIMEZONE_SHANGHAI}', 'UTC', CAST('{start_date}' AS TIMESTAMP))
                AND b.strategy_name = 'DEFAULT_FALLBACK'
        ),

        -- GET KILL STREAK LENGTH:

        base_kills AS (
            SELECT 
                user_id, 
                activity_date, 
                fish_type, 
                bet_time
            FROM base_data 
            WHERE killed = 1
        ),

        calculate_islands AS (
            SELECT 
                user_id,
                activity_date,
                fish_type,
                bet_time,
                -- GLOBAL: ignores fish_type. Checks if ANY fish was killed recently.
                CASE 
                    WHEN DATEDIFF(SECOND, LAG(bet_time) OVER(PARTITION BY user_id ORDER BY bet_time), bet_time) > {STREAK_KILL_THRESH} 
                        OR LAG(bet_time) OVER(PARTITION BY user_id ORDER BY bet_time) IS NULL 
                    THEN 1 ELSE 0 
                END AS is_new_global_streak,
                
                -- TYPE-SPECIFIC: isolated by fish_type. Only checks previous kill of SAME type.
                CASE 
                    WHEN DATEDIFF(SECOND, LAG(bet_time) OVER(PARTITION BY user_id, fish_type ORDER BY bet_time), bet_time) > {STREAK_KILL_THRESH} 
                        OR LAG(bet_time) OVER(PARTITION BY user_id, fish_type ORDER BY bet_time) IS NULL 
                    THEN 1 ELSE 0 
                END AS is_new_type_streak
            FROM base_kills
        ),

        streak_ids AS (
            SELECT
                user_id,
                activity_date,
                fish_type,
                bet_time,
                SUM(is_new_global_streak) OVER(PARTITION BY user_id ORDER BY bet_time ROWS UNBOUNDED PRECEDING) AS global_streak_id,
                SUM(is_new_type_streak) OVER(PARTITION BY user_id, fish_type ORDER BY bet_time ROWS UNBOUNDED PRECEDING) AS type_streak_id
            FROM calculate_islands
        ),

        streak_lengths AS (
            SELECT 
                user_id, 
                activity_date,
                fish_type,
                -- Calculate the length of the specific streak instance this row belongs to
                COUNT(*) OVER(PARTITION BY user_id, global_streak_id) AS global_streak_len,
                COUNT(*) OVER(PARTITION BY user_id, fish_type, type_streak_id) AS type_streak_len
            FROM streak_ids
        ),

        max_kill_streak_length AS (
        SELECT 
            user_id,
            activity_date,
            MAX(global_streak_len) AS max_kill_streak,
            MAX(CASE WHEN fish_type = 'low' THEN type_streak_len END) AS max_kill_streak_low,
            MAX(CASE WHEN fish_type = 'medium' THEN type_streak_len END) AS max_kill_streak_medium,
            MAX(CASE WHEN fish_type = 'high' THEN type_streak_len END) AS max_kill_streak_high,
            MAX(CASE WHEN fish_type = 'ultra' THEN type_streak_len END) AS max_kill_streak_ultra,

            AVG(global_streak_len) AS avg_kill_streak,
            AVG(CASE WHEN fish_type = 'low' THEN type_streak_len END) AS avg_kill_streak_low,
            AVG(CASE WHEN fish_type = 'medium' THEN type_streak_len END) AS avg_kill_streak_medium,
            AVG(CASE WHEN fish_type = 'high' THEN type_streak_len END) AS avg_kill_streak_high,
            AVG(CASE WHEN fish_type = 'ultra' THEN type_streak_len END) AS avg_kill_streak_ultra
        FROM streak_lengths
        GROUP BY user_id, activity_date
        ),
        
        -- 2. DETERMINE  RETURN/NON-RETURN USERS
        user_activity AS (
            SELECT DISTINCT
                user_id,
                activity_date
            FROM base_data
        ),

        user_next_bet_date AS (
            SELECT
                user_id,
                activity_date,
                LEAD(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS next_bet_date
            FROM user_activity
        ),

        retention_users AS (
            SELECT
                t1.activity_date,
                t1.user_id,
                CASE 
                    WHEN next_bet_date <= DATEADD(day, {RETENTION_DAYS}, activity_date) 
                    THEN 'return' 
                    ELSE 'non-return' 
                END AS return_user
            FROM user_next_bet_date AS t1
        ),

        -- 3. AGGREGATE STATS BY ASSIGNED DAILY GROUP
        daily_stats AS (
            SELECT
                b.activity_date,
                r.return_user,
                COUNT(DISTINCT b.user_id) AS num_users
            FROM base_data AS b
            INNER JOIN retention_users AS r ON b.user_id = r.user_id AND b.activity_date = r.activity_date
            GROUP BY b.activity_date, r.return_user
        ),

        user_daily_stats AS (
            SELECT
                t1.user_id,
                t1.activity_date,
                t2.return_user,
                COUNT(DISTINCT t1.room_id) AS num_rooms,

                -- Bullets info:
                COUNT(t1.user_id) AS num_bullets,
                SUM(CASE WHEN t1.fish_type = 'low' THEN 1 END) AS num_hits_fish_low,
                SUM(CASE WHEN t1.fish_type = 'medium' THEN 1 END) AS num_hits_fish_medium,
                SUM(CASE WHEN t1.fish_type = 'high' THEN 1 END) AS num_hits_fish_high,
                SUM(CASE WHEN t1.fish_type = 'ultra' THEN 1 END) AS num_hits_fish_ultra,

                -- killed Fish info:
                SUM(t1.killed) AS num_killed_bullets,
                SUM(CASE WHEN t1.fish_type = 'low' THEN t1.killed END) AS num_killed_fish_low,
                SUM(CASE WHEN t1.fish_type = 'medium' THEN t1.killed END) AS num_killed_fish_medium,
                SUM(CASE WHEN t1.fish_type = 'high' THEN t1.killed END) AS num_killed_fish_high,
                SUM(CASE WHEN t1.fish_type = 'ultra' THEN t1.killed END) AS num_killed_fish_ultra,
                
                SUM(t1.bet) AS daily_total_bet,
                SUM(t1.profit) AS total_profit,
                MAX(t1.profit) AS max_profit,
                ROUND(CAST(SUM(t1.payout) AS FLOAT) / NULLIF(SUM(t1.bet), 0), 3) AS rtp,
                STDDEV(t1.profit) / NULLIF(ABS(AVG(t1.profit)), 0) AS profit_coef_var,
                MAX(CASE WHEN t1.killed >= 1 THEN 1 ELSE 0 END) AS user_killed_fish

            FROM base_data AS t1
            INNER JOIN retention_users AS t2 ON t1.user_id = t2.user_id AND t1.activity_date = t2.activity_date
            GROUP BY t1.activity_date, t1.user_id, t2.return_user
            HAVING MAX(t1.killed) > 0
        ),

        -- 4. GET BET SESSION STATS:
        user_session_id AS (
            SELECT
                t.activity_date,
                t.user_id,
                t.bet_time,
                t.killed,
                t.fish_type,
                SUM(CASE 
                        WHEN DATEDIFF(SECOND, t.prev_bet_time, t.bet_time) < {STREAK_SESSION_THRESH} THEN 0 
                        ELSE 1 
                    END) OVER (
                        PARTITION BY t.activity_date, t.user_id 
                        ORDER BY t.bet_time 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS session_id,
                ROW_NUMBER() OVER (
                        PARTITION BY t.activity_date, t.user_id 
                        ORDER BY t.bet_time 
                    ) AS bet_index
                
            FROM base_data t
        ),

        user_session_length AS (
            SELECT
                t.activity_date,
                t.user_id,
                t.session_id,

                DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 THEN t.bet_time END)) AS seconds_to_kill_fish,
                DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'low' THEN t.bet_time END)) AS seconds_to_kill_fish_low,
                DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'medium' THEN t.bet_time END)) AS seconds_to_kill_fish_medium,
                DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'high' THEN t.bet_time END)) AS seconds_to_kill_fish_high,
                DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'ultra' THEN t.bet_time END)) AS seconds_to_kill_fish_ultra,

                MIN(CASE WHEN t.killed = 1 THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish,
                MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'low' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_low,
                MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'medium' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_medium,
                MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'high' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_high,
                MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'ultra' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_ultra,

                COUNT(t.user_id) AS session_length
            FROM user_session_id t
            GROUP BY t.user_id, t.activity_date, t.session_id
        ),

        user_session_stats AS (
            SELECT
                t.activity_date,
                t.user_id,
                COUNT(DISTINCT t.session_id) AS num_streak_sessions,
                AVG(CAST(t.session_length AS FLOAT)) AS avg_streak_length,
                MAX(t.session_length) AS max_streak_length,
                MIN(t.session_length) AS min_streak_length,

                AVG(seconds_to_kill_fish) AS seconds_to_kill_fish,
                AVG(seconds_to_kill_fish_low) AS seconds_to_kill_fish_low,
                AVG(seconds_to_kill_fish_medium) AS seconds_to_kill_fish_medium,
                AVG(seconds_to_kill_fish_high) AS seconds_to_kill_fish_high,
                AVG(seconds_to_kill_fish_ultra) AS seconds_to_kill_fish_ultra,

                AVG(bets_to_kill_fish) AS bets_to_kill_fish,
                AVG(bets_to_kill_fish_low) AS bets_to_kill_fish_low,
                AVG(bets_to_kill_fish_medium) AS bets_to_kill_fish_medium,
                AVG(bets_to_kill_fish_high) AS bets_to_kill_fish_high,
                AVG(bets_to_kill_fish_ultra) AS bets_to_kill_fish_ultra

            FROM user_session_length t
            GROUP BY t.user_id, t.activity_date
        )

        -- 6. FINAL JOIN & FORMATTING (Fully Restored)
        SELECT
            t1.activity_date,
            t1.user_id,
            t1.return_user,
            t2.num_users,
            t1.num_rooms,
            t1.daily_total_bet,

            t1.num_killed_bullets,
            t1.num_killed_fish_low,
            t1.num_killed_fish_medium,
            t1.num_killed_fish_high,
            t1.num_killed_fish_ultra,
            t1.num_killed_fish_low * 1.0 / t1.num_killed_bullets AS killed_fish_low_ratio,
            t1.num_killed_fish_medium * 1.0 / t1.num_killed_bullets AS killed_fish_medium_ratio,
            t1.num_killed_fish_high * 1.0 / t1.num_killed_bullets AS killed_fish_high_ratio,
            t1.num_killed_fish_ultra * 1.0 / t1.num_killed_bullets AS killed_fish_ultra_ratio,

            t1.num_bullets,
            t1.num_hits_fish_low,
            t1.num_hits_fish_medium,
            t1.num_hits_fish_high,
            t1.num_hits_fish_ultra,
            t1.num_hits_fish_low * 1.0 / t1.num_bullets AS hits_fish_low_ratio,
            t1.num_hits_fish_medium * 1.0 / t1.num_bullets AS hits_fish_medium_ratio,
            t1.num_hits_fish_high * 1.0 / t1.num_bullets AS hits_fish_high_ratio,
            t1.num_hits_fish_ultra * 1.0 / t1.num_bullets AS hits_fish_ultra_ratio,
            
            t1.total_profit,
            t1.max_profit,
            t1.rtp,
            t1.profit_coef_var,
            
            SUM(CASE WHEN t1.total_profit > 1 THEN 1 END) OVER (PARTITION BY t1.activity_date, t1.return_user) AS num_users_pos_profit,
            ROUND(CAST(t1.num_killed_bullets AS FLOAT) / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio,

            SUM(t1.user_killed_fish) OVER (PARTITION BY t1.activity_date, t1.return_user) AS num_users_killed_fish,

            t3.num_streak_sessions,
            t3.avg_streak_length,
            t3.max_streak_length,
            t3.min_streak_length,

            t3.seconds_to_kill_fish,
            t3.seconds_to_kill_fish_low,
            t3.seconds_to_kill_fish_medium,
            t3.seconds_to_kill_fish_high,
            t3.seconds_to_kill_fish_ultra,

            t3.bets_to_kill_fish,
            t3.bets_to_kill_fish_low,
            t3.bets_to_kill_fish_medium,
            t3.bets_to_kill_fish_high,
            t3.bets_to_kill_fish_ultra,

            t4.max_kill_streak,
            t4.max_kill_streak_low,
            t4.max_kill_streak_medium,
            t4.max_kill_streak_high,
            t4.max_kill_streak_ultra,

            t4.avg_kill_streak,
            t4.avg_kill_streak_low,
            t4.avg_kill_streak_medium,
            t4.avg_kill_streak_high,
            t4.avg_kill_streak_ultra

        FROM user_daily_stats AS t1
        INNER JOIN daily_stats AS t2 ON t1.activity_date = t2.activity_date AND t1.return_user = t2.return_user
        INNER JOIN user_session_stats AS t3 ON t1.activity_date = t3.activity_date AND t1.user_id = t3.user_id
        INNER JOIN max_kill_streak_length AS t4 ON t1.activity_date = t4.activity_date AND t1.user_id = t4.user_id
        ORDER BY t1.activity_date, t1.user_id, t1.return_user
        ;

        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    parser = argparse.ArgumentParser(description="ETL Game Stats Daily by User Group")
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

    # Initialize Scheduler with a default 3-day lookback
    scheduler = ETLScheduler(
        redshift_loader,
        f"{DEFAULT_ETL_OUTPUT}/jobs/output_fish_hunter",
        lookback_days=3,
        overwrite=args.overwrite,
    )

    scheduler.run_incremental_job(
        job_name="daily_stats_return_user",
        query_func=lambda start_date: generate_query(start_date),
        key_cols=["activity_date", "user_id"],
        date_col="activity_date",
        partition_level="none",
        lookback=3,
    )

    redshift_loader.close()
