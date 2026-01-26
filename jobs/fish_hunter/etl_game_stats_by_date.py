import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, ETLScheduler, RedshiftBackend

DATE_START = "2025-10-20"
DATE_END = "2027-12-1"

return_user_days = 30
retention_days = 3
DATE_START_HOUR = 6


def generate_query(stats_agg_col: str, start_date: str, end_date: str = DATE_END):

    query = dedent(
        f"""
        -- 1. FETCH RAW DATA (Keep strictly RAW columns to enable Index Scans)
        WITH base_data AS (
            SELECT
                b.user_id,
                b.room_id,
                b.strategy_name, -- Keep Raw
                b.partition_ab[0] as partition_val, -- Extract partition once here
                b.payout,
                b.bet,
                b.fish_value,
                b.killed,
                b.profit,
                CAST(DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS DATE) AS activity_date,
                CAST(DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS DATE) AS activity_week,
                CAST(DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS DATE) AS activity_month
            FROM public.bullet b
            WHERE
                b.currency_type = 'CNY'
                AND b.op_code not in ('B26', 'TST','TSB','TSO')

                -- ---------------------------------------------------------
                -- FAST FILTERING: Transform the INPUTS, not the COLUMN
                -- ---------------------------------------------------------
                
                -- 1. Reverse the date math for START
                -- Logic: We want events where (EventTime + UserDays) >= Start
                -- So: EventTime >= Start - UserDays
                AND b.created_at >= CONVERT_TIMEZONE('Asia/Shanghai', 'UTC', 
                       DATEADD(day, -{return_user_days}, CAST('{start_date}' AS TIMESTAMP)))

                -- 2. Reverse the date math for END
                -- Logic: We want events where (EventTime - RetentionDays) < End
                -- So: EventTime < End + RetentionDays
                AND b.created_at < CONVERT_TIMEZONE('Asia/Shanghai', 'UTC', 
                       DATEADD(day, {retention_days}, CAST('{end_date}' AS TIMESTAMP)))
        ),

        -- 2. DETERMINE USER DAILY GROUP (Logic applied inside SUM)
        user_daily_group AS (
            SELECT
                user_id,
                activity_date,
                activity_week,
                activity_month,
                CASE
                    -- Performance fix: Check Raw Strategy + Partition Value here
                    WHEN SUM(CASE WHEN strategy_name = 'BOOST_POOL' AND partition_val = 'c2mta7-ls8vqx-HyJf5k-event' THEN 1 ELSE 0 END) > 0 THEN 'BOOST_POOL_2'
                    WHEN SUM(CASE WHEN strategy_name = 'BOOST_POOL' THEN 1 ELSE 0 END) > 0 THEN 'BOOST_POOL'
                    WHEN SUM(CASE WHEN strategy_name = 'DYNAMIC_RTP' THEN 1 ELSE 0 END) > 0 THEN 'DYNAMIC_RTP'
                    ELSE 'DEFAULT_FALLBACK'
                END AS daily_group,
                LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS bj_date_last_bet
            FROM base_data
            GROUP BY user_id, activity_date, activity_week, activity_month
        ),

        -- 3. CALCULATE RETENTION (Pure Date Math)
        retention_stats AS (
            SELECT
                t1.daily_group,
                t1.{stats_agg_col},
                COUNT(DISTINCT t1.user_id) AS num_users_day0,
                COUNT(DISTINCT t2.user_id) AS num_users_day1,
                COUNT(DISTINCT t3.user_id) AS num_users_day3,
                COUNT(DISTINCT t4.user_id) AS num_users_day7
                -- COUNT(DISTINCT t5.user_id) AS num_users_week1,
                -- COUNT(DISTINCT t6.user_id) AS num_users_month1
            FROM user_daily_group t1
            LEFT JOIN user_daily_group t2
                ON t1.user_id = t2.user_id AND t2.activity_date = DATEADD(day, -1, t1.activity_date)
            LEFT JOIN user_daily_group t3
                ON t1.user_id = t3.user_id AND t3.activity_date = DATEADD(day, -3, t1.activity_date)
            LEFT JOIN user_daily_group t4
                ON t1.user_id = t4.user_id AND t4.activity_date = DATEADD(day, -7, t1.activity_date)
            -- LEFT JOIN user_daily_group t5
            --    ON t1.user_id = t5.user_id AND t5.activity_week = DATEADD(week, -1, t1.activity_week)
            -- LEFT JOIN user_daily_group t6
            --    ON t1.user_id = t6.user_id AND t6.activity_month = DATEADD(month, -1, t1.activity_month)
            GROUP BY t1.daily_group, t1.{stats_agg_col}
        ),

        -- 4. AGGREGATE STATS BY ASSIGNED DAILY GROUP
        stats_by_date AS (
            SELECT
                u.daily_group,
                b.{stats_agg_col},
                COUNT(DISTINCT b.user_id) AS num_users,
                COUNT(DISTINCT
                    CASE
                        WHEN
                            DATEDIFF('day', u.bj_date_last_bet, u.activity_date) > {return_user_days}
                        THEN u.user_id
                    END
                ) AS num_return_users,
                COUNT(DISTINCT CASE WHEN b.killed = 1 THEN b.user_id END) AS num_users_killed_fish,
                ROUND(CAST(SUM(b.payout) AS FLOAT) / NULLIF(SUM(b.bet), 0), 3) AS group_rtp
            FROM base_data b
            JOIN user_daily_group u ON b.user_id = u.user_id AND b.activity_date = u.activity_date
            GROUP BY u.daily_group, b.{stats_agg_col}
        ),

        stats_by_user_date AS (
            SELECT
                b.user_id,
                u.daily_group,
                b.{stats_agg_col},
                COUNT(DISTINCT b.user_id || '-' || b.room_id) AS num_rooms,
                COUNT(b.user_id) AS num_bullets,
                SUM(b.bet) AS total_bet,
                SUM(b.killed) AS num_killed_bullets,
                ROUND(CAST(SUM(b.payout) AS FLOAT) / NULLIF(SUM(b.bet), 0), 3) AS user_rtp,
                AVG(b.fish_value) AS avg_fish_value,
                AVG(CASE WHEN b.fish_value > 19 AND b.fish_value < 201 THEN b.fish_value END) AS avg_fish_value_20_200,
                AVG(CASE WHEN b.killed = 1 THEN b.fish_value END) AS avg_killed_fish_value,
                AVG(CASE WHEN b.fish_value > 19 AND b.fish_value < 201 AND b.killed = 1 THEN b.fish_value END) AS avg_killed_fish_value_20_200,
                AVG(b.profit) AS bullet_avg_profit,
                AVG(CASE WHEN b.killed = 1 THEN b.profit END) AS bullet_kill_avg_profit,
                SUM(b.profit) AS total_profit
            FROM base_data b
            JOIN user_daily_group u ON b.user_id = u.user_id AND b.activity_date = u.activity_date
            GROUP BY b.user_id, u.daily_group, b.{stats_agg_col}
        )

        -- 6. FINAL JOIN & FORMATTING (Fully Restored)
        SELECT
            t1.user_id,
            t1.daily_group,
            t1.{stats_agg_col} AS activity_date,
            t1.num_rooms,
            t1.num_bullets,
            t1.total_bet,
            t1.num_killed_bullets,
            t1.user_rtp,
            t1.avg_fish_value,
            t1.avg_fish_value_20_200,
            t1.avg_killed_fish_value,
            t1.avg_killed_fish_value_20_200,
            t1.bullet_avg_profit,
            t1.bullet_kill_avg_profit,
            t1.total_profit,

            t2.num_users_day0,
            t2.num_users_day1,
            t2.num_users_day3,
            t2.num_users_day7,
            -- t2.num_users_week1,
            -- t2.num_users_month1,

            t3.num_users,
            t3.num_return_users,
            t3.num_users_killed_fish,
            t3.group_rtp,

            ROUND(CAST(t2.num_users_day1 AS FLOAT) / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day1,
            ROUND(CAST(t2.num_users_day3 AS FLOAT) / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day3,
            ROUND(CAST(t2.num_users_day7 AS FLOAT) / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day7,
            ROUND(CAST(t1.num_killed_bullets AS FLOAT) / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio
        FROM stats_by_user_date t1
        INNER JOIN retention_stats t2 
            ON t1.{stats_agg_col} = t2.{stats_agg_col} 
            AND t1.daily_group = t2.daily_group
        INNER JOIN stats_by_date t3 
            ON t1.{stats_agg_col} = t3.{stats_agg_col} 
            AND t1.daily_group = t3.daily_group
        ORDER BY t1.{stats_agg_col}, t1.daily_group
        ;

        """
    )

    return query


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="production-redshift-cluster.cwiqzcm13zcn.ap-southeast-1.redshift.amazonaws.com",
            database="transform-agfish-game",
            user="anaylsis_user",
            password="oZ4ztMx0yEXPLbJL733L",
            port=5439,
        )
    )

    # Initialize Scheduler with a default 3-day lookback
    scheduler = ETLScheduler(redshift_loader, f"{LOCAL_ROOT}/jobs/output_fish_hunter", lookback_days=3)

    scheduler.run_incremental_job(
        job_name="daily_stats",
        query_func=lambda start_date: generate_query("activity_date", start_date),
        key_cols=["activity_date", "user_id", "daily_group"],
        date_col="activity_date",
        partition_level="none",
    )

    # Overrides to 7 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="weekly_stats",
        query_func=lambda start_date: generate_query("activity_week", start_date),
        key_cols=["activity_date", "user_id", "daily_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=7,
    )

    # Overrides to 31 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="monthly_stats",
        query_func=lambda start_date: generate_query("activity_month", start_date),
        key_cols=["activity_date", "user_id", "daily_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=31,
    )

    redshift_loader.close()
