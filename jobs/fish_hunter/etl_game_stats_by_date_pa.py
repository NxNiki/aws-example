import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.etl import AthenaBackend, DataLoader, ETLScheduler

DATE_END = "2027-12-1"

return_user_days = 30
retention_days = 3


def generate_query(stats_agg_col: str, start_date: str, end_date: str = DATE_END):
    query = dedent(
        f"""
        -- 1. FETCH RAW DATA (Keep Date as Date Object)
        WITH base_data AS (
            SELECT
                b.loginname AS user_id,
                b.roomid AS room_id,
                b.account + b.cus_account AS payout,
                b.account AS bet,
                b.fishcost / b.betx AS fish_value,
                b.hunted AS killed,
                b.cus_account AS profit,
                -- Convert to Beijing time and truncate to date object 
                CAST(DATE_TRUNC('day', b.billtime AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai' - INTERVAL '6' HOUR) AS DATE) AS activity_date,
                CAST(DATE_TRUNC('week', b.billtime AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai' - INTERVAL '6' HOUR) AS DATE) AS activity_week,
                CAST(DATE_TRUNC('month', b.billtime AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai' - INTERVAL '6' HOUR) AS DATE) AS activity_month
            FROM agfish.hunterorders b
            WHERE
                b.currency = 'CNY'
                -- avoid converting billtime to increase speed
                AND b.billtime >= (TIMESTAMP '{start_date} 06:00:00' AT TIME ZONE 'Asia/Shanghai' AT TIME ZONE 'UTC') - INTERVAL '{return_user_days}' DAY
                AND b.billtime < (TIMESTAMP '{end_date} 06:00:00' AT TIME ZONE 'Asia/Shanghai' AT TIME ZONE 'UTC') + INTERVAL '{retention_days}' DAY
                AND b.gametype = 'HM3D'
                AND b.account != 0
                AND b.fishcost != 0
                AND b.ordertype = 1
                AND b.remark = b.gametype
                AND b.weaponid IS NULL
        ),

        -- 2. DETERMINE USER DAILY GROUP (Logic applied inside SUM)
        user_daily_group AS (
            SELECT
                user_id,
                activity_date,
                activity_week,
                activity_month,
                'PA' AS daily_group,
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
                ON t1.user_id = t2.user_id AND t2.activity_date = DATE_ADD('day', -1, t1.activity_date)
            LEFT JOIN user_daily_group t3
                ON t1.user_id = t3.user_id AND t3.activity_date = DATE_ADD('day', -3, t1.activity_date)
            LEFT JOIN user_daily_group t4
                ON t1.user_id = t4.user_id AND t4.activity_date = DATE_ADD('day', -7, t1.activity_date)
            -- LEFT JOIN user_daily_group t5
            --    ON t1.user_id = t5.user_id AND t5.activity_week = DATE_ADD(week, -1, t1.activity_week)
            -- LEFT JOIN user_daily_group t6
            --    ON t1.user_id = t6.user_id AND t6.activity_month = DATE_ADD(month, -1, t1.activity_month)
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
                            DATE_DIFF('day', u.bj_date_last_bet, u.activity_date) > {return_user_days}
                        THEN u.user_id
                    END
                ) AS num_return_users,
                COUNT(DISTINCT CASE WHEN b.killed = 1 THEN b.user_id END) AS num_users_killed_fish,
                ROUND(SUM(b.payout) * 1.0 / NULLIF(SUM(b.bet), 0), 3) AS group_rtp
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
                ROUND(SUM(b.payout) * 1.0 / NULLIF(SUM(b.bet), 0), 3) AS user_rtp,
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

            ROUND(t2.num_users_day1 * 1.0 / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day1,
            ROUND(t2.num_users_day3 * 1.0 / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day3,
            ROUND(t2.num_users_day7 * 1.0 / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day7,
            ROUND(t1.num_killed_bullets * 1.0 / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio
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


def execute_query(output_file, stats_agg_col):

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/{output_file}.parquet"
    query = generate_query(stats_agg_col)
    df_rs = data_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs.head(10))
    data_loader.close()


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-pa_fish_hunter",
        )
    )
    # Initialize Scheduler with a default 3-day lookback
    scheduler = ETLScheduler(data_loader, f"{LOCAL_ROOT}/jobs/output_fish_hunter", lookback_days=3)

    scheduler.run_incremental_job(
        job_name="daily_stats_pa",
        query_func=lambda start_date: generate_query("activity_date", start_date),
        key_cols=["activity_date", "user_id", "daily_group"],
        date_col="activity_date",
        partition_level="none",
    )

    # Overrides to 7 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="weekly_stats_pa",
        query_func=lambda start_date: generate_query("activity_week", start_date),
        key_cols=["activity_date", "user_id", "daily_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=7,
    )

    # Overrides to 31 days because weekly data takes longer to settle
    scheduler.run_incremental_job(
        job_name="monthly_stats_pa",
        query_func=lambda start_date: generate_query("activity_month", start_date),
        key_cols=["activity_date", "user_id", "daily_group"],
        date_col="activity_date",  # Always check max activity_date
        partition_level="none",
        lookback=31,
    )
