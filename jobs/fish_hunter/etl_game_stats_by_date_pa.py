import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.etl import AthenaBackend, DataLoader

DATE_START = "2025-10-20"
DATE_END = "2027-12-1"

return_user_days = 30
retention_days = 3


def generate_query(stats_agg_col):
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
                DATE_TRUNC('day', b.billtime AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai') AS activity_date,
                DATE_TRUNC('week', b.billtime AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai') AS activity_week,
                DATE_TRUNC('month', b.billtime AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Shanghai') AS activity_month
            FROM agfish.hunterorders b
            WHERE
                b.currency = 'CNY'
                -- avoid converting billtime to increase speed
                AND b.billtime >= (TIMESTAMP '{DATE_START} 00:00:00' AT TIME ZONE 'Asia/Shanghai' AT TIME ZONE 'UTC') - INTERVAL '{return_user_days}' DAY
                AND b.billtime < (TIMESTAMP '{DATE_END} 00:00:00' AT TIME ZONE 'Asia/Shanghai' AT TIME ZONE 'UTC') + INTERVAL '{retention_days}' DAY
                AND b.gametype = 'HM3D'
                AND b.account != 0
                AND b.fishcost != 0
                AND b.ordertype = 1
                AND b.remark = b.gametype
                AND b.weaponid IS NULL
        ),

        -- 2. DETERMINE USER PREVIOUS BET DATE
        user_daily_map AS (
            SELECT
                user_id,
                activity_date,
                activity_week,
                activity_month,
                LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS bj_date_last_bet
            FROM base_data
            GROUP BY user_id, activity_date, activity_week, activity_month
        ),

        -- 3. CALCULATE RETENTION (Pure Date Math)
        retention_stats AS (
            SELECT
                t1.{stats_agg_col},
                COUNT(DISTINCT t1.user_id) AS num_users_day0,
                COUNT(DISTINCT t2.user_id) AS num_users_day1,
                COUNT(DISTINCT t3.user_id) AS num_users_day3
            FROM user_daily_map t1
            LEFT JOIN user_daily_map t2
                ON t1.user_id = t2.user_id
                AND t2.activity_date = DATE_ADD('day', -1, t1.activity_date)
            LEFT JOIN user_daily_map t3
                ON t1.user_id = t3.user_id
                AND t3.activity_date = DATE_ADD('day', -3, t1.activity_date)
            GROUP BY t1.{stats_agg_col}
        ),

        -- 4. AGGREGATE STATS BY ASSIGNED DAILY GROUP
        stats_by_daily_group AS (
            SELECT
                b.{stats_agg_col},
                COUNT(DISTINCT b.user_id) AS num_users,
                COUNT(DISTINCT
                    CASE
                        WHEN
                            DATE_DIFF('day', u.bj_date_last_bet, u.activity_date) > {return_user_days}
                        THEN u.user_id
                    END
                ) AS num_return_users,
                COUNT(DISTINCT b.user_id || '-' || b.room_id) AS total_num_rooms,
                COUNT(b.user_id) AS num_bullets,
                SUM(b.bet) AS daily_total_bet,
                SUM(b.killed) AS num_killed_bullets,
                COUNT(DISTINCT CASE WHEN b.killed = 1 THEN b.user_id END) AS num_users_killed_fish,
                ROUND(CAST(SUM(b.payout) AS DOUBLE) / NULLIF(SUM(b.bet), 0), 3) AS daily_group_rtp
            FROM base_data b
            JOIN user_daily_map u ON b.user_id = u.user_id AND b.activity_date = u.activity_date
            WHERE b.activity_date >= DATE '{DATE_START}'
                AND b.activity_date < DATE '{DATE_END}'
            GROUP BY b.{stats_agg_col}
        ),

        -- 5. AGGREGATE STATS BY RAW STRATEGY NAME
        stats_by_strategy AS (
            SELECT
                t.{stats_agg_col},
                COUNT(DISTINCT t.user_id) AS group_num_users,
                ROUND(CAST(SUM(t.payout) AS DOUBLE) / NULLIF(SUM(t.bet), 0), 3) AS group_rtp,
                AVG(t.fish_value) AS avg_fish_value,
                AVG(CASE WHEN t.fish_value > 19 AND t.fish_value < 201 THEN t.fish_value END) AS avg_fish_value_20_200,
                AVG(CASE WHEN t.killed = 1 THEN t.fish_value END) AS avg_killed_fish_value,
                AVG(CASE WHEN t.fish_value > 19 AND t.fish_value < 201 AND t.killed = 1 THEN t.fish_value END) AS avg_killed_fish_value_20_200,
                AVG(t.profit) AS bullet_avg_profit,
                AVG(CASE WHEN t.killed = 1 THEN t.profit END) AS bullet_kill_avg_profit,
                SUM(t.profit) AS total_profit
            FROM base_data t
            WHERE t.activity_date >= DATE '{DATE_START}'
                AND t.activity_date < DATE '{DATE_END}'
            GROUP BY t.{stats_agg_col}
        )

        -- 6. FINAL JOIN & FORMATTING
        SELECT
            'PA' AS daily_group,
            date_format(t1.{stats_agg_col}, '%Y-%m-%d') AS activity_date,
            t1.num_users,
            t1.num_return_users,
            t1.total_num_rooms,
            t1.num_bullets,
            t1.daily_total_bet,
            t1.num_killed_bullets,
            t1.num_users_killed_fish,
            t1.daily_group_rtp,

            t2.num_users_day0,
            t2.num_users_day1,
            t2.num_users_day3,

            t3.group_num_users,
            t3.group_rtp,
            t3.avg_fish_value,
            t3.avg_fish_value_20_200,
            t3.avg_killed_fish_value,
            t3.avg_killed_fish_value_20_200,
            t3.bullet_avg_profit,
            t3.bullet_kill_avg_profit,
            t3.total_profit,
            
            ROUND(CAST(t2.num_users_day1 AS DOUBLE) / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day1,
            ROUND(CAST(t2.num_users_day3 AS DOUBLE) / NULLIF(t2.num_users_day0, 0), 3) AS retention_rate_day3,

            ROUND(CAST(t1.daily_total_bet AS DOUBLE) / NULLIF(t2.num_users_day0, 0), 3) AS total_bet_per_user,
            ROUND(CAST(t3.total_profit AS DOUBLE) / NULLIF(t2.num_users_day0, 0), 3) AS total_profit_per_user,
            ROUND(CAST(t1.total_num_rooms AS DOUBLE) / NULLIF(t2.num_users_day0, 0), 3) AS total_rooms_per_user,

            ROUND(CAST(t1.num_killed_bullets AS DOUBLE) / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio,
            ROUND(CAST(t1.num_bullets AS DOUBLE) / NULLIF(t1.num_users, 0), 3) AS bullets_per_user,
            ROUND(CAST(t1.num_killed_bullets AS DOUBLE) / NULLIF(t1.num_users, 0), 3) AS killed_bullets_per_user
        FROM stats_by_daily_group t1
        INNER JOIN retention_stats t2 
            ON t1.{stats_agg_col} = t2.{stats_agg_col} 
        INNER JOIN stats_by_strategy t3 
            ON t1.{stats_agg_col} = t3.{stats_agg_col} 
        ORDER BY t1.{stats_agg_col}
        ;
        """
    )

    return query


def execute_query(output_file, stats_agg_col):
    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-fish_hunter/{output_file}",
        )
    )
    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/{output_file}.parquet"
    query = generate_query(stats_agg_col)
    df_rs = data_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs.head(10))
    data_loader.close()


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    execute_query("bullet_stats_by_date_pa", "activity_date")
    execute_query("bullet_stats_by_week_pa", "activity_week")
    execute_query("bullet_stats_by_month_pa", "activity_month")
