import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.etl import AthenaBackend, DataLoader

DATE_START = "2025-10-20"
DATE_END = "2025-12-1"

query = dedent(
    f"""
    WITH params AS (
    SELECT
        'CNY' AS target_currency,
        TIMESTAMP '{DATE_START}' AS start_date,
        TIMESTAMP '{DATE_END}' AS end_date,
        3 AS retention_days,
        30 AS return_user_days
    ),

    -- 1. FETCH RAW DATA (Keep Date as Date Object)
    base_data AS (
        SELECT
            b.loginname AS user_id,
            b.roomid AS room_id,
            b.account + b.cus_account AS payout,
            b.account AS bet,
            b.fishcost / b.betx AS fish_value,
            b.hunted AS killed,
            b.cus_account AS profit,
            -- Convert to Beijing time and truncate to date object 
            DATE_TRUNC('day', DATE_ADD('hour', 8, b.billtime)) AS bj_date_raw
        FROM agfish.hunterorders b
        CROSS JOIN params p
        WHERE
            b.currency = p.target_currency
            AND b.billtime >= DATE_ADD('hour', -8, DATE_ADD('day', -p.return_user_days, p.start_date))
            AND b.billtime < DATE_ADD('hour', -8, DATE_ADD('day', p.retention_days, p.end_date))
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
            bj_date_raw,
            LAG(bj_date_raw) OVER (PARTITION BY user_id ORDER BY bj_date_raw) AS bj_date_last_bet
        FROM base_data
        GROUP BY user_id, bj_date_raw
    ),

    -- 3. CALCULATE RETENTION (Pure Date Math)
    retention_stats AS (
        SELECT
            t1.bj_date_raw,
            COUNT(DISTINCT t1.user_id) AS num_users_day0,
            COUNT(DISTINCT t2.user_id) AS num_users_day1,
            COUNT(DISTINCT t3.user_id) AS num_users_day3
        FROM user_daily_map t1
        CROSS JOIN params p
        LEFT JOIN user_daily_map t2
            ON t1.user_id = t2.user_id
            AND t2.bj_date_raw = DATE_ADD('day', 1, t1.bj_date_raw)
        LEFT JOIN user_daily_map t3
            ON t1.user_id = t3.user_id
            AND t3.bj_date_raw = DATE_ADD('day', 3, t1.bj_date_raw)
        WHERE t1.bj_date_raw < p.end_date
        GROUP BY t1.bj_date_raw
    ),

    -- 4. AGGREGATE STATS BY ASSIGNED DAILY GROUP
    stats_by_daily_group AS (
        SELECT
            b.bj_date_raw,
            COUNT(DISTINCT b.user_id) AS num_users,
            COUNT(DISTINCT
                CASE
                    WHEN
                        DATE_DIFF('day', u.bj_date_last_bet, u.bj_date_raw) > p.return_user_days
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
        JOIN user_daily_map u ON b.user_id = u.user_id AND b.bj_date_raw = u.bj_date_raw
        CROSS JOIN params p
        WHERE b.bj_date_raw >= p.start_date AND b.bj_date_raw < p.end_date
        GROUP BY b.bj_date_raw
    ),

    -- 5. AGGREGATE STATS BY RAW STRATEGY NAME
    stats_by_strategy AS (
        SELECT
            t.bj_date_raw,
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
        CROSS JOIN params p
        WHERE t.bj_date_raw >= p.start_date AND t.bj_date_raw < p.end_date
        GROUP BY t.bj_date_raw
    )

    -- 6. FINAL JOIN & FORMATTING
    SELECT
        'PA' AS daily_group,
        -- CONVERT TO STRING HERE AT THE END
        date_format(t1.bj_date_raw, '%Y-%m-%d') AS bj_date,
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
        ROUND(CAST(t1.num_killed_bullets AS DOUBLE) / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio,
        ROUND(CAST(t1.num_bullets AS DOUBLE) / NULLIF(t1.num_users, 0), 3) AS bullets_per_user,
        ROUND(CAST(t1.num_killed_bullets AS DOUBLE) / NULLIF(t1.num_users, 0), 3) AS killed_bullets_per_user
    FROM stats_by_daily_group t1
    INNER JOIN retention_stats t2 
        ON t1.bj_date_raw = t2.bj_date_raw 
    INNER JOIN stats_by_strategy t3 
        ON t1.bj_date_raw = t3.bj_date_raw 
    ORDER BY t1.bj_date_raw
    ;
    """
)


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-fish_hunter/bullet_stats_by_date_pa",
        )
    )

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/bullet_stats_by_date_pa.parquet"
    df_rs = data_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)
    data_loader.close()
