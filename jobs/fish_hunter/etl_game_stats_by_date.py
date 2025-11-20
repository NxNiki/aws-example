import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

DATE_START = "2025-10-20"
DATE_END = "2025-12-1"

query = dedent(
    f"""
    WITH params AS (
    SELECT
        'CNY'::TEXT AS target_currency,
        '{DATE_START}'::DATE AS start_date,
        '{DATE_END}'::DATE AS end_date,
        3 AS retention_days
    ),

    -- 1. FETCH RAW DATA (Keep Date as Date Object)
    base_data AS (
        SELECT
            b.user_id,
            b.strategy_name,
            b.payout,
            b.bet,
            b.fish_value,
            b.killed,
            b.profit,
            -- Convert to Beijing time and truncate to date object 
            TRUNC(DATEADD(hour, 8, b.created_at)) AS bj_date_raw
        FROM public.bullet b
        CROSS JOIN params p
        WHERE
            b.currency_type = p.target_currency
            AND DATEADD(hour, 8, b.created_at) >= p.start_date
            AND DATEADD(hour, 8, b.created_at) < DATEADD(day, p.retention_days, p.end_date)
    ),

    -- 2. DETERMINE USER DAILY GROUP
    user_daily_map AS (
        SELECT
            user_id,
            bj_date_raw,
            CASE
                WHEN SUM(CASE WHEN strategy_name = 'BOOST_POOL' THEN 1 ELSE 0 END) > 0 THEN 'BOOST_POOL'
                WHEN SUM(CASE WHEN strategy_name = 'DYNAMIC_RTP' THEN 1 ELSE 0 END) > 0 THEN 'DYNAMIC_RTP'
                ELSE 'DEFAULT_FALLBACK'
            END AS daily_group
        FROM base_data
        GROUP BY user_id, bj_date_raw
    ),

    -- 3. CALCULATE RETENTION (Pure Date Math)
    retention_stats AS (
        SELECT
            t1.daily_group,
            t1.bj_date_raw,
            COUNT(DISTINCT t1.user_id) AS num_users_day0,
            COUNT(DISTINCT t2.user_id) AS num_users_day1,
            COUNT(DISTINCT t3.user_id) AS num_users_day3
        FROM user_daily_map t1
        CROSS JOIN params p
        LEFT JOIN user_daily_map t2
            ON t1.user_id = t2.user_id
            AND t2.bj_date_raw = DATEADD(day, 1, t1.bj_date_raw)
        LEFT JOIN user_daily_map t3
            ON t1.user_id = t3.user_id
            AND t3.bj_date_raw = DATEADD(day, 3, t1.bj_date_raw)
        WHERE t1.bj_date_raw < p.end_date
        GROUP BY t1.daily_group, t1.bj_date_raw
    ),

    -- 4. AGGREGATE STATS BY ASSIGNED DAILY GROUP
    stats_by_daily_group AS (
        SELECT
            u.daily_group,
            b.bj_date_raw,
            COUNT(DISTINCT b.user_id) AS num_users,
            COUNT(b.user_id) AS num_bullets,
            SUM(b.bet) AS daily_total_bet,
            SUM(b.killed) AS num_killed_bullets,
            COUNT(DISTINCT CASE WHEN b.killed = 1 THEN b.user_id END) AS num_users_killed_fish,
            ROUND(CAST(SUM(b.payout) AS FLOAT) / NULLIF(SUM(b.bet), 0), 3) AS daily_group_rtp
        FROM base_data b
        JOIN user_daily_map u ON b.user_id = u.user_id AND b.bj_date_raw = u.bj_date_raw
        CROSS JOIN params p
        WHERE b.bj_date_raw < p.end_date
        GROUP BY u.daily_group, b.bj_date_raw
    ),

    -- 5. AGGREGATE STATS BY RAW STRATEGY NAME
    stats_by_strategy AS (
        SELECT
            t.strategy_name,
            t.bj_date_raw,
            COUNT(DISTINCT t.user_id) AS group_num_users,
            ROUND(CAST(SUM(t.payout) AS FLOAT) / NULLIF(SUM(t.bet), 0), 3) AS group_rtp,
            AVG(t.fish_value) AS avg_fish_value,
            AVG(CASE WHEN t.fish_value > 19 AND t.fish_value < 201 THEN t.fish_value END) AS avg_fish_value_20_200,
            AVG(CASE WHEN t.killed = 1 THEN t.fish_value END) AS avg_killed_fish_value,
            AVG(CASE WHEN t.fish_value > 19 AND t.fish_value < 201 AND t.killed = 1 THEN t.fish_value END) AS avg_killed_fish_value_20_200,
            AVG(t.profit) AS avg_profit,
            AVG(CASE WHEN t.killed = 1 THEN t.profit END) AS avg_killed_profit
        FROM base_data t
        CROSS JOIN params p
        WHERE t.bj_date_raw < p.end_date
        GROUP BY t.bj_date_raw, t.strategy_name
    )

    -- 6. FINAL JOIN & FORMATTING
    SELECT
        t1.daily_group,
        -- CONVERT TO STRING HERE AT THE END
        TO_CHAR(t1.bj_date_raw, 'YYYY-mm-dd') AS bj_date,
        t1.num_users,
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
        t3.avg_profit,
        t3.avg_killed_profit,
        ROUND(CAST(t1.num_killed_bullets AS FLOAT) / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio,
        ROUND(CAST(t1.num_bullets AS FLOAT) / NULLIF(t1.num_users, 0), 3) AS bullets_per_user,
        ROUND(CAST(t1.num_killed_bullets AS FLOAT) / NULLIF(t1.num_users, 0), 3) AS killed_bullets_per_user
    FROM stats_by_daily_group t1
    INNER JOIN retention_stats t2 
        ON t1.bj_date_raw = t2.bj_date_raw 
        AND t1.daily_group = t2.daily_group
    INNER JOIN stats_by_strategy t3 
        ON t1.bj_date_raw = t3.bj_date_raw 
        AND t1.daily_group = t3.strategy_name
    ORDER BY t1.bj_date_raw, t1.daily_group;

    """
)


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

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/bullet_stats_by_date.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)
    redshift_loader.close()
