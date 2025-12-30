import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

DATE_START = "2025-10-31"
DATE_END = "2027-12-1"

return_user_days = 30
retention_days = 3


def generate_query(stats_agg_col):

    query = dedent(
        f"""
        -- Check game stats between retention users and non-retention users.

        -- 1. FETCH RAW DATA (Keep strictly RAW columns to enable Index Scans)
        WITH base_data AS (
            SELECT
                b.user_id,
                b.room_id,
                b.payout,
                b.bet,
                b.fish_value,
                b.killed,
                b.profit,
                DATE_TRUNC('day', DATEADD(hour, -6, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS activity_date
            FROM public.bullet b
            WHERE
                b.currency_type = 'CNY'
                -- ---------------------------------------------------------
                -- FAST FILTERING: Transform the INPUTS, not the COLUMN
                -- ---------------------------------------------------------
                -- Logic: We want events where (EventTime + UserDays) >= Start
                -- So: EventTime >= Start - UserDays
                AND b.created_at >= CONVERT_TIMEZONE('Asia/Shanghai', 'UTC',
                    DATEADD(day, -{return_user_days}, CAST('{DATE_START}' AS TIMESTAMP)))
                -- Logic: We want events where (EventTime - RetentionDays) < End
                -- So: EventTime < End + RetentionDays
                AND b.created_at < CONVERT_TIMEZONE('Asia/Shanghai', 'UTC',
                    DATEADD(day, {retention_days}, CAST('{DATE_END}' AS TIMESTAMP)))
                AND b.strategy_name = 'DEFAULT_FALLBACK'
        ),

        
        -- 2. DETERMINE USER DAILY GROUP (Logic applied inside SUM)
        user_activity AS (
            SELECT DISTINCT
                user_id,
                activity_date
            FROM base_data
        ),

        user_last_bet AS (
            SELECT
                user_id,
                activity_date,
                LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS last_bet_date
            FROM user_activity
        ),

        -- 3. CALCULATE RETENTION (Pure Date Math)
        retention_users AS (
            SELECT
                t1.activity_date,
                t1.user_id,
        --         CASE
        --             WHEN DATEDIFF('day', t1.last_bet_date, t1.activity_date) > 30 THEN 1
        --             ELSE 0
        --         END AS is_30_day_return_user,
                CASE
                    WHEN COUNT(t2.user_id) > 0 THEN 'return'
                    ELSE 'non-return'
                END AS return_user
            FROM user_last_bet AS t1
            LEFT JOIN user_last_bet AS t2
                ON
                    t1.user_id = t2.user_id
                    AND t1.activity_date < t2.activity_date
                    AND t2.activity_date <= DATEADD(DAY, 7, t1.activity_date)
            GROUP BY t1.user_id, t1.activity_date, t1.last_bet_date
        ),

        -- 4. AGGREGATE STATS BY ASSIGNED DAILY GROUP
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
                r.user_id,
                b.activity_date,
                r.return_user,
                COUNT(DISTINCT b.room_id) AS num_rooms,
                COUNT(b.user_id) AS num_bullets,
                SUM(b.bet) AS daily_total_bet,
                SUM(b.killed) AS num_killed_bullets,
                SUM(b.profit) AS total_profit,
                MAX(b.profit) AS max_profit,
                ROUND(CAST(SUM(b.payout) AS FLOAT) / NULLIF(SUM(b.bet), 0), 3) AS rtp,
                STDDEV(b.profit) / NULLIF(ABS(AVG(b.profit)), 0) AS profit_coef_var
            FROM base_data AS b
            INNER JOIN retention_users AS r ON b.user_id = r.user_id AND b.activity_date = r.activity_date
            GROUP BY b.activity_date, r.user_id, r.return_user
            HAVING MAX(b.killed) > 0
        ),

        user_kill_fish AS (
            SELECT
                b.user_id,
                b.activity_date,
                MAX(CASE WHEN b.killed >= 1 THEN 1 ELSE 0 END) AS user_killed_fish
            FROM base_data AS b
            GROUP BY b.activity_date, b.user_id
        )

        -- 6. FINAL JOIN & FORMATTING (Fully Restored)
        SELECT
            t1.activity_date,
            t1.user_id,
            t1.return_user,
            t2.num_users,
            t1.num_rooms,
            t1.num_bullets,
            t1.daily_total_bet,
            t1.num_killed_bullets,
            t1.total_profit,
            t1.max_profit,
            t1.rtp,
            t1.profit_coef_var,
        --     SUM(t3.is_30_day_return_user) OVER (PARTITION BY t1.activity_date, t1.return_user) AS num_30_day_return_user,
            SUM(t3.user_killed_fish) OVER (PARTITION BY t1.activity_date, t1.return_user) AS num_users_killed_fish,
            ROUND(CAST(t1.num_killed_bullets AS FLOAT) / NULLIF(t1.num_bullets, 0), 3) AS bullet_kill_ratio
        FROM user_daily_stats AS t1
        INNER JOIN daily_stats AS t2 ON t1.activity_date = t2.activity_date AND t1.return_user = t2.return_user
        INNER JOIN user_kill_fish AS t3 ON t1.activity_date = t3.activity_date AND t1.user_id = t3.user_id
        -- INNER JOIN retention_users AS t3 ON t1.activity_date = t3.activity_date AND t1.return_user = t3.return_user
        ORDER BY t1.activity_date, t1.user_id, t1.return_user
        ;

        """
    )

    return query


def execute_query(redshift_loader, stats_agg_col, output_file):

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/{output_file}.parquet"
    query = generate_query(stats_agg_col)
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)


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

    execute_query(redshift_loader, "activity_date", "bullet_stats_by_date_return_user")

    redshift_loader.close()
