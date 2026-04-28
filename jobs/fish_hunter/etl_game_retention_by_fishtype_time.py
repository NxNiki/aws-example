import os
from textwrap import dedent

from bituslabs_ds.config import (
    DATE_START_HOUR,
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
from bituslabs_ds.etl import DataLoader, RedshiftBackend

DATE_START = "2025-11-30"
DATE_END = "2027-12-1"

retention_days = 30
streak_session_thresh = 600


def generate_query():

    query = dedent(
        f"""
        WITH base_data AS (
            SELECT
                b.user_id,
                CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', b.event_timestamp) AS bet_time,
                b.killed,
                CASE
                    WHEN b.fish_value <= 10 THEN 'low'
                    WHEN b.fish_value <= 130 THEN 'medium'
                    WHEN b.fish_value <= 200 THEN 'high'
                    ELSE 'ultra'
                END as fish_type,
                DATEDIFF(second, LAG(b.event_timestamp) OVER (PARTITION BY b.user_id ORDER BY b.event_timestamp), b.event_timestamp) AS delta_bet_time,
                DATE_TRUNC('day', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', b.event_timestamp))) AS activity_date,
                DATE_TRUNC('week', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', b.event_timestamp))) AS activity_week,
                DATE_TRUNC('month', DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', '{TIMEZONE_SHANGHAI}', b.event_timestamp))) AS activity_month
            FROM public.bullet b
            WHERE
                b.currency_type IN {ETL_CURRENCY_CODES}
            AND b.op_code NOT IN {ETL_EXCLUDED_OP_CODES}
            -- ---------------------------------------------------------
            -- FAST FILTERING: Transform the INPUTS, not the COLUMN
            -- ---------------------------------------------------------
            -- Logic: We want events where (EventTime + UserDays) >= Start
            -- So: EventTime >= Start - UserDays
            AND b.event_timestamp >= CONVERT_TIMEZONE('{TIMEZONE_SHANGHAI}', 'UTC', CAST('{DATE_START}' AS TIMESTAMP))
            -- Logic: We want events where (EventTime - RetentionDays) < End
            -- So: EventTime < End + RetentionDays
            AND b.event_timestamp < CONVERT_TIMEZONE('{TIMEZONE_SHANGHAI}', 'UTC', DATEADD(day, {retention_days}, CAST('{DATE_END}' AS TIMESTAMP)))
            AND b.strategy_name = 'DEFAULT_FALLBACK'
        ),

        session_data AS (
            SELECT
                t.user_id,
                t.activity_date,
                t.bet_time,
                t.fish_type,
                t.killed,
                SUM(
                    CASE
                        WHEN delta_bet_time <= {streak_session_thresh} THEN 0
                        ELSE 1
                    END
                ) OVER (PARTITION BY t.user_id ORDER BY t.bet_time ROWS UNBOUNDED PRECEDING)
                AS session_id
            FROM base_data t
        ),

        session_time AS (
            SELECT
                t.user_id,
                t.activity_date,
                t.fish_type,
                t.bet_time,
                DATEDIFF(milliseconds , MIN(t.bet_time) OVER(PARTITION BY t.user_id, t.session_id), t.bet_time) / 1000.0 AS session_seconds,
                SUM(t.killed) OVER(PARTITION BY t.user_id, t.session_id, ORDER BY t.bet_time ROWS UNBOUNDED PRECEDING) AS session_killed_fish_all_types
                SUM(t.killed) OVER(PARTITION BY t.user_id, t.session_id, t.fish_type ORDER BY t.bet_time ROWS UNBOUNDED PRECEDING) AS session_killed_fish
            FROM session_data t
        ),

        session_time_window AS (
            SELECT
                t.user_id,
                t.activity_date,
                t.bet_time,
                t.fish_type,

                CASE
                    WHEN session_seconds <= 60 THEN '0-1_min'
                    WHEN session_seconds <= 180 THEN '1-3_min'
                    WHEN session_seconds <= 300 THEN '3-5_min'
                    WHEN session_seconds <= 600 THEN '5-10_min'
                    ELSE '>10_min'
                END AS time_window,

                CASE
                    WHEN session_killed_fish = 0 THEN 'kill_fish_0'
                    WHEN session_killed_fish = 1 THEN 'kill_fish_1'
                    WHEN session_killed_fish <= 3 THEN 'kill_fish_2-3'
                    WHEN session_killed_fish <= 5 THEN 'kill_fish_4-5'
                    WHEN session_killed_fish <= 10 THEN 'kill_fish_6-10'
                    ELSE 'kill_fish_>10'
                END AS kill_fish

            FROM session_time t
        ),

        daily_bet AS (
            SELECT
                t.user_id,
                t.activity_date,
                t.activity_week,
                t.activity_month,
                MIN(t.bet_time) AS daily_first_bet_time,
                MAX(t.bet_time) AS daily_last_bet_time
            FROM base_data t
            GROUP BY t.user_id, t.activity_date, t.activity_week, t.activity_month
        ),

        user_retention AS (
            SELECT
                t1.user_id,
                t1.activity_date,
                t1.activity_week,
                t1.activity_month,
                t1.daily_first_bet_time,
                t1.daily_last_bet_time,
                -- Returns 1 if at least one activity exists in the [24h, 48h) window, else 0
                SIGN(COUNT(DISTINCT t2.activity_date)) AS day1_return,
                -- Returns 1 if at least one activity exists in the [72h, 96h) window, else 0
                SIGN(COUNT(DISTINCT t3.activity_date)) AS day3_return,
                -- Returns 1 if activity exists in the following calendar week
                SIGN(COUNT(DISTINCT t4.activity_week)) AS week1_return,
                -- Returns 1 if activity exists in the following calendar month
                SIGN(COUNT(DISTINCT t5.activity_month)) AS month1_return
            FROM daily_bet AS t1
                    LEFT JOIN daily_bet AS t2
                            ON t1.user_id = t2.user_id
                                AND t2.daily_last_bet_time >= DATE_ADD('hour', 24, t1.daily_first_bet_time)
                                AND t2.daily_first_bet_time < DATE_ADD('hour', 48, t1.daily_first_bet_time)
                    LEFT JOIN daily_bet AS t3
                            ON t1.user_id = t3.user_id
                                AND t3.daily_last_bet_time >= DATE_ADD('hour', 72, t1.daily_first_bet_time)
                                AND t3.daily_first_bet_time < DATE_ADD('hour', 96, t1.daily_first_bet_time)
                    LEFT JOIN daily_bet AS t4
                            ON t1.user_id = t4.user_id
                                AND t4.activity_week = DATE_ADD('week', 1, t1.activity_week)
                    LEFT JOIN daily_bet AS t5
                            ON t1.user_id = t5.user_id
                                AND t5.activity_month = DATE_ADD('month', 1, t1.activity_month)
            GROUP BY
                t1.user_id,
                t1.activity_date,
                t1.activity_week,
                t1.activity_month,
                t1.daily_first_bet_time,
                t1.daily_last_bet_time
        )

        select
            t1.fish_type,
            t1.time_window,
            t1.kill_fish,
            count(distinct t1.user_id) as num_users,
            count(distinct case when t2.day1_return = 1 then t1.user_id end) * 1.0 / count(distinct t1.user_id) as day1_retention_ratio,
            count(distinct case when t2.day3_return = 1 then t1.user_id end) * 1.0 / count(distinct t1.user_id) as day3_retention_ratio,
            count(distinct case when t2.week1_return = 1 then t1.user_id end) * 1.0 / count(distinct t1.user_id) as week1_retention_ratio,
            count(distinct case when t2.month1_return = 1 then t1.user_id end) * 1.0 / count(distinct t1.user_id) as month1_retention_ratio
        from session_time_window t1
        join user_retention t2 on t1.user_id = t2.user_id and t1.activity_date = t2.activity_date
        group by t1.time_window, t1.kill_fish, t1.fish_type
        order by t1.fish_type, t1.time_window, t1.kill_fish
        ;
        """
    )

    return query


def execute_query(redshift_loader, output_file):

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/{output_file}.parquet"
    query = generate_query()
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database="transform-agfish-game",
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
        )
    )

    execute_query(redshift_loader, "retention_by_time_fishtype")

    redshift_loader.close()
