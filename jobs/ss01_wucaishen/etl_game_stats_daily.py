import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

# TODO:
# add max/min user daily profit

query = dedent(
    """
    WITH target_settings AS (
    -- Set the desired report start date (00:00:00) in Beijing Time (UTC+8)
    SELECT
        '2025-11-01'::TIMESTAMP AS start_date_bj,
        30 AS retention_threshold
    ),

    -- CTE 1: Daily Activity - Simplifies the base data needed for retention
    daily_activity AS (
        SELECT 
            t.user_id,
            t.bet_amount,
            t.actual_payout,
            t.bet_type,
            DATE_TRUNC('day', DATEADD('hour', 8, t.created_at)) AS activity_date
        FROM public.fct_bet_orders AS t, target_settings AS ts
        WHERE
            t.created_at >= DATEADD('hour', -8, ts.start_date_bj)
            AND t.status = 'COMPLETED'
            AND t.currency_type = 'CNY'
            AND t.op_code != 'B26'
    ),

    -- CTE 2: Metrics - Calculates the core metrics (Volume, RTP, etc.)
    daily_metrics AS (
        SELECT
            activity_date AS report_date,
            COUNT(DISTINCT user_id) AS total_users,
            SUM(bet_amount) AS total_bet,
            SUM(actual_payout) AS total_payout,

            -- RTP
            COALESCE(SUM(actual_payout) * 1.0 / NULLIF(SUM(bet_amount), 0), 0) AS rtp,

            -- Game Type Counts & Payouts
            COUNT(CASE WHEN bet_type = 'BASE' THEN 1 END) AS num_base_game,
            COUNT(CASE WHEN bet_type = 'FREE' THEN 1 END) AS num_free_game,
            SUM(CASE WHEN bet_type = 'BASE' THEN bet_amount ELSE 0 END) AS total_bet_bg,
            SUM(CASE WHEN bet_type = 'FREE' THEN bet_amount ELSE 0 END) AS total_bet_fg,
            SUM(CASE WHEN bet_type = 'BASE' THEN actual_payout ELSE 0 END) AS total_payout_bg,
            SUM(CASE WHEN bet_type = 'FREE' THEN actual_payout ELSE 0 END) AS total_payout_fg
        FROM daily_activity
        GROUP BY report_date
    ),

    -- CTE 3: Initial Cohort - Gets the unique users for the retention base day
    initial_cohort AS (
        SELECT DISTINCT
            activity_date AS report_date,
            user_id
        FROM daily_activity
    ),

    -- CTE 4: Retention Data - Joins the initial cohort back to all activity
    retention_data AS (
        SELECT
            t1.report_date,
            COUNT(DISTINCT t1.user_id) AS num_users_cohort, -- Redundant, but useful for clarity
            COUNT(DISTINCT t2.user_id) AS num_users_day1,
            COUNT(DISTINCT t3.user_id) AS num_users_day3
        FROM initial_cohort AS t1

        -- LEFT JOIN for Day 1 Retention (user_id and date offset)
        LEFT JOIN daily_activity AS t2
            ON
                t1.user_id = t2.user_id
                AND t2.activity_date = DATEADD('day', 1, t1.report_date)

        -- LEFT JOIN for Day 3 Retention (user_id and date offset)
        LEFT JOIN daily_activity AS t3
            ON
                t1.user_id = t3.user_id
                AND t3.activity_date = DATEADD('day', 3, t1.report_date)

        GROUP BY t1.report_date
    )

    -- Final SELECT: Combine Metrics and Retention
    SELECT
        TRUNC(m.report_date) AS bj_date,
        m.total_users,
        m.total_bet,
        m.total_payout,
        m.total_payout - m.total_bet As total_profit,
        m.rtp,
        m.num_base_game,
        m.num_free_game,
        m.total_bet_bg,
        m.total_bet_fg,
        m.total_payout_bg,
        m.total_payout_fg,
        m.total_payout_bg - m.total_bet_bg AS total_profit_bg,
        m.total_payout_fg - m.total_bet_fg AS total_profit_fg,

        -- Retention Counts
        r.num_users_day1,
        r.num_users_day3,

        -- Retention Rates
        ROUND(COALESCE(r.num_users_day1 * 1.0 / NULLIF(m.total_users, 0), 0), 3) AS retention_rate_day1,
        ROUND(COALESCE(r.num_users_day3 * 1.0 / NULLIF(m.total_users, 0), 0), 3) AS retention_rate_day3,


        ROUND(COALESCE(m.total_bet * 1.0 / NULLIF(m.total_users, 0), 0), 3) AS total_bet_per_user,
        ROUND(COALESCE((m.total_payout - m.total_bet) * 1.0 / NULLIF(m.total_users, 0), 0), 3) AS total_profit_per_user

    FROM daily_metrics AS m
    INNER JOIN retention_data AS r
        ON m.report_date = r.report_date

    ORDER BY m.report_date;
    """
)


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="production-redshift-cluster.cwiqzcm13zcn.ap-southeast-1.redshift.amazonaws.com",
            database="slot-machine",
            user="anaylsis_user",
            password="oZ4ztMx0yEXPLbJL733L",
            port=5439,
        )
    )

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_date.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)
    redshift_loader.close()
