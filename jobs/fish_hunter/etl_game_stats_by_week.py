import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

query = dedent(
    """
    WITH target_settings AS (
        SELECT
            '2025-10-01'::TIMESTAMP AS start_date_bj,
            30 AS retention_threshold
    ),

    weekly_activity AS (
        SELECT DISTINCT
            user_id,
            DATE_TRUNC('day', CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', event_timestamp)) AS bj_date,
            -- This value is ALWAYS the Monday of the week
            DATE_TRUNC('week', CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', event_timestamp)) AS bj_week_start
        FROM public.bullet
        WHERE
            event_timestamp
            >= DATE_ADD(
                'day',
                -(SELECT t.retention_threshold FROM target_settings AS t) - 1,
                CONVERT_TIMEZONE('Asia/Shanghai', 'UTC', (SELECT t.start_date_bj FROM target_settings AS t))
            )
    ),

    users_last_bet AS (
        SELECT
            user_id,
            bj_date,
            bj_week_start,
            CASE
                WHEN
                    DATE_DIFF('day', LAG(bj_date) OVER (PARTITION BY user_id ORDER BY bj_date), bj_date)
                    > (SELECT t.retention_threshold FROM target_settings AS t)
                    THEN 1
                ELSE 0
            END AS is_return_user
        FROM weekly_activity
        WHERE bj_date >= CONVERT_TIMEZONE('Asia/Shanghai', 'UTC', (SELECT t.start_date_bj FROM target_settings AS t))
    )

    SELECT
        -- This will output: 2025-10-06, 2025-10-13, etc.
        TRUNC(lb.bj_week_start) AS week_commencing_monday,

        -- Total Active Users this week
        COUNT(DISTINCT lb.user_id) AS num_users,

        -- Return Users (no bet in last 30 days)
        -- SUM(lb.is_return_user) AS num_return_users
        COUNT(DISTINCT CASE WHEN lb.is_return_user = 1 THEN user_id END) AS num_return_users
    FROM users_last_bet AS lb
    GROUP BY 1
    ORDER BY 1;
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

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/bullet_stats_by_week.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)
    redshift_loader.close()
