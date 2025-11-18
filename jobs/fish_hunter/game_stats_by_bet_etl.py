from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT
from bituslabs_ds.etl import DataLoader, RedshiftBackend

GROUP_COL = "session_group"

query = dedent(
    f"""
    WITH user_events AS (
        SELECT
            t.user_id,
            t.event_id,
            t.strategy_name,
            t.event_timestamp,
            LAG(t.event_timestamp, 1) OVER (
                PARTITION BY t.user_id ORDER BY t.event_timestamp
            ) AS prev_event_time,
            LEAD(t.event_timestamp, 1) OVER (
                PARTITION BY t.user_id ORDER BY t.event_timestamp
            ) AS next_event_time
        FROM public.bullet AS t
        WHERE
            t.currency_type = 'CNY'
            AND t.event_timestamp + INTERVAL '8 hours' >= '2025-11-01'
            AND t.event_timestamp + INTERVAL '8 hours' < '2025-12-01'
    ),

    session_start_flag AS (
        SELECT
            user_id,
            event_id,
            event_timestamp,
            CASE
                WHEN prev_event_time IS NULL THEN 1
                WHEN
                    DATEDIFF(SECOND, prev_event_time, event_timestamp) > 1800
                    THEN 1
                ELSE 0
            END AS is_session_start_flag
        FROM user_events
    ),

    session_metadata AS (
        SELECT
            user_id,
            event_id,
            event_timestamp,
            is_session_start_flag,
            SUM(is_session_start_flag) OVER (
                PARTITION BY user_id
                ORDER BY event_timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_id
        FROM session_start_flag
    ),

    user_event_session AS (
        SELECT
            m.user_id,
            m.event_id,
            m.session_id,
            ROW_NUMBER() OVER (PARTITION BY m.user_id, m.session_id ORDER BY m.event_timestamp) AS bet_index,
            MIN(m.event_timestamp)
                OVER (PARTITION BY m.user_id, m.session_id)
                AS session_start_time,
            MAX(m.event_timestamp)
                OVER (PARTITION BY m.user_id, m.session_id)
                AS session_end_time
        FROM session_metadata AS m
    ),

    user_event_session_group AS (
        SELECT
            u.user_id,
            m.session_id,
            CASE
                WHEN SUM(CASE WHEN u.strategy_name = 'BOOST_POOL' THEN 1 ELSE 0 END) > 0 THEN 'BOOST'
                WHEN SUM(CASE WHEN u.strategy_name = 'DYNAMIC_RTP' THEN 1 ELSE 0 END) > 0 THEN 'DYNA_RTP'
                ELSE 'DEFAULT'
            END AS session_group
        FROM user_events AS u
        INNER JOIN user_event_session AS m
            ON u.user_id = m.user_id AND u.event_id = m.event_id
        GROUP BY u.user_id, m.session_id
    ),

    user_stats AS (
        SELECT
            u.bet_index,
            g.{GROUP_COL},
            COUNT(*) AS num_sessions,
            AVG(t.payout / t.bet) AS rtp_mean,
            AVG(
                CASE
                    WHEN
                        t.fish_value > 19 AND t.fish_value < 201
                        THEN t.payout / t.bet
                END
            ) AS rtp_mean_20_200,
            STDDEV(t.payout / t.bet) AS rtp_std,
            STDDEV(
                CASE
                    WHEN
                        t.fish_value > 19 AND t.fish_value < 201
                        THEN t.payout / t.bet
                END
            ) AS rtp_std_20_200,
            AVG(t.multiplier) AS multiplier_mean,
            ROUND(AVG(CAST(t.bullet_level AS FLOAT)), 3) AS bullet_level_mean,
            ROUND(AVG(CAST(t.fish_value AS FLOAT)), 3) AS fish_value_mean,
            AVG(t.bet) AS bet_mean,
            AVG(t.payout) AS payout_mean,
            SUM(t.bullet_level) AS bullet_level_sum,
            SUM(t.fish_value) AS fish_value_sum,
            SUM(t.bet) AS bet_sum,
            SUM(t.payout) AS payout_sum,
            MAX(t.multiplier) AS multiplier_max,
            MAX(t.payout / t.bet) AS rtp_max,
            MAX(t.bullet_level) AS bullet_level_max,
            MAX(t.fish_value) AS fish_value_max,
            MAX(t.bet) AS bet_max,
            MAX(t.payout) AS payout_max,
            MIN(t.multiplier) AS multiplier_min,
            MIN(t.payout / t.bet) AS rtp_min,
            MIN(t.bullet_level) AS bullet_level_min,
            MIN(t.fish_value) AS fish_value_min,
            MIN(t.bet) AS bet_min,
            MIN(t.payout) AS payout_min,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date
        FROM public.bullet t
        JOIN user_event_session u ON t.user_id = u.user_id AND t.event_id = u.event_id
        JOIN user_event_session_group g ON u.user_id = g.user_id AND u.session_id = g.session_id
        GROUP BY session_start_date, u.bet_index, g.{GROUP_COL}
    ),

    -- Compute each median separately (Redshift doesn’t allow multiple different ORDER BYs)
    user_median1 AS (
        SELECT
            u.bet_index,
            g.{GROUP_COL},
            -- PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.bullet_level) AS bullet_level_median
            MEDIAN(t.bullet_level) AS bullet_level_median,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date
        FROM user_event_session AS u
        JOIN public.bullet AS t ON u.user_id = t.user_id AND u.event_id = t.event_id
        JOIN user_event_session_group AS g ON u.user_id = g.user_id AND u.session_id = g.session_id
        GROUP BY session_start_date, u.bet_index, g.{GROUP_COL}
    ),

    user_median2 AS (
        SELECT
            u.bet_index,
            g.{GROUP_COL},
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.fish_value)
                AS fish_value_median,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date
        FROM user_event_session AS u
        JOIN public.bullet AS t ON u.user_id = t.user_id AND u.event_id = t.event_id
        JOIN user_event_session_group AS g ON u.user_id = g.user_id AND u.session_id = g.session_id
        GROUP BY session_start_date, u.bet_index, g.{GROUP_COL}
    ),

    user_median3 AS (
        SELECT
            u.bet_index,
            g.{GROUP_COL},
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.bet) AS bet_median,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date
        FROM user_event_session AS u
        JOIN public.bullet AS t ON u.user_id = t.user_id AND u.event_id = t.event_id
        JOIN user_event_session_group AS g ON u.user_id = g.user_id AND u.session_id = g.session_id
        GROUP BY session_start_date, u.bet_index, g.{GROUP_COL}
    ),

    user_median4 AS (
        SELECT
            u.bet_index,
            g.{GROUP_COL},
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.payout) AS payout_median,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date
        FROM user_event_session AS u
        JOIN public.bullet AS t ON u.user_id = t.user_id AND u.event_id = t.event_id
        JOIN user_event_session_group AS g ON u.user_id = g.user_id AND u.session_id = g.session_id
        GROUP BY session_start_date, u.bet_index, g.{GROUP_COL}
    )

    -- Final output: Join all CTEs
    SELECT
        s.*,
        m1.bullet_level_median,
        m2.fish_value_median,
        m3.bet_median,
        m4.payout_median
    FROM user_stats AS s
    LEFT JOIN user_median1 AS m1
        ON
            s.bet_index = m1.bet_index
            AND s.session_start_date = m1.session_start_date
            AND s.{GROUP_COL} = m1.{GROUP_COL}
    LEFT JOIN user_median2 AS m2
        ON
            s.bet_index = m2.bet_index
            AND s.session_start_date = m2.session_start_date
            AND s.{GROUP_COL} = m2.{GROUP_COL}
    LEFT JOIN user_median3 AS m3
        ON
            s.bet_index = m3.bet_index
            AND s.session_start_date = m3.session_start_date
            AND s.{GROUP_COL} = m3.{GROUP_COL}
    LEFT JOIN user_median4 AS m4
        ON
            s.bet_index = m4.bet_index
            AND s.session_start_date = m4.session_start_date
            AND s.{GROUP_COL} = m4.{GROUP_COL}
    ORDER BY s.session_start_date, s.bet_index, s.{GROUP_COL};

    """
)

if __name__ == "__main__":

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="production-redshift-cluster.cwiqzcm13zcn.ap-southeast-1.redshift.amazonaws.com",
            database="transform-agfish-game",
            user="anaylsis_user",
            password="oZ4ztMx0yEXPLbJL733L",
            port=5439,
        )
    )

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/bullet_stats_by_index.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)
    redshift_loader.close()
