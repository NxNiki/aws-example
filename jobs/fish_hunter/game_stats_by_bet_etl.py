from textwrap import dedent

from bituslabs_ds.etl import DataLoader, RedshiftBackend

query = dedent(
    f"""
    WITH user_events AS (
        SELECT
            t.*,
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

    user_event_ssession AS (
        SELECT
            u.*,
            m.session_id,
            m.is_session_start_flag,
            ROW_NUMBER() OVER (PARTITION BY m.user_id, m.session_id) AS bet_index,
            MIN(u.event_timestamp)
                OVER (PARTITION BY u.user_id, m.session_id)
                AS session_start_time,
            MAX(u.event_timestamp)
                OVER (PARTITION BY u.user_id, m.session_id)
                AS session_end_time,
            DATEDIFF(SECOND, u.event_timestamp, u.next_event_time)
                AS time_diff_next,
            DATEDIFF(SECOND, u.prev_event_time, u.event_timestamp) AS time_diff_prev
        FROM user_events AS u
        INNER JOIN session_metadata AS m
            ON u.user_id = m.user_id AND u.event_id = m.event_id
    ),

    user_stats AS (
        SELECT
            u.bet_index,
            u.strategy_name,
            COUNT(*) AS num_sessions,
            AVG(u.payout / u.bet) AS rtp_mean,
            AVG(
                CASE
                    WHEN
                        u.fish_value > 19 AND u.fish_value < 201
                        THEN u.payout / u.bet
                END
            ) AS rtp_mean_20_200,
            STDDEV(u.payout / u.bet) AS rtp_std,
            STDDEV(
                CASE
                    WHEN
                        u.fish_value > 19 AND u.fish_value < 201
                        THEN u.payout / u.bet
                END
            ) AS rtp_std_20_200,
            AVG(u.multiplier) AS multiplier_mean,
            ROUND(AVG(CAST(u.bullet_level AS FLOAT)), 3) AS bullet_level_mean,
            ROUND(AVG(CAST(u.fish_value AS FLOAT)), 3) AS fish_value_mean,
            AVG(u.bet) AS bet_mean,
            AVG(u.payout) AS payout_mean,
            SUM(u.bullet_level) AS bullet_level_sum,
            SUM(u.fish_value) AS fish_value_sum,
            SUM(u.bet) AS bet_sum,
            SUM(u.payout) AS payout_sum,
            MAX(u.multiplier) AS multiplier_max,
            MAX(u.payout / u.bet) AS rtp_max,
            MAX(u.bullet_level) AS bullet_level_max,
            MAX(u.fish_value) AS fish_value_max,
            MAX(u.bet) AS bet_max,
            MAX(u.payout) AS payout_max,
            MIN(u.multiplier) AS multiplier_min,
            MIN(u.payout / u.bet) AS rtp_min,
            MIN(u.bullet_level) AS bullet_level_min,
            MIN(u.fish_value) AS fish_value_min,
            MIN(u.bet) AS bet_min,
            MIN(u.payout) AS payout_min,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date
        FROM user_event_ssession AS u
        GROUP BY session_start_date, u.bet_index, u.strategy_name
    ),

    -- Compute each median separately (Redshift doesn’t allow multiple different ORDER BYs)
    user_median1 AS (
        SELECT
            u.bet_index,
            u.strategy_name,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date,
            -- PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY u.bullet_level) AS bullet_level_median
            MEDIAN(u.bullet_level) AS bullet_level_median
        FROM user_event_ssession AS u
        GROUP BY session_start_date, u.bet_index, u.strategy_name
    ),

    user_median2 AS (
        SELECT
            u.bet_index,
            u.strategy_name,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY u.fish_value)
                AS fish_value_median
        FROM user_event_ssession AS u
        GROUP BY session_start_date, u.bet_index, u.strategy_name
    ),

    user_median3 AS (
        SELECT
            u.bet_index,
            u.strategy_name,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY u.bet) AS bet_median
        FROM user_event_ssession AS u
        GROUP BY session_start_date, u.bet_index, u.strategy_name
    ),

    user_median4 AS (
        SELECT
            u.bet_index,
            u.strategy_name,
            TO_CHAR(u.session_start_time + INTERVAL '8 hours', 'YYYY-MM-DD')
                AS session_start_date,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY u.payout) AS payout_median
        FROM user_event_ssession AS u
        GROUP BY session_start_date, u.bet_index, u.strategy_name
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
            AND s.strategy_name = m1.strategy_name
    LEFT JOIN user_median2 AS m2
        ON
            s.bet_index = m2.bet_index
            AND s.session_start_date = m2.session_start_date
            AND s.strategy_name = m2.strategy_name
    LEFT JOIN user_median3 AS m3
        ON
            s.bet_index = m3.bet_index
            AND s.session_start_date = m3.session_start_date
            AND s.strategy_name = m3.strategy_name
    LEFT JOIN user_median4 AS m4
        ON
            s.bet_index = m4.bet_index
            AND s.session_start_date = m4.session_start_date
            AND s.strategy_name = m4.strategy_name
    ORDER BY s.session_start_date, s.bet_index, s.strategy_name;

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
    df_rs = redshift_loader.query_to_df("SELECT * FROM public.bullet LIMIT 10;")
    print(df_rs)
    redshift_loader.close()
