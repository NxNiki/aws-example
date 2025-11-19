from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

GROUP_COL = "session_group"

query = dedent(
    f"""
    WITH base_filtered_events AS (
        SELECT
            user_id,
            event_id,
            strategy_name,
            event_timestamp,
            payout,
            bet,
            bullet_level,
            fish_value,
            multiplier,
            -- Convert to Analysis Timezone (e.g., UTC+8) once here
            TO_CHAR(event_timestamp + INTERVAL '8 hours', 'YYYY-MM-DD') AS session_date_str,
            LAG(event_timestamp, 1) OVER (
                PARTITION BY user_id ORDER BY event_timestamp
            ) AS prev_event_time
        FROM public.bullet
        WHERE
            currency_type = 'CNY'
            AND event_timestamp + INTERVAL '8 hours' >= '2025-11-01'
            AND event_timestamp + INTERVAL '8 hours' < '2025-12-01'
    ),

    session_identification AS (
        SELECT
            *,
            -- Session Flag Logic
            CASE
                WHEN prev_event_time IS NULL THEN 1
                WHEN DATEDIFF(SECOND, prev_event_time, event_timestamp) > 1800 THEN 1
                ELSE 0
            END AS is_session_start,
            -- Create Session ID
            SUM(
                CASE
                    WHEN prev_event_time IS NULL THEN 1
                    WHEN DATEDIFF(SECOND, prev_event_time, event_timestamp) > 1800 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY user_id
                ORDER BY event_timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_id
        FROM base_filtered_events
    ),

    session_enriched AS (
        SELECT
            user_id,
            event_id,
            session_id,
            session_date_str,
            strategy_name,
            payout,
            bet,
            bullet_level,
            fish_value,
            multiplier,
            -- Window Metrics
            SUM(payout) OVER (PARTITION BY user_id, session_id ORDER BY event_timestamp ROWS UNBOUNDED PRECEDING) AS cum_payout,
            SUM(bet) OVER (PARTITION BY user_id, session_id ORDER BY event_timestamp ROWS UNBOUNDED PRECEDING) AS cum_bet,
            ROW_NUMBER() OVER (PARTITION BY user_id, session_id ORDER BY event_timestamp) AS bet_index,
            MIN(event_timestamp) OVER (PARTITION BY user_id, session_id) AS session_start_time
        FROM session_identification
    ),

    session_grouping AS (
        SELECT
            user_id,
            session_id,
            CASE
                WHEN COUNT(CASE WHEN strategy_name = 'BOOST_POOL' THEN 1 END) > 0 THEN 'BOOST'
                WHEN COUNT(CASE WHEN strategy_name = 'DYNAMIC_RTP' THEN 1 END) > 0 THEN 'DYNA_RTP'
                ELSE 'DEFAULT'
            END AS session_group
        FROM session_enriched
        GROUP BY 1, 2
    ),

    -- THE KEY OPTIMIZATION: 
    -- Consolidate everything into one filtered dataset for aggregation.
    -- We never touch public.bullet again after this.
    analysis_base AS (
        SELECT
            e.*,
            g.session_group
        FROM session_enriched e
        INNER JOIN session_grouping g
            ON e.user_id = g.user_id AND e.session_id = g.session_id
    ),

    user_stats AS (
        SELECT
            session_date_str AS session_start_date,
            bet_index,
            {GROUP_COL},
            COUNT(*) AS num_sessions,
            
            -- Basic Averages
            AVG(payout / NULLIF(bet, 0)) AS rtp_mean,
            STDDEV(payout / NULLIF(bet, 0)) AS rtp_std,
            AVG(cum_payout / NULLIF(cum_bet, 0)) AS cum_rtp,
            
            -- Conditional Aggregates (20-200 range)
            AVG(CASE WHEN fish_value > 19 AND fish_value < 201 THEN payout / NULLIF(bet, 0) END) AS rtp_mean_20_200,
            STDDEV(CASE WHEN fish_value > 19 AND fish_value < 201 THEN payout / NULLIF(bet, 0) END) AS rtp_std_20_200,

            -- Raw Metrics
            AVG(multiplier) AS multiplier_mean,
            ROUND(AVG(CAST(bullet_level AS FLOAT)), 3) AS bullet_level_mean,
            ROUND(AVG(CAST(fish_value AS FLOAT)), 3) AS fish_value_mean,
            AVG(bet) AS bet_mean,
            AVG(payout) AS payout_mean,

            -- Sums
            SUM(bullet_level) AS bullet_level_sum,
            SUM(fish_value) AS fish_value_sum,
            SUM(bet) AS bet_sum,
            SUM(payout) AS payout_sum,

            -- Max/Min
            MAX(multiplier) AS multiplier_max,
            MAX(payout / NULLIF(bet, 0)) AS rtp_max,
            MAX(bullet_level) AS bullet_level_max,
            MAX(fish_value) AS fish_value_max,
            MAX(bet) AS bet_max,
            MAX(payout) AS payout_max,
            
            MIN(multiplier) AS multiplier_min,
            MIN(payout / NULLIF(bet, 0)) AS rtp_min,
            MIN(bullet_level) AS bullet_level_min,
            MIN(fish_value) AS fish_value_min,
            MIN(bet) AS bet_min,
            MIN(payout) AS payout_min

        FROM analysis_base
        GROUP BY 1, 2, 3
    ),

    -- Median calculations now query 'analysis_base', NOT 'public.bullet'
    user_median1 AS (
        SELECT session_date_str, bet_index, {GROUP_COL},
               MEDIAN(bullet_level) AS bullet_level_median
        FROM analysis_base GROUP BY 1, 2, 3
    ),
    user_median2 AS (
        SELECT session_date_str, bet_index, {GROUP_COL},
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fish_value) AS fish_value_median
        FROM analysis_base GROUP BY 1, 2, 3
    ),
    user_median3 AS (
        SELECT session_date_str, bet_index, {GROUP_COL},
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bet) AS bet_median
        FROM analysis_base GROUP BY 1, 2, 3
    ),
    user_median4 AS (
        SELECT session_date_str, bet_index, {GROUP_COL},
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY payout) AS payout_median
        FROM analysis_base GROUP BY 1, 2, 3
    )

    SELECT
        s.*,
        m1.bullet_level_median,
        m2.fish_value_median,
        m3.bet_median,
        m4.payout_median
    FROM user_stats s
    LEFT JOIN user_median1 m1 ON s.bet_index = m1.bet_index AND s.session_start_date = m1.session_date_str AND s.{GROUP_COL} = m1.{GROUP_COL}
    LEFT JOIN user_median2 m2 ON s.bet_index = m2.bet_index AND s.session_start_date = m2.session_date_str AND s.{GROUP_COL} = m2.{GROUP_COL}
    LEFT JOIN user_median3 m3 ON s.bet_index = m3.bet_index AND s.session_start_date = m3.session_date_str AND s.{GROUP_COL} = m3.{GROUP_COL}
    LEFT JOIN user_median4 m4 ON s.bet_index = m4.bet_index AND s.session_start_date = m4.session_date_str AND s.{GROUP_COL} = m4.{GROUP_COL}
    ORDER BY s.session_start_date, s.bet_index, s.{GROUP_COL};
    """
)

if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename="game_stats_by_bet_etl_pa.log")

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
