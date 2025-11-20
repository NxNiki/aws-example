from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.etl import AthenaBackend, DataLoader

query = dedent(
    f"""
    WITH UserEvents AS (
    -- Selects base event data and calculates previous/next event times for session detection
    SELECT
        t.loginname,
        t.billno,
        t.billtime,
        t.creditseq,
        t.account,       -- Required for final calculations
        t.cus_account,   -- Required for final calculations
        t.betx,          -- Required for final calculations
        t.fishcost,      -- Required for final calculations
        -- Window function to get the timestamp of the previous event for the same user
        LAG(t.billtime) OVER (
            PARTITION BY t.loginname 
            ORDER BY t.billtime, t.creditseq
        ) AS prev_event_time,
        -- Window function to get the timestamp of the next event for the same user
        LEAD(t.billtime) OVER (
            PARTITION BY t.loginname 
            ORDER BY t.billtime, t.creditseq
        ) AS next_event_time
    FROM 
        agfish.hunterorders t
    WHERE  
        -- Filter events for the specified date range (adjusting for the +8 hour offset)
        t.billtime + INTERVAL '8' HOUR >= TIMESTAMP '2025-11-01 00:00:00'
        -- Standard game-specific filters
        AND t.currency = 'CNY'
        AND t.gametype = 'HM3D'
        AND t.account != 0      -- Must have a bet amount
        AND t.fishcost != 0     -- Must have a fish cost
        -- Ensures standard order types and no weapon usage
        AND t.ordertype = 1
        AND t.remark = t.gametype
        AND t.weaponid IS NULL
    ),

    UserEventsWithSession AS (
        -- Determines session starts and assigns a unique session ID per user
        SELECT
            ue.loginname,
            ue.billno,
            ue.billtime,
            ue.account,
            ue.cus_account,
            ue.betx,
            ue.fishcost,
            ue.prev_event_time,
            ue.creditseq,
            -- Session start flag: 1 for the first event or if the gap is > 1800 seconds (0.5 hours)
            CASE
                WHEN ue.prev_event_time IS NULL THEN 1
                WHEN DATE_DIFF('second', ue.prev_event_time, ue.billtime) > 1800 THEN 1
                ELSE 0
            END AS is_session_start_flag,
            
            -- Session ID: Cumulative sum of the session start flag, partitioned by user
            SUM(
                CASE
                    WHEN ue.prev_event_time IS NULL THEN 1
                    WHEN DATE_DIFF('second', ue.prev_event_time, ue.billtime) > 1800 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY ue.loginname 
                ORDER BY ue.billtime, ue.creditseq 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_id
        FROM 
            UserEvents ue
    ),

    UserEventsWithBetIndex AS (
        -- Assigns a sequential bet index within each session and calculates session boundaries
        SELECT
            ues.loginname,
            ues.billno,
            ues.session_id,
            ues.billtime,
            ues.account,
            ues.cus_account,
            ues.betx,
            ues.fishcost,
            SUM(account) OVER (
                PARTITION BY ues.loginname, ues.session_id 
                ORDER BY ues.billtime, ues.creditseq
            ) AS cum_bet,
            SUM(account + cus_account) OVER (
                PARTITION BY ues.loginname, ues.session_id 
                ORDER BY ues.billtime, ues.creditseq
            ) AS cum_payout,
            -- Bet Index: The Nth bet within a specific session for a user
            ROW_NUMBER() OVER (
                PARTITION BY ues.loginname, ues.session_id 
                ORDER BY ues.billtime, ues.creditseq
            ) AS bet_index,
            -- Session start time
            MIN(ues.billtime) OVER (
                PARTITION BY ues.loginname, ues.session_id
            ) AS session_start_time,
            -- Session end time
            MAX(ues.billtime) OVER (
                PARTITION BY ues.loginname, ues.session_id
            ) AS session_end_time
        FROM 
            UserEventsWithSession ues
    )

    -- Final SELECT: Aggregates metrics by bet index and session start date
    SELECT
        t.bet_index,
        'PA' AS session_group, 
        COUNT(*) AS num_sessions,
        
        -- Date formatting, adjusted for the +8 hour offset
        DATE_FORMAT(t.session_start_time + INTERVAL '8' HOUR, '%Y-%m-%d') AS session_start_date,
        
        ---------------------------------------------------
        -- RETURN TO PLAYER (RTP) METRICS: (Payout / Bet) --
        ---------------------------------------------------
        AVG((t.account + t.cus_account) / t.account) AS rtp_mean,
        STDDEV((t.account + t.cus_account) / t.account) AS rtp_std,
        
        -- RTP Filtered for Fish Value (FishCost / BetX) between 20 and 200
        AVG(CASE
            WHEN t.fishcost / t.betx > 19 AND t.fishcost / t.betx < 201
                THEN (t.account + t.cus_account) / t.account
        END) AS rtp_mean_20_200,
        STDDEV(CASE
            WHEN t.fishcost / t.betx > 19 AND t.fishcost / t.betx < 201
                THEN (t.account + t.cus_account) / t.account
        END) AS rtp_std_20_200,

        AVG(t.cum_payout / t.cum_bet) AS cum_rtp_mean,
        STDDEV(t.cum_payout / t.cum_bet) AS cum_rtp_std,
        
        ---------------------------------------------------
        -- BETTING/PAYOUT METRICS --
        ---------------------------------------------------
        AVG(t.account) AS bet_mean,
        SUM(t.account) AS bet_sum,
        MAX(t.account) AS bet_max,
        MIN(t.account) AS bet_min,
        APPROX_PERCENTILE(t.account, .5) AS bet_median,

        AVG(t.account + t.cus_account) AS payout_mean,
        SUM(t.account + t.cus_account) AS payout_sum,
        MAX(t.account + t.cus_account) AS payout_max,
        MIN(t.account + t.cus_account) AS payout_min,
        APPROX_PERCENTILE(t.account + t.cus_account, .5) AS payout_median,
        
        ---------------------------------------------------
        -- GAME-SPECIFIC METRICS (Multiplier, Bullet Level, Fish Value) --
        ---------------------------------------------------
        -- Multiplier (betx)
        AVG(t.betx) AS multiplier_mean,
        SUM(t.betx) AS multiplier_sum, 
        MAX(t.betx) AS multiplier_max,
        MIN(t.betx) AS multiplier_min,

        -- Bullet Level (Bet / Multiplier) - Assuming t.account is the total bet value
        ROUND(AVG(t.account / t.betx), 3) AS bullet_level_mean,
        MAX(t.account / t.betx) AS bullet_level_max,
        MIN(t.account / t.betx) AS bullet_level_min,
        APPROX_PERCENTILE((t.account / t.betx), .5) AS bullet_level_median,
        
        -- Fish Value (Fish Cost / Multiplier)
        ROUND(AVG(t.fishcost / t.betx), 3) AS fish_value_mean,
        SUM(t.fishcost / t.betx) AS fish_value_sum,
        MAX(t.fishcost / t.betx) AS fish_value_max,
        MIN(t.fishcost / t.betx) AS fish_value_min,
        APPROX_PERCENTILE(t.fishcost / t.betx, .5) AS fish_value_median,
        
        -- RTP Max/Min
        MAX((t.account + t.cus_account) / t.account) AS rtp_max,
        MIN((t.account + t.cus_account) / t.account) AS rtp_min

    FROM 
        UserEventsWithBetIndex t
    GROUP BY 
        t.bet_index, 
        DATE_FORMAT(t.session_start_time + INTERVAL '8' HOUR, '%Y-%m-%d') -- session_start_date
    ORDER BY 
        DATE_FORMAT(t.session_start_time + INTERVAL '8' HOUR, '%Y-%m-%d'),
        t.bet_index;

    """
)

if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename="game_stats_by_bet_etl_pa.log")

    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-fish_hunter/bullet_stats_by_index_pa",
        )
    )

    file_path = f"{LOCAL_ROOT}/jobs/output_fish_hunter/bullet_stats_by_index_pa.parquet"
    df_rs = data_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)
    data_loader.close()
