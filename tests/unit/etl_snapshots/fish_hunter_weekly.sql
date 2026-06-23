-- 1. FETCH RAW DATA (Keep strictly RAW columns to enable Index Scans)
WITH base_data AS (
    SELECT
        b.user_id,
        b.room_id,
        b.bullet_id,
        b.strategy_name, -- Keep Raw
        b.event_timestamp AS bet_time,
        b.payout,
        b.bet,
        b.fish_value,
        CASE
            WHEN b.fish_value <= 10 THEN 'low'
            WHEN b.fish_value <= 130 THEN 'medium'
            WHEN b.fish_value <= 200 THEN 'high'
            ELSE 'ultra'
        END as fish_type,
        b.killed,
        b.profit,
        LAG(b.event_timestamp) OVER (PARTITION BY b.user_id ORDER BY b.bullet_id, b.event_timestamp) AS prev_bet_time,
        LAG(b.bet) OVER (PARTITION BY b.user_id ORDER BY b.bullet_id, b.event_timestamp) AS prev_bet_amount,
        CAST(DATE_TRUNC('day', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS DATE) AS activity_date,
        CAST(DATE_TRUNC('week', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS DATE) AS activity_week,
        CAST(DATE_TRUNC('month', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', b.created_at))) AS DATE) AS activity_month
    FROM public.bullet b
    WHERE
        b.currency_type IN ('CNY')
        AND b.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')

        -- ---------------------------------------------------------
        -- FAST FILTERING: Transform the INPUTS, not the COLUMN
        -- ---------------------------------------------------------

        -- 1. Reverse the date math for START (use effective_start so monthly/weekly get full period)
        -- Logic: We want events where (EventTime + UserDays) >= Start
        -- So: EventTime >= Start - UserDays
        AND b.created_at >= CONVERT_TIMEZONE('Asia/Shanghai', 'UTC',
               DATEADD(day, -30, CAST('2024-12-30' AS TIMESTAMP)))

),

-- 2. DETERMINE USER DAILY GROUP (Logic applied inside SUM)
user_daily_group AS (
    SELECT
        user_id,
        activity_date,
        activity_week,
        activity_month,
        -- Assign the whole user-day to its highest-priority strategy_name.
        -- A single bet in a higher tier claims the day. See generate_query docstring.
        CASE
            WHEN MAX(CASE WHEN strategy_name = 'RISK_CONTROLLED' THEN 1 ELSE 0 END) > 0 THEN 'RISK_CONTROLLED'
            WHEN MAX(CASE WHEN strategy_name = 'BOOST_POOL' THEN 1 ELSE 0 END) > 0 THEN 'BOOST_POOL'
            WHEN MAX(CASE WHEN strategy_name IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3') THEN 1 ELSE 0 END) > 0
                THEN MIN(CASE WHEN strategy_name IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3') THEN strategy_name END)
            ELSE 'DEFAULT_FALLBACK'
        END AS daily_group,
        LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS bj_date_last_bet
    FROM base_data
    GROUP BY user_id, activity_date, activity_week, activity_month
),

-- GET KILL STREAK LENGTH:
base_kills AS (
    SELECT 
        user_id, 
        activity_date, 
        fish_type, 
        bullet_id,
        bet_time
    FROM base_data 
    WHERE killed = 1
),

calculate_islands AS (
    SELECT
        user_id,
        activity_date,
        fish_type,
        bullet_id,
        bet_time,
        -- GLOBAL: ignores fish_type. Checks if ANY fish was killed recently.
        CASE 
            WHEN DATEDIFF(SECOND, LAG(bet_time) OVER(PARTITION BY user_id ORDER BY bullet_id, bet_time), bet_time) > 3 
                OR LAG(bet_time) OVER(PARTITION BY user_id ORDER BY bullet_id, bet_time) IS NULL 
            THEN 1 ELSE 0 
        END AS is_new_global_streak,

        -- TYPE-SPECIFIC: isolated by fish_type. Only checks previous kill of SAME type.
        CASE 
            WHEN DATEDIFF(SECOND, LAG(bet_time) OVER(PARTITION BY user_id, fish_type ORDER BY bullet_id, bet_time), bet_time) > 3 
                OR LAG(bet_time) OVER(PARTITION BY user_id, fish_type ORDER BY bullet_id, bet_time) IS NULL 
            THEN 1 ELSE 0 
        END AS is_new_type_streak
    FROM base_kills
),

streak_ids AS (
    SELECT
        user_id,
        activity_date,
        fish_type,
        bet_time,
        SUM(is_new_global_streak) OVER(PARTITION BY user_id ORDER BY bullet_id, bet_time ROWS UNBOUNDED PRECEDING) AS global_streak_id,
        SUM(is_new_type_streak) OVER(PARTITION BY user_id, fish_type ORDER BY bullet_id, bet_time ROWS UNBOUNDED PRECEDING) AS type_streak_id
    FROM calculate_islands
),

streak_lengths AS (
    SELECT 
        user_id, 
        activity_date,
        fish_type,
        -- Calculate the length of the specific streak instance this row belongs to
        COUNT(*) OVER(PARTITION BY user_id, global_streak_id) AS global_streak_len,
        COUNT(*) OVER(PARTITION BY user_id, fish_type, type_streak_id) AS type_streak_len
    FROM streak_ids
),

max_kill_streak_length AS (
    SELECT 
        user_id,
        activity_date,
        MAX(global_streak_len) AS max_kill_streak,
        MAX(CASE WHEN fish_type = 'low' THEN type_streak_len END) AS max_kill_streak_low,
        MAX(CASE WHEN fish_type = 'medium' THEN type_streak_len END) AS max_kill_streak_medium,
        MAX(CASE WHEN fish_type = 'high' THEN type_streak_len END) AS max_kill_streak_high,
        MAX(CASE WHEN fish_type = 'ultra' THEN type_streak_len END) AS max_kill_streak_ultra,

        AVG(global_streak_len) AS avg_kill_streak,
        AVG(CASE WHEN fish_type = 'low' THEN type_streak_len END) AS avg_kill_streak_low,
        AVG(CASE WHEN fish_type = 'medium' THEN type_streak_len END) AS avg_kill_streak_medium,
        AVG(CASE WHEN fish_type = 'high' THEN type_streak_len END) AS avg_kill_streak_high,
        AVG(CASE WHEN fish_type = 'ultra' THEN type_streak_len END) AS avg_kill_streak_ultra
    FROM streak_lengths
    GROUP BY user_id, activity_date
),

-- GET BET SESSION STATS:
user_session_id AS (
    SELECT
        t.activity_date,
        t.user_id,
        t.bet_time,
        t.killed,
        t.fish_type,
        SUM(CASE 
                WHEN DATEDIFF(SECOND, t.prev_bet_time, t.bet_time) < 600 THEN 0 
                ELSE 1 
            END) OVER (
                PARTITION BY t.activity_date, t.user_id 
                ORDER BY t.bullet_id, t.bet_time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_id,
        ROW_NUMBER() OVER (
                PARTITION BY t.activity_date, t.user_id 
                ORDER BY t.bullet_id, t.bet_time 
            ) AS bet_index

    FROM base_data t
),

user_session_length AS (
    SELECT
        t.activity_date,
        t.user_id,
        t.session_id,

        DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 THEN t.bet_time END)) AS seconds_to_kill_fish,
        DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'low' THEN t.bet_time END)) AS seconds_to_kill_fish_low,
        DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'medium' THEN t.bet_time END)) AS seconds_to_kill_fish_medium,
        DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'high' THEN t.bet_time END)) AS seconds_to_kill_fish_high,
        DATEDIFF(SECOND, MIN(t.bet_time), MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'ultra' THEN t.bet_time END)) AS seconds_to_kill_fish_ultra,

        MIN(CASE WHEN t.killed = 1 THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish,
        MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'low' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_low,
        MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'medium' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_medium,
        MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'high' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_high,
        MIN(CASE WHEN t.killed = 1 AND t.fish_type = 'ultra' THEN t.bet_index END) - MIN(bet_index) AS bets_to_kill_fish_ultra,

        COUNT(t.user_id) AS session_length
    FROM user_session_id t
    GROUP BY t.user_id, t.activity_date, t.session_id
),

user_session_stats AS (
    SELECT
        t.activity_date,
        t.user_id,
        COUNT(DISTINCT t.session_id) AS num_streak_sessions,
        AVG(CAST(t.session_length AS FLOAT)) AS avg_streak_length,
        MAX(t.session_length) AS max_streak_length,
        MIN(t.session_length) AS min_streak_length,

        AVG(seconds_to_kill_fish) AS seconds_to_kill_fish,
        AVG(seconds_to_kill_fish_low) AS seconds_to_kill_fish_low,
        AVG(seconds_to_kill_fish_medium) AS seconds_to_kill_fish_medium,
        AVG(seconds_to_kill_fish_high) AS seconds_to_kill_fish_high,
        AVG(seconds_to_kill_fish_ultra) AS seconds_to_kill_fish_ultra,

        AVG(bets_to_kill_fish) AS bets_to_kill_fish,
        AVG(bets_to_kill_fish_low) AS bets_to_kill_fish_low,
        AVG(bets_to_kill_fish_medium) AS bets_to_kill_fish_medium,
        AVG(bets_to_kill_fish_high) AS bets_to_kill_fish_high,
        AVG(bets_to_kill_fish_ultra) AS bets_to_kill_fish_ultra

    FROM user_session_length t
    GROUP BY t.user_id, t.activity_date
),

user_session_stats_agg AS (
    SELECT
        u.daily_group,
        t.user_id,
        u.activity_week,
        SUM(t.num_streak_sessions)      AS user_num_streak_sessions,
        AVG(t.avg_streak_length)        AS user_avg_streak_length,
        MAX(t.max_streak_length)        AS user_max_streak_length,
        MIN(t.min_streak_length)        AS user_min_streak_length,

        AVG(t.seconds_to_kill_fish)         AS user_seconds_to_kill_fish,
        AVG(t.seconds_to_kill_fish_low)     AS user_seconds_to_kill_fish_low,
        AVG(t.seconds_to_kill_fish_medium)  AS user_seconds_to_kill_fish_medium,
        AVG(t.seconds_to_kill_fish_high)    AS user_seconds_to_kill_fish_high,
        AVG(t.seconds_to_kill_fish_ultra)   AS user_seconds_to_kill_fish_ultra,

        AVG(t.bets_to_kill_fish)        AS user_bets_to_kill_fish,
        AVG(t.bets_to_kill_fish_low)    AS user_bets_to_kill_fish_low,
        AVG(t.bets_to_kill_fish_medium) AS user_bets_to_kill_fish_medium,
        AVG(t.bets_to_kill_fish_high)   AS user_bets_to_kill_fish_high,
        AVG(t.bets_to_kill_fish_ultra)  AS user_bets_to_kill_fish_ultra
    FROM user_session_stats t
    JOIN user_daily_group u ON t.user_id = u.user_id AND t.activity_date = u.activity_date
    GROUP BY u.daily_group, t.user_id, u.activity_week
),

max_kill_streak_length_agg AS (
    SELECT
        u.daily_group,
        t.user_id,
        u.activity_week,
        MAX(t.max_kill_streak)          AS user_max_kill_streak,
        MAX(t.max_kill_streak_low)      AS user_max_kill_streak_low,
        MAX(t.max_kill_streak_medium)   AS user_max_kill_streak_medium,
        MAX(t.max_kill_streak_high)     AS user_max_kill_streak_high,
        MAX(t.max_kill_streak_ultra)    AS user_max_kill_streak_ultra,

        AVG(t.avg_kill_streak)          AS user_avg_kill_streak,
        AVG(t.avg_kill_streak_low)      AS user_avg_kill_streak_low,
        AVG(t.avg_kill_streak_medium)   AS user_avg_kill_streak_medium,
        AVG(t.avg_kill_streak_high)     AS user_avg_kill_streak_high,
        AVG(t.avg_kill_streak_ultra)    AS user_avg_kill_streak_ultra
    FROM max_kill_streak_length t
    JOIN user_daily_group u ON t.user_id = u.user_id AND t.activity_date = u.activity_date
    GROUP BY u.daily_group, t.user_id, u.activity_week
),

-- 4. USER-LEVEL STATS BY ASSIGNED DAILY GROUP
stats_by_user_date AS (
    SELECT
        b.user_id,
        u.daily_group,
        b.activity_week,
        COUNT(DISTINCT b.user_id || '-' || b.room_id) AS user_num_rooms,
        COUNT(b.user_id)                              AS user_num_bets,

        -- Bullets info by fish type:
        SUM(CASE WHEN b.fish_type = 'low'    THEN 1 END) AS user_num_hits_fish_low,
        SUM(CASE WHEN b.fish_type = 'medium' THEN 1 END) AS user_num_hits_fish_medium,
        SUM(CASE WHEN b.fish_type = 'high'   THEN 1 END) AS user_num_hits_fish_high,
        SUM(CASE WHEN b.fish_type = 'ultra'  THEN 1 END) AS user_num_hits_fish_ultra,

        -- Killed fish info by fish type:
        SUM(b.killed)                                                          AS user_num_killed_bullets,
        SUM(CASE WHEN b.fish_type = 'low'    THEN b.killed END)               AS user_num_killed_fish_low,
        SUM(CASE WHEN b.fish_type = 'medium' THEN b.killed END)               AS user_num_killed_fish_medium,
        SUM(CASE WHEN b.fish_type = 'high'   THEN b.killed END)               AS user_num_killed_fish_high,
        SUM(CASE WHEN b.fish_type = 'ultra'  THEN b.killed END)               AS user_num_killed_fish_ultra,

        SUM(b.bet)                                                             AS user_total_bet,
        AVG(b.bet)                                                             AS user_avg_bet_amount,
        AVG(CASE WHEN EXTRACT(EPOCH FROM (b.bet_time - b.prev_bet_time)) <= 1800 THEN GREATEST(EXTRACT(EPOCH FROM (b.bet_time - b.prev_bet_time)), 0.25) END)
                                                                               AS user_avg_delta_t_seconds,
        SUM(b.payout)                                                          AS user_total_payout,
        SUM(b.profit)                                                          AS user_total_profit,
        MAX(b.profit)                                                          AS user_max_profit,
        ROUND(CAST(SUM(b.payout) AS FLOAT) / NULLIF(SUM(b.bet), 0), 3)        AS user_rtp,
        STDDEV(b.profit) / NULLIF(ABS(AVG(b.profit)), 0)                      AS user_profit_coef_var,
        MAX(CASE WHEN b.killed >= 1 THEN 1 ELSE 0 END)                        AS user_killed_fish,

        AVG(b.fish_value)                                                      AS user_avg_fish_value,
        AVG(CASE WHEN b.fish_value > 19 AND b.fish_value < 201 THEN b.fish_value END)                          AS user_avg_fish_value_20_200,
        AVG(CASE WHEN b.killed = 1 THEN b.fish_value END)                     AS user_avg_killed_fish_value,
        AVG(CASE WHEN b.fish_value > 19 AND b.fish_value < 201 AND b.killed = 1 THEN b.fish_value END)         AS user_avg_killed_fish_value_20_200,
        AVG(b.profit)                                                          AS user_bullet_avg_profit,
        AVG(CASE WHEN b.killed = 1 THEN b.profit END)                         AS user_bullet_kill_avg_profit,

        -- delta bet amount metrics:
        SUM(CASE WHEN (b.bet - b.prev_bet_amount) > 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_pos_delta_bet,
        SUM(CASE WHEN (b.bet - b.prev_bet_amount) < 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_neg_delta_bet,
        AVG(CASE WHEN (b.bet - b.prev_bet_amount) > 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_pos_delta_bet_avg,
        AVG(CASE WHEN (b.bet - b.prev_bet_amount) < 0 THEN (b.bet - b.prev_bet_amount) END) AS user_accu_neg_delta_bet_avg,
        SUM(b.bet - b.prev_bet_amount) AS user_accu_delta_bet,
        AVG(b.bet - b.prev_bet_amount) AS user_accu_delta_bet_avg,
        COUNT(CASE WHEN (b.bet - b.prev_bet_amount) > 0 THEN 1 END) AS user_pos_delta_bet_num,
        COUNT(CASE WHEN (b.bet - b.prev_bet_amount) < 0 THEN 1 END) AS user_neg_delta_bet_num
    FROM base_data b
    JOIN user_daily_group u ON b.user_id = u.user_id AND b.activity_date = u.activity_date
    GROUP BY b.user_id, u.daily_group, b.activity_week
    -- Include ALL betting users, not only fish-killers. Kill-specific metrics
    -- already degrade to NULL/0 for non-killers (CASE WHEN killed / NULLIF), while
    -- downstream user counts & retention (day0_num_users, num_active_users, …) need
    -- the full active-user set; a `HAVING MAX(b.killed) > 0` here undercounted them.
    -- Segment to killers downstream via the user_killed_fish flag when needed.
),

user_first_bet AS (
    SELECT
        user_id,
        MIN(CAST(DATE_TRUNC('day', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', created_at))) AS DATE)) AS first_bet_date
    FROM public.bullet
    WHERE currency_type IN ('CNY')
      AND op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
    GROUP BY user_id
)

-- 5. FINAL JOIN & FORMATTING
SELECT
    t1.user_id,
    t1.daily_group,
    t1.activity_week AS activity_date,
    CASE
        WHEN DATEDIFF('day', fb.first_bet_date, t1.activity_week) <= 3 THEN 'new'
        WHEN DATEDIFF('day', fb.first_bet_date, t1.activity_week) <= 7 THEN 'beginner'
        ELSE 'old'
    END AS user_group,

    -- Core user-level counts (user_ prefix = one row per user per period):
    t1.user_num_rooms,
    t1.user_num_bets,
    t1.user_num_killed_bullets,
    t1.user_killed_fish,

    -- Bullet/hit counts by fish type:
    t1.user_num_hits_fish_low,
    t1.user_num_hits_fish_medium,
    t1.user_num_hits_fish_high,
    t1.user_num_hits_fish_ultra,
    t1.user_num_hits_fish_low    * 1.0 / NULLIF(t1.user_num_bets, 0) AS user_hits_fish_low_ratio,
    t1.user_num_hits_fish_medium * 1.0 / NULLIF(t1.user_num_bets, 0) AS user_hits_fish_medium_ratio,
    t1.user_num_hits_fish_high   * 1.0 / NULLIF(t1.user_num_bets, 0) AS user_hits_fish_high_ratio,
    t1.user_num_hits_fish_ultra  * 1.0 / NULLIF(t1.user_num_bets, 0) AS user_hits_fish_ultra_ratio,

    -- Kill counts by fish type:
    t1.user_num_killed_fish_low,
    t1.user_num_killed_fish_medium,
    t1.user_num_killed_fish_high,
    t1.user_num_killed_fish_ultra,
    t1.user_num_killed_fish_low    * 1.0 / NULLIF(t1.user_num_killed_bullets, 0) AS user_killed_fish_low_ratio,
    t1.user_num_killed_fish_medium * 1.0 / NULLIF(t1.user_num_killed_bullets, 0) AS user_killed_fish_medium_ratio,
    t1.user_num_killed_fish_high   * 1.0 / NULLIF(t1.user_num_killed_bullets, 0) AS user_killed_fish_high_ratio,
    t1.user_num_killed_fish_ultra  * 1.0 / NULLIF(t1.user_num_killed_bullets, 0) AS user_killed_fish_ultra_ratio,

    -- Bet / payout / profit:
    t1.user_total_bet,
    t1.user_avg_bet_amount,
    t1.user_avg_delta_t_seconds,
    t1.user_total_payout,
    t1.user_total_profit,
    t1.user_max_profit,
    t1.user_rtp,
    t1.user_profit_coef_var,
    ROUND(CAST(t1.user_num_killed_bullets AS FLOAT) / NULLIF(t1.user_num_bets, 0), 3) AS user_bullet_kill_ratio,

    -- Fish value:
    t1.user_avg_fish_value,
    t1.user_avg_fish_value_20_200,
    t1.user_avg_killed_fish_value,
    t1.user_avg_killed_fish_value_20_200,
    t1.user_bullet_avg_profit,
    t1.user_bullet_kill_avg_profit,

    -- delta bet amount metrics:
    t1.user_accu_pos_delta_bet,
    t1.user_accu_neg_delta_bet,
    t1.user_accu_pos_delta_bet_avg,
    t1.user_accu_neg_delta_bet_avg,
    t1.user_accu_delta_bet,
    t1.user_accu_delta_bet_avg,
    t1.user_pos_delta_bet_num,
    t1.user_neg_delta_bet_num,

    -- Session stats:
    t4.user_num_streak_sessions,
    t4.user_avg_streak_length,
    t4.user_max_streak_length,
    t4.user_min_streak_length,
    t4.user_seconds_to_kill_fish,
    t4.user_seconds_to_kill_fish_low,
    t4.user_seconds_to_kill_fish_medium,
    t4.user_seconds_to_kill_fish_high,
    t4.user_seconds_to_kill_fish_ultra,
    t4.user_bets_to_kill_fish,
    t4.user_bets_to_kill_fish_low,
    t4.user_bets_to_kill_fish_medium,
    t4.user_bets_to_kill_fish_high,
    t4.user_bets_to_kill_fish_ultra,

    -- Kill streak stats:
    t5.user_max_kill_streak,
    t5.user_max_kill_streak_low,
    t5.user_max_kill_streak_medium,
    t5.user_max_kill_streak_high,
    t5.user_max_kill_streak_ultra,
    t5.user_avg_kill_streak,
    t5.user_avg_kill_streak_low,
    t5.user_avg_kill_streak_medium,
    t5.user_avg_kill_streak_high,
    t5.user_avg_kill_streak_ultra
FROM stats_by_user_date t1
LEFT JOIN user_first_bet AS fb ON t1.user_id = fb.user_id
LEFT JOIN user_session_stats_agg t4
    ON t1.user_id = t4.user_id
    AND t1.activity_week = t4.activity_week
    AND t1.daily_group = t4.daily_group
LEFT JOIN max_kill_streak_length_agg t5
    ON t1.user_id = t5.user_id
    AND t1.activity_week = t5.activity_week
    AND t1.daily_group = t5.daily_group
WHERE t1.activity_week >= '2024-12-30'
ORDER BY t1.activity_week, t1.daily_group
;
