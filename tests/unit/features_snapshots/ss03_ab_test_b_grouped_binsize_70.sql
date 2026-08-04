WITH user_bets AS (
    SELECT
        t.spin_id,
        t.user_id,
        t.math_table_id,
        t.created_at,
        t.bet_type,
        -- ignore bet amount in free game (0). This will influence avg and std stats
        CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END AS bet_amount,
        t.actual_payout AS payout,
        t.balance_after_bet,
        t.balance_after_payout,
        LAG(t.created_at, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_time,
        CASE
            WHEN t.bet_type = 'BASE' THEN 1
            ELSE 0
        END AS is_new_game_group,
        CASE
            WHEN t.partition_ab[0] = 'jojpin-9mokha-rexQug' THEN 'AI'
            WHEN t.partition_ab[0] = '4f1a46ca-7baa-4452-9a40-ef21d9b33b57' THEN 'AB_TEST_A'
            WHEN t.partition_ab[0] = '4a04df21-c749-4808-8e55-3a0b74c084d2' THEN 'AB_TEST_B'
            ELSE 'Default'
        END AS ai_group
    FROM public.fct_bet_orders AS t
    WHERE
        t.created_at >= '2026-06-10'
        AND t.created_at < '2026-07-01'
        AND t.currency_type IN ('CNY')
        AND t.status = 'COMPLETED'
        AND t.game_id = 'SS03'
        AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
        AND (t.partition_ab[0] = '4a04df21-c749-4808-8e55-3a0b74c084d2')
),

free_game_group AS (
    SELECT
        t.spin_id,
        t.user_id,
        t.ai_group,
        t.math_table_id,
        t.created_at,
        t.bet_type,
        t.bet_amount,
        t.payout,
        t.balance_after_bet,
        t.balance_after_payout,
        t.prev_bet_time,
        SUM(t.is_new_game_group) OVER (
            PARTITION BY t.user_id
            ORDER BY t.spin_id, t.created_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS fg_group
    FROM user_bets AS t
),

agg_free_game AS (
    SELECT
        t.user_id,
        t.ai_group,
        t.math_table_id,
        MIN(t.spin_id) AS spin_id,
        MIN(t.created_at) AS min_created_at,
        MAX(t.created_at) AS max_created_at,
        SUM(CASE WHEN t.bet_type = 'FREE' THEN 1 ELSE 0 END) AS fg_rounds,
        MIN(t.prev_bet_time) AS prev_bet_time,
        SUM(t.bet_amount) AS bet_amount,
        SUM(t.payout) AS payout,
        MIN(t.balance_after_bet) AS balance_after_bet,
        MAX(t.balance_after_payout) AS balance_after_payout
    FROM free_game_group AS t
    GROUP BY t.user_id, t.ai_group, t.math_table_id, t.fg_group
),

delta_stats AS (
    SELECT
        t.user_id,
        t.ai_group,
        t.math_table_id,
        t.spin_id,
        t.min_created_at,
        t.max_created_at,
        t.fg_rounds,
        t.bet_amount,
        t.payout,
        t.balance_after_bet,
        DATEDIFF(SECONDS, t.prev_bet_time, t.min_created_at) AS delta_t_seconds,
        t.bet_amount - LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_bet_amount,
        t.payout - LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_payout,
        t.balance_after_bet
            + t.bet_amount
            - LAG(t.balance_after_payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS balance_transaction,
        CASE WHEN t.payout > t.bet_amount THEN 1 ELSE 0 END AS is_win,
        CASE WHEN t.payout < t.bet_amount THEN 1 ELSE 0 END AS is_lose,
        CASE
            WHEN LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) > LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                THEN 1
            ELSE 0
        END AS prev_win,
        CASE
            WHEN LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) < LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                THEN 1
            ELSE 0
        END AS prev_lose
    FROM agg_free_game AS t
),

user_group AS (
    SELECT
        t.user_id,
        t.ai_group,
        t.math_table_id,
        t.spin_id,
        t.min_created_at,
        t.max_created_at,
        t.fg_rounds,
        t.delta_t_seconds,
        t.bet_amount,
        t.delta_bet_amount,
        t.payout,
        t.delta_payout,
        t.balance_after_bet,
        t.balance_transaction,
        t.is_win,
        t.is_lose,
        t.prev_win,
        t.prev_lose,
        LAST_VALUE(
            CASE
                WHEN t.delta_t_seconds > 43200 OR t.delta_t_seconds IS NULL
                    THEN t.min_created_at
            END
            IGNORE NULLS
        )
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS session_start_ts,
        SUM(CASE WHEN t.delta_t_seconds <= 200 THEN 0 ELSE 1 END)
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS streak_group,
        SUM(
            CASE
                WHEN t.delta_t_seconds <= 200 AND t.prev_win = 1 AND t.is_win = 1 THEN 0
                ELSE 1
            END
        )
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS win_streak_group,
        SUM(
            CASE
                WHEN t.delta_t_seconds <= 200 AND t.prev_lose = 1 AND t.is_lose = 1 THEN 0
                ELSE 1
            END
        )
            OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS lose_streak_group
    FROM delta_stats AS t
),

raw_stats AS (
    SELECT
        t.user_id,
        t.ai_group,
        t.math_table_id,
        CAST(t.session_start_ts AS DATE) AS session_start_date,
        DENSE_RANK() OVER (
            PARTITION BY t.user_id, CAST(t.session_start_ts AS DATE)
            ORDER BY t.session_start_ts
        ) - 1 AS session_group,
        t.session_start_ts,
        t.min_created_at,
        t.max_created_at,
        CAST(t.min_created_at AS DATE) AS activity_date,
        t.spin_id,
        t.fg_rounds,
        t.bet_amount,
        t.delta_bet_amount,
        t.payout,
        t.delta_payout,
        t.balance_after_bet,
        t.delta_t_seconds,
        -- ignore delta t larger than the configured gap threshold (default 1 hour).
        CASE WHEN t.delta_t_seconds <= 3600 THEN t.delta_t_seconds END AS delta_t_seconds_nogap,
        t.payout - t.bet_amount AS profit,
        CASE WHEN t.balance_transaction > .1 THEN t.balance_transaction END AS deposit,
        CASE WHEN t.balance_transaction < -.1 THEN -t.balance_transaction END AS withdraw,
        ROW_NUMBER() OVER (PARTITION BY t.user_id, t.streak_group ORDER BY t.spin_id, t.min_created_at) AS streak,
        CASE
            WHEN t.payout > t.bet_amount
                THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.win_streak_group ORDER BY t.spin_id, t.min_created_at)
        END AS win_streak,
        CASE
            WHEN t.payout < t.bet_amount
                THEN ROW_NUMBER() OVER (PARTITION BY t.user_id, t.lose_streak_group ORDER BY t.spin_id, t.min_created_at)
        END AS lose_streak,
        ROW_NUMBER() OVER (PARTITION BY t.user_id, t.session_start_ts ORDER BY t.spin_id, t.min_created_at)
            AS session_bet_index
    FROM user_group AS t
),

binned AS (
    SELECT
        t.*,
        ROUND((t.session_bet_index - 1) / 70) AS agg_group
    FROM raw_stats AS t
),

perc_delta_t AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_t_seconds) AS delta_t_seconds_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_t_seconds) AS delta_t_seconds_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_t_seconds) AS delta_t_seconds_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_delta_t_ng AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_t_seconds_nogap) AS delta_t_seconds_nogap_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_t_seconds_nogap) AS delta_t_seconds_nogap_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_t_seconds_nogap) AS delta_t_seconds_nogap_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_bet_amount AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bet_amount) AS bet_amount_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY bet_amount) AS bet_amount_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bet_amount) AS bet_amount_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_delta_bet AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_bet_amount) AS delta_bet_amount_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_bet_amount) AS delta_bet_amount_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_bet_amount) AS delta_bet_amount_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_payout AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY payout) AS payout_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY payout) AS payout_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY payout) AS payout_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_rtp AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY payout * 1.0 / NULLIF(bet_amount, 0)) AS rtp_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_profit AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY profit) AS profit_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY profit) AS profit_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY profit) AS profit_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_delta_payout AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delta_payout) AS delta_payout_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY delta_payout) AS delta_payout_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delta_payout) AS delta_payout_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_balance AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY balance_after_bet) AS balance_after_bet_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY balance_after_bet) AS balance_after_bet_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY balance_after_bet) AS balance_after_bet_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_streak AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY streak) AS streak_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY streak) AS streak_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY streak) AS streak_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_win_streak AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY win_streak) AS win_streak_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY win_streak) AS win_streak_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY win_streak) AS win_streak_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),
perc_lose_streak AS (
    SELECT
        user_id, ai_group, math_table_id, session_start_date, session_group, agg_group,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY lose_streak) AS lose_streak_p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY lose_streak) AS lose_streak_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY lose_streak) AS lose_streak_p75
    FROM binned
    GROUP BY user_id, ai_group, math_table_id, session_start_date, session_group, agg_group
),

stats_base AS (
    SELECT
        t.user_id,
        t.ai_group,
        t.math_table_id,
        t.session_start_date,
        t.session_group,
        t.agg_group,
        MIN(t.activity_date) AS activity_date,
        MIN(t.min_created_at) AS min_created_at,
        MAX(t.max_created_at) AS max_created_at,
        COUNT(t.user_id) AS bet_rounds,
        SUM(t.fg_rounds) AS fg_rounds,
        AVG(t.delta_t_seconds) AS delta_t_seconds_avg,
        STDDEV(t.delta_t_seconds) AS delta_t_seconds_std,
        MIN(t.delta_t_seconds) AS delta_t_seconds_min,
        MAX(t.delta_t_seconds) AS delta_t_seconds_max,
        AVG(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_avg,
        STDDEV(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_std,
        MIN(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_min,
        MAX(t.delta_t_seconds_nogap) AS delta_t_seconds_nogap_max,
        AVG(t.bet_amount) AS bet_amount_avg,
        STDDEV(t.bet_amount) AS bet_amount_std,
        MIN(t.bet_amount) AS bet_amount_min,
        MAX(t.bet_amount) AS bet_amount_max,
        AVG(t.delta_bet_amount) AS delta_bet_amount_avg,
        STDDEV(t.delta_bet_amount) AS delta_bet_amount_std,
        MIN(t.delta_bet_amount) AS delta_bet_amount_min,
        MAX(t.delta_bet_amount) AS delta_bet_amount_max,
        SUM(CASE WHEN t.delta_bet_amount > 0 THEN t.delta_bet_amount ELSE 0 END) AS accum_pos_delta_bet_amount,
        SUM(CASE WHEN t.delta_bet_amount < 0 THEN t.delta_bet_amount ELSE 0 END) AS accum_neg_delta_bet_amount,
        AVG(t.payout) AS payout_avg,
        STDDEV(t.payout) AS payout_std,
        MIN(t.payout) AS payout_min,
        MAX(t.payout) AS payout_max,
        SUM(CASE WHEN t.payout > 0 THEN 1 END) * 1.0 / NULLIF(COUNT(t.user_id), 0) AS payout_rate,
        AVG(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_mean,
        MAX(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_max,
        MIN(t.payout * 1.0 / NULLIF(t.bet_amount, 0)) AS rtp_min,
        AVG(t.profit) AS profit_avg,
        STDDEV(t.profit) AS profit_std,
        MIN(t.profit) AS profit_min,
        MAX(t.profit) AS profit_max,
        SUM(CASE WHEN t.profit > 0 THEN t.profit ELSE 0 END) AS accum_pos_profit,
        SUM(CASE WHEN t.profit < 0 THEN t.profit ELSE 0 END) AS accum_neg_profit,
        SUM(CASE WHEN t.profit > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(t.user_id), 0) AS profit_rate,
        AVG(t.delta_payout) AS delta_payout_avg,
        STDDEV(t.delta_payout) AS delta_payout_std,
        MIN(t.delta_payout) AS delta_payout_min,
        MAX(t.delta_payout) AS delta_payout_max,
        AVG(t.balance_after_bet) AS balance_after_bet_avg,
        STDDEV(t.balance_after_bet) AS balance_after_bet_std,
        MIN(t.balance_after_bet) AS balance_after_bet_min,
        MAX(t.balance_after_bet) AS balance_after_bet_max,
        SUM(CASE WHEN t.deposit > 0 THEN 1 ELSE 0 END) AS num_deposit,
        SUM(CASE WHEN t.deposit > 0 THEN t.deposit ELSE 0 END) AS accum_deposit,
        SUM(CASE WHEN t.withdraw > 0 THEN 1 ELSE 0 END) AS num_withdraw,
        SUM(CASE WHEN t.withdraw > 0 THEN t.withdraw ELSE 0 END) AS accum_withdraw,
        AVG(t.streak) AS streak_avg,
        STDDEV(t.streak) AS streak_std,
        MIN(t.streak) AS streak_min,
        MAX(t.streak) AS streak_max,
        AVG(t.win_streak) AS win_streak_avg,
        STDDEV(t.win_streak) AS win_streak_std,
        MIN(t.win_streak) AS win_streak_min,
        MAX(t.win_streak) AS win_streak_max,
        AVG(t.lose_streak) AS lose_streak_avg,
        STDDEV(t.lose_streak) AS lose_streak_std,
        MIN(t.lose_streak) AS lose_streak_min,
        MAX(t.lose_streak) AS lose_streak_max
    FROM binned AS t
    GROUP BY
        t.user_id,
        t.ai_group,
        t.math_table_id,
        t.session_start_date,
        t.session_group,
        t.agg_group
)
SELECT
    b.user_id,
    b.ai_group,
    b.math_table_id,
    b.session_start_date,
    b.session_group,
    b.agg_group,
    b.activity_date,
    b.min_created_at,
    b.max_created_at,
    b.bet_rounds,
    b.fg_rounds,
    -- delta bet time
    b.delta_t_seconds_avg,
    p1.delta_t_seconds_p25,
    p1.delta_t_seconds_median,
    p1.delta_t_seconds_p75,
    b.delta_t_seconds_std,
    b.delta_t_seconds_min,
    b.delta_t_seconds_max,
    -- delta bet time nogap
    b.delta_t_seconds_nogap_avg,
    p2.delta_t_seconds_nogap_p25,
    p2.delta_t_seconds_nogap_median,
    p2.delta_t_seconds_nogap_p75,
    b.delta_t_seconds_nogap_std,
    b.delta_t_seconds_nogap_min,
    b.delta_t_seconds_nogap_max,
    -- bet amount
    b.bet_amount_avg,
    p3.bet_amount_p25,
    p3.bet_amount_median,
    p3.bet_amount_p75,
    b.bet_amount_std,
    b.bet_amount_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_cv,
    b.bet_amount_min,
    b.bet_amount_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_drawdown_ratio,
    b.bet_amount_max,
    b.bet_amount_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS bet_amount_spike_ratio,
    -- delta bet amount
    b.delta_bet_amount_avg,
    p4.delta_bet_amount_p25,
    p4.delta_bet_amount_median,
    p4.delta_bet_amount_p75,
    b.delta_bet_amount_std,
    COALESCE(b.delta_bet_amount_std * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_cv,
    b.delta_bet_amount_min,
    COALESCE(b.delta_bet_amount_min * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_drawdown_ratio,
    b.delta_bet_amount_max,
    COALESCE(b.delta_bet_amount_max * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_bet_amount_spike_ratio,
    -- accumulative delta bet
    b.accum_pos_delta_bet_amount,
    b.accum_neg_delta_bet_amount,
    b.accum_pos_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_delta_bet_amount_ratio,
    b.accum_neg_delta_bet_amount * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_delta_bet_amount_ratio,
    -- payout
    b.payout_avg,
    p5.payout_p25,
    p5.payout_median,
    p5.payout_p75,
    b.payout_std,
    b.payout_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_cv,
    b.payout_min,
    b.payout_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_drawdown_ratio,
    b.payout_max,
    b.payout_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS payout_spike_ratio,
    b.payout_rate,
    -- rtp
    b.rtp_mean,
    b.rtp_max,
    b.rtp_min,
    p6.rtp_p25,
    p6.rtp_median,
    p6.rtp_p75,
    -- profit
    b.profit_avg,
    p7.profit_p25,
    p7.profit_median,
    p7.profit_p75,
    b.profit_std,
    b.profit_std * 1.0 / NULLIF(b.bet_amount_avg, 0) AS profit_cv,
    b.profit_min,
    b.profit_min * 1.0 / NULLIF(b.bet_amount_avg, 0) AS min_profit_ratio,
    b.profit_max,
    b.profit_max * 1.0 / NULLIF(b.bet_amount_avg, 0) AS max_profit_ratio,
    b.accum_pos_profit,
    b.accum_pos_profit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_pos_profit_ratio,
    b.accum_neg_profit,
    b.accum_neg_profit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_neg_profit_ratio,
    b.profit_rate,
    -- delta payout
    b.delta_payout_avg,
    p8.delta_payout_p25,
    p8.delta_payout_median,
    p8.delta_payout_p75,
    b.delta_payout_std,
    COALESCE(b.delta_payout_std * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_cv,
    b.delta_payout_min,
    COALESCE(b.delta_payout_min * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_drawdown_ratio,
    b.delta_payout_max,
    COALESCE(b.delta_payout_max * 1.0 / NULLIF(b.bet_amount_avg, 0), 0) AS delta_payout_spike_ratio,
    -- balance after bet
    b.balance_after_bet_avg,
    p9.balance_after_bet_p25,
    p9.balance_after_bet_median,
    p9.balance_after_bet_p75,
    b.balance_after_bet_std,
    b.balance_after_bet_std * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_cv,
    b.balance_after_bet_min,
    b.balance_after_bet_min * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_drawdown_ratio,
    b.balance_after_bet_max,
    b.balance_after_bet_max * 1.0 / NULLIF(b.balance_after_bet_avg, 0) AS balance_after_bet_spike_ratio,
    -- deposit/withdraw
    b.num_deposit,
    b.accum_deposit,
    b.accum_deposit * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_deposit_ratio,
    b.num_withdraw,
    b.accum_withdraw,
    b.accum_withdraw * 1.0 / NULLIF(b.bet_amount_avg, 0) AS accum_withdraw_ratio,
    -- streaks
    COALESCE(b.streak_avg, 0) AS streak_avg,
    COALESCE(p10.streak_p25, 0) AS streak_p25,
    COALESCE(p10.streak_median, 0) AS streak_median,
    COALESCE(p10.streak_p75, 0) AS streak_p75,
    COALESCE(b.streak_std, 0) AS streak_std,
    COALESCE(b.streak_min, 0) AS streak_min,
    COALESCE(b.streak_max, 0) AS streak_max,
    COALESCE(b.win_streak_avg, 0) AS win_streak_avg,
    COALESCE(p11.win_streak_p25, 0) AS win_streak_p25,
    COALESCE(p11.win_streak_median, 0) AS win_streak_median,
    COALESCE(p11.win_streak_p75, 0) AS win_streak_p75,
    COALESCE(b.win_streak_std, 0) AS win_streak_std,
    COALESCE(b.win_streak_min, 0) AS win_streak_min,
    COALESCE(b.win_streak_max, 0) AS win_streak_max,
    COALESCE(b.lose_streak_avg, 0) AS lose_streak_avg,
    COALESCE(p12.lose_streak_p25, 0) AS lose_streak_p25,
    COALESCE(p12.lose_streak_median, 0) AS lose_streak_median,
    COALESCE(p12.lose_streak_p75, 0) AS lose_streak_p75,
    COALESCE(b.lose_streak_std, 0) AS lose_streak_std,
    COALESCE(b.lose_streak_min, 0) AS lose_streak_min,
    COALESCE(b.lose_streak_max, 0) AS lose_streak_max
FROM stats_base AS b
    JOIN perc_delta_t AS p1
        ON b.user_id = p1.user_id
            AND b.ai_group = p1.ai_group
            AND b.math_table_id = p1.math_table_id
            AND b.session_start_date = p1.session_start_date
            AND b.session_group = p1.session_group
            AND b.agg_group = p1.agg_group
    JOIN perc_delta_t_ng AS p2
        ON b.user_id = p2.user_id
            AND b.ai_group = p2.ai_group
            AND b.math_table_id = p2.math_table_id
            AND b.session_start_date = p2.session_start_date
            AND b.session_group = p2.session_group
            AND b.agg_group = p2.agg_group
    JOIN perc_bet_amount AS p3
        ON b.user_id = p3.user_id
            AND b.ai_group = p3.ai_group
            AND b.math_table_id = p3.math_table_id
            AND b.session_start_date = p3.session_start_date
            AND b.session_group = p3.session_group
            AND b.agg_group = p3.agg_group
    JOIN perc_delta_bet AS p4
        ON b.user_id = p4.user_id
            AND b.ai_group = p4.ai_group
            AND b.math_table_id = p4.math_table_id
            AND b.session_start_date = p4.session_start_date
            AND b.session_group = p4.session_group
            AND b.agg_group = p4.agg_group
    JOIN perc_payout AS p5
        ON b.user_id = p5.user_id
            AND b.ai_group = p5.ai_group
            AND b.math_table_id = p5.math_table_id
            AND b.session_start_date = p5.session_start_date
            AND b.session_group = p5.session_group
            AND b.agg_group = p5.agg_group
    JOIN perc_rtp AS p6
        ON b.user_id = p6.user_id
            AND b.ai_group = p6.ai_group
            AND b.math_table_id = p6.math_table_id
            AND b.session_start_date = p6.session_start_date
            AND b.session_group = p6.session_group
            AND b.agg_group = p6.agg_group
    JOIN perc_profit AS p7
        ON b.user_id = p7.user_id
            AND b.ai_group = p7.ai_group
            AND b.math_table_id = p7.math_table_id
            AND b.session_start_date = p7.session_start_date
            AND b.session_group = p7.session_group
            AND b.agg_group = p7.agg_group
    JOIN perc_delta_payout AS p8
        ON b.user_id = p8.user_id
            AND b.ai_group = p8.ai_group
            AND b.math_table_id = p8.math_table_id
            AND b.session_start_date = p8.session_start_date
            AND b.session_group = p8.session_group
            AND b.agg_group = p8.agg_group
    JOIN perc_balance AS p9
        ON b.user_id = p9.user_id
            AND b.ai_group = p9.ai_group
            AND b.math_table_id = p9.math_table_id
            AND b.session_start_date = p9.session_start_date
            AND b.session_group = p9.session_group
            AND b.agg_group = p9.agg_group
    JOIN perc_streak AS p10
        ON b.user_id = p10.user_id
            AND b.ai_group = p10.ai_group
            AND b.math_table_id = p10.math_table_id
            AND b.session_start_date = p10.session_start_date
            AND b.session_group = p10.session_group
            AND b.agg_group = p10.agg_group
    JOIN perc_win_streak AS p11
        ON b.user_id = p11.user_id
            AND b.ai_group = p11.ai_group
            AND b.math_table_id = p11.math_table_id
            AND b.session_start_date = p11.session_start_date
            AND b.session_group = p11.session_group
            AND b.agg_group = p11.agg_group
    JOIN perc_lose_streak AS p12
        ON b.user_id = p12.user_id
            AND b.ai_group = p12.ai_group
            AND b.math_table_id = p12.math_table_id
            AND b.session_start_date = p12.session_start_date
            AND b.session_group = p12.session_group
            AND b.agg_group = p12.agg_group
ORDER BY
    b.user_id,
    b.ai_group,
    b.math_table_id,
    b.session_start_date,
    b.session_group,
    b.agg_group
;
