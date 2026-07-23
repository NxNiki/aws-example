WITH user_bets AS (
SELECT
    t.user_id,
    t.spin_id,
    t.created_at,
    t.math_table_id AS mathtable,
    t.bet_amount,
    t.actual_payout AS payout,
    t.bet_type,
    t.actual_payout - t.bet_amount AS profit,
    CAST(DATE_TRUNC('day', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_date,
    CAST(DATE_TRUNC('week', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_week,
    CAST(DATE_TRUNC('month', DATEADD(hour, -0, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS DATE) AS activity_month,
    CASE
        WHEN t.partition_ab[0] = 'jojpin-9mokha-rexQug' THEN 'AI'
        ELSE 'Default'
    END AS ab_group,
    LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_created_at,
    EXTRACT(EPOCH FROM (t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at))) AS delta_t_seconds,
    LAG(t.bet_type) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_type,
    LAG(t.bet_amount) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_amount,
    CASE
        WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) IS NULL THEN 0
        WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) <> t.math_table_id THEN 1
        ELSE 0
    END AS mathtable_change
FROM
    public.fct_bet_orders AS t
WHERE
    t.game_id = 'SS01'
    AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) >= '2025-01-01'
    AND t.currency_type IN ('CNY')
    AND t.status = 'COMPLETED'
    AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
),

user_stats AS (
    SELECT
        t.activity_date,
        t.ab_group,
        t.mathtable,
        t.user_id,

        -- total number of bets:
        COUNT(t.user_id) AS user_num_bets,
        COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS user_num_bets_bg,
        COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS user_num_bets_fg,

        -- total bet amount:
        SUM(t.bet_amount) AS user_total_bet,
        SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS user_total_bet_bg,
        AVG(t.bet_amount) AS user_avg_bet_amount,

        -- total payout amount:
        SUM(t.payout) AS user_total_payout,
        SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS user_total_payout_bg,
        SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS user_total_payout_fg,

        SUM(CASE WHEN t.bet_type = 'FREE' AND t.prev_bet_type = 'BASE' THEN t.prev_bet_amount END) AS user_total_bet_fg,

        COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS user_num_bets_with_payout,
        COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS user_num_bets_bg_with_payout,
        COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS user_num_bets_fg_with_payout,

        -- delta-t between consecutive BASE bets: AVG plus its exact
        -- denominator so the average recombines across grain rows.
        AVG(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= 1800 THEN GREATEST(t.delta_t_seconds, 1) END)
            AS user_avg_delta_t_seconds_bg,
        COUNT(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= 1800 THEN 1 END)
            AS user_num_delta_t_bg,

        SUM(t.mathtable_change) AS user_mathtable_change,

        -- delta bet amount metrics:
        SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet,
        SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet,
        AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet_avg,
        AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet_avg,
        SUM(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet,
        AVG(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet_avg,
        COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN 1 END) AS user_pos_delta_bet_num,
        COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN 1 END) AS user_neg_delta_bet_num,
        COUNT(t.prev_bet_amount) AS user_num_delta_bet

    FROM user_bets AS t
    GROUP BY t.activity_date, t.ab_group, t.mathtable, t.user_id
)

SELECT
    us.activity_date AS activity_date,
    us.ab_group,
    us.mathtable,
    us.user_id,
    us.user_mathtable_change,
    -- DataMetrics input columns (user-level raw stats):
    us.user_num_bets,
    us.user_num_bets_bg,
    us.user_num_bets_fg,
    us.user_total_bet,
    us.user_total_bet_bg,
    us.user_avg_bet_amount,
    us.user_total_payout,
    us.user_total_payout_bg,
    us.user_total_payout_fg,
    us.user_total_bet_fg,
    us.user_num_bets_with_payout,
    us.user_num_bets_bg_with_payout,
    us.user_num_bets_fg_with_payout,
    us.user_total_payout * 1.0 / NULLIF(us.user_total_bet, 0) AS user_rtp,

    -- User-level derived (not computed by DataMetrics):
    us.user_avg_delta_t_seconds_bg,
    us.user_num_delta_t_bg,
    us.user_num_bets_fg * 1.0 / NULLIF(us.user_num_bets, 0) AS user_fg_ratio,
    (us.user_total_payout - us.user_total_bet) AS user_total_profit,
    us.user_total_payout_bg * 1.0 / NULLIF(us.user_total_bet, 0) AS user_rtp_bg,
    us.user_total_payout_fg * 1.0 / NULLIF(us.user_total_bet_fg, 0) AS user_rtp_fg,
    us.user_num_bets_with_payout * 1.0 / NULLIF(us.user_num_bets, 0) AS user_hit_rate,
    us.user_num_bets_bg_with_payout * 1.0 / NULLIF(us.user_num_bets_bg, 0) AS user_hit_rate_bg,
    us.user_num_bets_fg_with_payout * 1.0 / NULLIF(us.user_num_bets_fg, 0) AS user_hit_rate_fg,

    -- delta bet amount metrics:
    us.user_accu_pos_delta_bet,
    us.user_accu_neg_delta_bet,
    us.user_accu_pos_delta_bet_avg,
    us.user_accu_neg_delta_bet_avg,
    us.user_accu_delta_bet,
    us.user_accu_delta_bet_avg,
    us.user_pos_delta_bet_num,
    us.user_neg_delta_bet_num,
    us.user_num_delta_bet

FROM user_stats AS us
WHERE us.activity_date >= '2025-01-01'
ORDER BY us.activity_date DESC, us.ab_group DESC, us.mathtable DESC, us.user_id DESC;
