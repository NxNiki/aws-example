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
    t.partition_ab[0] AS ab_group_id,
    LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_created_at,
    EXTRACT(EPOCH FROM (t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at))) AS delta_t_seconds,
    LAG(t.bet_type) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_type,
    LAG(t.bet_amount) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) AS prev_bet_amount,
    COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count,
    CASE
        WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) IS NULL THEN 0
        WHEN LAG(t.math_table_id) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.created_at) <> t.math_table_id THEN 1
        ELSE 0
    END AS mathtable_change
FROM
    public.fct_bet_orders AS t
WHERE
    t.game_id = 'SS06'
    AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) >= '2025-01-01'
    AND t.status = 'COMPLETED'
    AND t.currency_type IN ('CNY')
    AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
),

user_bets_group AS (
    SELECT

        t.created_at,
        t.activity_date,
        t.activity_week,
        t.activity_month,
        t.user_id,
        t.bet_amount,
        t.payout,
        t.bet_type,
        t.profit,
        t.delta_t_seconds,
        t.user_bet_count,
        t.mathtable_change,
        t.prev_bet_type,
        t.prev_bet_amount,
        CASE WHEN t.bet_type = 'FREE' AND t.prev_bet_type = 'BASE' THEN t.prev_bet_amount END AS fg_session_trigger_bet,

        CASE
            WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' THEN 'AI'
            WHEN t.ab_group_id = '4f1a46ca-7baa-4452-9a40-ef21d9b33b57' THEN 'AB_TEST_A'
            WHEN t.ab_group_id = '4a04df21-c749-4808-8e55-3a0b74c084d2' THEN 'AB_TEST_B'
            ELSE 'Default'
        END AS ai_group
    FROM
        user_bets AS t

    UNION ALL

    -- Per-mathtable variants. AB_TEST_A/B users are excluded so they don't
    -- get double-counted into Default_<mathtable> alongside the Default cohort.
    SELECT

        t.created_at,
        t.activity_date,
        t.activity_week,
        t.activity_month,
        t.user_id,
        t.bet_amount,
        t.payout,
        t.bet_type,
        t.profit,
        t.delta_t_seconds,
        t.user_bet_count,
        t.mathtable_change,
        t.prev_bet_type,
        t.prev_bet_amount,
        CASE WHEN t.bet_type = 'FREE' AND t.prev_bet_type = 'BASE' THEN t.prev_bet_amount END AS fg_session_trigger_bet,

        CASE
            WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' THEN t.mathtable
            ELSE CONCAT('Default_', t.mathtable)
        END AS ai_group
    FROM
        user_bets AS t
    WHERE t.ab_group_id IS NULL
       OR t.ab_group_id NOT IN ('4f1a46ca-7baa-4452-9a40-ef21d9b33b57', '4a04df21-c749-4808-8e55-3a0b74c084d2')
),

user_stats AS (
    SELECT
        t.activity_month,
        t.ai_group,
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

        SUM(t.fg_session_trigger_bet) AS user_total_bet_fg,

        COUNT(CASE WHEN t.payout > 0 THEN 1 END) AS user_num_bets_with_payout,
        COUNT(CASE WHEN t.bet_type = 'BASE' AND t.payout > 0 THEN 1 END) AS user_num_bets_bg_with_payout,
        COUNT(CASE WHEN t.bet_type = 'FREE' AND t.payout > 0 THEN 1 END) AS user_num_bets_fg_with_payout,

        AVG(CASE WHEN t.bet_type = 'BASE' AND t.prev_bet_type = 'BASE' AND t.delta_t_seconds <= 1800 THEN GREATEST(t.delta_t_seconds, 1) END)
            AS user_avg_delta_t_seconds_bg,

        SUM(t.mathtable_change) AS user_mathtable_change,

        -- delta bet amount metrics:
        SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet,
        SUM(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet,
        AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_pos_delta_bet_avg,
        AVG(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN (t.bet_amount - t.prev_bet_amount) END) AS user_accu_neg_delta_bet_avg,
        SUM(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet,
        AVG(t.bet_amount - t.prev_bet_amount) AS user_accu_delta_bet_avg,
        COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) > 0 THEN 1 END) AS user_pos_delta_bet_num,
        COUNT(CASE WHEN (t.bet_amount - t.prev_bet_amount) < 0 THEN 1 END) AS user_neg_delta_bet_num

    FROM user_bets_group AS t
    GROUP BY t.activity_month, t.ai_group, t.user_id
),

user_mathtable_last_spin_ai AS (
    -- Last spin per (period, AI user, mathtable). Restricted to AI bets so the
    -- downstream metric only applies to AI per-mathtable ai_groups.
    SELECT
        t.activity_month,
        t.user_id,
        t.mathtable,
        MAX(t.spin_id) AS mathtable_last_spin_id
    FROM user_bets AS t
    WHERE t.ab_group_id = 'jojpin-9mokha-rexQug'
    GROUP BY t.activity_month, t.user_id, t.mathtable
),

user_remaining_bet AS (
    -- Avg bet on bets the AI user made AFTER their last bet on the math table,
    -- keyed by ai_group (= mathtable for the AI per-mathtable variant). All
    -- other ai_groups are absent here, so the LEFT JOIN below yields NULL.
    SELECT
        m.activity_month,
        m.user_id,
        m.mathtable AS ai_group,
        AVG(b.bet_amount) AS user_avg_remaining_bet_amount
    FROM user_mathtable_last_spin_ai AS m
    LEFT JOIN user_bets AS b
        ON b.user_id = m.user_id
        AND b.activity_month = m.activity_month
        AND b.spin_id > m.mathtable_last_spin_id
        AND b.ab_group_id = 'jojpin-9mokha-rexQug'
    GROUP BY m.activity_month, m.user_id, m.mathtable
)

SELECT
    us.activity_month AS activity_date,
    us.ai_group,
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

    -- average bet amount on bets after the user's last bet on the ai_group's math table
    -- (only populated for AI per-mathtable ai_groups; NULL for combined / Default / AB_TEST groups)
    rb.user_avg_remaining_bet_amount

FROM user_stats AS us
LEFT JOIN user_remaining_bet AS rb
    ON rb.user_id = us.user_id
    AND rb.activity_month = us.activity_month
    AND rb.ai_group = us.ai_group
WHERE us.activity_month >= '2025-01-01'
ORDER BY us.activity_month DESC, us.ai_group DESC, us.user_id DESC;
