
WITH user_bets AS (
    SELECT
        spin_id,
        user_id,
        math_table_id,
        created_at,
        bet_type,
        CASE WHEN bet_type = 'BASE' THEN bet_amount END AS bet_amount,
        actual_payout AS payout,
        balance_after_bet,
        balance_after_payout,
        LAG(created_at, 1) OVER (PARTITION BY user_id ORDER BY spin_id, created_at) AS prev_bet_time,
        CASE WHEN bet_type = 'BASE' THEN 1 ELSE 0 END AS is_new_game_group,
        ai_group
    FROM (
        SELECT
            t.spin_id,
            t.user_id,
            t.math_table_id,
            t.created_at,
            t.bet_type,
            t.bet_amount,
            t.actual_payout,
            t.balance_after_bet,
            t.balance_after_payout,
            t.partition_ab,
            t.currency_type,
            t.status,
            t.op_code,
            CASE
            WHEN CAST(t.created_at + INTERVAL '8' HOUR AS DATE) >= DATE '2026-08-04' THEN CASE
                WHEN CAST(t.user_id AS BIGINT) % 10 >= 8 THEN 'AI'
                WHEN CAST(t.user_id AS BIGINT) % 10 >= 6 THEN 'AB_TEST_B'
                WHEN CAST(t.user_id AS BIGINT) % 10 >= 4 THEN 'AB_TEST_A'
                ELSE 'Default'
            END
            ELSE CASE
                WHEN get_json_object(CAST(t.partition_ab AS STRING), '$[0]') = 'jojpin-9mokha-rexQug' THEN 'AI'
                WHEN get_json_object(CAST(t.partition_ab AS STRING), '$[0]') = '4f1a46ca-7baa-4452-9a40-ef21d9b33b57' THEN 'AB_TEST_A'
                WHEN get_json_object(CAST(t.partition_ab AS STRING), '$[0]') = '4a04df21-c749-4808-8e55-3a0b74c084d2' THEN 'AB_TEST_B'
                ELSE 'Default'
            END
        END AS ai_group
        FROM bet_order_raw AS t
        WHERE
            t.created_at >= TIMESTAMP '2026-06-17 00:00:00'
            AND t.created_at < TIMESTAMP '2026-08-01 00:00:00'
            AND t.currency_type IN ('CNY')
            AND t.status = 'COMPLETED'
            AND t.op_code NOT IN ('B26', 'TST', 'TSB', 'TSO')
    )
    WHERE ai_group = 'AB_TEST_B'
),

free_game_group AS (
    SELECT
        t.spin_id,
        t.user_id,
        t.math_table_id,
        t.created_at,
        t.bet_type,
        t.bet_amount,
        t.payout,
        t.balance_after_bet,
        t.balance_after_payout,
        t.prev_bet_time,
        t.is_new_game_group,
        t.ai_group,
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
        unix_timestamp(t.min_created_at) - unix_timestamp(t.prev_bet_time) AS delta_t_seconds,
        t.bet_amount - LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_bet_amount,
        t.payout - LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS delta_payout,
        t.balance_after_bet
            + t.bet_amount
            - LAG(t.balance_after_payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at) AS balance_transaction,
        CASE WHEN t.payout > t.bet_amount THEN 1 ELSE 0 END AS is_win,
        CASE WHEN t.payout < t.bet_amount THEN 1 ELSE 0 END AS is_lose,
        CASE
            WHEN LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                 > LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                THEN 1
            ELSE 0
        END AS prev_win,
        CASE
            WHEN LAG(t.payout, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
                 < LAG(t.bet_amount, 1) OVER (PARTITION BY t.user_id ORDER BY t.spin_id, t.min_created_at)
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
        t.bet_amount,
        t.payout,
        t.balance_after_bet,
        t.delta_t_seconds,
        t.delta_bet_amount,
        t.delta_payout,
        t.balance_transaction,
        t.is_win,
        t.is_lose,
        t.prev_win,
        t.prev_lose,
        LAST(
            CASE
                WHEN t.delta_t_seconds > 43200 OR t.delta_t_seconds IS NULL
                    THEN t.min_created_at
            END,
            TRUE
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
)

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
WHERE CAST(t.session_start_ts + INTERVAL '8' HOUR AS DATE)
    NOT IN (DATE '2026-08-03', DATE '2026-08-04')
