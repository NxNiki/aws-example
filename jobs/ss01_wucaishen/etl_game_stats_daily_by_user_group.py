import os
from textwrap import dedent

import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

# TODO:
# add max/min user daily profit

query = dedent(
    """
    WITH user_bets AS (
    SELECT
        t.user_id,
        m.mathtable,
        t.bet_amount,
        t.actual_payout AS payout,
        t.bet_type,
        t.actual_payout - t.bet_amount AS profit,
        TRUNC(CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at)) AS activity_date,
        t.partition_ab[0] AS ab_group_id,
        t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS delta_t,
        COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count
    FROM
        public.fct_bet_orders AS t
    LEFT JOIN
        public.dim_math_talbes AS m
        ON
            t.user_id = m.user_id
            AND t.game_id = m.game_id
            AND t.created_at >= m.start_time
            AND (t.created_at <= m.end_time OR m.is_current IS TRUE)
    WHERE
        CONVERT_TIMEZONE('UTC', 'America/Los_Angeles', t.created_at) >= '2025-12-01 17:00:00'
        AND t.currency_type = 'CNY'
        AND t.status = 'COMPLETED'
        AND t.game_id = 'SS01'
        AND t.op_code != 'B26'
    ),

    user_bets_group AS (
        SELECT
            t.activity_date,
            t.user_id,
            t.bet_amount,
            t.payout,
            t.bet_type,
            t.profit,
            t.delta_t,
            t.user_bet_count,
            CASE
                WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' THEN 'Default'
                ELSE 'AI'
            END AS ai_group
        FROM
            user_bets AS t

        UNION ALL

        SELECT
            t.activity_date,
            t.user_id,
            t.bet_amount,
            t.payout,
            t.bet_type,
            t.profit,
            t.delta_t,
            t.user_bet_count,
            t.mathtable AS ai_group
        FROM
            user_bets AS t
        WHERE
            t.ab_group_id = 'jojpin-9mokha-rexQug'
    ),

    daily_login AS (
        SELECT DISTINCT
            t.activity_date,
            t.user_id,
            t.ai_group
        FROM
            user_bets_group AS t
    ),

    user_retention AS (
        SELECT
            t1.activity_date,
            t1.ai_group,
            COUNT(t1.user_id) AS day0_num_users,
            COUNT(t2.user_id) AS day1_num_users,
            COUNT(t3.user_id) AS day3_num_users
        FROM
            daily_login AS t1
        LEFT JOIN daily_login AS t2
            ON t2.activity_date = DATE_ADD('day', -1, t1.activity_date) AND t1.user_id = t2.user_id
        LEFT JOIN daily_login AS t3
            ON t3.activity_date = DATE_ADD('day', -3, t1.activity_date) AND t1.user_id = t3.user_id
        GROUP BY t1.activity_date, t1.ai_group
    ),

    user_stats AS (
        SELECT
            t.activity_date,
            t.ai_group,
            t.user_id,

            -- total number of bets:
            COUNT(t.user_id) AS user_num_bets,
            COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS user_num_bets_bg,
            COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS user_num_bets_fg,

            -- total bet amount:
            SUM(t.bet_amount) AS user_total_bet,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS user_total_bet_bg,

            -- total payout amount:
            SUM(t.payout) AS user_total_payout,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS user_total_payout_bg,
            SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS user_total_payout_fg,

            AVG(CASE WHEN EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 THEN EXTRACT(EPOCH FROM t.delta_t) END)
                AS user_avg_delta_t_seconds

        FROM user_bets_group AS t
        WHERE t.user_bet_count >= 40
        GROUP BY t.activity_date, t.ai_group, t.user_id
    ),

    group_stats AS (
        SELECT
            t.activity_date,
            t.ai_group,
            COUNT(DISTINCT t.user_id) AS num_active_users,
            COUNT(t.user_id) AS total_num_bets,
            COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS total_num_bets_bg,
            COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS total_num_bets_fg,

            -- total bet amount:
            SUM(t.bet_amount) AS total_bet,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS total_bet_bg,

            SUM(t.payout) AS total_payout,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS total_payout_bg,
            SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS total_payout_fg
        FROM user_bets_group AS t
        WHERE t.user_bet_count >= 40
        GROUP BY t.activity_date, t.ai_group
    )

    SELECT
        us.activity_date,
        us.ai_group,
        us.user_id,

        gs.num_active_users,

        gs.total_num_bets,
        gs.total_num_bets_bg,
        gs.total_num_bets_fg,

        gs.total_bet,
        gs.total_bet_bg,

        gs.total_payout,
        gs.total_payout_bg,
        gs.total_payout_fg,

        us.user_avg_delta_t_seconds,

        -- Retention:
        ur.day0_num_users,
        ur.day1_num_users,
        ur.day3_num_users,
        ur.day1_num_users * 1.0 / NULLIF(ur.day0_num_users, 0) AS retention_rate_day1,
        ur.day3_num_users * 1.0 / NULLIF(ur.day0_num_users, 0) AS retention_rate_day3,

        -- free game ratio
        gs.total_num_bets_fg * 1.0 / NULLIF(gs.total_num_bets, 0) AS fg_ratio,

        -- Number of bets and total bet per user:
        us.user_num_bets,
        us.user_num_bets_bg,
        us.user_num_bets_fg,

        us.user_total_bet,

        -- Profit Calculations
        (gs.total_payout - gs.total_bet) AS total_profit,
        (us.user_total_payout - us.user_total_bet) AS user_total_profit,

        -- RTP (Return to Player) Calculations (Fix: NULLIF for bet amounts AND RTP numerator error)
        gs.total_payout / NULLIF(gs.total_bet, 0) AS rtp,
        -- total bet for base game is same to total bet and free game has 0 bet amount:
        gs.total_payout_bg / NULLIF(gs.total_bet, 0) AS rtp_bg,
        us.user_total_payout / NULLIF(us.user_total_bet, 0) AS user_rtp,
        us.user_total_payout_bg / NULLIF(us.user_total_bet, 0) AS user_rtp_bg

    FROM user_stats AS us
    INNER JOIN group_stats AS gs
        ON us.activity_date = gs.activity_date AND us.ai_group = gs.ai_group
    INNER JOIN user_retention AS ur
        ON us.activity_date = ur.activity_date AND us.ai_group = ur.ai_group
    ORDER BY us.activity_date DESC, us.ai_group DESC, us.user_id DESC;

    """
)


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="production-redshift-cluster.cwiqzcm13zcn.ap-southeast-1.redshift.amazonaws.com",
            database="slot-machine",
            user="anaylsis_user",
            password="oZ4ztMx0yEXPLbJL733L",
            port=5439,
        )
    )

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_date_user.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(
        df_rs[
            [
                "activity_date",
                "user_id",
                "ai_group",
                "num_active_users",
                "user_num_bets",
                "user_total_bet",
                "rtp",
                "rtp_bg",
                "day0_num_users",
                "day1_num_users",
                "day3_num_users",
            ]
        ]
    )

    redshift_loader.close()
