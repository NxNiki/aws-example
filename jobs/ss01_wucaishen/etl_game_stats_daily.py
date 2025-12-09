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

    daily_login AS (
        SELECT DISTINCT
            t.activity_date,
            t.user_id,
            t.ab_group_id,
            t.mathtable
        FROM
            user_bets AS t
    ),

    user_retention AS (
        SELECT
            t1.activity_date,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id != 'jojpin-9mokha-rexQug' THEN t1.user_id END) AS day0_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id != 'jojpin-9mokha-rexQug' THEN t2.user_id END) AS day1_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id != 'jojpin-9mokha-rexQug' THEN t3.user_id END) AS day3_num_users,

            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' THEN t1.user_id END) AS ai_day0_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' THEN t2.user_id END) AS ai_day1_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' THEN t3.user_id END) AS ai_day3_num_users,

            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'giftShop' THEN t1.user_id END) AS ai_giftshop_day0_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'giftShop' THEN t2.user_id END) AS ai_giftshop_day1_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'giftShop' THEN t3.user_id END) AS ai_giftshop_day3_num_users,

            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'newBee' THEN t1.user_id END) AS ai_newbee_day0_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'newBee' THEN t2.user_id END) AS ai_newbee_day1_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'newBee' THEN t3.user_id END) AS ai_newbee_day3_num_users,

            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'carousels' THEN t1.user_id END) AS ai_carousels_day0_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'carousels' THEN t2.user_id END) AS ai_carousels_day1_num_users,
            COUNT(DISTINCT CASE WHEN t1.ab_group_id = 'jojpin-9mokha-rexQug' AND t1.mathtable = 'carousels' THEN t3.user_id END) AS ai_carousels_day3_num_users
        FROM
            daily_login AS t1
        LEFT JOIN daily_login AS t2
            ON t2.activity_date = DATE_ADD('day', 1, t1.activity_date) AND t1.user_id = t2.user_id
        LEFT JOIN daily_login AS t3
            ON t3.activity_date = DATE_ADD('day', 3, t1.activity_date) AND t1.user_id = t3.user_id
        GROUP BY t1.activity_date
    ),

    daily_stats AS (
        SELECT
            t.activity_date,

            -- number of users:
            COUNT(DISTINCT t.user_id) AS total_daily_users,
            COUNT(DISTINCT CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' THEN t.user_id END) AS default_group_users,
            COUNT(DISTINCT CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' THEN t.user_id END) AS ai_group_users,
            COUNT(
                DISTINCT CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'giftShop' THEN t.user_id END
            ) AS ai0_giftshop_users,
            COUNT(
                DISTINCT CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'carousels' THEN t.user_id END
            ) AS ai2_carousels_users,
            COUNT(DISTINCT CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'newBee' THEN t.user_id END)
                AS ai1_newbee_users,

            -- total number of bets:
            COUNT(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' THEN t.user_id END) AS default_num_bets,
            COUNT(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' THEN t.user_id END) AS ai_num_bets,
            COUNT(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'giftShop' THEN t.user_id END)
                AS ai0_giftshop_num_bets,
            COUNT(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'carousels' THEN t.user_id END)
                AS ai2_carousels_num_bets,
            COUNT(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'newBee' THEN t.user_id END)
                AS ai1_newbee_num_bets,

            COUNT(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' THEN t.user_id END)
                AS default_num_bets_bg,
            COUNT(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' THEN t.user_id END)
                AS ai_num_bets_bg,
            COUNT(
                CASE
                    WHEN
                        t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' AND t.mathtable = 'giftShop'
                        THEN t.user_id
                END
            )
                AS ai0_giftshop_num_bets_bg,
            COUNT(
                CASE
                    WHEN
                        t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' AND t.mathtable = 'carousels'
                        THEN t.user_id
                END
            )
                AS ai2_carousels_num_bets_bg,
            COUNT(
                CASE
                    WHEN
                        t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' AND t.mathtable = 'newBee'
                        THEN t.user_id
                END
            )
                AS ai1_newbee_num_bets_bg,

            COUNT(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' THEN t.user_id END)
                AS default_num_bets_fg,
            COUNT(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' THEN t.user_id END)
                AS ai_num_bets_fg,
            COUNT(
                CASE
                    WHEN
                        t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' AND t.mathtable = 'giftShop'
                        THEN t.user_id
                END
            )
                AS ai0_giftshop_num_bets_fg,
            COUNT(
                CASE
                    WHEN
                        t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' AND t.mathtable = 'carousels'
                        THEN t.user_id
                END
            )
                AS ai2_carousels_num_bets_fg,
            COUNT(
                CASE
                    WHEN
                        t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' AND t.mathtable = 'newBee'
                        THEN t.user_id
                END
            )
                AS ai1_newbee_num_bets_fg,

            -- total bet amount:
            SUM(t.bet_amount) AS total_bet,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS total_bet_bg,
            SUM(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' THEN t.bet_amount END) AS default_total_bet,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' THEN t.bet_amount END) AS ai_total_bet,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'giftShop' THEN t.bet_amount END)
                AS ai0_giftshop_total_bet,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'carousels' THEN t.bet_amount END)
                AS ai2_carousels_total_bet,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'newBee' THEN t.bet_amount END)
                AS ai1_newbee_total_bet,

            -- total payout amount:
            SUM(t.payout) AS total_payout,
            SUM(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' THEN t.payout END) AS default_total_payout,
            SUM(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' THEN t.payout END)
                AS default_total_payout_bg,
            SUM(CASE WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' THEN t.payout END)
                AS default_total_payout_fg,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' THEN t.payout END) AS ai_total_payout,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'BASE' THEN t.payout END)
                AS ai_total_payout_bg,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.bet_type = 'FREE' THEN t.payout END)
                AS ai_total_payout_fg,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'giftShop' THEN t.payout END)
                AS ai0_giftshop_total_payout,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'carousels' THEN t.payout END)
                AS ai2_carousels_total_payout,
            SUM(CASE WHEN t.ab_group_id = 'jojpin-9mokha-rexQug' AND t.mathtable = 'newBee' THEN t.payout END)
                AS ai1_newbee_total_payout,
            AVG(CASE WHEN EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 THEN EXTRACT(EPOCH FROM t.delta_t) END)
                AS avg_delta_t_seconds,
            AVG(
                CASE
                    WHEN
                        EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 AND t.ab_group_id != 'jojpin-9mokha-rexQug'
                        THEN EXTRACT(EPOCH FROM t.delta_t)
                END
            ) AS default_avg_delta_t,
            AVG(
                CASE
                    WHEN
                        EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 AND t.ab_group_id = 'jojpin-9mokha-rexQug'
                        THEN EXTRACT(EPOCH FROM t.delta_t)
                END
            ) AS ai_avg_delta_t
        FROM user_bets AS t
        WHERE t.user_bet_count >= 40
        GROUP BY t.activity_date
    )

    SELECT
        ds.activity_date,
        ds.total_daily_users,
        ds.ai_group_users,
        ds.default_group_users,
        ds.ai0_giftshop_users,
        ds.ai1_newbee_users,
        ds.ai2_carousels_users,

        ds.total_bet,
        ds.total_bet_bg,
        ds.default_total_bet,
        ds.ai_total_bet,
        ds.ai0_giftshop_total_bet,
        ds.ai1_newbee_total_bet,
        ds.ai2_carousels_total_bet,

        ds.total_payout,
        ds.default_total_payout,
        ds.default_total_payout_bg,
        ds.default_total_payout_fg,
        ds.ai_total_payout,
        ds.ai_total_payout_bg,
        ds.ai_total_payout_fg,
        ds.ai0_giftshop_total_payout,
        ds.ai1_newbee_total_payout,
        ds.ai2_carousels_total_payout,

        ds.avg_delta_t_seconds,
        ds.default_avg_delta_t,
        ds.ai_avg_delta_t,

        -- free game ratio
        ds.default_num_bets_fg * 1.0 / NULLIF(ds.default_num_bets, 0) AS default_fg_ratio,
        ds.ai_num_bets_fg * 1.0 / NULLIF(ds.ai_num_bets, 0) AS ai_fg_ratio,
        ds.ai0_giftshop_num_bets_fg * 1.0 / NULLIF(ds.ai0_giftshop_num_bets, 0) AS ai0_fiftshop_fg_ratio,

        -- Profit Calculations
        ds.ai1_newbee_num_bets_fg * 1.0 / NULLIF(ds.ai1_newbee_num_bets, 0) AS ai1_newbee_fg_ratio,
        ds.ai2_carousels_num_bets_fg * 1.0 / NULLIF(ds.ai2_carousels_num_bets, 0) AS ai2_carousels_fg_ratio,
        
        -- Number of bets per user:
        ds.default_num_bets / NULLIF(ds.default_group_users, 0) AS default_num_bets_per_user,
        ds.ai_num_bets / NULLIF(ds.ai_group_users, 0) AS ai_num_bets_per_user,
        ds.ai0_giftshop_num_bets / NULLIF(ds.ai0_giftshop_users, 0) AS ai0_giftshop_num_bets_per_user,

        -- Per-User Profit
        ds.ai1_newbee_num_bets / NULLIF(ds.ai1_newbee_users, 0) AS ai1_newbee_num_bets_per_user,
        ds.ai2_carousels_num_bets / NULLIF(ds.ai2_carousels_users, 0) AS ai2_carousels_num_bets_per_user,

        ds.default_total_bet / NULLIF(ds.default_group_users, 0) AS default_total_bet_per_user,
        ds.ai_total_bet / NULLIF(ds.ai_group_users, 0) AS ai_total_bet_per_user,
        (ds.default_total_payout - ds.default_total_bet) AS default_total_profit,

        -- Retention:
        ur.day0_num_users,
        ur.day1_num_users,
        ur.day3_num_users,

        ur.ai_day0_num_users,
        ur.ai_day1_num_users,
        ur.ai_day3_num_users,

        ai_giftshop_day0_num_users,
        ai_giftshop_day1_num_users,
        ai_giftshop_day3_num_users,

        ai_newbee_day0_num_users,
        ai_newbee_day1_num_users,
        ai_newbee_day3_num_users,

        ai_carousels_day0_num_users,
        ai_carousels_day1_num_users,
        ai_carousels_day3_num_users,


        (ds.ai_total_payout - ds.ai_total_bet) AS ai_total_profit,
        (ds.ai0_giftshop_total_payout - ds.ai0_giftshop_total_bet) AS ai0_giftshop_total_profit,
        (ds.ai1_newbee_total_payout - ds.ai1_newbee_total_bet) AS ai1_newbee_total_profit,
        (ds.ai2_carousels_total_payout - ds.ai2_carousels_total_bet) AS ai2_carousels_total_profit,
        (ds.default_total_payout - ds.default_total_bet) / NULLIF(ds.default_group_users, 0) AS default_profit_per_user,
        (ds.ai_total_payout - ds.ai_total_bet) / NULLIF(ds.ai_group_users, 0) AS ai_profit_per_user,

        -- RTP (Return to Player) Calculations (Fix: NULLIF for bet amounts AND RTP numerator error)
        ds.default_total_payout / NULLIF(ds.default_total_bet, 0) AS default_rtp,
        -- total bet for base game is same to total bet and free game has 0 bet amount:
        ds.default_total_payout_bg / NULLIF(ds.default_total_bet, 0) AS default_rtp_bg,

        ds.ai_total_payout / NULLIF(ds.ai_total_bet, 0) AS ai_rtp,
        ds.ai_total_payout_bg / NULLIF(ds.ai_total_bet, 0) AS ai_rtp_bg

    FROM daily_stats AS ds
    INNER JOIN user_retention AS ur
        ON ds.activity_date = ur.activity_date
    ORDER BY ds.activity_date DESC;

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

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/stats_by_date.parquet"
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df_rs)

    redshift_loader.close()
