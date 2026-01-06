import os
from textwrap import dedent

import pandas as pd

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

DATE_START_HOUR = 6


def generate_query():

    query = dedent(
        f"""
        WITH user_bets AS (
            SELECT
                t.user_id,
                t.created_at,
                TRUNC(DATEADD(hour, -{DATE_START_HOUR}, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS activity_date,
                t.script_id AS mathtable,
                t.bet_amount,
                t.actual_payout AS payout,
                t.bet_type,
                t.actual_payout - t.bet_amount AS profit,
                t.partition_ab[0] AS ab_group_id,
                t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS delta_t,
                COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count,
                CASE
                    WHEN LAG(t.script_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at) IS NULL THEN 0
                    WHEN LAG(t.script_id) OVER (PARTITION BY t.user_id ORDER BY t.created_at) <> t.script_id THEN 1
                    ELSE 0
                    END AS mathtable_change
            FROM
                public.fct_bet_orders AS t
            WHERE
                CONVERT_TIMEZONE('UTC', 'America/Los_Angeles', t.created_at) >= '2025-12-21 06:00:00'
            AND t.currency_type = 'CNY'
            AND t.status = 'COMPLETED'
            AND t.game_id = 'SS01'
            AND t.op_code not in ('B26','TST','TSB','TSO')
        ),

            user_bets_group AS (
                SELECT
                    t.created_at,
                    t.activity_date,
                    t.user_id,
                    t.bet_amount,
                    t.payout,
                    t.bet_type,
                    t.profit,
                    t.delta_t,
                    t.user_bet_count,
                    t.mathtable_change,
                    CASE
                        WHEN t.ab_group_id != 'jojpin-9mokha-rexQug' THEN 'Default'
                        ELSE 'AI'
                        END AS ai_group
                FROM
                    user_bets AS t

                UNION ALL

                SELECT
                    t.created_at,
                    t.activity_date,
                    t.user_id,
                    t.bet_amount,
                    t.payout,
                    t.bet_type,
                    t.profit,
                    t.delta_t,
                    t.user_bet_count,
                    t.mathtable_change,
                    t.mathtable AS ai_group
                FROM
                    user_bets AS t
                WHERE
                    t.ab_group_id = 'jojpin-9mokha-rexQug'
            ),

            daily_login AS (
                SELECT
                    t.user_id,
                    MIN(t.created_at) as first_bet_time,
                    MAX(t.created_at) as last_bet_time,
                    t.ai_group
                FROM
                    user_bets_group AS t
                GROUP BY t.user_id, t.ai_group, t.activity_date
            ),

            user_retention AS (
                SELECT
                    t1.ai_group,
                    COUNT(DISTINCT t1.user_id) AS day0_num_users,
                    COUNT(DISTINCT t2.user_id) AS day1_num_users,
                    COUNT(DISTINCT t3.user_id) AS day3_num_users
                FROM
                    daily_login AS t1
                        LEFT JOIN daily_login AS t2
                                ON t2.first_bet_time < DATE_ADD('hour', 48, t1.first_bet_time) AND t2.last_bet_time >= DATE_ADD('hour', 24, t1.first_bet_time) AND t1.user_id = t2.user_id
                        LEFT JOIN daily_login AS t3
                                ON t3.first_bet_time < DATE_ADD('hour', 96, t1.first_bet_time) AND t3.last_bet_time >= DATE_ADD('hour', 72, t1.first_bet_time) AND t1.user_id = t3.user_id
                GROUP BY t1.ai_group
            ),

            user_stats AS (
                SELECT
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
                        AS user_avg_delta_t_seconds,

                    SUM(t.mathtable_change) AS user_mathtable_change

                FROM user_bets_group AS t
                WHERE t.user_bet_count >= 40
                GROUP BY t.ai_group, t.user_id
            ),

            group_stats AS (
                SELECT
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
                GROUP BY t.ai_group
            )

        SELECT
            us.ai_group,
            us.user_id,
            us.user_mathtable_change,

            gs.num_active_users,

            gs.total_num_bets,
            gs.total_num_bets_bg,
            gs.total_num_bets_fg,
            gs.total_num_bets * 1.0 / NULLIF(gs.num_active_users, 0) AS total_num_bets_per_user,

            gs.total_bet,
            gs.total_bet_bg,
            gs.total_bet * 1.0 / NULLIF(gs.num_active_users, 0) AS total_bet_per_user,
            gs.total_bet * 1.0 / gs.total_num_bets_bg AS avg_bet_amount_bg,

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

            -- Number of bets and total bet per user:
            us.user_num_bets,
            us.user_num_bets_bg,
            us.user_num_bets_fg,

            us.user_total_bet,
            us.user_total_bet * 1.0 / NULLIF(us.user_num_bets_bg, 0) AS user_avg_bet_amount_bg,

            -- free game ratio
            us.user_num_bets_fg * 1.0 / NULLIF(us.user_num_bets, 0) AS user_fg_ratio,
            gs.total_num_bets_fg * 1.0 / NULLIF(gs.total_num_bets, 0) AS fg_ratio,

            -- Profit Calculations
            (gs.total_payout - gs.total_bet) AS total_profit,
            (gs.total_payout - gs.total_bet) * 1.0 / gs.num_active_users AS total_profit_per_user,
            (us.user_total_payout - us.user_total_bet) AS user_total_profit,

            -- RTP (Return to Player) Calculations (Fix: NULLIF for bet amounts AND RTP numerator error)
            gs.total_payout / NULLIF(gs.total_bet, 0) AS rtp,
            -- total bet for base game is same to total bet and free game has 0 bet amount:
            gs.total_payout_bg / NULLIF(gs.total_bet, 0) AS rtp_bg,
            us.user_total_payout / NULLIF(us.user_total_bet, 0) AS user_rtp,
            us.user_total_payout_bg / NULLIF(us.user_total_bet, 0) AS user_rtp_bg

        FROM user_stats AS us
                INNER JOIN group_stats AS gs
                            ON us.ai_group = gs.ai_group
                INNER JOIN user_retention AS ur
                            ON  us.ai_group = ur.ai_group
        ORDER BY us.ai_group DESC, us.user_id DESC;



        """
    )

    return query


def execute_query(redshift_loader, output_file):

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/{output_file}.parquet"
    query = generate_query()
    df_rs = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(
        df_rs[
            [
                "user_id",
                "ai_group",
                "user_mathtable_change",
                "num_active_users",
                "user_num_bets",
                "user_total_bet",
                "user_avg_bet_amount_bg",
                "rtp",
                "rtp_bg",
                "day0_num_users",
                "day1_num_users",
                "day3_num_users",
            ]
        ]
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

    execute_query(redshift_loader, "all_user_stats")

    redshift_loader.close()
