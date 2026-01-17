import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.etl import DataLoader, RedshiftBackend

# TODO:
# add max/min user daily profit

query = dedent(
    """
    WITH user_bets AS (
    SELECT
        t.user_id,
        t.script_id AS mathtable,
        t.bet_amount,
        t.actual_payout AS payout,
        t.bet_type,
        t.actual_payout - t.bet_amount AS profit,
        TRUNC(DATEADD(hour, -6, CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at))) AS activity_date,
        t.partition_ab[0] AS ab_group_id,
        t.created_at - LAG(t.created_at) OVER (PARTITION BY t.user_id ORDER BY t.created_at) AS delta_t,
        COUNT(t.user_id) OVER (PARTITION BY t.user_id) AS user_bet_count
    FROM
        public.fct_bet_orders AS t
    WHERE
        CONVERT_TIMEZONE('UTC', 'America/Los_Angeles', t.created_at) >= '2026-01-10 06:00:00'
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
            COUNT(distinct t1.user_id) AS day0_num_users,
            COUNT(distinct t2.user_id) AS day1_num_users,
            COUNT(distinct t3.user_id) AS day3_num_users
        FROM
            daily_login AS t1
        LEFT JOIN daily_login AS t2
            ON t2.activity_date = DATE_ADD('day', -1, t1.activity_date) AND t1.user_id = t2.user_id
        LEFT JOIN daily_login AS t3
            ON t3.activity_date = DATE_ADD('day', -3, t1.activity_date) AND t1.user_id = t3.user_id
        GROUP BY t1.activity_date, t1.ai_group
    ),

    daily_stats AS (
        SELECT
            t.activity_date,
            t.ai_group,

            COUNT(DISTINCT t.user_id) AS num_active_users,

            -- total number of bets:
            COUNT(t.user_id) AS num_bets,
            COUNT(CASE WHEN t.bet_type = 'BASE' THEN t.user_id END) AS num_bets_bg,
            COUNT(CASE WHEN t.bet_type = 'FREE' THEN t.user_id END) AS num_bets_fg,

            -- total bet amount:
            SUM(t.bet_amount) AS total_bet,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.bet_amount END) AS total_bet_bg,

            -- total payout amount:
            SUM(t.payout) AS total_payout,
            SUM(CASE WHEN t.bet_type = 'BASE' THEN t.payout END) AS total_payout_bg,
            SUM(CASE WHEN t.bet_type = 'FREE' THEN t.payout END) AS total_payout_fg,

            AVG(CASE WHEN EXTRACT(EPOCH FROM t.delta_t) BETWEEN 0 AND 86400 THEN EXTRACT(EPOCH FROM t.delta_t) END)
                AS avg_delta_t_seconds

        FROM user_bets_group AS t
        WHERE t.user_bet_count >= 40
        GROUP BY t.activity_date, t.ai_group
    )

    SELECT
        ds.activity_date,
        ds.ai_group,

        ds.num_active_users,

        ds.num_bets,
        ds.num_bets_bg,
        ds.num_bets_fg,

        ds.total_bet,
        ds.total_bet_bg,

        ds.total_payout,
        ds.total_payout_bg,
        ds.total_payout_fg,

        ds.avg_delta_t_seconds,

        -- free game ratio
        ur.day0_num_users,

        -- Number of bets and total bet per user:
        ur.day1_num_users,
        ur.day3_num_users,

        -- Profit Calculations
        ds.num_bets_fg * 1.0 / NULLIF(ds.num_bets, 0) AS fg_ratio,

        -- Per-User Bet
        ds.num_bets / NULLIF(ds.num_active_users, 0) AS num_bets_per_user,
        ds.total_bet / NULLIF(ds.num_active_users, 0) AS total_bet_per_user,
        (ds.total_payout - ds.total_bet) AS total_profit,

        -- RTP (Return to Player) Calculations (Fix: NULLIF for bet amounts AND RTP numerator error)
        ds.total_payout / NULLIF(ds.total_bet, 0) AS rtp,
        -- total bet for base game is same to total bet and free game has 0 bet amount:
        ds.total_payout_bg / NULLIF(ds.total_bet, 0) AS rtp_bg

    FROM daily_stats AS ds
    INNER JOIN user_retention AS ur
        ON ds.activity_date = ur.activity_date AND ds.ai_group = ur.ai_group
    ORDER BY ds.activity_date DESC, ds.ai_group DESC
    ;

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
    print(
        df_rs[
            [
                "activity_date",
                "ai_group",
                "num_active_users",
                "num_bets_per_user",
                "total_bet_per_user",
                "rtp",
                "rtp_bg",
                "day0_num_users",
                "day1_num_users",
                "day3_num_users",
            ]
        ]
    )

    redshift_loader.close()
