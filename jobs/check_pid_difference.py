import os
from textwrap import dedent

from bituslabs_ds.config import (
    DEFAULT_BASTION_IP,
    LOCAL_ROOT,
    REDSHIFT_HOST,
    REDSHIFT_PORT,
    S3_BUCKET,
    get_redshift_password,
    get_redshift_user,
    setup_logging,
)
from bituslabs_ds.etl import AthenaBackend, DataLoader, RedshiftBackend

pids_to_ignore = {
    "B26",  # test server
    "LV1",
    "E52",
    "ER5",
    "LQ6",
    "LF1",
    "DS2",
    "MD6",
    "JL1",
    "ER2",
    "F19",
    "KD9",
    "LL0",
    "LQ5",
    "LJ8",
    "KY5",
    "JL5",
    "E41",
    "FS8",
    "LT4",
    "LO1",
    "J31",
    "JW3",
    "D70",
    "MJ4",
    "E31",
    "ED9",
    "KJ1",
    "G63",
    "H48",
    "KI6",
    "N27",
    "GN7",
    "GP7",
    "FE0",
    "KG1",
    "LO7",
    "FR7",
    "N89",
    "KE6",
    "LK2",
    "FN8",
    "MJ0",
    "LP2",
    "LX9",
    "EH9",
    "KU5",
    "A05",
    "LB9",
    "HC3",
    "LN1",
    "KW9",
    "HX1",
    "GN9",
    "LO5",
    "LJ3",
    "D13",
    "LO3",
    "A51",
    "DV5",
    "MI1",
    "LK1",
    "JW4",
    "HS9",
    "LR0",
    "HC2",
    "GG2",
    "E59",
    "HS7",
    "LO0",
    "R09",
    "CJ9",
    "DW1",
    "J75",
    "JQ9",
    "JS6",
    "MU8",
    "E04",
    "JJ3",
    "EE3",
    "LN7",
}

# Queries for different products
QUERIES = {
    "fish_hunter": {
        "redshift_query": dedent(
            """
            SELECT
                distinct t.op_code AS product_id
            FROM
                public.bullet AS t
            WHERE
                t.currency_type = 'CNY'
                AND t.game_id = 'FM01'
                AND t.op_code not in ('B26','TST','TSB','TSO') 
                -- AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) < '2025-12-29 06:00:00'
            """
        ),
        "athena_query": dedent(
            """
            SELECT
                distinct t.productid AS product_id
            FROM
                agfish.hunterorders AS t
            WHERE
                t.currency = 'CNY'
                AND t.gametype = 'HM3D'
                AND t.account != 0      -- Must have a bet amount
                AND t.fishcost != 0     -- Must have a fish cost
                -- Ensures standard order types and no weapon usage
                AND t.ordertype = 1
                AND t.remark = t.gametype
                AND t.weaponid IS NULL
                AND t.billtime > TIMESTAMP '2024-01-01 00:00:00'
            """
        ),
        "athena_database": "agfish",
        "athena_output": f"s3://{S3_BUCKET}/ds-data-fish_hunter/product_id_pa",
        "redshift_database": "transform-agfish-game",
        "output_path_redshift": f"{LOCAL_ROOT}/jobs/output_fish_hunter/product_id.parquet",
        "output_path_athena": f"{LOCAL_ROOT}/jobs/output_fish_hunter/product_id_pa.parquet",
    },
    "ss01_wucaishen": {
        "redshift_query": dedent(
            """
            SELECT
                distinct t.op_code AS product_id
            FROM
                public.fct_bet_orders AS t
            WHERE
                t.currency_type = 'CNY'
                AND t.status = 'COMPLETED'
                AND t.game_id = 'SS01'
                AND t.op_code not in ('B26','TST','TSB','TSO') 
                -- AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) < '2025-12-29 06:00:00'
            """
        ),
        "athena_query": dedent(
            """
            SELECT
                distinct t.productid AS product_id
            FROM
                ag_share_data.slotorders AS t
            WHERE
                t.currency = 'CNY'
                AND gametype = 'SB28' 
                AND flag != -8.0
            """
        ),
        "athena_database": "ag_share_data",
        "athena_output": f"s3://{S3_BUCKET}/ds-data-ss01/product_id_pa",
        "redshift_database": "slot-machine",
        "output_path_redshift": f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/product_id.parquet",
        "output_path_athena": f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/product_id_pa.parquet",
    },
    "ss03_majiang_streak": {
        "redshift_query": dedent(
            """
            SELECT
                distinct t.op_code AS product_id
            FROM
                public.fct_bet_orders AS t
            WHERE
                t.currency_type = 'CNY'
                AND t.status = 'COMPLETED'
                AND t.game_id = 'SS03'
                AND t.op_code not in ('B26','TST','TSB','TSO') 
                -- AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) < '2025-12-29 06:00:00'
            """
        ),
        "athena_query": dedent(
            """
            SELECT
                distinct t.productid AS product_id
            FROM
                ag_share_data.slotorders AS t
            WHERE
                t.currency = 'CNY'
                AND gametype = 'SB28' 
                AND flag != -8.0
            """
        ),
        "athena_database": "ag_share_data",
        "athena_output": f"s3://{S3_BUCKET}/ds-data-ss03_majiang_streak/product_id_pa",
        "redshift_database": "slot-machine",
        "output_path_redshift": f"{LOCAL_ROOT}/jobs/output_ss03_majiang_streak/product_id.parquet",
        "output_path_athena": f"{LOCAL_ROOT}/jobs/output_ss03_majiang_streak/product_id_pa.parquet",
    },
}


def run_pid_diff_check(
    athena_database,
    athena_output_location,
    athena_query,
    redshift_database,
    redshift_query,
    output_path_redshift,
    output_path_athena,
    pids_to_ignore,
    ctas_approach=False,
):
    # Athena DataLoader
    data_loader = DataLoader(
        backend=AthenaBackend(
            database=athena_database,
            output_location=athena_output_location,
            ctas_approach=ctas_approach,
        )
    )

    # Redshift DataLoader
    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host=REDSHIFT_HOST,
            database=redshift_database,
            user=get_redshift_user(),
            password=get_redshift_password(),
            port=REDSHIFT_PORT,
            bastion_ip=DEFAULT_BASTION_IP,
        )
    )

    try:
        df = redshift_loader.query_to_df(query=redshift_query, local_cache=output_path_redshift, reload=True)
        print(df)
    finally:
        # Always attempt to close connection, but do not suppress exceptions.
        redshift_loader.close()

    try:
        df_pa = data_loader.query_to_df(query=athena_query, local_cache=output_path_athena, reload=False)
        print(df_pa)
    finally:
        # Always attempt to close connection, but do not suppress exceptions.
        data_loader.close()

    diff_product_ids = set(df_pa["product_id"]) - set(df["product_id"]) - pids_to_ignore

    return sorted(diff_product_ids)


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    diff_product_ids_fishhunter = run_pid_diff_check(
        athena_database=QUERIES["fish_hunter"]["athena_database"],
        athena_output_location=QUERIES["fish_hunter"]["athena_output"],
        athena_query=QUERIES["fish_hunter"]["athena_query"],
        redshift_database=QUERIES["fish_hunter"]["redshift_database"],
        redshift_query=QUERIES["fish_hunter"]["redshift_query"],
        output_path_redshift=QUERIES["fish_hunter"]["output_path_redshift"],
        output_path_athena=QUERIES["fish_hunter"]["output_path_athena"],
        pids_to_ignore=pids_to_ignore,
        ctas_approach=True,  # Only fish_hunter needs CTAS
    )

    diff_product_ids_ss01_wucaishen = run_pid_diff_check(
        athena_database=QUERIES["ss01_wucaishen"]["athena_database"],
        athena_output_location=QUERIES["ss01_wucaishen"]["athena_output"],
        athena_query=QUERIES["ss01_wucaishen"]["athena_query"],
        redshift_database=QUERIES["ss01_wucaishen"]["redshift_database"],
        redshift_query=QUERIES["ss01_wucaishen"]["redshift_query"],
        output_path_redshift=QUERIES["ss01_wucaishen"]["output_path_redshift"],
        output_path_athena=QUERIES["ss01_wucaishen"]["output_path_athena"],
        pids_to_ignore=pids_to_ignore,
        ctas_approach=False,
    )

    diff_product_ids_ss03_majiang_streak = run_pid_diff_check(
        athena_database=QUERIES["ss03_majiang_streak"]["athena_database"],
        athena_output_location=QUERIES["ss03_majiang_streak"]["athena_output"],
        athena_query=QUERIES["ss03_majiang_streak"]["athena_query"],
        redshift_database=QUERIES["ss03_majiang_streak"]["redshift_database"],
        redshift_query=QUERIES["ss03_majiang_streak"]["redshift_query"],
        output_path_redshift=QUERIES["ss03_majiang_streak"]["output_path_redshift"],
        output_path_athena=QUERIES["ss03_majiang_streak"]["output_path_athena"],
        pids_to_ignore=pids_to_ignore,
        ctas_approach=False,
    )

    print("ss01_wucaishen 未开通PID：")
    print("--------------------------------")
    print(diff_product_ids_ss01_wucaishen)
    print("--------------------------------")

    print("ss03_majiang_streak 未开通PID：")
    print("--------------------------------")
    print(diff_product_ids_ss03_majiang_streak)
    print("--------------------------------")

    print("捕鱼未开通PID：")
    print("--------------------------------")
    print(diff_product_ids_fishhunter)
    print("--------------------------------")
