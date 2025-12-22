import os
from textwrap import dedent

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.etl import AthenaBackend, DataLoader, RedshiftBackend

query_pa = dedent(
    f"""
    SELECT
        distinct t.productid AS product_id
    FROM
        ag_share_data.slotorders AS t
    WHERE
        t.currency = 'CNY'
        AND gametype = 'SB28' 
        AND flag != -8.0
    """
)

query = dedent(
    f"""
    SELECT
        distinct t.op_code AS product_id
    FROM
        public.fct_bet_orders AS t
    WHERE
        t.currency_type = 'CNY'
        AND t.status = 'COMPLETED'
        AND t.game_id = 'SS01'
        AND t.op_code != 'B26'
        -- AND CONVERT_TIMEZONE('UTC', 'Asia/Shanghai', t.created_at) < '2025-12-19 06:00:00'
    """
)


if __name__ == "__main__":

    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")

    data_loader = DataLoader(
        backend=AthenaBackend(
            database="agfish",
            output_location=f"s3://{S3_BUCKET}/ds-data-ss01/product_id_pa",
        )
    )

    redshift_loader = DataLoader(
        backend=RedshiftBackend(
            host="production-redshift-cluster.cwiqzcm13zcn.ap-southeast-1.redshift.amazonaws.com",
            database="slot-machine",
            user="anaylsis_user",
            password="oZ4ztMx0yEXPLbJL733L",
            port=5439,
        )
    )

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/product_id.parquet"
    df = redshift_loader.query_to_df(query=query, local_cache=file_path, reload=True)
    print(df)

    file_path = f"{LOCAL_ROOT}/jobs/output_ss01_wucaishen/product_id_pa.parquet"
    df_pa = data_loader.query_to_df(query=query_pa, local_cache=file_path, reload=False)
    print(df_pa)
    data_loader.close()

    pids_to_ignore = {
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

    # Get the elements in df_pa, but not in df, both have only one column named 'product_id'
    diff_product_ids = set(df_pa["product_id"]) - set(df["product_id"]) - pids_to_ignore
    print("Product IDs in df_pa but not in df:")
    print(diff_product_ids)
