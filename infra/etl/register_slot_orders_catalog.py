"""Register the Athena table ``bituslabs_ds.slot_orders_ab_group``.

Points the Glue catalog at the ``output_slot_orders_ab_group/orders`` dataset
(slot bet orders + the policy-correct ``ab_group`` column — see
``jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py``), the same
external-parquet-table pattern as ``bituslabs_ds.fish_hunter_daily_stats``.
Column types are read from an actual parquet footer, so run this AFTER the
first backfill. Partition projection (game_id enum × period date) means no
crawler and no MSCK — new partitions are queryable the moment they land.

Usage:
    poetry run python infra/etl/register_slot_orders_catalog.py
"""

from typing import Any

import boto3
import polars as pl

from bituslabs_ds.config import REGION, S3_BUCKET
from bituslabs_ds.s3_utils import expand_paths_to_files

DATABASE = "bituslabs_ds"
TABLE = "slot_orders_ab_group"
LOCATION = f"s3://{S3_BUCKET}/etl-results/jobs/output_slot_orders_ab_group/orders/"
GAMES = ["SS01", "SS01A", "SS02", "SS03", "SS06"]
PERIOD_RANGE_START = "2025-11-24"  # earliest slot cold data (SS01)

_HIVE_TYPES: dict[Any, str] = {
    pl.Int64: "bigint",
    pl.Int32: "int",
    pl.Float64: "double",
    pl.String: "string",
    pl.Date: "date",
    pl.Boolean: "boolean",
}


def _columns() -> list[dict]:
    files = expand_paths_to_files([LOCATION.rstrip("/")])
    data_files = [f for f in files if f.endswith(".parquet")]
    if not data_files:
        raise SystemExit(f"No parquet under {LOCATION} — run the backfill first.")
    schema = pl.scan_parquet(data_files[0]).collect_schema()
    cols = []
    for name, dtype in schema.items():
        if name in ("game_id", "period"):
            continue
        if isinstance(dtype, pl.Datetime):
            hive = "timestamp"
        else:
            hive = _HIVE_TYPES.get(dtype.base_type(), "string")
        cols.append({"Name": name, "Type": hive})
    return cols


def main() -> None:
    table_input: dict[str, Any] = {
        "Name": TABLE,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "EXTERNAL": "TRUE",
            "classification": "parquet",
            "projection.enabled": "true",
            "projection.game_id.type": "enum",
            "projection.game_id.values": ",".join(GAMES),
            "projection.period.type": "date",
            "projection.period.format": "yyyy-MM-dd",
            "projection.period.range": f"{PERIOD_RANGE_START},NOW+1DAYS",
            "storage.location.template": LOCATION + "game_id=${game_id}/period=${period}",
        },
        "PartitionKeys": [
            {"Name": "game_id", "Type": "string"},
            {"Name": "period", "Type": "string"},
        ],
        "StorageDescriptor": {
            "Columns": _columns(),
            "Location": LOCATION,
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"},
        },
    }
    glue = boto3.client("glue", region_name=REGION)
    try:
        glue.update_table(DatabaseName=DATABASE, TableInput=table_input)
        print(f"Updated {DATABASE}.{TABLE}")
    except glue.exceptions.EntityNotFoundException:
        glue.create_table(DatabaseName=DATABASE, TableInput=table_input)
        print(f"Created {DATABASE}.{TABLE}")
    print(f"  location: {LOCATION}")
    for c in table_input["StorageDescriptor"]["Columns"]:
        print(f"  {c['Name']}: {c['Type']}")


if __name__ == "__main__":
    main()
