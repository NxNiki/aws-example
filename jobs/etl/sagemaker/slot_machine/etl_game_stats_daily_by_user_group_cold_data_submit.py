"""Submit etl_game_stats_daily_by_user_group_cold_data.py as SageMaker PySpark
processing jobs, one per slot game.

Runs from a local machine or a scheduler. Jobs execute under the DataScience
notebook role (the only principal the slotmachine source bucket trusts) and
run sequentially.

Without ``--start``/``--end`` this is the daily incremental run: it recomputes
a rolling window (last LOOKBACK_DAYS Beijing days through tomorrow), and the
job's dynamic period= overwrite replaces exactly the recomputed periods -- no
compaction or key-dedup step needed. Weekly/monthly stay correct because the
job truncates the window start to the period start and rescans full periods.

    # daily incremental, all games
    poetry run python etl_game_stats_daily_by_user_group_cold_data_submit.py
    # explicit backfill window for a subset
    poetry run python ..._submit.py SS02 SS03 --start 2025-01-01 --end 2026-07-29
"""

import argparse
from datetime import datetime, timedelta, timezone

import boto3
from sagemaker.session import Session

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor

LOOKBACK_DAYS = 3

# NOT the repo-wide SAGEMAKER_ROLE: the slotmachine bucket policy (owned by
# another account) only trusts this role -- see feature_engineer_life_cycle_submit.py.
ROLE = "arn:aws:iam::338568447110:role/service-role/AmazonSageMaker-ExecutionRole-20250102T151291"

# game_id partitioning makes each per-game run prune to its own partition.
INPUT_ROOT = "s3://slotmachine-production-data-warehouse/transformed_data/partition_cold_data/bet_order"

GAME_OUTPUT_ROOTS = {
    "SS01": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss01_wucaishen_v2_cold_data",
    "SS01A": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss01a_golden_goal_v2_cold_data",
    "SS02": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss02_deepdive_v2_cold_data",
    "SS03": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_mahjiang_streak_v2_cold_data",
    "SS06": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss06_pocket_soccer_v2_cold_data",
}

AGG = "all"  # all | daily | weekly | monthly

INSTANCE_TYPE = "ml.m5.4xlarge"  # 16 vCPU, 64 GB memory per node
INSTANCE_COUNT = 3
# Shuffle spill from the per-user window functions lands on local disk; the
# SageMaker default of 30 GB per node is far too small for long windows.
VOLUME_SIZE_GB = 300

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("games", nargs="*", choices=sorted(GAME_OUTPUT_ROOTS), help="subset of games (default all)")
parser.add_argument("--start", help="activity date >= this (default: rolling lookback window)")
parser.add_argument("--end", help="activity date < this (default: tomorrow in Beijing time)")
args = parser.parse_args()

bj_today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
output_start = args.start or str(bj_today - timedelta(days=LOOKBACK_DAYS))
output_end = args.end or str(bj_today + timedelta(days=1))

games = args.games or list(GAME_OUTPUT_ROOTS)

for game_id in games:
    output_root = GAME_OUTPUT_ROOTS[game_id]
    processor = spark_processor(
        f"{game_id.lower()}-game-stats-cold-data",
        Session(boto3.Session(region_name=REGION)),
        volume_size_gb=VOLUME_SIZE_GB,
        max_runtime_hours=6,
    )
    print(f"\n===== submitting {game_id} -> {output_root} =====")
    processor.run(
        submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_game_stats_daily_by_user_group_cold_data.py",
        submit_py_files=SPARK_COMMON_PY_FILES,
        arguments=[
            "--game-id",
            game_id,
            "--input-root",
            INPUT_ROOT,
            "--output-root",
            output_root,
            "--output-start",
            output_start,
            "--output-end",
            output_end,
            "--agg",
            AGG,
        ],
        spark_event_logs_s3_uri=f"{output_root}/spark-event-logs",
        logs=True,
        wait=True,
    )
