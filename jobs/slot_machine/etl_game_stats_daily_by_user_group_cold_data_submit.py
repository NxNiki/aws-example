"""Submit etl_game_stats_daily_by_user_group_cold_data.py as SageMaker PySpark
processing jobs, one per slot game.

Runs from a local machine. Jobs execute under the DataScience notebook role
(the only principal the slotmachine source bucket trusts) and run sequentially;
pass game ids as CLI args to run a subset, e.g.:

    poetry run python etl_game_stats_daily_by_user_group_cold_data_submit.py SS02 SS03
"""

import sys

import boto3
from sagemaker.session import Session
from sagemaker.spark.processing import PySparkProcessor

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET

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

OUTPUT_START = "2025-01-01"  # keep rows with activity date >= this
OUTPUT_END = "2026-07-29"  # exclusive; align to Monday / 1st so weekly/monthly periods are complete
AGG = "all"  # all | daily | weekly | monthly

INSTANCE_TYPE = "ml.m5.4xlarge"  # 16 vCPU, 64 GB memory per node
INSTANCE_COUNT = 3
# Shuffle spill from the per-user window functions lands on local disk; the
# SageMaker default of 30 GB per node is far too small for long windows.
VOLUME_SIZE_GB = 300

games = sys.argv[1:] or list(GAME_OUTPUT_ROOTS)
unknown = [g for g in games if g not in GAME_OUTPUT_ROOTS]
if unknown:
    raise SystemExit(f"unknown game ids: {unknown}; choose from {sorted(GAME_OUTPUT_ROOTS)}")

for game_id in games:
    output_root = GAME_OUTPUT_ROOTS[game_id]
    processor = PySparkProcessor(
        base_job_name=f"{game_id.lower()}-game-stats-cold-data",
        framework_version="3.3",
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        volume_size_in_gb=VOLUME_SIZE_GB,
        max_runtime_in_seconds=6 * 60 * 60,
        sagemaker_session=Session(boto3.Session(region_name=REGION)),
    )
    print(f"\n===== submitting {game_id} -> {output_root} =====")
    processor.run(
        submit_app=f"{LOCAL_ROOT}/jobs/slot_machine/etl_game_stats_daily_by_user_group_cold_data.py",
        arguments=[
            "--game-id",
            game_id,
            "--input-root",
            INPUT_ROOT,
            "--output-root",
            output_root,
            "--output-start",
            OUTPUT_START,
            "--output-end",
            OUTPUT_END,
            "--agg",
            AGG,
        ],
        spark_event_logs_s3_uri=f"{output_root}/spark-event-logs",
        logs=True,
        wait=True,
    )
