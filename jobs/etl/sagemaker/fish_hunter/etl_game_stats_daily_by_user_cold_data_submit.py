"""Submit etl_game_stats_daily_by_user_cold_data.py as a SageMaker PySpark processing job.

Manual/backfill launcher; the scheduled daily run is the ``etl-fm01`` step of
the ``slot-cold-data-daily`` pipeline (infra/etl/deploy_slot_cold_data_pipeline.py).
Reads the fish_bullets_group_tag bullets dataset (build it first with
etl_fish_bullets_group_tag_submit.py); the daily/weekly/monthly per-user
stats are written to S3 under OUTPUT_ROOT.
"""

import boto3
from sagemaker.session import Session

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor

INPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_fish_bullets_group_tag/bullets"
OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_fish_hunter_v3_cold_data"

OUTPUT_START = "2026-01-24"  # keep rows with activity date >= this
OUTPUT_END = "2026-08-15"  # exclusive; align to Monday / 1st so weekly/monthly periods are complete
AGG = "all"  # all | daily | weekly | monthly

# Shuffle spill from the per-user window functions lands on local disk; full
# history needs far more than the 100 GB daily-incremental default.
VOLUME_SIZE_GB = 300

# True blocks and streams the job logs (per-level summaries) to this terminal;
# False submits and returns -- watch the job in SageMaker console -> Processing jobs.
WAIT = True

processor = spark_processor(
    "fm01-game-stats-cold-data",
    Session(boto3.Session(region_name=REGION)),
    volume_size_gb=VOLUME_SIZE_GB,
    max_runtime_hours=6,
)

processor.run(
    submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/fish_hunter/etl_game_stats_daily_by_user_cold_data.py",
    submit_py_files=SPARK_COMMON_PY_FILES,
    arguments=[
        "--input-root",
        INPUT_ROOT,
        "--output-root",
        OUTPUT_ROOT,
        "--output-start",
        OUTPUT_START,
        "--output-end",
        OUTPUT_END,
        "--agg",
        AGG,
    ],
    spark_event_logs_s3_uri=f"{OUTPUT_ROOT}/spark-event-logs",
    logs=WAIT,
    wait=WAIT,
)
