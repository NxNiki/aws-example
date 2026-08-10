"""Create/update the daily game cold-data ETL schedule.

Builds a SageMaker Pipeline (``slot-cold-data-daily`` -- the name predates the
fish_hunter step and is pinned by the admin-managed EventBridge schedule) with
one PySpark processing step per slot game plus a final fish_hunter (FM01)
step, chained sequentially to stay inside the account's 4x ml.m5.4xlarge
processing quota, and an EventBridge Scheduler rule that starts it once a day.
Steps pass no date window, so each run recomputes the job's rolling
incremental window (last 3 Beijing days; dynamic period= overwrite leaves
history untouched).

Shared pipeline/schedule plumbing lives in ``bituslabs_ds.sagemaker_etl``.

Usage:
    poetry run python infra/etl/deploy_slot_cold_data_pipeline.py            # upsert pipeline + schedule
    poetry run python infra/etl/deploy_slot_cold_data_pipeline.py --run-now  # also start one execution
"""

import argparse

import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

from bituslabs_ds.config import LOCAL_ROOT, REGION, S3_BUCKET
from bituslabs_ds.sagemaker_etl import SPARK_COMMON_PY_FILES, spark_processor, upsert_pipeline_with_schedule

PIPELINE_NAME = "slot-cold-data-daily"
SCHEDULE_NAME = "slot-cold-data-daily"
# 9:10 AM LA = ~00:10 next day Beijing: the whole Beijing day is complete in
# the warehouse (with 10 min of margin) before the orders copy runs.
SCHEDULE_CRON = "cron(10 9 * * ? *)"
SCHEDULER_ROLE_NAME = "slot-cold-data-scheduler-role"

SLOT_INPUT_ROOT = "s3://slotmachine-production-data-warehouse/transformed_data/partition_cold_data/bet_order"
# The orders dataset (raw bets + policy-correct ab_group; backs the Athena
# table bituslabs_ds.slot_orders_ab_group). Built FIRST each day; every other
# slot step reads it instead of the warehouse, so ab_group is derived once.
ORDERS_OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_slot_orders_ab_group"
ORDERS_DATA_ROOT = f"{ORDERS_OUTPUT_ROOT}/orders"
GAME_OUTPUT_ROOTS = {
    "SS01": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss01_wucaishen_v2_cold_data",
    "SS01A": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss01a_golden_goal_v2_cold_data",
    "SS02": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss02_deepdive_v2_cold_data",
    "SS03": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss03_mahjiang_streak_v2_cold_data",
    "SS06": f"s3://{S3_BUCKET}/etl-results/jobs/output_ss06_pocket_soccer_v2_cold_data",
}

FISH_INPUT_ROOT = "s3://oceanhunter-production-data-warehouse/transformed_data/cold_data/bullet"
FISH_OUTPUT_ROOT = f"s3://{S3_BUCKET}/etl-results/jobs/output_fish_hunter_v2_cold_data"

# Cohort variants appended after the base games, (game_id, ab_group, min_run):
# only the ab_group partition's bets are scanned and only stretches of
# >= min_run consecutive same-mathtable bets survive, written to the game's
# _<group>_run<N> root. ss03's AI mathtable-combo dashboard reads the SS03
# AI run30 dataset.
GAME_VARIANTS = [
    ("SS03", "AI", 30),
]


def build_pipeline(session: PipelineSession) -> Pipeline:
    steps: list[ProcessingStep] = []

    # Orders first: everything downstream reads it (rolling 3-day window).
    orders_processor = spark_processor("slot-orders-ab-group-daily", session)
    orders_step_args = orders_processor.run(
        submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_slot_orders_ab_group.py",
        submit_py_files=SPARK_COMMON_PY_FILES,
        arguments=[
            "--input-root",
            SLOT_INPUT_ROOT,
            "--output-root",
            ORDERS_OUTPUT_ROOT,
        ],
    )
    steps.append(ProcessingStep(name="etl-slot-orders-ab-group", step_args=orders_step_args))

    for game_id, output_root in GAME_OUTPUT_ROOTS.items():
        processor = spark_processor(f"{game_id.lower()}-cold-data-daily", session)
        step_args = processor.run(
            submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_game_stats_daily_by_user_group_cold_data.py",
            submit_py_files=SPARK_COMMON_PY_FILES,
            arguments=[
                "--game-id",
                game_id,
                "--input-root",
                ORDERS_DATA_ROOT,
                "--output-root",
                output_root,
            ],
        )
        steps.append(
            ProcessingStep(
                name=f"etl-{game_id.lower()}",
                step_args=step_args,
                depends_on=[steps[-1].name],
            )
        )

    for game_id, ab_group, min_run in GAME_VARIANTS:
        variant = f"{ab_group.lower()}-run{min_run}"
        processor = spark_processor(f"{game_id.lower()}-{variant}-cold-data-daily", session)
        step_args = processor.run(
            submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/slot_machine/etl_game_stats_daily_by_user_group_cold_data.py",
            submit_py_files=SPARK_COMMON_PY_FILES,
            arguments=[
                "--game-id",
                game_id,
                "--input-root",
                ORDERS_DATA_ROOT,
                "--output-root",
                f"{GAME_OUTPUT_ROOTS[game_id]}_{ab_group.lower()}_run{min_run}",
                "--min-mathtable-run",
                str(min_run),
                "--ab-group",
                ab_group,
            ],
        )
        steps.append(
            ProcessingStep(
                name=f"etl-{game_id.lower()}-{variant}",
                step_args=step_args,
                depends_on=[steps[-1].name],
            )
        )

    # fish_hunter (FM01) runs last: same 3-node footprint, different source
    # bucket and job script. The bullet table is higher-volume than bet_order
    # and the monthly level rescans the whole current month plus the 30-day
    # kill-streak lookback, hence the bigger spill volume and runtime cap.
    fish_processor = spark_processor("fm01-cold-data-daily", session, volume_size_gb=200, max_runtime_hours=3)
    fish_step_args = fish_processor.run(
        submit_app=f"{LOCAL_ROOT}/jobs/etl/sagemaker/fish_hunter/etl_game_stats_daily_by_user_cold_data.py",
        submit_py_files=SPARK_COMMON_PY_FILES,
        arguments=[
            "--input-root",
            FISH_INPUT_ROOT,
            "--output-root",
            FISH_OUTPUT_ROOT,
        ],
    )
    steps.append(ProcessingStep(name="etl-fm01", step_args=fish_step_args, depends_on=[steps[-1].name]))
    return Pipeline(name=PIPELINE_NAME, steps=steps, sagemaker_session=session)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-now", action="store_true", help="start one pipeline execution after upsert")
    args = parser.parse_args()

    session = PipelineSession(boto_session=boto3.Session(region_name=REGION))
    upsert_pipeline_with_schedule(
        build_pipeline(session),
        schedule_name=SCHEDULE_NAME,
        cron=SCHEDULE_CRON,
        scheduler_role_name=SCHEDULER_ROLE_NAME,
        description="Daily slot-machine + fish_hunter cold-data ETL (rolling 3-day incremental window)",
        run_now=args.run_now,
    )


if __name__ == "__main__":
    main()
