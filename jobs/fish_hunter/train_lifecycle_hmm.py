"""Train the FM01 player-lifecycle IO-HMM on the S3 feature table.

ETL job: retrains the lifecycle churn model (Low/Engaged/Lapsed + absorbing
STOP) on the ``selected_hmm_features_fm01_cny`` CSV produced by
feature_engineer_life_cycle.py, and uploads the two artifacts to S3 under
``<features-root>/hmm/<run-date>/``:
``selected_hmm_features_fm01_cny_iohmm_labels.csv`` (per user-bet-day: stage,
p_stop, risk tier) and ``..._iohmm_model.json`` (deployable artifact for
io_hmm_infer-style online scoring).

Behavior: downloads the single-file CSV export from S3, runs
bituslabs_ds.models.hmm_player_states.io_hmm.fit_iohmm (Golden 7 features,
drops each user's first bet-day, H=46 churn horizon, seed=42), then uploads
the outputs. Runs locally in minutes -- no SageMaker submission needed.

Run: poetry run python jobs/fish_hunter/train_lifecycle_hmm.py [--max-users N]
"""

import argparse
from datetime import date
from pathlib import Path

import boto3

from bituslabs_ds.config import REGION, S3_BUCKET
from bituslabs_ds.models.hmm_player_states.io_hmm import fit_iohmm
from bituslabs_ds.s3_utils import list_s3_files, parse_s3_path, upload_file_to_s3

FEATURES_ROOT = f"s3://{S3_BUCKET}/etl-results/lifecycle_feature_engineering/fm01_cny"
CSV_NAME = "selected_hmm_features_fm01_cny"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", default=FEATURES_ROOT, help="s3://... root of the feature outputs")
    parser.add_argument("--work-dir", default="jobs/output_fish_hunter/lifecycle_hmm", help="local working directory")
    parser.add_argument("--n-behavior", type=int, default=3)
    parser.add_argument("--max-users", type=int, default=None, help="subsample for a smoke run (no artifacts written)")
    return parser.parse_args()


def download_features_csv(features_root: str, work_dir: Path) -> Path:
    bucket, prefix = parse_s3_path(f"{features_root}/csv/{CSV_NAME}/")
    parts = [u for u in list_s3_files(bucket, prefix) if u.endswith(".csv")]
    if len(parts) != 1:
        raise ValueError(f"expected exactly 1 csv part under s3://{bucket}/{prefix}, found {len(parts)}")
    local = work_dir / f"{CSV_NAME}.csv"
    part_bucket, part_key = parse_s3_path(parts[0])
    print(f"downloading {parts[0]} -> {local}")
    boto3.client("s3", region_name=REGION).download_file(part_bucket, part_key, str(local))
    return local


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_features_csv(args.features_root, work_dir)
    fit_iohmm(str(csv_path), n_behavior=args.n_behavior, max_users=args.max_users)

    if args.max_users is not None:
        print("smoke run only -- artifacts not uploaded")
        return

    bucket, root_prefix = parse_s3_path(args.features_root)
    run_prefix = f"{root_prefix}/hmm/{date.today().isoformat()}"
    for suffix in ["_iohmm_labels.csv", "_iohmm_model.json"]:
        local = csv_path.with_name(csv_path.stem + suffix)
        uri = upload_file_to_s3(local, bucket, f"{run_prefix}/{local.name}")
        print("uploaded:", uri)


if __name__ == "__main__":
    main()
