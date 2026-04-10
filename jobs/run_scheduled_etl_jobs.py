"""
Run scheduled ETL/report jobs in a fixed sequence.

Intended usage:
  poetry run python jobs/run_scheduled_etl_jobs.py [--skip-daily_report] [--overwrite]

This orchestrator is code-level job logic and should live under `jobs/`.
Infrastructure tooling (EventBridge/ECS/Step Functions/Terraform/CDK) should call
this script as a single entrypoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from bituslabs_ds.config import LOCAL_ROOT

PROJECT_ROOT = Path(LOCAL_ROOT)
JOBS_DIR = PROJECT_ROOT / "jobs"


def _run_python_script(script_path: Path, script_args: list[str]) -> bool:
    """Execute one Python script and stream output."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")])

    cmd = [sys.executable, str(script_path), *script_args]
    print(f"\n{'=' * 72}")
    print(f"Running: {script_path.relative_to(PROJECT_ROOT)}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 72}")

    started = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - started

    if result.returncode == 0:
        print(f"[OK] Finished in {elapsed:.1f}s")
        return True

    print(f"[ERROR] Failed with exit code {result.returncode} after {elapsed:.1f}s")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scheduled ETL/report jobs.")
    parser.add_argument(
        "--skip-daily_report",
        action="store_true",
        help="Skip the first job in the scheduled jobs list (operation daily report).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to ETL jobs that support it.",
    )
    args = parser.parse_args()

    # Set args per script directly in this list, and mark whether --overwrite is supported.
    jobs: list[tuple[str, Path, list[str], bool]] = [
        (
            "operation daily report",
            JOBS_DIR / "operation_daily_report" / "run_daily_report.py",
            ["--lookback-days", "1", "--send-slack"],
            False,
        ),
        (
            "ss01_wucaishen ETL",
            JOBS_DIR / "ss01_wucaishen" / "etl_game_stats_daily_by_user_group.py",
            [],
            True,
        ),
        (
            "ss02_deepdive ETL",
            JOBS_DIR / "ss02_deepdive" / "etl_game_stats_daily_by_user_group.py",
            [],
            True,
        ),
        (
            "fish_hunter ETL",
            JOBS_DIR / "fish_hunter" / "etl_game_stats_daily_by_user.py",
            [],
            True,
        ),
        (
            "operation daily weekly report ETL",
            JOBS_DIR / "operation_daily_report" / "etl_weekly_report_all_games.py",
            [],
            False,
        ),
    ]

    if args.skip_daily_report:
        print("[INFO] --skip-daily_report enabled, skipping first scheduled job.")
        jobs = jobs[1:]

    failures: list[str] = []

    for name, script_path, base_args, supports_overwrite in jobs:
        if not script_path.exists():
            print(f"[ERROR] Missing script: {script_path}")
            failures.append(name)
            continue

        script_args = [*base_args]
        if args.overwrite and supports_overwrite:
            script_args.append("--overwrite")

        ok = _run_python_script(script_path, script_args)
        if not ok:
            failures.append(name)

    print("\n" + "-" * 72)
    if failures:
        print("[RESULT] Completed with failures")
        for name in failures:
            print(f"  - {name}")
        return 1

    print("[RESULT] All scheduled jobs completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
