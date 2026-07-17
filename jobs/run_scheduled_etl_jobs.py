"""
Run scheduled ETL/report jobs in a fixed sequence.

Intended usage:
  poetry run python jobs/run_scheduled_etl_jobs.py [--skip-daily_report] [--lookback-days N]

Always runs the ETLs incrementally. To force a full reload of a game's data,
run that job directly with its own --overwrite flag (long full reloads over the
bastion tunnel are fragile, so overwrite one job at a time and verify).

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
        "--lookback-days",
        type=int,
        default=3,
        help="Lookback days passed to operation_daily_report (default: 3).",
    )
    args = parser.parse_args()

    # Set args per script directly in this list.
    jobs: list[tuple[str, Path, list[str]]] = [
        (
            # --skip-ss01 disables the SS01 stats daily-report table and the
            # HG/PA ETLs that only feed it; the job still runs the PID
            # difference check (sent to Slack).
            "operation daily report (PID check only)",
            JOBS_DIR / "operation_daily_report" / "run_daily_report.py",
            ["--lookback-days", str(args.lookback_days), "--send-slack", "--skip-ss01"],
        ),
        (
            "ss01_wucaishen ETL",
            JOBS_DIR / "ss01_wucaishen" / "etl_game_stats_daily_by_user_group.py",
            [],
        ),
        (
            "ss02_deepdive ETL",
            JOBS_DIR / "ss02_deepdive" / "etl_game_stats_daily_by_user_group.py",
            [],
        ),
        (
            "ss03_mahjiang_streak ETL",
            JOBS_DIR / "ss03_mahjiang_streak" / "etl_game_stats_daily_by_user_group.py",
            [],
        ),
        (
            "ss06_pocket_soccer ETL",
            JOBS_DIR / "ss06_pocket_soccer" / "etl_game_stats_daily_by_user_group.py",
            [],
        ),
        (
            "fish_hunter ETL",
            JOBS_DIR / "fish_hunter" / "etl_game_stats_daily_by_user.py",
            [],
        ),
        (
            "operation daily weekly report ETL",
            JOBS_DIR / "operation_daily_report" / "etl_weekly_report_all_games.py",
            [],
        ),
    ]

    if args.skip_daily_report:
        print("[INFO] --skip-daily_report enabled, skipping first scheduled job.")
        jobs = jobs[1:]

    failures: list[str] = []

    for name, script_path, script_args in jobs:
        if not script_path.exists():
            print(f"[ERROR] Missing script: {script_path}")
            failures.append(name)
            continue

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
