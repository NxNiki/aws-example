"""
Run scheduled ETL/report jobs in a fixed sequence.

Intended usage:
  poetry run python jobs/run_scheduled_etl_jobs.py [--bastion-ip IP]

This orchestrator is code-level job logic and should live under `jobs/`.
Infrastructure tooling (EventBridge/ECS/Step Functions/Terraform/CDK) should call
this script as a single entrypoint.
"""

from __future__ import annotations

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
    # Set args per script directly in this list, e.g. ["--overwrite"] or ["--bastion-ip", "..."].
    jobs: list[tuple[str, Path, list[str]]] = [
        (
            "operation daily report",
            JOBS_DIR / "operation_daily_report" / "run_daily_report.py",
            ["--lookback-days", "1", "--send-slack"],
        ),
        (
            "ss01_wucaishen ETL",
            JOBS_DIR / "ss01_wucaishen" / "etl_game_stats_daily_by_user_group.py",
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
