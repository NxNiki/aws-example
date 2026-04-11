"""
ETL dispatcher: run one or more ETL jobs.

Usage:
  python etl_dispatcher.py <job_path> [job_path ...] [-- job_args...]
  python etl_dispatcher.py job1 job2 -- --bastion-ip 13.215.212.244

Examples:
  python etl_dispatcher.py fish_hunter/etl_game_stats_daily_by_user.py -- --bastion-ip 13.215.212.244
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from bituslabs_ds.config import LOCAL_ROOT


def _run_one_job(job_relative_path: str, job_args: list) -> bool:
    """Run a single ETL job. Returns True on success, False on failure."""
    jobs_root = LOCAL_ROOT / "jobs"
    job_file_path = (jobs_root / job_relative_path).resolve()

    if not job_file_path.exists():
        print(f"Error: Job file not found at {job_file_path}")
        return False

    sys.path.append(str(job_file_path.parent))
    sys.path.append(str(jobs_root))

    module_name = job_file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, job_file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Failed to create module spec for {job_file_path}")
        return False
    job_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job_module)

    try:
        if hasattr(job_module, "main"):
            job_module.main()
        else:
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(LOCAL_ROOT), os.environ.get("PYTHONPATH", "")])}
            result = subprocess.run(
                [sys.executable, str(job_file_path)] + job_args,
                cwd=str(LOCAL_ROOT),
                env={k: (v or "") for k, v in env.items()},
            )
            if result.returncode != 0:
                return False
        print(f"--- Job '{module_name}' completed successfully ---")
        return True
    except Exception as e:
        print(f"--- Job '{module_name}' failed: ---")
        print(e)
        return False


def run_jobs():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(1)

    # Split: job paths (until --) and shared args (after --)
    if "--" in argv:
        sep = argv.index("--")
        job_paths = argv[:sep]
        job_args = argv[sep + 1 :]
    else:
        # No --: last segment starting with - begins args
        job_paths = []
        job_args = []
        for i, a in enumerate(argv):
            if a.startswith("-"):
                job_paths = argv[:i]
                job_args = argv[i:]
                break
        if not job_paths:
            job_paths = argv

    if not job_paths:
        print("Usage: etl_dispatcher.py <job_path> [job_path ...] [-- job_args...]")
        sys.exit(1)

    failed = 0
    for path in job_paths:
        if not _run_one_job(path.strip(), job_args):
            failed += 1

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    run_jobs()
