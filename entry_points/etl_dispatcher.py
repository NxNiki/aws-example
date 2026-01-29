import importlib.util
import os
import sys
from pathlib import Path

from bituslabs_ds.config import LOCAL_ROOT


def run_job():
    if len(sys.argv) < 2:
        print("Usage: python etl_dispatcher.py <path_to_job_file>")
        print("Example: python etl_dispatcher.py ss01_wucaishen/etl_game_stats_daily_by_user_group.py")
        sys.exit(1)

    # 1. Resolve the path to the job script
    # This allows you to pass paths like 'fish_hunter/daily_job.py'
    job_relative_path = sys.argv[1]
    jobs_root = LOCAL_ROOT / "jobs"
    job_file_path = (jobs_root / job_relative_path).resolve()

    if not job_file_path.exists():
        print(f"Error: Job file not found at {job_file_path}")
        sys.exit(1)

    # 2. Add the job's directory to sys.path so it can find local modules/config
    sys.path.append(str(job_file_path.parent))
    sys.path.append(str(jobs_root))

    # 3. Dynamically load and execute the module
    module_name = job_file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, job_file_path)
    job_module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(job_module)
        # Check if there is a main() function to call, otherwise it just runs on import
        if hasattr(job_module, "main"):
            job_module.main()
        print(f"--- Job '{module_name}' completed successfully ---")
    except Exception as e:
        print(f"--- Job '{module_name}' failed with error: ---")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    run_job()
