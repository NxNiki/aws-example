"""
Combined daily report script.

Runs in sequence:
  1. ETL game stats daily by group (HG, Redshift → stats_by_date.parquet)
  2. ETL game stats daily by group PA (Athena → stats_by_day_pa.parquet)
  3. Display daily report (SS01 metrics table)
  4. Check PID difference (Redshift vs Athena for fish_hunter, ss01, ss03)

Outputs: jobs/operation_daily_report/output/
Final report is printed cleanly at the end (no logging mixed in).

Usage:
  poetry run python jobs/operation_daily_report/run_daily_report.py [--bastion-ip IP]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from bituslabs_ds.config import LOCAL_ROOT, setup_logging

OP_DIR = Path(__file__).resolve().parent
LOG_DIR = Path(LOCAL_ROOT) / "jobs" / "log"


def _run_script(script_path: Path, args: list[str]) -> bool:
    """Run a Python script. Returns True on success."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(LOCAL_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, cwd=str(LOCAL_ROOT), env=env)
    return result.returncode == 0


def _get_daily_report(lookback_days: int) -> str:
    """Call display_daily_report logic and return report string."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "display_daily_report",
        OP_DIR / "display_daily_report.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output_dir = OP_DIR / "output"
    df_hg, df_pa = mod._load_data(output_dir)
    return mod.generate_daily_report(df_hg, df_pa, lookback_days=lookback_days)


def _get_pid_report(bastion_ip: str | None, reload_athena: bool = False) -> str:
    """Call check_pid_difference and return report string."""
    import importlib.util

    from bituslabs_ds.config import DEFAULT_BASTION_IP

    spec = importlib.util.spec_from_file_location(
        "check_pid_difference",
        OP_DIR / "check_pid_difference.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = mod.main(bastion_ip=bastion_ip or DEFAULT_BASTION_IP, reload_athena=reload_athena)
    return mod.format_pid_report(results)


def main():
    setup_logging(
        str(LOG_DIR),
        log_filename="operation_daily_report.log",
    )

    parser = argparse.ArgumentParser(
        description="Generate daily report: ETL stats, display report, and PID difference check"
    )
    parser.add_argument(
        "--bastion-ip",
        type=str,
        default=None,
        help="Bastion IP for Redshift tunnel (passed to ETL and check_pid_difference)",
    )
    parser.add_argument(
        "--skip-hg-etl",
        action="store_true",
        help="Skip HG ETL (Redshift → stats_by_date)",
    )
    parser.add_argument(
        "--skip-pa-etl",
        dest="skip_pa_etl",
        action="store_true",
        help="Skip PA ETL (default: skip). PA data is static, used as year-ago comparison",
    )
    parser.add_argument(
        "--no-skip-pa-etl",
        dest="skip_pa_etl",
        action="store_false",
        help="Run PA ETL (override default skip)",
    )
    parser.set_defaults(skip_pa_etl=True)
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip step 3 (display daily report)",
    )
    parser.add_argument(
        "--skip-pid-check",
        action="store_true",
        help="Skip step 4 (check PID difference)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="Number of days for daily report (default: 1)",
    )
    parser.add_argument(
        "--reload-cached-etl",
        action="store_true",
        help="Force re-run ETL for stats_by_day_pa, fishhunter_product_id_pa, slot_product_id_pa (default: use cache)",
    )
    args = parser.parse_args()

    extra_args = []
    if args.bastion_ip:
        extra_args = ["--bastion-ip", args.bastion_ip]

    output_dir = OP_DIR / "output"
    stats_by_day_pa = output_dir / "stats_by_day_pa.parquet"

    # Run PA ETL when: --reload-cached-etl, or not --skip-pa-etl, or file missing (and display needs it)
    run_pa_etl = args.reload_cached_etl or not args.skip_pa_etl
    if not run_pa_etl and not args.skip_report and not stats_by_day_pa.exists():
        run_pa_etl = True  # Display needs stats_by_day_pa; run PA ETL if file missing

    pa_etl_args = ["--reload"] if args.reload_cached_etl else []

    # --- ETL steps (subprocess) ---
    scripts = [
        (
            "ETL game stats daily by group (HG)",
            OP_DIR / "etl_game_stats_daily_by_group.py",
            not args.skip_hg_etl,
            extra_args,
        ),
        ("ETL game stats daily by group PA", OP_DIR / "etl_game_stats_daily_by_group_pa.py", run_pa_etl, pa_etl_args),
    ]

    failed = []
    for name, path, run_it, script_args in scripts:
        if not run_it:
            print(f"[SKIP] {name}")
            continue
        if not path.exists():
            print(f"[ERROR] Script not found: {path}")
            failed.append(name)
            continue
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        if not _run_script(path, script_args):
            failed.append(name)
        else:
            print(f"[OK] {name} completed")

    # --- Report steps (import and call, capture output) ---
    report_chunks = []

    if not args.skip_report:
        print(f"\n{'='*60}\nRunning: Display daily report\n{'='*60}")
        try:
            daily_report = _get_daily_report(lookback_days=args.lookback_days)
            report_chunks.append(daily_report)
            print("[OK] Display daily report completed")
        except Exception as e:
            print(f"[ERROR] Display daily report failed: {e}")
            failed.append("Display daily report")

    if not args.skip_pid_check:
        print(f"\n{'='*60}\nRunning: Check PID difference\n{'='*60}")
        import io

        _saved_stdout = sys.stdout
        pid_report = None
        try:
            sys.stdout = io.StringIO()  # Suppress DataFrame/logging prints from check_pid
            pid_report = _get_pid_report(args.bastion_ip, reload_athena=args.reload_cached_etl)
        except Exception as e:
            sys.stdout = _saved_stdout
            print(f"[ERROR] Check PID difference failed: {e}")
            failed.append("Check PID difference")
            pid_report = None
        else:
            sys.stdout = _saved_stdout
        if pid_report is not None:
            report_chunks.append(pid_report)
            print("[OK] Check PID difference completed")

    # --- Final combined report (clean output for submission) ---
    if report_chunks:
        print("\n" + "=" * 60)
        print("FINAL REPORT FOR SUBMISSION")
        print("=" * 60)
        print("\n".join(report_chunks))
        print("=" * 60)
        print("END OF REPORT")
        print("=" * 60)

    if failed:
        print(f"\n[FAILED] {len(failed)} step(s): {', '.join(failed)}")
        sys.exit(1)
    print("\n[OK] Daily report completed successfully")


if __name__ == "__main__":
    main()
