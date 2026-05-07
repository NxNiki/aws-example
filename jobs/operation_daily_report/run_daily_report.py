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
  poetry run python jobs/operation_daily_report/run_daily_report.py
  poetry run python jobs/operation_daily_report/run_daily_report.py --send-slack  # send report to Slack

Environment variables for Slack (when --send-slack):
  SLACK_USER_TOKEN  - Slack User OAuth token (e.g. xoxp-...); messages appear as you. Preferred.
  SLACK_BOT_TOKEN   - Slack Bot OAuth token (e.g. xoxb-...); alternative if SLACK_USER_TOKEN not set
  SLACK_CHANNEL_ID  - Slack channel ID (e.g. C01234567) or channel name (e.g. #daily-reports)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from bituslabs_ds.config import LOCAL_ROOT, setup_logging

OP_DIR = Path(__file__).resolve().parent
LOG_DIR = Path(LOCAL_ROOT) / "jobs" / "log"


def _send_report_to_slack(report_text: str) -> bool:
    """Send report to Slack. Returns True on success. Uses SLACK_USER_TOKEN or SLACK_BOT_TOKEN."""
    token = os.environ.get("SLACK_USER_TOKEN") or os.environ.get("SLACK_BOT_TOKEN")
    chan = os.environ.get("SLACK_CHANNEL_ID")
    if not token:
        print("[WARN] SLACK_USER_TOKEN or SLACK_BOT_TOKEN not set; skipping Slack send")
        return False
    if not chan:
        print("[WARN] SLACK_CHANNEL_ID not set; skipping Slack send")
        return False

    try:
        client = WebClient(token=token)
        # Slack text field has 40k char limit; split if needed
        max_len = 39_000
        if len(report_text) <= max_len:
            client.chat_postMessage(channel=chan, text=f"📊 *Daily Report*\n\n```\n{report_text}\n```")
        else:
            chunks = [report_text[i : i + max_len] for i in range(0, len(report_text), max_len)]
            for i, chunk in enumerate(chunks):
                prefix = "📊 *Daily Report* " if i == 0 else ""
                client.chat_postMessage(channel=chan, text=f"{prefix}({i + 1}/{len(chunks)})\n```\n{chunk}\n```")
        print(f"[OK] Report sent to Slack channel {chan}")
        return True
    except SlackApiError as e:
        print(f"[ERROR] Slack API error: {e.response['error']}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send to Slack: {e}")
        return False


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


def _get_pid_report(reload_athena: bool = False) -> str:
    """Call check_pid_difference and return report string."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_pid_difference",
        OP_DIR / "check_pid_difference.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = mod.main(reload_athena=reload_athena)
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
        "--run-pa-etl",
        action="store_true",
        help="Run PA ETL (default: skip). PA data is static, used as year-ago comparison",
    )
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
    parser.add_argument(
        "--send-slack",
        action="store_true",
        help="Send the final report to Slack (requires SLACK_USER_TOKEN or SLACK_BOT_TOKEN and SLACK_CHANNEL_ID)",
    )
    args = parser.parse_args()

    output_dir = OP_DIR / "output"
    stats_by_day_pa = output_dir / "stats_by_day_pa.parquet"

    # Run PA ETL when: --reload-cached-etl, --run-pa-etl, or file missing (and display needs it)
    run_pa_etl = args.reload_cached_etl or args.run_pa_etl
    if not run_pa_etl and not args.skip_report and not stats_by_day_pa.exists():
        run_pa_etl = True  # Display needs stats_by_day_pa; run PA ETL if file missing

    pa_etl_args = ["--reload"] if args.reload_cached_etl else []

    # --- ETL steps (subprocess) ---
    scripts = [
        (
            "ETL game stats daily by group (HG)",
            OP_DIR / "etl_game_stats_daily_by_group.py",
            True,
            [],
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
            pid_report = _get_pid_report(reload_athena=args.reload_cached_etl)
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
    full_report = "\n".join(report_chunks) if report_chunks else ""
    if report_chunks:
        print("\n" + "=" * 60)
        print("FINAL REPORT FOR SUBMISSION")
        print("=" * 60)
        print(full_report)
        print("=" * 60)
        print("END OF REPORT")
        print("=" * 60)

    # --- Send to Slack if requested ---
    if args.send_slack and full_report:
        print("\nSending report to Slack...")
        _send_report_to_slack(full_report)

    if failed:
        print(f"\n[FAILED] {len(failed)} step(s): {', '.join(failed)}")
        sys.exit(1)
    print("\n[OK] Daily report completed successfully")


if __name__ == "__main__":
    main()
