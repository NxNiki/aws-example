#!/bin/bash
# Install launchd schedule for operation daily report on macOS.
# Runs daily at the time configured in infra/com.operation.daily-report.plist (local time).
#
# Usage: bash infra/setup_daily_report_schedule_mac.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_NAME="com.operation.daily-report"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME.plist"
PLIST_DEST="$LAUNCH_AGENTS/$PLIST_NAME.plist"

# Ensure log dir exists
mkdir -p "$PROJECT_ROOT/jobs/log"

# Update paths in plist (in case project is in different location)
TMP_PLIST=$(mktemp)
sed "s|/Users/niuxin/Documents/aws-example|$PROJECT_ROOT|g" "$PLIST_SRC" > "$TMP_PLIST"

echo "Installing launchd job: $PLIST_NAME"
echo "  Schedule: daily (time set in plist)"
echo "  Logs: $PROJECT_ROOT/jobs/log/daily-report-*.log"
echo ""

# Unload if already loaded
if launchctl list "$PLIST_NAME" &>/dev/null; then
  echo "Unloading existing job..."
  launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

# Copy and load
cp "$TMP_PLIST" "$PLIST_DEST"
rm -f "$TMP_PLIST"
launchctl load "$PLIST_DEST"
echo "Done. Job is now scheduled."

echo ""
echo "Commands:"
echo "  Check status:  launchctl list $PLIST_NAME"
echo "  Unload/stop:   launchctl unload $PLIST_DEST"
echo "  Reload:        launchctl unload $PLIST_DEST && launchctl load $PLIST_DEST"
echo "  Change time:   edit infra/com.operation.daily-report.plist (Hour 0-23, Minute 0-59), then re-run this script"
