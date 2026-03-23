#!/bin/zsh

set -euo pipefail

APP_DIR="/Users/stephaniechen/aws-example/jobs/weekly_report_dashboard"
APP_URL="http://127.0.0.1:8051"
LOG_FILE="$APP_DIR/weekly_report_dashboard.log"

cd "$APP_DIR"

if curl -fsS "$APP_URL" >/dev/null 2>&1; then
  open "$APP_URL"
  exit 0
fi

nohup env PORT=8051 STRICT_PORT=1 DASH_DEBUG=0 python3 "$APP_DIR/weekly_report_dashboard.py" >"$LOG_FILE" 2>&1 &

for _ in {1..30}; do
  if curl -fsS "$APP_URL" >/dev/null 2>&1; then
    open "$APP_URL"
    exit 0
  fi
  sleep 1
done

echo "Weekly Report did not start successfully. Check $LOG_FILE"
exit 1
