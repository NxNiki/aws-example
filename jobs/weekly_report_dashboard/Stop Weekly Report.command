#!/bin/zsh

set -euo pipefail

APP_PATTERN="/Users/stephaniechen/aws-example/jobs/weekly_report_dashboard/weekly_report_dashboard.py"

pkill -f "$APP_PATTERN" || true
