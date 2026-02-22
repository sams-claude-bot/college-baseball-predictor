#!/bin/bash
# Weekly Power Rankings — Monday 12 PM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_weekly_power_rankings.log"

echo "=== Power Rankings $(date) ===" >> "$LOG"
python3 -u scripts/power_rankings.py --store --min-games 3 -v >> "$LOG" 2>&1

echo "--- Restart dashboard ---" >> "$LOG"
sudo systemctl restart college-baseball-dashboard.service >> "$LOG" 2>&1

echo "=== Done $(date) ===" >> "$LOG"
