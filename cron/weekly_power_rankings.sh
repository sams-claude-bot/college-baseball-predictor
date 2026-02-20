#!/bin/bash
# Weekly Power Rankings — Monday 12 PM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_weekly_power_rankings.log"

echo "=== Power Rankings $(date) ===" >> "$LOG"
python3 -u scripts/power_rankings.py --store --min-games 3 -v >> "$LOG" 2>&1

echo "--- Restart dashboard ---" >> "$LOG"
sudo systemctl restart college-baseball-dashboard.service >> "$LOG" 2>&1

echo "--- Git commit ---" >> "$LOG"
git add -A && git commit -m "Weekly power rankings $(date +%Y-%m-%d)" --author="sams-claude-bot <sams-claude-bot@users.noreply.github.com>" >> "$LOG" 2>&1 && git push origin master >> "$LOG" 2>&1 || echo "Nothing to commit" >> "$LOG"
echo "=== Done $(date) ===" >> "$LOG"
