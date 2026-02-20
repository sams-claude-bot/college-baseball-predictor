#!/bin/bash
# Weekly Model Training — Sunday 9:30 PM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_weekly_training.log"

echo "=== Weekly Training $(date) ===" >> "$LOG"
PYTHONPATH=. python3 scripts/train_all_models.py >> "$LOG" 2>&1

echo "--- Git commit ---" >> "$LOG"
git add -A && git commit -m "Weekly model training $(date +%Y-%m-%d)" --author="sams-claude-bot <sams-claude-bot@users.noreply.github.com>" >> "$LOG" 2>&1 && git push origin master >> "$LOG" 2>&1 || echo "Nothing to commit" >> "$LOG"

echo "--- Restart dashboard ---" >> "$LOG"
sudo systemctl restart college-baseball-dashboard.service >> "$LOG" 2>&1
echo "=== Done $(date) ===" >> "$LOG"
