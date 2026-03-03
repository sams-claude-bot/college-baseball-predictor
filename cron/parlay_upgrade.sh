#!/bin/bash
# Parlay upgrade check - runs after odds scrape
# No AI needed - pure Python

set -euo pipefail
cd /home/sam/college-baseball-predictor

DATE=$(date +%Y-%m-%d)
LOG="logs/cron/${DATE}_parlay_upgrade.log"
mkdir -p logs/cron

echo "=== Parlay Upgrade Check: $DATE ===" >> "$LOG"
echo "Started: $(date)" >> "$LOG"

python3 -m scripts.betting.upgrade_parlay >> "$LOG" 2>&1
RC=$?

echo "Finished: $(date) (exit $RC)" >> "$LOG"
exit $RC
