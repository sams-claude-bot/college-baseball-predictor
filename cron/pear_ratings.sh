#!/bin/bash
# PEAR ratings weekly fetch
# No AI needed - pure Python + HTTP

set -euo pipefail
cd /home/sam/college-baseball-predictor

DATE=$(date +%Y-%m-%d)
LOG="logs/cron/${DATE}_pear_ratings.log"
mkdir -p logs/cron

echo "=== PEAR Ratings Fetch: $DATE ===" >> "$LOG"
echo "Started: $(date)" >> "$LOG"

python3 scripts/fetch_pear_ratings.py >> "$LOG" 2>&1
RC=$?

echo "Finished: $(date) (exit $RC)" >> "$LOG"
exit $RC
