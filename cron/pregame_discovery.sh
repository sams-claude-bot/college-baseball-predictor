#!/bin/bash
# Pre-game SB discovery pipeline
# Runs daily before first game time
# No AI needed - pure Python + HTTP

set -euo pipefail
cd /home/sam/college-baseball-predictor

DATE=$(date +%Y-%m-%d)
LOG="logs/cron/${DATE}_pregame_discovery.log"
mkdir -p logs/cron

echo "=== Pre-Game Discovery: $DATE ===" >> "$LOG"
echo "Started: $(date)" >> "$LOG"

PYTHONPATH=. python3 scripts/d1b_pregame_discovery.py --date "$DATE" -v >> "$LOG" 2>&1
RC=$?

echo "Finished: $(date) (exit $RC)" >> "$LOG"
exit $RC
