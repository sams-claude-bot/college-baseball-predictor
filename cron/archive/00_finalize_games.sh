#!/bin/bash
# 0/6 Finalize Games — 4:45 AM CT (runs before late scores)
# Scores yesterday's games from D1BB team pages, marks postponed/canceled
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_00_finalize_games.log"

echo "=== Finalize Games $(date) ===" >> "$LOG"
python3 -u scripts/finalize_games.py --verbose >> "$LOG" 2>&1
echo "=== Done $(date) ===" >> "$LOG"
