#!/bin/bash
# 2/6 Stats Scrape — 1:00 AM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_02_stats_scrape.log"

echo "=== Stats Scrape $(date) ===" >> "$LOG"
python3 -u scripts/d1bb_scraper.py --all-d1 --delay 2 >> "$LOG" 2>&1
echo "--- Verification ---" >> "$LOG"
python3 -c "
import sqlite3
from datetime import datetime, timedelta
db = sqlite3.connect('data/baseball.db')
cutoff = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
fresh = db.execute('SELECT COUNT(DISTINCT team_id) FROM player_stats WHERE updated_at > ?', (cutoff,)).fetchone()[0]
woba = db.execute('SELECT COUNT(DISTINCT team_id) FROM player_stats WHERE woba IS NOT NULL').fetchone()[0]
print(f'Teams updated: {fresh}/311')
print(f'Teams with wOBA: {woba}')
" >> "$LOG" 2>&1
echo "=== Done $(date) ===" >> "$LOG"
