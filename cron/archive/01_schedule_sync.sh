#!/bin/bash
# 1/6 Schedule Sync — 12:30 AM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_01_schedule_sync.log"

echo "=== Schedule Sync $(date) ===" >> "$LOG"
python3 -u scripts/d1bb_team_sync.py --delay 0.3 >> "$LOG" 2>&1
echo "--- Verification ---" >> "$LOG"
python3 -c "
import sqlite3
from datetime import datetime, timedelta
db = sqlite3.connect('data/baseball.db')
for i in range(3):
    d = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
    games = db.execute('SELECT COUNT(*) FROM games WHERE date = ?', (d,)).fetchone()[0]
    print(f'{d}: {games} games')
" >> "$LOG" 2>&1
# backfill_missing_games.py removed — d1bb_team_sync now covers this via ScheduleGateway

echo "=== Done $(date) ===" >> "$LOG"
