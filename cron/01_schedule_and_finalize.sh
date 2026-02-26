#!/bin/bash
# 1/6 Schedule + Finalize + Late Catchup — 5:00 AM CT
# Merged from: 01_schedule_sync + 00_finalize_games + 01b_late_scores
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_01_schedule_and_finalize.log"
YESTERDAY=$(date -d yesterday +%Y-%m-%d)

echo "=== Schedule+Finalize $(date) ===" >> "$LOG"

echo "--- Step 1: Schedule Sync (today+next few days) ---" >> "$LOG"
python3 -u scripts/d1bb_team_sync.py --delay 0.3 >> "$LOG" 2>&1
# backfill_missing_games.py removed — d1bb_team_sync now covers this via ScheduleGateway
# (was re-fetching the same D1BB team pages and duplicating work)

echo "--- Step 2: Finalize Yesterday ---" >> "$LOG"
python3 -u scripts/finalize_games.py --date "$YESTERDAY" --verbose >> "$LOG" 2>&1

echo "--- Verification ---" >> "$LOG"
python3 - <<'PY' >> "$LOG" 2>&1
import sqlite3
from datetime import datetime, timedelta

db = sqlite3.connect('data/baseball.db')
today = datetime.now().strftime('%Y-%m-%d')
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

for i in range(3):
    d = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
    games = db.execute('SELECT COUNT(*) FROM games WHERE date=?', (d,)).fetchone()[0]
    print(f'{d}: {games} games')

final_y = db.execute('SELECT COUNT(*) FROM games WHERE date=? AND status="final"', (yesterday,)).fetchone()[0]
total_y = db.execute('SELECT COUNT(*) FROM games WHERE date=?', (yesterday,)).fetchone()[0]
left_y = db.execute('SELECT COUNT(*) FROM games WHERE date=? AND status NOT IN ("final","postponed","canceled")', (yesterday,)).fetchone()[0]
print(f'Yesterday final: {final_y}/{total_y} | unresolved: {left_y}')
PY

echo "=== Done $(date) ===" >> "$LOG"
