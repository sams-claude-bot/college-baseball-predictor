#!/bin/bash
# 3/6 Derived Stats & Snapshots — 1:45 AM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_03_derived_stats.log"

echo "=== Derived Stats $(date) ===" >> "$LOG"
python3 scripts/compute_pitching_quality.py >> "$LOG" 2>&1
python3 scripts/compute_batting_quality.py >> "$LOG" 2>&1
python3 scripts/aggregate_team_stats.py >> "$LOG" 2>&1
python3 scripts/snapshot_stats.py >> "$LOG" 2>&1
python3 scripts/compute_sos.py >> "$LOG" 2>&1
python3 scripts/compute_rpi.py >> "$LOG" 2>&1

echo "--- Verification ---" >> "$LOG"
python3 -c "
import sqlite3
from datetime import datetime
db = sqlite3.connect('data/baseball.db')
today = datetime.now().strftime('%Y-%m-%d')
pq = db.execute('SELECT COUNT(*) FROM team_pitching_quality').fetchone()[0]
bq = db.execute('SELECT COUNT(*) FROM team_batting_quality').fetchone()[0]
for t in ['player_stats_snapshots','team_batting_quality_snapshots','team_pitching_quality_snapshots']:
    n = db.execute(f'SELECT COUNT(*) FROM {t} WHERE snapshot_date=?',(today,)).fetchone()[0]
    print(f'{t}: {n} rows today')
print(f'Pitching quality: {pq} teams')
print(f'Batting quality: {bq} teams')
" >> "$LOG" 2>&1

echo "=== Done $(date) ===" >> "$LOG"
