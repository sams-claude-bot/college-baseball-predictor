#!/bin/bash
# 1b/6 Late Score Catchup — 5:00 AM CT
# Catches Hawaii/West Coast games that finished after midnight CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_01b_late_scores.log"

YESTERDAY=$(date -d yesterday +%Y-%m-%d)

echo "=== Late Score Catchup $(date) ===" >> "$LOG"
echo "Fetching final scores for $YESTERDAY..." >> "$LOG"
python3 -u scripts/d1bb_schedule.py --date "$YESTERDAY" >> "$LOG" 2>&1

# Show any still-unfinished games
echo "--- Remaining non-final games for $YESTERDAY ---" >> "$LOG"
python3 -c "
import sqlite3
db = sqlite3.connect('data/baseball.db')
rows = db.execute('''
    SELECT id, home_team_id, away_team_id, home_score, away_score, status
    FROM games WHERE date = '$YESTERDAY' AND status != 'final'
''').fetchall()
if rows:
    for r in rows:
        print(f'  {r[0]}: {r[1]} vs {r[2]} ({r[5]})')
else:
    print('  All games final.')
" >> "$LOG" 2>&1

# Elo/eval/bets handled by 04_nightly_eval.sh which now runs after this

echo "=== Done $(date) ===" >> "$LOG"
