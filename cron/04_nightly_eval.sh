#!/bin/bash
# 4/6 Nightly Eval — 2:30 AM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_04_nightly_eval.log"
YESTERDAY=$(date -d 'yesterday' +%Y-%m-%d)

echo "=== Nightly Eval $(date) — processing $YESTERDAY ===" >> "$LOG"

echo "--- Backup ---" >> "$LOG"
python3 scripts/backup_db.py >> "$LOG" 2>&1

echo "--- Evaluate bets ---" >> "$LOG"
python3 scripts/record_daily_bets.py evaluate >> "$LOG" 2>&1

echo "--- Update Elo ---" >> "$LOG"
python3 scripts/update_elo.py --date "$YESTERDAY" >> "$LOG" 2>&1

echo "--- Evaluate model predictions ---" >> "$LOG"
PYTHONPATH=. python3 scripts/predict_and_track.py evaluate >> "$LOG" 2>&1

echo "--- Verification ---" >> "$LOG"
python3 -c "
import sqlite3
from datetime import datetime, timedelta
db = sqlite3.connect('data/baseball.db')
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
scored = db.execute('SELECT COUNT(*) FROM games WHERE date=? AND home_score IS NOT NULL', (YESTERDAY,)).fetchone()[0]
total = db.execute('SELECT COUNT(*) FROM games WHERE date=?', (YESTERDAY,)).fetchone()[0]
print(f'Games yesterday: {scored}/{total} have scores')
ml = db.execute('SELECT COUNT(*), SUM(CASE WHEN won=1 THEN 1 ELSE 0 END), COALESCE(SUM(profit),0) FROM tracked_bets WHERE won IS NOT NULL').fetchone()
if ml[0]: print(f'ML all-time: {ml[1]}/{ml[0]} wins, \${ml[2]:+.2f}')
else: print('ML: no evaluated bets')
elo = db.execute('SELECT COUNT(*) FROM elo_ratings WHERE updated_at > datetime(\"now\", \"-2 hours\")').fetchone()[0]
print(f'Elo updated: {elo}')
unevaled = db.execute('SELECT COUNT(*) FROM model_predictions mp JOIN games g ON mp.game_id=g.id WHERE g.status=\"final\" AND mp.was_correct IS NULL').fetchone()[0]
print(f'Unevaluated predictions on final games: {unevaled}')
" >> "$LOG" 2>&1

echo "--- Git commit ---" >> "$LOG"
git add -A && git commit -m "Nightly eval $YESTERDAY" --author="sams-claude-bot <sams-claude-bot@users.noreply.github.com>" >> "$LOG" 2>&1 && git push origin master >> "$LOG" 2>&1 || echo "Nothing to commit" >> "$LOG"
echo "=== Done $(date) ===" >> "$LOG"
