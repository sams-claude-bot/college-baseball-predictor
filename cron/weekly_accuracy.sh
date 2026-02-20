#!/bin/bash
# Weekly Accuracy Report — Sunday 10 PM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_weekly_accuracy.log"

echo "=== Weekly Accuracy $(date) ===" >> "$LOG"

PYTHONPATH=. python3 scripts/predict_and_track.py evaluate >> "$LOG" 2>&1
PYTHONPATH=. python3 scripts/predict_and_track.py accuracy >> "$LOG" 2>&1

echo "--- Detailed verification ---" >> "$LOG"
python3 -c "
import sqlite3
db = sqlite3.connect('data/baseball.db')
models = db.execute('''SELECT model_name,
    COUNT(*) as total,
    SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as correct,
    ROUND(100.0*SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END)/COUNT(*), 1) as pct
    FROM model_predictions WHERE was_correct IS NOT NULL
    GROUP BY model_name ORDER BY pct DESC''').fetchall()
for m in models:
    print(f'{m[0]}: {m[1]} predictions, {m[2]} correct ({m[3]}%)')
ml = db.execute('SELECT COUNT(*), SUM(CASE WHEN won=1 THEN 1 ELSE 0 END), COALESCE(SUM(profit),0) FROM tracked_bets WHERE won IS NOT NULL').fetchone()
if ml[0]: print(f'ML P&L: {ml[1]}/{ml[0]} ({100*ml[1]/ml[0]:.0f}%) \${ml[2]:+.2f}')
else: print('ML P&L: no data')
for bt in ['spread','total']:
    r = db.execute('SELECT COUNT(*), SUM(CASE WHEN won=1 THEN 1 ELSE 0 END), COALESCE(SUM(profit),0) FROM tracked_bets_spreads WHERE won IS NOT NULL AND bet_type=?', (bt,)).fetchone()
    if r[0]: print(f'{bt} P&L: {r[1]}/{r[0]} ({100*r[1]/r[0]:.0f}%) \${r[2]:+.2f}')
    else: print(f'{bt} P&L: no data')
" >> "$LOG" 2>&1

echo "--- Git commit ---" >> "$LOG"
git add -A && git commit -m "Weekly accuracy report $(date +%Y-%m-%d)" --author="sams-claude-bot <sams-claude-bot@users.noreply.github.com>" >> "$LOG" 2>&1 && git push origin master >> "$LOG" 2>&1 || echo "Nothing to commit" >> "$LOG"
echo "=== Done $(date) ===" >> "$LOG"
