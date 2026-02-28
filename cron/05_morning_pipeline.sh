#!/bin/bash
# 6/6 Morning Pipeline — 8:15 AM CT
# (5/6 DK Odds is AI-based, not in system cron)
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_05_morning_pipeline.log"

echo "=== Morning Pipeline $(date) ===" >> "$LOG"

echo "--- Weather ---" >> "$LOG"
python3 -u scripts/weather.py fetch --upcoming >> "$LOG" 2>&1

echo "--- Infer Starters ---" >> "$LOG"
python3 -u scripts/infer_starters.py >> "$LOG" 2>&1

echo "--- Update Model Calibration ---" >> "$LOG"
python3 scripts/update_model_calibration.py >> "$LOG" 2>&1 || true

echo "--- Predictions ---" >> "$LOG"
PYTHONPATH=. python3 scripts/predict_and_track.py predict --refresh-existing >> "$LOG" 2>&1

echo "--- Bet selection ---" >> "$LOG"
python3 scripts/bet_selection_v2.py record >> "$LOG" 2>&1

echo "--- Parlay upgrade check ---" >> "$LOG"
python3 -m scripts.betting.upgrade_parlay >> "$LOG" 2>&1 || true

echo "--- Restart dashboard ---" >> "$LOG"
sudo systemctl restart college-baseball-dashboard.service >> "$LOG" 2>&1

echo "--- Verification ---" >> "$LOG"
python3 -c "
import sqlite3
from datetime import datetime, timedelta
db = sqlite3.connect('data/baseball.db')
today = datetime.now().strftime('%Y-%m-%d')
cutoff = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
lines = db.execute('SELECT COUNT(*) FROM betting_lines WHERE date = ?', (today,)).fetchone()[0]
weather = db.execute('SELECT COUNT(*) FROM game_weather gw JOIN games g ON gw.game_id=g.id WHERE g.date BETWEEN date(\"now\") AND date(\"now\",\"+2 days\")').fetchone()[0]
preds = db.execute('SELECT COUNT(DISTINCT game_id) FROM model_predictions WHERE predicted_at > ?', (cutoff,)).fetchone()[0]
conf = db.execute('SELECT COUNT(*) FROM tracked_confident_bets WHERE date=?', (today,)).fetchone()[0]
ml = db.execute('SELECT COUNT(*) FROM tracked_bets WHERE date=?', (today,)).fetchone()[0]
print(f'Lines: {lines} | Weather: {weather} games | Predictions: {preds} games | Bets: {conf} consensus, {ml} ML')
if lines == 0: print('WARNING: No odds — predictions ran without DK lines')
" >> "$LOG" 2>&1

echo "--- Prediction Sanity Check ---" >> "$LOG"
python3 scripts/prediction_sanity_check.py >> "$LOG" 2>&1 || true

echo "=== Done $(date) ===" >> "$LOG"
