#!/bin/bash
# Check overnight pipeline results — run manually or from heartbeat
cd /home/sam/college-baseball-predictor
TODAY=$(date +%Y-%m-%d)
echo "=== Overnight Pipeline Status: $TODAY ==="
echo ""

JOBS="01_schedule_sync 02_stats_scrape 03_derived_stats 04_nightly_eval full_train 05_morning_pipeline"

for job in $JOBS; do
    LOG="logs/cron/${TODAY}_${job}.log"
    if [ ! -f "$LOG" ]; then
        echo "❌ $job: NO LOG (did not run?)"
        continue
    fi
    
    SIZE=$(wc -c < "$LOG")
    if [ "$SIZE" -lt 10 ]; then
        echo "❌ $job: EMPTY LOG"
        continue
    fi
    
    # Check for errors
    ERRORS=$(grep -ci "error\|traceback\|failed\|exception" "$LOG" 2>/dev/null || echo 0)
    LAST_LINE=$(tail -1 "$LOG")
    
    if echo "$LAST_LINE" | grep -qi "done\|ok\|complete\|finish"; then
        if [ "$ERRORS" -gt 0 ]; then
            echo "⚠️  $job: COMPLETED with $ERRORS error(s)"
        else
            echo "✅ $job: OK"
        fi
    else
        echo "❌ $job: MAY HAVE FAILED (last line: $LAST_LINE)"
    fi
done

echo ""
echo "=== Quick DB Health ==="
python3 -c "
import sqlite3
from datetime import datetime, timedelta
db = sqlite3.connect('data/baseball.db')
today = datetime.now().strftime('%Y-%m-%d')
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

# Games
final_yesterday = db.execute('SELECT COUNT(*) FROM games WHERE date=? AND status=\"final\"', (yesterday,)).fetchone()[0]
total_yesterday = db.execute('SELECT COUNT(*) FROM games WHERE date=?', (yesterday,)).fetchone()[0]
scheduled_today = db.execute('SELECT COUNT(*) FROM games WHERE date=? AND status=\"scheduled\"', (today,)).fetchone()[0]

# Predictions
preds_today = db.execute('SELECT COUNT(DISTINCT game_id) FROM model_predictions WHERE predicted_at > ?', (today,)).fetchone()[0]

# Elo
elo_updated = db.execute('SELECT COUNT(*) FROM elo_ratings WHERE updated_at > datetime(\"now\", \"-12 hours\")').fetchone()[0]

# Stats freshness
fresh_stats = db.execute('SELECT COUNT(DISTINCT team_id) FROM player_stats WHERE updated_at > ?', (today,)).fetchone()[0]

# Model files
import os
from pathlib import Path
models = {'nn_slim': 'data/nn_slim_model_finetuned.pt', 'xgb': 'data/xgb_moneyline.pkl', 'lgb': 'data/lgb_moneyline.pkl'}
model_status = []
for name, path in models.items():
    if os.path.exists(path):
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        age_hrs = (datetime.now() - mtime).total_seconds() / 3600
        model_status.append(f'{name}: {age_hrs:.0f}h ago')
    else:
        model_status.append(f'{name}: MISSING')

print(f'Yesterday: {final_yesterday}/{total_yesterday} games final')
print(f'Today: {scheduled_today} games scheduled, {preds_today} predicted')
print(f'Elo: {elo_updated} teams updated overnight')
print(f'Stats: {fresh_stats} teams refreshed')
print(f'Models: {\" | \".join(model_status)}')

# P&L
ml = db.execute('SELECT COUNT(*), SUM(CASE WHEN won=1 THEN 1 ELSE 0 END), COALESCE(SUM(profit),0) FROM tracked_bets WHERE won IS NOT NULL').fetchone()
if ml[0]:
    print(f'P&L: {ml[1]}/{ml[0]} wins (\${ml[2]:+.2f})')

# Unevaluated
unevaled = db.execute('SELECT COUNT(*) FROM model_predictions mp JOIN games g ON mp.game_id=g.id WHERE g.status=\"final\" AND mp.was_correct IS NULL').fetchone()[0]
if unevaled > 0:
    print(f'⚠️  {unevaled} unevaluated predictions on final games')
"

echo ""
echo "=== Log Sizes ==="
for job in $JOBS; do
    LOG="logs/cron/${TODAY}_${job}.log"
    if [ -f "$LOG" ]; then
        du -h "$LOG" | cut -f1 | tr -d '\n'
        echo "  $job"
    fi
done
