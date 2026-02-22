#!/bin/bash
# Full model training — train on all historical, finetune on all 2026
# Slot: after nightly data pipeline (2:30 AM), before morning predictions (8:15 AM)
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_full_train.log"

echo "=== Full Train $(date) ===" >> "$LOG"

echo "--- Training all models (--full-train) ---" >> "$LOG"
python3 -u scripts/train_all_models.py --full-train --no-gpu >> "$LOG" 2>&1

echo "--- Verification ---" >> "$LOG"
python3 -c "
import os, torch, pickle
from pathlib import Path
data = Path('data')
models = {
    'nn_slim_win': data/'nn_slim_model.pt',
    'nn_slim_win_ft': data/'nn_slim_model_finetuned.pt',
    'nn_slim_totals': data/'nn_slim_totals.pt',
    'nn_slim_totals_ft': data/'nn_slim_totals_finetuned.pt',
    'xgb_moneyline': data/'xgb_moneyline.pkl',
    'lgb_moneyline': data/'lgb_moneyline.pkl',
}
for name, path in models.items():
    if path.exists():
        mtime = os.path.getmtime(path)
        from datetime import datetime
        ts = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
        print(f'  {name}: updated {ts}')
    else:
        print(f'  {name}: MISSING')
" >> "$LOG" 2>&1

echo "=== Done $(date) ===" >> "$LOG"
