#!/bin/bash
# Weekly Model Training — Sunday 9:30 PM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_weekly_training.log"

echo "=== Weekly Training $(date) ===" >> "$LOG"
PYTHONPATH=. python3 scripts/train_all_models.py >> "$LOG" 2>&1

echo "--- Retrain meta-ensemble ---" >> "$LOG"
PYTHONPATH=. python3 scripts/train_meta_ensemble.py >> "$LOG" 2>&1

echo "--- Update calibration ---" >> "$LOG"
PYTHONPATH=. python3 scripts/update_model_calibration.py >> "$LOG" 2>&1

echo "--- Restart dashboard ---" >> "$LOG"
sudo systemctl restart college-baseball-dashboard.service >> "$LOG" 2>&1
echo "=== Done $(date) ===" >> "$LOG"
