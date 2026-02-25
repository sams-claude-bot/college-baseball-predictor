#!/bin/bash
# Weekly Power Rankings — Monday 12 PM CT
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_weekly_power_rankings.log"
LIVE_DB="data/baseball.db"
SNAPSHOT_DB="data/baseball_powerrank_snapshot.db"

cleanup_snapshot() {
  if [[ -f "$SNAPSHOT_DB" ]]; then
    echo "[cleanup] Removing snapshot $SNAPSHOT_DB" >> "$LOG"
    rm -f "$SNAPSHOT_DB"
  fi
}
trap cleanup_snapshot EXIT

echo "=== Power Rankings $(date) ===" >> "$LOG"
echo "[snapshot] Creating SQLite backup snapshot: $SNAPSHOT_DB" >> "$LOG"
rm -f "$SNAPSHOT_DB"
sqlite3 "$LIVE_DB" ".backup $SNAPSHOT_DB" >> "$LOG" 2>&1

echo "[compute] Running power rankings against snapshot, final write to live DB" >> "$LOG"
python3 -u scripts/power_rankings.py \
  --store \
  --min-games 3 \
  --read-db "$SNAPSHOT_DB" \
  --write-db "$LIVE_DB" \
  -v >> "$LOG" 2>&1

echo "--- Restart dashboard ---" >> "$LOG"
sudo systemctl restart college-baseball-dashboard.service >> "$LOG" 2>&1

echo "=== Done $(date) ===" >> "$LOG"
