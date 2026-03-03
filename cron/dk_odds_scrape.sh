#!/usr/bin/env bash
# DraftKings NCAA Baseball Odds — pure headless Playwright scrape (no AI)
# System cron: 0 8 * * *
set -euo pipefail
cd /home/sam/college-baseball-predictor

echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting DK odds scrape"

RAW_FILE="/tmp/dk_odds_raw_$(date +%Y%m%d).txt"

# Step 1: Scrape + parse in one Python script
python3 scripts/dk_headless_scrape.py "$RAW_FILE"

if [ ! -s "data/dk_odds_today.json" ]; then
    echo "ERROR: No odds JSON produced"
    exit 1
fi

# Step 2: Load into DB
python3 scripts/dk_odds_scraper.py load data/dk_odds_today.json

echo "$(date '+%Y-%m-%d %H:%M:%S') — DK odds scrape complete"
