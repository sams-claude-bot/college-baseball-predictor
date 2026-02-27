#!/bin/bash
# Scrape lineup data from D1Baseball for all major conference teams
# Run weekly on Mondays to update rotation patterns

cd /home/sam/college-baseball-predictor

echo "=== D1Baseball Lineup Scraper - $(date) ==="

# Major conferences to scrape
for conf in SEC ACC "Big 12" "Big Ten"; do
    echo ""
    echo "--- Scraping $conf ---"
    python3 scripts/d1b_lineups.py --conference "$conf" 2>&1
    sleep 10  # Be nice to D1Baseball servers
done

echo ""
echo "=== Updating starter inference ==="
python3 scripts/infer_starters.py

echo ""
echo "=== Done - $(date) ==="
