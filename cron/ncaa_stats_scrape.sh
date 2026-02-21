#!/bin/bash
# NCAA Team Stats Scraper — uses openclaw browser CLI (no AI agent needed)
# Navigates stats.ncaa.org, extracts table data via JS, loads into DB
#
# Usage:
#   ./cron/ncaa_stats_scrape.sh                    # All stats, 2021-2025
#   ./cron/ncaa_stats_scrape.sh 2025               # All stats, just 2025
#   ./cron/ncaa_stats_scrape.sh 2025 era,obp       # Specific stats + season
set -euo pipefail
cd /home/sam/college-baseball-predictor
LOG="logs/cron/$(date +%Y-%m-%d)_ncaa_stats.log"
DATA_DIR="data/ncaa"
PROFILE="openclaw"
mkdir -p "$DATA_DIR" "$(dirname "$LOG")"

echo "=== NCAA Stats Scrape $(date) ===" >> "$LOG"

# Stat codes
declare -A STATS
STATS[era]=211
STATS[batting_avg]=210
STATS[fielding_pct]=212
STATS[scoring]=213
STATS[slugging]=321
STATS[obp]=504
STATS[k_per_9]=425
STATS[whip]=597
STATS[k_bb_ratio]=591

# Season → ranking_period (final stats)
declare -A PERIODS
PERIODS[2025]=104
PERIODS[2024]=100
PERIODS[2023]=96
PERIODS[2022]=92
PERIODS[2021]=88

SEASONS="${1:-2021,2022,2023,2024,2025}"
STAT_LIST="${2:-all}"

SUCCESS=0
FAILED=0
COMBINED="[]"

IFS=',' read -ra SEASON_ARR <<< "$SEASONS"

for SEASON in "${SEASON_ARR[@]}"; do
    PERIOD="${PERIODS[$SEASON]:-}"
    if [ -z "$PERIOD" ]; then
        echo "  No ranking period for $SEASON, skipping" >> "$LOG"
        continue
    fi
    
    if [ "$STAT_LIST" = "all" ]; then
        STAT_KEYS=("${!STATS[@]}")
    else
        IFS=',' read -ra STAT_KEYS <<< "$STAT_LIST"
    fi
    
    for STAT in "${STAT_KEYS[@]}"; do
        CODE="${STATS[$STAT]:-}"
        if [ -z "$CODE" ]; then
            echo "  Unknown stat: $STAT" >> "$LOG"
            FAILED=$((FAILED+1))
            continue
        fi
        
        URL="https://stats.ncaa.org/rankings/national_ranking?academic_year=${SEASON}&division=1&ranking_period=${PERIOD}&sport_code=MBA&stat_seq=${CODE}"
        echo "  [$SEASON] $STAT (code=$CODE)..." >> "$LOG"
        
        # Navigate
        openclaw browser navigate --browser-profile "$PROFILE" --json "$URL" >> "$LOG" 2>&1 || {
            echo "    Navigation failed" >> "$LOG"
            FAILED=$((FAILED+1))
            sleep 2
            continue
        }
        sleep 4
        
        # Show all entries (-1 = show all)
        openclaw browser evaluate --browser-profile "$PROFILE" --json \
            --fn '() => { const sel = document.querySelector("select[name=\"rankings_table_length\"]") || document.querySelector("select[name=\"stat_grid_length\"]"); if (!sel) return 0; sel.value = "-1"; sel.dispatchEvent(new Event("change")); return -1; }' \
            >> "$LOG" 2>&1 || true
        sleep 3
        
        # Extract table data
        RESULT=$(openclaw browser evaluate --browser-profile "$PROFILE" --json \
            --fn '() => { const rows = document.querySelectorAll("#rankings_table tbody tr, #stat_grid tbody tr"); const data = []; for (const row of rows) { const cells = row.querySelectorAll("td"); if (cells.length < 4) continue; const rank = cells[0]?.textContent?.trim(); if (rank === "Reclassifying" || !rank) continue; const team = cells[1]?.textContent?.trim(); const games = cells[2]?.textContent?.trim(); const record = cells[3]?.textContent?.trim(); const value = cells[cells.length - 1]?.textContent?.trim(); if (team && value) data.push({rank: rank === "-" ? null : parseInt(rank) || null, team, games: parseInt(games) || null, record: record || null, value: parseFloat(value) || null}); } return data; }' \
            2>>"$LOG") || {
            echo "    Extraction failed" >> "$LOG"
            FAILED=$((FAILED+1))
            sleep 2
            continue
        }
        
        # Parse result from openclaw JSON wrapper
        DATA=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    wrapper = json.load(sys.stdin)
    data = wrapper.get('result', [])
    if isinstance(data, str):
        data = json.loads(data)
    print(json.dumps(data))
except:
    print('[]')
" 2>>"$LOG")
        
        COUNT=$(echo "$DATA" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
        echo "    Got $COUNT teams" >> "$LOG"
        
        if [ "$COUNT" -gt "10" ]; then
            echo "$DATA" > "$DATA_DIR/${SEASON}_${STAT}.json"
            SUCCESS=$((SUCCESS+1))
            
            # Append to combined
            COMBINED=$(python3 -c "
import json, sys
combined = json.loads(sys.argv[1])
combined.append({'stat_name': '$STAT', 'season': $SEASON, 'teams': json.loads(sys.argv[2])})
print(json.dumps(combined))
" "$COMBINED" "$DATA" 2>>"$LOG")
        else
            echo "    Too few teams ($COUNT), skipping" >> "$LOG"
            FAILED=$((FAILED+1))
        fi
        
        sleep 2
    done
done

# Save combined and load into DB
echo "$COMBINED" > "$DATA_DIR/combined_latest.json"
CATS=$(echo "$COMBINED" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
echo "  Loading $CATS stat categories into DB..." >> "$LOG"
python3 -u scripts/ncaa_stats_scraper.py load "$DATA_DIR/combined_latest.json" >> "$LOG" 2>&1 || true

echo "=== Done: $SUCCESS success, $FAILED failed $(date) ===" >> "$LOG"
echo "Results: $SUCCESS scraped, $FAILED failed"
