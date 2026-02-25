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
STATS[slugging]=327
STATS[obp]=589
STATS[k_per_9]=425
STATS[whip]=597
STATS[k_bb_ratio]=591

# Season → ranking_period (final stats)
# For in-progress season (2026), auto-detect from NCAA dropdown
declare -A PERIODS
PERIODS[2025]=104
PERIODS[2024]=100
PERIODS[2023]=96
PERIODS[2022]=92
PERIODS[2021]=88

# Auto-detect 2026 ranking period from NCAA dropdown
detect_2026_rp() {
    echo "  Auto-detecting 2026 ranking period..." >> "$LOG"
    openclaw browser navigate --browser-profile "$PROFILE" --json \
        "https://stats.ncaa.org/rankings/change_sport_year_div?sport_code=MBA&academic_year=2026&division=1" \
        >> "$LOG" 2>&1 || { echo "  Navigation failed for RP detection" >> "$LOG"; return 1; }
    sleep 4
    
    RP_JSON=$(openclaw browser evaluate --browser-profile "$PROFILE" --json \
        --fn '() => {
            const sel = document.querySelector("select#rp");
            if (!sel) return [];
            return Array.from(sel.options)
                .filter(o => !o.textContent.includes(" - "))
                .map(o => ({value: parseFloat(o.value), text: o.textContent.trim()}));
        }' 2>>"$LOG") || { echo "  RP extraction failed" >> "$LOG"; return 1; }
    
    RP=$(echo "$RP_JSON" | python3 -c "
import sys, json
try:
    wrapper = json.load(sys.stdin)
    data = wrapper.get('result', [])
    if isinstance(data, str):
        data = json.loads(data)
    if data:
        best = max(data, key=lambda o: o['value'])
        print(int(best['value']))
    else:
        print('')
except:
    print('')
" 2>>"$LOG")
    
    if [ -n "$RP" ] && [ "$RP" != "" ]; then
        echo "  Detected 2026 ranking period: $RP" >> "$LOG"
        PERIODS[2026]="$RP"
        return 0
    else
        echo "  Could not detect 2026 ranking period" >> "$LOG"
        return 1
    fi
}

SEASONS="${1:-2026}"
STAT_LIST="${2:-all}"

# Auto-detect 2026 RP if needed
if echo "$SEASONS" | grep -q "2026"; then
    detect_2026_rp || { echo "  FATAL: Cannot detect 2026 ranking period" >> "$LOG"; exit 1; }
fi

SUCCESS=0
FAILED=0

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
        
        # Parse result from openclaw JSON wrapper and save to per-stat file
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
        else
            echo "    Too few teams ($COUNT), skipping" >> "$LOG"
            FAILED=$((FAILED+1))
        fi
        
        sleep 2
    done
done

# Build combined JSON from individual files and load into DB
echo "  Building combined JSON from individual files..." >> "$LOG"
python3 -c "
import json, glob, sys
combined = []
for season in '${SEASONS}'.split(','):
    for f in sorted(glob.glob(f'$DATA_DIR/{season}_*.json')):
        stat_name = f.rsplit('_', 1)[1].replace('.json', '')
        # Handle multi-word stat names like k_bb_ratio
        parts = f.split('/')[-1].replace('.json', '').split('_')
        season_str = parts[0]
        stat_name = '_'.join(parts[1:])
        with open(f) as fh:
            teams = json.load(fh)
        combined.append({'stat_name': stat_name, 'season': int(season_str), 'teams': teams})
with open('$DATA_DIR/combined_latest.json', 'w') as fh:
    json.dump(combined, fh)
print(f'Combined {len(combined)} categories')
" >> "$LOG" 2>&1

CATS=$(python3 -c "import json; print(len(json.load(open('$DATA_DIR/combined_latest.json'))))" 2>/dev/null || echo "0")
echo "  Loading $CATS stat categories into DB..." >> "$LOG"
python3 -u scripts/ncaa_stats_scraper.py load "$DATA_DIR/combined_latest.json" >> "$LOG" 2>&1 || true

echo "=== Done: $SUCCESS success, $FAILED failed $(date) ===" >> "$LOG"
echo "Results: $SUCCESS scraped, $FAILED failed"
