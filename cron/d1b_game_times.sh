#!/bin/bash
# D1Baseball Game Time Scraper — uses openclaw browser CLI (no AI agent needed)
# Scrapes d1baseball.com/scores/?date= to fill missing game times in DB
#
# D1B covers ~150+ games/day vs ESPN API's ~102 cap
# D1B timestamps are Eastern time encoded as pseudo-UTC
# Converts ET → CT (subtract 1h standard time, adjust for DST)
#
# Usage:
#   ./cron/d1b_game_times.sh              # Today's games
#   ./cron/d1b_game_times.sh 2026-02-28   # Specific date
#   ./cron/d1b_game_times.sh tomorrow     # Tomorrow's games
set -euo pipefail
cd /home/sam/college-baseball-predictor

DATE="${1:-today}"
if [ "$DATE" = "today" ]; then
    DATE=$(date +%Y-%m-%d)
elif [ "$DATE" = "tomorrow" ]; then
    DATE=$(date -d "+1 day" +%Y-%m-%d)
fi

# D1B URL format: YYYYMMDD (no dashes)
D1B_DATE=$(echo "$DATE" | tr -d '-')
URL="https://d1baseball.com/scores/?date=${D1B_DATE}"

LOG="logs/cron/$(date +%Y-%m-%d)_d1b_times.log"
PROFILE="openclaw"
DB="data/baseball.db"
MAPPING="data/d1b_slug_mapping.json"

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

log "=== D1B Game Times Scrape: $DATE ==="
log "URL: $URL"

# Check how many games are missing times
MISSING=$(sqlite3 "$DB" "SELECT COUNT(*) FROM games WHERE date='$DATE' AND (time IS NULL OR time='' OR time='TBA' OR time='TBD')")
TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM games WHERE date='$DATE'")
log "Games for $DATE: $TOTAL total, $MISSING missing times"

if [ "$MISSING" -eq 0 ]; then
    log "No missing times — skipping scrape"
    echo "D1B times: $DATE — 0 missing, skipped"
    exit 0
fi

# Navigate to D1B scores page
log "Opening D1B scores page..."
openclaw browser navigate --browser-profile "$PROFILE" --json "$URL" >> "$LOG" 2>&1 || {
    log "ERROR: Navigation failed"
    echo "D1B times: navigation failed" >&2
    exit 1
}

# Wait for score tiles to load (D1B uses JS rendering)
sleep 5

# Check if tiles loaded
TILE_COUNT=$(openclaw browser evaluate --browser-profile "$PROFILE" --json \
    --fn '() => document.querySelectorAll(".d1-score-tile").length' 2>>"$LOG" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',0))" 2>>"$LOG") || TILE_COUNT=0

if [ "$TILE_COUNT" -eq 0 ]; then
    log "No score tiles found — page may not have loaded. Retrying..."
    sleep 5
    TILE_COUNT=$(openclaw browser evaluate --browser-profile "$PROFILE" --json \
        --fn '() => document.querySelectorAll(".d1-score-tile").length' 2>>"$LOG" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',0))" 2>>"$LOG") || TILE_COUNT=0
fi

if [ "$TILE_COUNT" -eq 0 ]; then
    log "ERROR: No score tiles after retry"
    echo "D1B times: no tiles found" >&2
    exit 1
fi

log "Found $TILE_COUNT score tiles"

# Extract game data: away slug, home slug, Unix timestamp
GAMES_JSON=$(openclaw browser evaluate --browser-profile "$PROFILE" --json --fn '() => {
    const tiles = document.querySelectorAll(".d1-score-tile");
    const seen = new Set();
    const games = [];
    tiles.forEach(t => {
        const key = t.dataset.key;
        if (seen.has(key)) return;
        seen.add(key);
        const away = (t.querySelector(".team-1 a[href*=\"/team/\"]") || {}).href || "";
        const home = (t.querySelector(".team-2 a[href*=\"/team/\"]") || {}).href || "";
        const awaySlug = away.match(/\/team\/([^/]+)/)?.[1] || "";
        const homeSlug = home.match(/\/team\/([^/]+)/)?.[1] || "";
        const ts = parseInt(t.dataset.matchupTime) || 0;
        if (awaySlug && homeSlug && ts > 0) {
            games.push({a: awaySlug, h: homeSlug, t: ts});
        }
    });
    return games;
}' 2>>"$LOG") || { log "ERROR: JS extraction failed"; exit 1; }

# Save raw scrape data
RAW_FILE="data/d1b_times_${DATE}.json"
echo "$GAMES_JSON" | python3 -c "
import sys, json
wrapper = json.load(sys.stdin)
games = wrapper.get('result', [])
print(json.dumps(games, indent=2))
" > "$RAW_FILE" 2>>"$LOG"

GAME_COUNT=$(python3 -c "import json; print(len(json.load(open('$RAW_FILE'))))" 2>>"$LOG")
log "Extracted $GAME_COUNT unique games from D1B"

# Process and update DB
python3 << PYEOF >> "$LOG" 2>&1
import json
import sqlite3
from datetime import datetime, date as date_mod
import pytz

DATE = "$DATE"
DB = "$DB"
MAPPING_FILE = "$MAPPING"
RAW_FILE = "$RAW_FILE"

# Load D1B games
with open(RAW_FILE) as f:
    d1b_games = json.load(f)

# Load slug mapping
with open(MAPPING_FILE) as f:
    slug_map = json.load(f)

# Determine midnight-UTC base for this date (D1B stores ET as fake-UTC)
dt = datetime.strptime(DATE, '%Y-%m-%d')
base_ts = int((dt - datetime(1970, 1, 1)).total_seconds())

# Check if date is in DST (Central)
ct = pytz.timezone('America/Chicago')
et = pytz.timezone('America/New_York')
# A noon timestamp on this date
noon_ct = ct.localize(datetime.strptime(DATE + ' 12:00', '%Y-%m-%d %H:%M'))
noon_et = et.localize(datetime.strptime(DATE + ' 12:00', '%Y-%m-%d %H:%M'))
# ET-CT offset in hours (1 in standard time, 1 in DST too since both shift)
et_ct_offset = 1  # Always 1 hour since both zones observe DST together

def ts_to_central(ts):
    """Convert D1B pseudo-UTC timestamp to Central time string."""
    seconds_from_midnight = ts - base_ts
    total_minutes = seconds_from_midnight // 60
    et_hours = total_minutes // 60
    et_minutes = total_minutes % 60
    
    # ET -> CT: subtract offset
    ct_hours = et_hours - et_ct_offset
    if ct_hours < 0:
        ct_hours += 24
    
    ampm = 'AM' if ct_hours < 12 else 'PM'
    display_hour = ct_hours
    if display_hour == 0:
        display_hour = 12
    elif display_hour > 12:
        display_hour -= 12
    return f"{display_hour}:{et_minutes:02d} {ampm}"

conn = sqlite3.connect(DB, timeout=30)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Get valid team IDs
cur.execute("SELECT id FROM teams")
valid_ids = {r['id'] for r in cur.fetchall()}

# Get games missing times
cur.execute("""
    SELECT id, away_team_id, home_team_id
    FROM games WHERE date = ? AND (time IS NULL OR time = '' OR time = 'TBA' OR time = 'TBD')
""", (DATE,))
missing = {(r['away_team_id'], r['home_team_id']): r['id'] for r in cur.fetchall()}

updated = 0
unmapped = []

for g in d1b_games:
    away_id = slug_map.get(g['a'])
    home_id = slug_map.get(g['h'])
    
    if not away_id or not home_id:
        unmapped.append(f"{g['a']}@{g['h']}")
        continue
    
    if away_id not in valid_ids or home_id not in valid_ids:
        continue
    
    time_str = ts_to_central(g['t'])
    
    # Try normal order
    game_id = missing.get((away_id, home_id))
    if not game_id:
        # Try swapped
        game_id = missing.get((home_id, away_id))
    
    if game_id:
        cur.execute("UPDATE games SET time = ? WHERE id = ?", (time_str, game_id))
        updated += 1

conn.commit()

# Final count
cur.execute("""
    SELECT COUNT(*) as cnt FROM games
    WHERE date = ? AND (time IS NULL OR time = '' OR time = 'TBA' OR time = 'TBD')
""", (DATE,))
still_missing = cur.fetchone()['cnt']
total = conn.execute("SELECT COUNT(*) FROM games WHERE date=?", (DATE,)).fetchone()[0]

print(f"Updated: {updated} games")
print(f"Coverage: {total - still_missing}/{total} ({100*(total-still_missing)//total}%)")
print(f"Still missing: {still_missing}")
if unmapped:
    print(f"Unmapped D1B slugs: {', '.join(set(unmapped))}")

conn.close()
PYEOF

# Clean up raw file (keep for debugging today only)
# Older files cleaned by separate maintenance

# Summary
REMAINING=$(sqlite3 "$DB" "SELECT COUNT(*) FROM games WHERE date='$DATE' AND (time IS NULL OR time='' OR time='TBA' OR time='TBD')")
FILLED=$((MISSING - REMAINING))
log "Done: filled $FILLED times, $REMAINING still missing"
echo "D1B times: $DATE — filled $FILLED/$MISSING, $REMAINING remaining"
