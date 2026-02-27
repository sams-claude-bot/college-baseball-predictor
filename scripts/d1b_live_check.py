#!/usr/bin/env python3
"""
D1Baseball Live Status Check — detect TBD games that have started.

Scrapes d1baseball.com/scores to find games marked as in-progress or completed,
then updates our DB so ESPN/SB pollers pick them up.

Designed to run every 15-30 min during game hours via cron.

Usage:
    python3 scripts/d1b_live_check.py              # Today
    python3 scripts/d1b_live_check.py 2026-02-27   # Specific date
"""
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytz

PROJECT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT / "data" / "baseball.db"
MAPPING = PROJECT / "data" / "d1b_slug_mapping.json"
PROFILE = "openclaw"
OPENCLAW_BIN = "/home/sam/.npm-global/bin/openclaw"

CT = pytz.timezone("America/Chicago")


def run_browser_cmd(args: list, timeout: int = 30) -> dict:
    """Run an openclaw browser CLI command and return parsed JSON."""
    cmd = [OPENCLAW_BIN, "browser"] + args + ["--browser-profile", PROFILE, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Browser command failed: {result.stderr[:200]}")
    return json.loads(result.stdout)


def scrape_d1b_status(date_str: str) -> list:
    """Scrape D1B scores page for game statuses.
    
    Returns list of dicts with: away_slug, home_slug, in_progress, is_over, 
    away_score, home_score, matchup_time
    """
    d1b_date = date_str.replace("-", "")
    url = f"https://d1baseball.com/scores/?date={d1b_date}"
    
    # Navigate
    run_browser_cmd(["navigate", url])
    
    # Wait for JS rendering
    import time
    time.sleep(5)
    
    # Extract status data from tiles
    js = """() => {
        const tiles = document.querySelectorAll('.d1-score-tile');
        const seen = new Set();
        const games = [];
        tiles.forEach(t => {
            const key = t.dataset.key;
            if (seen.has(key)) return;
            seen.add(key);
            
            const away = (t.querySelector('.team-1 a[href*="/team/"]') || {}).href || '';
            const home = (t.querySelector('.team-2 a[href*="/team/"]') || {}).href || '';
            const awaySlug = away.match(/\\/team\\/([^/]+)/)?.[1] || '';
            const homeSlug = home.match(/\\/team\\/([^/]+)/)?.[1] || '';
            
            const inProgress = t.dataset.inProgress === 'true' || t.dataset.inProgress === '1';
            const isOver = t.dataset.isOver === 'true' || t.dataset.isOver === '1';
            const ts = parseInt(t.dataset.matchupTime) || 0;
            
            // Get scores if available
            const awayScoreEl = t.querySelector('.team-1 .score, .team-1 .d1-score');
            const homeScoreEl = t.querySelector('.team-2 .score, .team-2 .d1-score');
            const awayScore = awayScoreEl ? parseInt(awayScoreEl.textContent.trim()) : null;
            const homeScore = homeScoreEl ? parseInt(homeScoreEl.textContent.trim()) : null;
            
            if (awaySlug && homeSlug) {
                games.push({
                    a: awaySlug, h: homeSlug,
                    ip: inProgress, over: isOver,
                    as: isNaN(awayScore) ? null : awayScore,
                    hs: isNaN(homeScore) ? null : homeScore,
                    t: ts
                });
            }
        });
        return games;
    }"""
    
    resp = run_browser_cmd(["evaluate", "--fn", js])
    return resp.get("result", [])


def ts_to_central(ts: int, date_str: str) -> str:
    """Convert D1B pseudo-UTC timestamp to Central time string."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    base_ts = int((dt - datetime(1970, 1, 1)).total_seconds())
    seconds_from_midnight = ts - base_ts
    total_minutes = seconds_from_midnight // 60
    et_hours = total_minutes // 60
    et_minutes = total_minutes % 60
    
    # ET -> CT: always 1 hour difference
    ct_hours = et_hours - 1
    if ct_hours < 0:
        ct_hours += 24
    
    ampm = 'AM' if ct_hours < 12 else 'PM'
    display_hour = ct_hours
    if display_hour == 0:
        display_hour = 12
    elif display_hour > 12:
        display_hour -= 12
    return f"{display_hour}:{et_minutes:02d} {ampm}"


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now(CT).strftime('%Y-%m-%d')
    
    # Load slug mapping
    with open(MAPPING) as f:
        slug_map = json.load(f)
    
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    
    # Get our TBD/scheduled games for this date
    tbd_games = conn.execute("""
        SELECT id, away_team_id, home_team_id, status, time
        FROM games 
        WHERE date = ? AND status = 'scheduled'
    """, (date_str,)).fetchall()
    
    if not tbd_games:
        print(f"No scheduled games for {date_str}")
        conn.close()
        return
    
    # Build lookup: (away_id, home_id) -> game row
    game_lookup = {}
    for g in tbd_games:
        game_lookup[(g['away_team_id'], g['home_team_id'])] = dict(g)
        # Also reversed
        game_lookup[(g['home_team_id'], g['away_team_id'])] = dict(g)
    
    print(f"Checking D1B for {len(tbd_games)} scheduled games on {date_str}...")
    
    try:
        d1b_games = scrape_d1b_status(date_str)
    except Exception as e:
        print(f"Error scraping D1B: {e}", file=sys.stderr)
        conn.close()
        return
    
    print(f"D1B returned {len(d1b_games)} games")
    
    updated_times = 0
    detected_live = 0
    
    for dg in d1b_games:
        away_id = slug_map.get(dg['a'])
        home_id = slug_map.get(dg['h'])
        if not away_id or not home_id:
            continue
        
        game = game_lookup.get((away_id, home_id))
        if not game:
            continue
        
        game_id = game['id']
        game_time = game['time']
        
        # Fill time if missing
        if (not game_time or game_time in ('TBD', 'TBA', '')) and dg['t'] > 0:
            time_str = ts_to_central(dg['t'], date_str)
            conn.execute("UPDATE games SET time = ? WHERE id = ?", (time_str, game_id))
            updated_times += 1
            print(f"  Time filled: {game_id} → {time_str}")
        
        # Detect games that have started
        if dg['ip'] or dg['over']:
            current_status = conn.execute(
                "SELECT status FROM games WHERE id = ?", (game_id,)
            ).fetchone()['status']
            
            if current_status == 'scheduled':
                new_status = 'final' if dg['over'] else 'in-progress'
                updates = {"status": new_status}
                
                # Set scores if D1B has them
                if dg['as'] is not None and dg['hs'] is not None:
                    updates['away_score'] = dg['as']
                    updates['home_score'] = dg['hs']
                
                set_clause = ', '.join(f"{k} = ?" for k in updates)
                conn.execute(
                    f"UPDATE games SET {set_clause} WHERE id = ?",
                    list(updates.values()) + [game_id]
                )
                detected_live += 1
                print(f"  Now {new_status}: {game_id}" + 
                      (f" ({dg['as']}-{dg['hs']})" if dg['as'] is not None else ""))
    
    conn.commit()
    conn.close()
    
    print(f"\nDone: {updated_times} times filled, {detected_live} games detected as started")


if __name__ == "__main__":
    main()
