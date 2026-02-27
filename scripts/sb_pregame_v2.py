#!/usr/bin/env python3
"""
StatBroadcast Pre-Game Discovery v2 — uses schedule.php for reliable event discovery.

For each home team with games that lack SB events:
1. Loads statbroadcast.com/events/schedule.php?live=0&gid={gid}
2. Filters to Baseball via JS dropdown
3. Extracts all event IDs + dates from the filtered table
4. Probes new events for metadata (xml_file, team names)
5. Matches to games in our DB
6. Saves incrementally after each school

Rate-limited (10s between schools) to avoid 403 blocks.
Detects 403 and backs off automatically.

Usage:
    python3 scripts/sb_pregame_v2.py              # Today
    python3 scripts/sb_pregame_v2.py --date 2026-02-28
    python3 scripts/sb_pregame_v2.py --all-dates   # Register events for ALL dates
    python3 scripts/sb_pregame_v2.py --delay 15     # Custom delay between schools
"""
import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html import unescape
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from statbroadcast_client import StatBroadcastClient
from statbroadcast_discovery import ensure_table, _upsert_sb_event, match_game
from team_resolver import TeamResolver

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / 'data' / 'baseball.db'
GROUP_IDS_PATH = PROJECT_ROOT / 'scripts' / 'sb_group_ids.json'
DEFAULT_DELAY = 10  # seconds between schools

# JS to filter to baseball + show 100 entries + extract event data
EXTRACT_JS = """() => {
    // Filter to baseball
    const sportSel = document.getElementById('sports');
    if (sportSel) {
        sportSel.value = 'M;bsgame';
        sportSel.dispatchEvent(new Event('change'));
    }
    // Show 100 entries
    const lenSel = document.querySelector('select[name=eventCalendar_length]');
    if (lenSel) {
        lenSel.value = '100';
        lenSel.dispatchEvent(new Event('change'));
    }
    // Check for 403
    if (document.title.includes('403') || document.body.innerText.includes('403 Forbidden')) {
        return {error: '403'};
    }
    // Extract events from DataTable
    const rows = document.querySelectorAll('#eventCalendar tbody tr');
    const events = [];
    rows.forEach(r => {
        if (r.style.display === 'none') return;
        const cells = r.querySelectorAll('td');
        const link = r.querySelector('a[href*="statb.us"]');
        if (link && cells.length >= 3) {
            const href = link.href;
            const id = href.match(/\\/b\\/(\\d+)/)?.[1] || href.match(/\\/(\\d+)/)?.[1];
            if (id) {
                events.push({
                    id: parseInt(id),
                    date: cells[0]?.textContent?.trim() || '',
                    matchup: cells[1]?.textContent?.trim() || '',
                    sport: cells[2]?.textContent?.trim() || ''
                });
            }
        }
    });
    return {events};
}"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def load_group_ids():
    with open(str(GROUP_IDS_PATH)) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, str)}


def parse_sb_date(date_str):
    """Convert SB date format '02-27-26' to '2026-02-27'."""
    try:
        parts = date_str.split('-')
        if len(parts) == 3:
            mm, dd, yy = parts
            return f"20{yy}-{mm}-{dd}"
    except Exception:
        pass
    return None


def scrape_schedule_page(gid):
    """
    Use openclaw browser CLI to load a school's schedule page,
    filter to baseball, and extract all event IDs with dates.
    
    Returns (events_list, error_str_or_None).
    On 403: returns ([], '403').
    """
    url = f"https://www.statbroadcast.com/events/schedule.php?live=0&gid={gid}"
    
    try:
        # Navigate
        nav = subprocess.run(
            ['openclaw', 'browser', 'navigate', url,
             '--browser-profile', 'openclaw', '--json'],
            capture_output=True, text=True, timeout=20
        )
        if nav.returncode != 0:
            logger.debug("Navigate failed for gid=%s", gid)
            return [], 'nav_failed'
        
        # Wait for page load
        time.sleep(3)
        
        # Extract events
        result = subprocess.run(
            ['openclaw', 'browser', 'evaluate',
             '--browser-profile', 'openclaw', '--json',
             '--fn', EXTRACT_JS],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode != 0:
            logger.debug("Evaluate failed for gid=%s", gid)
            return [], 'eval_failed'
        
        data = json.loads(result.stdout)
        payload = data.get('result', {})
        
        # Check for 403
        if isinstance(payload, dict) and payload.get('error') == '403':
            return [], '403'
        
        # Check for empty page (another sign of blocking)
        if isinstance(payload, dict) and 'events' in payload:
            events = payload['events']
        elif isinstance(payload, list):
            events = payload
        else:
            return [], 'bad_response'
        
        # Convert dates
        for e in events:
            e['iso_date'] = parse_sb_date(e.get('date', ''))
        
        return events, None
        
    except subprocess.TimeoutExpired:
        logger.warning("Browser timeout for gid=%s", gid)
        return [], 'timeout'
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Error scraping gid=%s: %s", gid, e)
        return [], str(e)


def probe_events(client, event_ids):
    """Probe events in parallel for metadata (xml_file, team names)."""
    results = {}
    
    def probe(eid):
        try:
            return client.get_event_info(eid)
        except Exception:
            return None
    
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(probe, eid): eid for eid in event_ids}
        for f in as_completed(futures):
            eid = futures[f]
            info = f.result()
            if info and info.get('sport') == 'bsgame':
                results[eid] = info
    
    return results


def register_event(conn, eid, info, resolver):
    """Register a single event in the DB. Returns game_id if matched."""
    game_id = match_game(info, conn, resolver)
    
    event_data = {
        'sb_event_id': eid,
        'game_id': game_id,
        'home_team': unescape(info.get('home', '') or ''),
        'visitor_team': unescape(info.get('visitor', '') or ''),
        'home_team_id': resolver.resolve(unescape(info.get('home', '') or '')),
        'visitor_team_id': resolver.resolve(unescape(info.get('visitor', '') or '')),
        'game_date': info.get('date', ''),
        'group_id': info.get('group_id', ''),
        'xml_file': info.get('xml_file', ''),
        'completed': 1 if info.get('completed') else 0,
    }
    
    _upsert_sb_event(conn, event_data)
    return game_id


def main():
    parser = argparse.ArgumentParser(description='SB Pre-Game Discovery v2')
    parser.add_argument('--date', default=None, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--all-dates', action='store_true',
                        help='Register events for all dates, not just target')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Seconds between schools (default: {DEFAULT_DELAY})')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    
    target_date = args.date or datetime.now().strftime('%Y-%m-%d')
    
    conn = get_db()
    ensure_table(conn)
    group_ids = load_group_ids()
    client = StatBroadcastClient()
    resolver = TeamResolver()
    
    # Get games that need SB events
    if args.all_dates:
        games = conn.execute("""
            SELECT g.id, g.home_team_id, g.away_team_id, g.date
            FROM games g
            LEFT JOIN statbroadcast_events se ON g.id = se.game_id
            WHERE g.status != 'cancelled' AND se.sb_event_id IS NULL
              AND g.date >= ?
            ORDER BY g.date
        """, (target_date,)).fetchall()
    else:
        games = conn.execute("""
            SELECT g.id, g.home_team_id, g.away_team_id, g.date
            FROM games g
            LEFT JOIN statbroadcast_events se ON g.id = se.game_id
            WHERE g.date = ? AND g.status != 'cancelled' AND se.sb_event_id IS NULL
            ORDER BY g.id
        """, (target_date,)).fetchall()
    
    games = [dict(r) for r in games]
    print(f"Games needing SB events: {len(games)} (date={'all future' if args.all_dates else target_date})")
    
    if not games:
        print("Nothing to do")
        return
    
    # Group by home team, get unique gids to scrape
    home_teams = {}
    for g in games:
        ht = g['home_team_id']
        gid = group_ids.get(ht)
        if gid:
            if gid not in home_teams:
                home_teams[gid] = {'team_id': ht, 'games': []}
            home_teams[gid]['games'].append(g)
    
    no_gid = len(set(g['home_team_id'] for g in games)) - len(home_teams)
    print(f"Unique home teams to scrape: {len(home_teams)} ({no_gid} without SB group ID)")
    print(f"Delay between schools: {args.delay}s")
    
    # Get already-registered event IDs
    existing_ids = set()
    for row in conn.execute("SELECT sb_event_id FROM statbroadcast_events"):
        existing_ids.add(row['sb_event_id'])
    
    # Scrape each school's schedule page
    total_found = 0
    total_new = 0
    total_registered = 0
    total_matched = 0
    consecutive_403 = 0
    schools_scraped = 0
    
    for i, (gid, info) in enumerate(home_teams.items()):
        logger.info("[%d/%d] Scraping %s (gid=%s)...",
                    i + 1, len(home_teams), info['team_id'], gid)
        
        events, error = scrape_schedule_page(gid)
        
        # Handle 403
        if error == '403':
            consecutive_403 += 1
            logger.warning("  403 Forbidden (consecutive: %d)", consecutive_403)
            print(f"  ⚠ 403 on {info['team_id']} (#{consecutive_403})")
            
            if consecutive_403 >= 3:
                print(f"\n🛑 Aborting: 3 consecutive 403s. Likely rate-limited.")
                print(f"   Schools completed: {schools_scraped}/{len(home_teams)}")
                print(f"   Try again later with: --delay {int(args.delay * 2)}")
                break
            
            # Back off and retry once
            backoff = 30 * consecutive_403
            print(f"  Backing off {backoff}s...")
            time.sleep(backoff)
            
            events, error = scrape_schedule_page(gid)
            if error == '403':
                print(f"\n🛑 Still 403 after backoff. Aborting.")
                print(f"   Schools completed: {schools_scraped}/{len(home_teams)}")
                break
        
        if error and error != '403':
            logger.warning("  Error: %s", error)
            time.sleep(args.delay)
            continue
        
        # Reset 403 counter on success
        consecutive_403 = 0
        schools_scraped += 1
        
        baseball_events = [e for e in events if e.get('sport') == 'BASE']
        total_found += len(baseball_events)
        
        # Find new events
        new_events = []
        for e in baseball_events:
            eid = e['id']
            if eid in existing_ids:
                continue
            if not args.all_dates and e.get('iso_date') != target_date:
                continue
            new_events.append(e)
        
        if baseball_events:
            logger.info("  Found %d baseball events (%d new)",
                       len(baseball_events), len(new_events))
        else:
            logger.info("  Found 0 baseball events")
        
        # Probe + register new events IMMEDIATELY (save as we go)
        if new_events:
            new_ids = [e['id'] for e in new_events]
            total_new += len(new_ids)
            
            probed = probe_events(client, new_ids)
            
            for eid, einfo in probed.items():
                game_id = register_event(conn, eid, einfo, resolver)
                existing_ids.add(eid)
                total_registered += 1
                if game_id:
                    total_matched += 1
                    print(f"  ✓ {einfo.get('visitor', '?')} @ {einfo.get('home', '?')} -> {game_id}")
                else:
                    logger.info("  Event %d registered (no game match)", eid)
        
        # Rate limit
        if i < len(home_teams) - 1:
            time.sleep(args.delay)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Schools scraped: {schools_scraped}/{len(home_teams)}")
    print(f"Baseball events found: {total_found}")
    print(f"New events: {total_new}")
    print(f"Registered: {total_registered}")
    print(f"Matched to games: {total_matched}")
    
    # Coverage stats
    total_sb = conn.execute(
        "SELECT COUNT(*) FROM statbroadcast_events WHERE game_date = ?",
        (target_date,)
    ).fetchone()[0]
    total_games = conn.execute(
        "SELECT COUNT(*) FROM games WHERE date = ?",
        (target_date,)
    ).fetchone()[0]
    print(f"SB coverage for {target_date}: {total_sb}/{total_games} ({100*total_sb//max(total_games,1)}%)")


if __name__ == '__main__':
    main()
