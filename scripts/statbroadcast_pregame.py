#!/usr/bin/env python3
"""
StatBroadcast Pre-Game Discovery — automatically find SB event IDs for upcoming games.

Runs daily before games start. For each game on the target date:
1. Looks up the home team's SB group ID
2. Scrapes the home team's athletic site for StatBroadcast links
3. Falls back to scanning the SB schedule page via browser CLI
4. Registers matched events in the statbroadcast_events table

Usage:
    # Discover events for today
    python3 scripts/statbroadcast_pregame.py

    # Discover events for tomorrow
    python3 scripts/statbroadcast_pregame.py --date 2026-02-27

    # Discover for a date range (e.g., upcoming weekend)
    python3 scripts/statbroadcast_pregame.py --date 2026-02-27 --days 3

    # Verbose output
    python3 scripts/statbroadcast_pregame.py -v
"""

import argparse
import codecs
import base64
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from statbroadcast_client import StatBroadcastClient, sb_decode
from statbroadcast_discovery import ensure_table, _upsert_sb_event, match_game
from team_resolver import TeamResolver

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / 'data' / 'baseball.db'
GROUP_IDS_PATH = PROJECT_ROOT / 'scripts' / 'sb_group_ids.json'

# Known athletic site URL patterns for SIDEARM-powered schools
# {team_id} is our slug (e.g., 'washington-state')
# Most SIDEARM sites follow predictable URL patterns
SIDEARM_PATTERNS = [
    # Direct slug patterns
    "https://{slug}athletics.com/sports/baseball/schedule",
    "https://{slug}sports.com/sports/baseball/schedule",
    "https://go{slug}.com/sports/baseball/schedule",
    "https://{slug}.com/sports/baseball/schedule",
    # Common abbreviation patterns
    "https://{abbrev}athletics.com/sports/baseball/schedule",
    "https://{abbrev}sports.com/sports/baseball/schedule",
    "https://go{abbrev}.com/sports/baseball/schedule",
]


def get_db():
    """Get DB connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_group_ids():
    """Load team_id -> sb_group_id mapping."""
    with open(str(GROUP_IDS_PATH)) as f:
        return json.load(f)


def get_games_for_date(conn, target_date):
    """Get all games for a target date that don't have SB events yet."""
    ensure_table(conn)

    rows = conn.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.status
        FROM games g
        LEFT JOIN statbroadcast_events se ON g.id = se.game_id
        WHERE g.date = ?
          AND g.status != 'cancelled'
          AND se.sb_event_id IS NULL
        ORDER BY g.id
    """, (target_date,)).fetchall()

    return [dict(r) for r in rows]


def scrape_sidearm_for_sb_links(team_id):
    """
    Try to scrape a team's athletic site for StatBroadcast event links.
    Returns list of event IDs found, or empty list.
    """
    slug = team_id.replace('-', '')
    abbrev = ''.join(w[0] for w in team_id.split('-') if w)

    urls_to_try = []
    for pattern in SIDEARM_PATTERNS:
        urls_to_try.append(pattern.format(slug=slug, abbrev=abbrev))

    event_ids = set()
    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=4) as resp:
                html = resp.read().decode('utf-8', errors='ignore')

                # Look for StatBroadcast broadcast links
                # Pattern 1: stats.statbroadcast.com/broadcast/?id=NNNNN
                for m in re.finditer(r'stats?\.statbroadcast\.com/broadcast/\?id=(\d+)', html):
                    event_ids.add(int(m.group(1)))

                # Pattern 2: statb.us/b/NNNNN (short URLs)
                for m in re.finditer(r'statb\.us/b/(\d+)', html):
                    event_ids.add(int(m.group(1)))

                if event_ids:
                    logger.debug("Found %d SB links on %s", len(event_ids), url)
                    break  # Got results, stop trying other URLs

        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            continue

    return sorted(event_ids)


def probe_event(client, event_id):
    """Probe a single event ID. Returns event info or None."""
    try:
        return client.get_event_info(event_id)
    except Exception:
        return None


def discover_from_sidearm(conn, games, group_ids, client, resolver):
    """
    Phase 1: Scrape SIDEARM athletic sites for SB links.
    Returns list of game_ids that were matched.
    """
    matched = []
    # Group games by home team to avoid scraping the same site multiple times
    home_teams = {}
    for g in games:
        ht = g['home_team_id']
        if ht not in home_teams:
            home_teams[ht] = []
        home_teams[ht].append(g)

    for home_team, team_games in home_teams.items():
        if home_team not in group_ids:
            logger.debug("No SB group ID for %s, skipping SIDEARM scrape", home_team)
            continue

        logger.info("Scraping SIDEARM for %s...", home_team)
        event_ids = scrape_sidearm_for_sb_links(home_team)

        if not event_ids:
            logger.debug("No SB links found for %s", home_team)
            continue

        logger.info("  Found %d SB event IDs for %s", len(event_ids), home_team)

        # Check each event ID for date/team match
        for eid in event_ids:
            info = probe_event(client, eid)
            if not info:
                continue
            if info.get('sport') != 'bsgame':
                continue

            game_id = match_game(info, conn, resolver)
            if game_id and any(g['id'] == game_id for g in team_games):
                _upsert_sb_event(conn, {
                    'sb_event_id': eid,
                    'game_id': game_id,
                    'home_team': unescape(info.get('home', '')),
                    'visitor_team': unescape(info.get('visitor', '')),
                    'home_team_id': resolver.resolve(unescape(info.get('home', ''))),
                    'visitor_team_id': resolver.resolve(unescape(info.get('visitor', ''))),
                    'game_date': info.get('date', ''),
                    'group_id': info.get('group_id', ''),
                    'xml_file': info.get('xml_file', ''),
                    'completed': 0,
                })
                logger.info("  ✓ Matched event %d -> %s", eid, game_id)
                matched.append(game_id)

    return matched


def discover_from_scan(conn, games, group_ids, client, resolver, scan_width=200):
    """
    Phase 2: For unmatched games, scan nearby event ID ranges.
    Uses the highest known event ID as an anchor and scans around it.
    """
    matched = []

    # Find the highest known event ID as anchor
    row = conn.execute("SELECT MAX(sb_event_id) FROM statbroadcast_events").fetchone()
    max_known = row[0] if row and row[0] else 650000

    # Scan a range around the anchor
    start = max(max_known - scan_width, 600000)
    end = max_known + scan_width
    probe_ids = list(range(start, end, 3))  # every 3rd ID

    logger.info("Scanning SB event IDs %d-%d (%d probes)...", start, end, len(probe_ids))

    target_dates = set(g['id'].split('_')[0] for g in games)

    found_count = 0
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(probe_event, client, eid): eid for eid in probe_ids}
        for f in as_completed(futures):
            info = f.result()
            if not info:
                continue
            if info.get('sport') != 'bsgame':
                continue
            if info.get('date') not in target_dates:
                continue

            found_count += 1
            game_id = match_game(info, conn, resolver)
            if game_id and any(g['id'] == game_id for g in games):
                _upsert_sb_event(conn, {
                    'sb_event_id': info['event_id'],
                    'game_id': game_id,
                    'home_team': unescape(info.get('home', '')),
                    'visitor_team': unescape(info.get('visitor', '')),
                    'home_team_id': resolver.resolve(unescape(info.get('home', ''))),
                    'visitor_team_id': resolver.resolve(unescape(info.get('visitor', ''))),
                    'game_date': info.get('date', ''),
                    'group_id': info.get('group_id', ''),
                    'xml_file': info.get('xml_file', ''),
                    'completed': 0,
                })
                logger.info("  ✓ Scan matched event %d -> %s", info['event_id'], game_id)
                matched.append(game_id)

    logger.info("Scan found %d baseball events for target dates, matched %d",
                found_count, len(matched))
    return matched


def discover_from_browser(conn, games, group_ids, client, resolver):
    """
    Phase 3: Last resort — use openclaw browser CLI to load SB schedule pages.
    Only used for games still unmatched after Phase 1 and 2.
    """
    matched = []

    # Group remaining by home team
    home_teams = {}
    for g in games:
        ht = g['home_team_id']
        gid = group_ids.get(ht)
        if gid and ht not in home_teams:
            home_teams[ht] = (gid, [])
        if gid:
            home_teams[ht][1].append(g)

    for home_team, (gid, team_games) in home_teams.items():
        logger.info("Browser scrape for %s (gid=%s)...", home_team, gid)

        url = "https://www.statbroadcast.com/events/statbroadcast.php?gid={}".format(gid)
        try:
            # Use openclaw browser CLI if available
            result = subprocess.run(
                ['openclaw', 'browser', 'eval', '--profile', 'openclaw', '--url', url,
                 '--wait', '5000',
                 '--js', '''
                    [...document.querySelectorAll('a[href*="broadcast/?id="]')]
                    .map(a => a.href.match(/id=(\\d+)/)?.[1])
                    .filter(Boolean)
                    .join(',')
                 '''],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                event_ids = [int(x) for x in result.stdout.strip().split(',') if x.isdigit()]
                logger.info("  Browser found %d event IDs", len(event_ids))

                for eid in event_ids:
                    info = probe_event(client, eid)
                    if not info or info.get('sport') != 'bsgame':
                        continue
                    game_id = match_game(info, conn, resolver)
                    if game_id and any(g['id'] == game_id for g in team_games):
                        _upsert_sb_event(conn, {
                            'sb_event_id': eid,
                            'game_id': game_id,
                            'home_team': unescape(info.get('home', '')),
                            'visitor_team': unescape(info.get('visitor', '')),
                            'home_team_id': resolver.resolve(unescape(info.get('home', ''))),
                            'visitor_team_id': resolver.resolve(unescape(info.get('visitor', ''))),
                            'game_date': info.get('date', ''),
                            'group_id': info.get('group_id', ''),
                            'xml_file': info.get('xml_file', ''),
                            'completed': 0,
                        })
                        logger.info("  ✓ Browser matched event %d -> %s", eid, game_id)
                        matched.append(game_id)
            else:
                logger.debug("  Browser returned no results for %s", gid)

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("  Browser scrape failed for %s: %s", gid, e)
            continue

    return matched


def discover_from_group_scan(conn, games, group_ids, client, resolver):
    """
    Smart scan: for each home team's group ID, probe nearby event IDs
    based on known event ranges for that group.
    """
    matched = []
    
    # Build reverse map: gid -> team_ids
    gid_to_teams = {}
    for team_id, gid in group_ids.items():
        gid_to_teams.setdefault(gid, []).append(team_id)
    
    # Get known event ranges per group
    existing = conn.execute("""
        SELECT group_id, MIN(sb_event_id) as min_id, MAX(sb_event_id) as max_id
        FROM statbroadcast_events
        WHERE group_id IS NOT NULL
        GROUP BY group_id
    """).fetchall()
    
    gid_ranges = {}
    for row in existing:
        gid_ranges[row[0]] = (row[1], row[2])
    
    # For each home team in unmatched games, scan near their known range
    home_teams = set()
    for g in games:
        ht = g['home_team_id']
        gid = group_ids.get(ht)
        if gid:
            home_teams.add((ht, gid))
    
    for team_id, gid in home_teams:
        known_range = gid_ranges.get(gid)
        if known_range:
            # Scan around known range for this group
            center = known_range[1]  # latest known event
            probe_ids = list(range(center - 20, center + 50))
        else:
            # No known range — skip, will be caught by broad scan
            continue
        
        logger.debug("Group scan for %s (gid=%s): probing %d-%d",
                     team_id, gid, probe_ids[0], probe_ids[-1])
        
        for eid in probe_ids:
            info = probe_event(client, eid)
            if not info or info.get('sport') != 'bsgame':
                continue
            if info.get('group_id') != gid:
                continue
            
            game_id = match_game(info, conn, resolver)
            if game_id and any(g['id'] == game_id for g in games):
                _upsert_sb_event(conn, {
                    'sb_event_id': eid,
                    'game_id': game_id,
                    'home_team': unescape(info.get('home', '')),
                    'visitor_team': unescape(info.get('visitor', '')),
                    'home_team_id': resolver.resolve(unescape(info.get('home', ''))),
                    'visitor_team_id': resolver.resolve(unescape(info.get('visitor', ''))),
                    'game_date': info.get('date', ''),
                    'group_id': info.get('group_id', ''),
                    'xml_file': info.get('xml_file', ''),
                    'completed': 0,
                })
                logger.info("  ✓ Group scan matched %d -> %s", eid, game_id)
                matched.append(game_id)
    
    return matched


def run_discovery(target_date, verbose=False):
    """
    Run the full discovery pipeline for a target date.
    Returns (total_games, matched_count, unmatched_games).
    """
    conn = get_db()
    ensure_table(conn)
    group_ids = load_group_ids()
    client = StatBroadcastClient()
    resolver = TeamResolver()

    games = get_games_for_date(conn, target_date)
    if not games:
        logger.info("No unmatched games for %s", target_date)
        return 0, 0, []

    logger.info("Discovering SB events for %s: %d games need matching", target_date, len(games))

    all_matched = set()

    # Phase 1: Broad event ID range scan (parallel, fast, most effective)
    # Scan a wide range centered on the latest known event ID
    phase1 = discover_from_scan(conn, games, group_ids, client, resolver,
                                scan_width=2000)
    all_matched.update(phase1)
    logger.info("Phase 1 (broad scan): matched %d/%d", len(phase1), len(games))

    # Phase 2: Group-specific targeted scan for remaining
    remaining = [g for g in games if g['id'] not in all_matched]
    if remaining:
        phase_g = discover_from_group_scan(conn, remaining, group_ids, client, resolver)
        all_matched.update(phase_g)
        if phase_g:
            logger.info("Phase 2 (group scan): matched %d more", len(phase_g))

    # Phase 3: SIDEARM scrape for remaining if under 20 unmatched
    remaining = [g for g in games if g['id'] not in all_matched]
    if remaining and len(remaining) <= 20:
        phase3 = discover_from_sidearm(conn, remaining, group_ids, client, resolver)
        all_matched.update(phase3)
        if phase3:
            logger.info("Phase 3 (SIDEARM): matched %d more", len(phase3))

    unmatched = [g for g in games if g['id'] not in all_matched]
    logger.info("Discovery complete: %d/%d matched, %d unmatched",
                len(all_matched), len(games), len(unmatched))

    if unmatched and verbose:
        for g in unmatched:
            logger.info("  Unmatched: %s", g['id'])

    conn.close()
    return len(games), len(all_matched), [g['id'] for g in unmatched]


def main():
    parser = argparse.ArgumentParser(description='StatBroadcast pre-game event discovery')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD, default: today)')
    parser.add_argument('--days', type=int, default=1, help='Number of days to discover (default: 1)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    if args.date:
        start_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        start_date = datetime.now()

    total_all = 0
    matched_all = 0
    unmatched_all = []

    for day_offset in range(args.days):
        target = (start_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        total, matched, unmatched = run_discovery(target, args.verbose)
        total_all += total
        matched_all += matched
        unmatched_all.extend(unmatched)

    print("\n=== Discovery Summary ===")
    print("Total games: {}".format(total_all))
    print("Matched: {}".format(matched_all))
    print("Unmatched: {}".format(len(unmatched_all)))
    if unmatched_all:
        print("Unmatched games:")
        for gid in unmatched_all:
            print("  {}".format(gid))


if __name__ == '__main__':
    main()
