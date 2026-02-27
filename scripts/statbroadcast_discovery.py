#!/usr/bin/env python3
"""
StatBroadcast Event Discovery — find SB event IDs for our scheduled games.

For each game in our DB on a target date, attempts to discover the matching
StatBroadcast event by scanning candidate event IDs from known group pages.

Usage:
    python3 scripts/statbroadcast_discovery.py --date 2026-02-26
    python3 scripts/statbroadcast_discovery.py --scan-range 652700 652800

As a library:
    from statbroadcast_discovery import discover_events, match_game
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from statbroadcast_client import StatBroadcastClient
from team_resolver import TeamResolver

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "baseball.db"
GROUP_IDS_PATH = SCRIPTS_DIR / "sb_group_ids.json"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS statbroadcast_events (
    sb_event_id INTEGER PRIMARY KEY,
    game_id TEXT,
    home_team TEXT,
    visitor_team TEXT,
    home_team_id TEXT,
    visitor_team_id TEXT,
    game_date TEXT,
    group_id TEXT,
    xml_file TEXT,
    completed INTEGER DEFAULT 0,
    discovered_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (game_id) REFERENCES games(id)
);
"""


def ensure_table(conn):
    """Create statbroadcast_events table if it doesn't exist."""
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Group ID helpers
# ---------------------------------------------------------------------------

def load_group_ids(path=None):
    """Load team_id -> sb_group_id mapping from JSON file.

    Returns dict like {"washington-state": "wsu", "byu": "byu", ...}
    """
    p = Path(path) if path else GROUP_IDS_PATH
    if not p.exists():
        logger.warning("Group IDs file not found: %s", p)
        return {}
    with open(p) as f:
        return json.load(f)


def invert_group_ids(group_ids):
    """Invert mapping to sb_group_id -> [team_id, ...] for reverse lookup."""
    result = {}
    for team_id, gid in group_ids.items():
        result.setdefault(gid, []).append(team_id)
    return result


# ---------------------------------------------------------------------------
# Game matching
# ---------------------------------------------------------------------------

def match_game(sb_event_info, conn, resolver=None):
    """
    Match a StatBroadcast event to a game in our DB.

    Args:
        sb_event_info: dict from client.get_event_info() with keys:
            home, visitor, date, sport, completed, group_id, xml_file, event_id
        conn: sqlite3 connection to our DB
        resolver: TeamResolver instance (created if None)

    Returns:
        game_id string or None if no match found
    """
    if not sb_event_info:
        return None

    # Only match baseball games
    sport = sb_event_info.get('sport', '')
    if sport and sport != 'bsgame':
        return None

    if resolver is None:
        resolver = TeamResolver()

    # Decode HTML entities (StatBroadcast XML uses &apos; etc.)
    try:
        from html import unescape
        home_name = unescape(sb_event_info.get('home', '') or '')
        visitor_name = unescape(sb_event_info.get('visitor', '') or '')
    except ImportError:
        home_name = sb_event_info.get('home', '')
        visitor_name = sb_event_info.get('visitor', '')
    game_date = sb_event_info.get('date', '')

    if not home_name or not visitor_name or not game_date:
        return None

    # Resolve team names to our DB IDs
    home_id = resolver.resolve(home_name)
    visitor_id = resolver.resolve(visitor_name)

    if not home_id or not visitor_id:
        logger.debug(
            "Could not resolve teams: home=%r -> %s, visitor=%r -> %s",
            home_name, home_id, visitor_name, visitor_id
        )
        return None

    # Search for matching game in DB by date + teams (either order)
    c = conn.cursor()
    row = c.execute("""
        SELECT id FROM games
        WHERE date = ?
          AND (
            (home_team_id = ? AND away_team_id = ?) OR
            (home_team_id = ? AND away_team_id = ?)
          )
        LIMIT 1
    """, (game_date, home_id, visitor_id, visitor_id, home_id)).fetchone()

    if row:
        game_id = row[0] if isinstance(row, tuple) else row['id']
        return game_id

    return None


# ---------------------------------------------------------------------------
# Discovery via event ID scanning
# ---------------------------------------------------------------------------

def probe_event(client, event_id):
    """Probe a single StatBroadcast event ID. Returns event info or None."""
    try:
        info = client.get_event_info(event_id)
        if info and info.get('sport') == 'bsgame':
            return info
    except Exception as e:
        logger.debug("Error probing event %d: %s", event_id, e)
    return None


def discover_events_by_scan(conn, start_id, end_id, target_date=None,
                            client=None, resolver=None):
    """
    Discover StatBroadcast events by scanning a range of event IDs.

    Args:
        conn: sqlite3 connection
        start_id: first event ID to try
        end_id: last event ID to try (exclusive)
        target_date: only match games on this date (YYYY-MM-DD string)
        client: StatBroadcastClient instance
        resolver: TeamResolver instance

    Returns:
        list of dicts with keys: sb_event_id, game_id, home_team, visitor_team, ...
    """
    if client is None:
        client = StatBroadcastClient()
    if resolver is None:
        resolver = TeamResolver()

    ensure_table(conn)
    discovered = []

    for eid in range(start_id, end_id):
        info = probe_event(client, eid)
        if not info:
            continue

        event_date = info.get('date', '')
        if target_date and event_date != target_date:
            continue

        game_id = match_game(info, conn, resolver)
        home_id = resolver.resolve(info.get('home', ''))
        visitor_id = resolver.resolve(info.get('visitor', ''))

        record = {
            'sb_event_id': info['event_id'],
            'game_id': game_id,
            'home_team': info.get('home', ''),
            'visitor_team': info.get('visitor', ''),
            'home_team_id': home_id,
            'visitor_team_id': visitor_id,
            'game_date': event_date,
            'group_id': info.get('group_id', ''),
            'xml_file': info.get('xml_file', ''),
            'completed': 1 if info.get('completed') else 0,
        }

        _upsert_sb_event(conn, record)
        discovered.append(record)

        logger.info(
            "Discovered event %d: %s @ %s (%s) -> game_id=%s",
            info['event_id'], info.get('visitor'), info.get('home'),
            event_date, game_id
        )

    return discovered


def discover_events_for_games(conn, target_date=None, group_ids=None,
                              client=None, resolver=None):
    """
    Discover StatBroadcast events for games in our DB on a target date.

    Strategy: For each game on the target date where we know the home team's
    SB group ID, try to find the matching SB event by querying candidate
    event IDs already stored, or by probing known patterns.

    Args:
        conn: sqlite3 connection
        target_date: date string YYYY-MM-DD (defaults to today)
        group_ids: dict of team_id -> sb_group_id
        client: StatBroadcastClient instance
        resolver: TeamResolver instance

    Returns:
        list of matched records
    """
    if client is None:
        client = StatBroadcastClient()
    if resolver is None:
        resolver = TeamResolver()
    if group_ids is None:
        group_ids = load_group_ids()
    if target_date is None:
        target_date = date.today().isoformat()

    ensure_table(conn)

    # Get games for target date
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, home_team_id, away_team_id FROM games WHERE date = ?",
        (target_date,)
    ).fetchall()

    if not rows:
        logger.info("No games found for %s", target_date)
        return []

    logger.info("Found %d games for %s", len(rows), target_date)

    # Check which games already have SB event mappings
    matched = []
    for row in rows:
        game_id = row[0] if isinstance(row, tuple) else row['id']
        existing = c.execute(
            "SELECT sb_event_id FROM statbroadcast_events WHERE game_id = ?",
            (game_id,)
        ).fetchone()
        if existing:
            logger.debug("Game %s already mapped to SB event %s", game_id,
                         existing[0] if isinstance(existing, tuple) else existing['sb_event_id'])
            continue

        home_id = row[1] if isinstance(row, tuple) else row['home_team_id']
        sb_gid = group_ids.get(home_id)
        if sb_gid:
            logger.info("Game %s: home team %s has SB group %s", game_id,
                        home_id, sb_gid)
        else:
            logger.debug("Game %s: home team %s has no SB group mapping",
                         game_id, home_id)

    return matched


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _upsert_sb_event(conn, record):
    """Insert or update a statbroadcast_events row."""
    conn.execute("""
        INSERT OR REPLACE INTO statbroadcast_events
            (sb_event_id, game_id, home_team, visitor_team,
             home_team_id, visitor_team_id, game_date, group_id,
             xml_file, completed, discovered_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        record['sb_event_id'],
        record.get('game_id'),
        record.get('home_team', ''),
        record.get('visitor_team', ''),
        record.get('home_team_id'),
        record.get('visitor_team_id'),
        record.get('game_date', ''),
        record.get('group_id', ''),
        record.get('xml_file', ''),
        record.get('completed', 0),
    ))
    conn.commit()


def get_active_events(conn):
    """Get non-completed SB events that have a matched game_id and are past start time.
    
    Only returns events for:
    - Games already in-progress or with situation data
    - Games whose start time has passed (within 10 min grace)
    - Games with TBD times during the general game window (11 AM - 11 PM CT)
    - Games on past dates that weren't marked completed (cleanup)
    """
    import pytz
    from datetime import datetime
    
    ct = pytz.timezone('America/Chicago')
    now = datetime.now(ct)
    today = now.strftime('%Y-%m-%d')
    now_hour = now.hour
    now_min = now.minute
    now_minutes = now_hour * 60 + now_min
    
    c = conn.cursor()
    rows = c.execute("""
        SELECT se.sb_event_id, se.game_id, se.home_team, se.visitor_team,
               se.home_team_id, se.visitor_team_id, se.game_date, se.group_id,
               se.xml_file, se.completed,
               g.time, g.status
        FROM statbroadcast_events se
        LEFT JOIN games g ON se.game_id = g.id
        WHERE se.completed = 0 AND se.game_id IS NOT NULL
    """).fetchall()
    
    result = []
    for row in rows:
        if isinstance(row, tuple):
            ev = {
                'sb_event_id': row[0], 'game_id': row[1],
                'home_team': row[2], 'visitor_team': row[3],
                'home_team_id': row[4], 'visitor_team_id': row[5],
                'game_date': row[6], 'group_id': row[7],
                'xml_file': row[8], 'completed': row[9],
            }
            game_time = row[10]
            game_status = row[11]
        else:
            ev = dict(row)
            game_time = row['time'] if 'time' in row.keys() else None
            game_status = row['status'] if 'status' in row.keys() else None
        
        # Always poll games already in progress
        if game_status == 'in-progress':
            result.append(ev)
            continue
        
        # Skip future dates
        game_date = ev.get('game_date', '')
        if game_date > today:
            continue
        
        # Past dates: include for cleanup (mark_completed will handle)
        if game_date < today:
            result.append(ev)
            continue
        
        # Today's games: check if past start time
        if game_time and game_time not in ('TBD', 'TBA', ''):
            try:
                parts = game_time.replace('\xa0', ' ').strip().split()
                time_part = parts[0]
                ampm = parts[1].upper()
                h, m = time_part.split(':')
                h, m = int(h), int(m)
                if ampm == 'PM' and h != 12:
                    h += 12
                elif ampm == 'AM' and h == 12:
                    h = 0
                game_minutes = h * 60 + m
                
                # Poll if within 10 min of start or past start
                if now_minutes >= game_minutes - 10:
                    result.append(ev)
            except (ValueError, IndexError):
                # Can't parse time, include it to be safe
                result.append(ev)
        else:
            # TBD time: poll during game window (11 AM - 11 PM CT)
            if 11 <= now_hour <= 23:
                result.append(ev)
    
    return result


def mark_completed(conn, sb_event_id):
    """Mark a StatBroadcast event as completed."""
    conn.execute(
        "UPDATE statbroadcast_events SET completed = 1 WHERE sb_event_id = ?",
        (sb_event_id,)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discover StatBroadcast events for scheduled games"
    )
    parser.add_argument('--date', default=None,
                        help='Target date (YYYY-MM-DD, default: today)')
    parser.add_argument('--scan-range', nargs=2, type=int, metavar=('START', 'END'),
                        help='Scan event ID range (e.g., 652700 652800)')
    parser.add_argument('--db', default=str(DB_PATH),
                        help='Database path')
    parser.add_argument('--group-ids', default=str(GROUP_IDS_PATH),
                        help='Group IDs JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)

    target_date = args.date or date.today().isoformat()
    client = StatBroadcastClient()

    if args.scan_range:
        start, end = args.scan_range
        logger.info("Scanning event IDs %d-%d for date %s", start, end,
                     target_date)
        results = discover_events_by_scan(
            conn, start, end, target_date=target_date, client=client
        )
    else:
        group_ids = load_group_ids(args.group_ids)
        logger.info("Discovering events for %s (%d group IDs loaded)",
                     target_date, len(group_ids))
        results = discover_events_for_games(
            conn, target_date=target_date, group_ids=group_ids, client=client
        )

    logger.info("Discovered %d events", len(results))
    for r in results:
        print(
            "  SB#{sb_event_id}: {visitor_team} @ {home_team} ({game_date}) "
            "-> {game_id}".format(**r)
        )

    conn.close()


if __name__ == '__main__':
    main()
