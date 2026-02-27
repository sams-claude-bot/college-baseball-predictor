#!/usr/bin/env python3
"""
StatBroadcast Season Scrape — browser-based discovery of ALL event IDs.

Loads each school's StatBroadcast schedule page via openclaw browser,
extracts all broadcast event IDs, probes the event API for metadata,
and stores them in statbroadcast_events table.

Usage:
    # Full scrape of all mapped schools
    python3 scripts/statbroadcast_season_scrape.py

    # Scrape only schools not scraped in the last N days
    python3 scripts/statbroadcast_season_scrape.py --stale-days 7

    # Scrape specific schools
    python3 scripts/statbroadcast_season_scrape.py --teams texas byu hawaii

    # Verify today's events still valid
    python3 scripts/statbroadcast_season_scrape.py --verify-today

    # Dry run (don't write to DB)
    python3 scripts/statbroadcast_season_scrape.py --dry-run
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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
SCRAPE_STATE_PATH = PROJECT_ROOT / 'data' / 'sb_scrape_state.json'

# Browser page load timeout
PAGE_TIMEOUT_MS = 10000
# Delay between browser loads to be polite
PAGE_DELAY_S = 2


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_group_ids(include_extra=True):
    """
    Load team_id -> gid mapping.
    
    If include_extra is True, also includes _extra_schools entries
    (keyed by SB name instead of team_id) so we scrape all schools.
    """
    with open(str(GROUP_IDS_PATH)) as f:
        data = json.load(f)
    
    # Separate main mapping from extra schools
    extra = data.pop('_extra_schools', {})
    
    if include_extra:
        # Add extra schools to the mapping (keyed by SB name)
        for sb_name, gid in extra.items():
            # Use a prefixed key to avoid collisions
            data[f"_extra:{sb_name}"] = gid
    
    return data


def load_scrape_state():
    """Load per-school scrape timestamps."""
    if SCRAPE_STATE_PATH.exists():
        with open(str(SCRAPE_STATE_PATH)) as f:
            return json.load(f)
    return {}


def save_scrape_state(state):
    with open(str(SCRAPE_STATE_PATH), 'w') as f:
        json.dump(state, f, indent=2)


def scrape_sb_schedule_page(gid):
    """
    Use openclaw browser CLI to load a StatBroadcast schedule page
    and extract all broadcast event IDs.

    Returns list of integer event IDs, or empty list on failure.
    """
    url = "https://www.statbroadcast.com/events/statbroadcast.php?gid={}".format(gid)

    js_extract = (
        '() => [...document.querySelectorAll("a[href*=\\"broadcast/?id=\\"]")]'
        '.map(a => a.href.match(/id=(\\d+)/)?.[1])'
        '.filter(Boolean).join(",")'
    )

    try:
        # Step 1: Navigate to the schedule page
        nav_result = subprocess.run(
            ['openclaw', 'browser', 'navigate', url,
             '--browser-profile', 'openclaw',
             '--timeout', str(PAGE_TIMEOUT_MS)],
            capture_output=True, text=True, timeout=30
        )
        if nav_result.returncode != 0:
            logger.debug("Navigate failed for gid=%s: %s",
                         gid, nav_result.stderr.strip())
            return []

        # Step 2: Wait for JS to render
        subprocess.run(
            ['openclaw', 'browser', 'wait', '--time', '5000',
             '--browser-profile', 'openclaw'],
            capture_output=True, text=True, timeout=15
        )

        # Step 3: Extract event IDs
        eval_result = subprocess.run(
            ['openclaw', 'browser', 'evaluate',
             '--browser-profile', 'openclaw',
             '--fn', js_extract],
            capture_output=True, text=True, timeout=15
        )

        if eval_result.returncode == 0 and eval_result.stdout.strip():
            raw = eval_result.stdout.strip().strip('"')
            event_ids = list(set(
                int(x) for x in raw.split(',') if x.strip().isdigit()
            ))
            return sorted(event_ids)
        else:
            logger.debug("Evaluate returned no results for gid=%s (rc=%d)",
                         gid, eval_result.returncode)
            return []

    except subprocess.TimeoutExpired:
        logger.warning("Browser timeout for gid=%s", gid)
        return []
    except FileNotFoundError:
        logger.error("openclaw CLI not found — cannot use browser scraping")
        return []
    except Exception as e:
        logger.warning("Browser error for gid=%s: %s", gid, e)
        return []


def probe_events_batch(client, event_ids):
    """
    Probe multiple event IDs in parallel for metadata.
    Returns list of event info dicts (only baseball events).
    """
    results = []

    def probe(eid):
        try:
            info = client.get_event_info(eid)
            if info and info.get('sport') == 'bsgame':
                return info
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(probe, eid): eid for eid in event_ids}
        for f in as_completed(futures):
            info = f.result()
            if info:
                results.append(info)

    return results


def register_events(conn, events, resolver, dry_run=False):
    """
    Register discovered events in the DB.
    Matches to existing games where possible.
    Returns (registered_count, matched_count).
    """
    registered = 0
    matched = 0

    for info in events:
        eid = info.get('event_id')
        if not eid:
            continue

        # Check if already registered
        existing = conn.execute(
            "SELECT sb_event_id FROM statbroadcast_events WHERE sb_event_id = ?",
            (eid,)
        ).fetchone()

        if existing:
            # Update date/completion status in case it changed
            if not dry_run:
                conn.execute("""
                    UPDATE statbroadcast_events
                    SET game_date = ?, completed = ?, xml_file = ?
                    WHERE sb_event_id = ?
                """, (
                    info.get('date', ''),
                    1 if info.get('completed') else 0,
                    info.get('xml_file', ''),
                    eid,
                ))
                conn.commit()
            continue

        # Try to match to a game in our DB
        game_id = match_game(info, conn, resolver)
        home_name = unescape(info.get('home', '') or '')
        vis_name = unescape(info.get('visitor', '') or '')

        event_data = {
            'sb_event_id': eid,
            'game_id': game_id,
            'home_team': home_name,
            'visitor_team': vis_name,
            'home_team_id': resolver.resolve(home_name),
            'visitor_team_id': resolver.resolve(vis_name),
            'game_date': info.get('date', ''),
            'group_id': info.get('group_id', ''),
            'xml_file': info.get('xml_file', ''),
            'completed': 1 if info.get('completed') else 0,
        }

        if not dry_run:
            _upsert_sb_event(conn, event_data)

        registered += 1
        if game_id:
            matched += 1
            logger.info("  ✓ Event %d: %s @ %s (%s) -> %s",
                         eid, vis_name, home_name, info.get('date'), game_id)
        else:
            logger.debug("  Event %d: %s @ %s (%s) — no game match",
                          eid, vis_name, home_name, info.get('date'))

    return registered, matched


def verify_today_events(conn, client):
    """
    Verify that today's registered events are still valid.
    Checks: date hasn't changed, event still exists, not completed prematurely.
    Returns (verified, issues) counts.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    rows = conn.execute("""
        SELECT sb_event_id, game_id, game_date, completed
        FROM statbroadcast_events
        WHERE game_date = ? AND completed = 0
    """, (today,)).fetchall()

    if not rows:
        logger.info("No events to verify for %s", today)
        return 0, 0

    verified = 0
    issues = 0

    for row in rows:
        eid = row['sb_event_id']
        info = client.get_event_info(eid)
        if not info:
            logger.warning("  ⚠ Event %d: API returned nothing (may not exist)", eid)
            issues += 1
            continue

        if info.get('date') != today:
            logger.warning("  ⚠ Event %d: date changed from %s to %s",
                           eid, today, info.get('date'))
            # Update the date
            conn.execute(
                "UPDATE statbroadcast_events SET game_date = ? WHERE sb_event_id = ?",
                (info.get('date', ''), eid)
            )
            conn.commit()
            issues += 1
        elif info.get('completed'):
            logger.info("  Event %d: already completed", eid)
            conn.execute(
                "UPDATE statbroadcast_events SET completed = 1 WHERE sb_event_id = ?",
                (eid,)
            )
            conn.commit()
        else:
            verified += 1

    logger.info("Verified %d events, %d issues", verified, issues)
    return verified, issues


def scrape_schools(team_ids, group_ids, conn, client, resolver,
                   dry_run=False, state=None):
    """
    Scrape StatBroadcast schedule pages for a list of schools.
    """
    total_registered = 0
    total_matched = 0
    total_events = 0

    for i, team_id in enumerate(team_ids):
        gid = group_ids.get(team_id)
        if not gid:
            continue

        logger.info("[%d/%d] Scraping %s (gid=%s)...",
                     i + 1, len(team_ids), team_id, gid)

        event_ids = scrape_sb_schedule_page(gid)
        if not event_ids:
            logger.info("  No events found")
            if state is not None:
                state[team_id] = {
                    'last_scraped': datetime.now().isoformat(),
                    'events_found': 0,
                }
            time.sleep(PAGE_DELAY_S)
            continue

        logger.info("  Found %d event IDs, probing...", len(event_ids))
        total_events += len(event_ids)

        # Probe for metadata
        events = probe_events_batch(client, event_ids)
        logger.info("  %d are baseball events", len(events))

        # Register
        reg, match = register_events(conn, events, resolver, dry_run)
        total_registered += reg
        total_matched += match

        if state is not None:
            state[team_id] = {
                'last_scraped': datetime.now().isoformat(),
                'events_found': len(event_ids),
                'baseball_events': len(events),
                'registered': reg,
                'matched': match,
            }

        time.sleep(PAGE_DELAY_S)

    return total_events, total_registered, total_matched


def refresh_mapping():
    """Re-run build_sb_mapping.py to update the group IDs mapping."""
    import subprocess
    script_path = PROJECT_ROOT / 'scripts' / 'build_sb_mapping.py'
    result = subprocess.run(
        ['python3', str(script_path), '--add-aliases'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("Mapping refreshed successfully")
        print(result.stdout)
    else:
        logger.error("Mapping refresh failed: %s", result.stderr)
        raise RuntimeError("Failed to refresh mapping")


def main():
    parser = argparse.ArgumentParser(description='StatBroadcast season scrape')
    parser.add_argument('--teams', nargs='+', help='Specific team IDs to scrape')
    parser.add_argument('--stale-days', type=int,
                        help='Only scrape schools not scraped in N days')
    parser.add_argument('--verify-today', action='store_true',
                        help='Verify today\'s events are still valid')
    parser.add_argument('--refresh-mapping', action='store_true',
                        help='Re-build the school mapping from SB index before scraping')
    parser.add_argument('--dry-run', action='store_true',
                        help='Don\'t write to DB')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    # Refresh mapping if requested
    if args.refresh_mapping:
        refresh_mapping()

    conn = get_db()
    ensure_table(conn)
    group_ids = load_group_ids()
    client = StatBroadcastClient()
    resolver = TeamResolver()

    # Verify mode
    if args.verify_today:
        verified, issues = verify_today_events(conn, client)
        print("\n=== Verify Summary ===")
        print("Verified: {}".format(verified))
        print("Issues: {}".format(issues))
        return

    # Determine which schools to scrape
    state = load_scrape_state()

    if args.teams:
        team_ids = args.teams
    elif args.stale_days:
        cutoff = (datetime.now() - timedelta(days=args.stale_days)).isoformat()
        team_ids = []
        for team_id in sorted(group_ids.keys()):
            school_state = state.get(team_id, {})
            last = school_state.get('last_scraped', '')
            if not last or last < cutoff:
                team_ids.append(team_id)
        logger.info("%d schools stale (>%d days)", len(team_ids), args.stale_days)
    else:
        team_ids = sorted(group_ids.keys())

    if not team_ids:
        print("No schools to scrape")
        return

    print("Scraping {} schools...".format(len(team_ids)))

    total_events, total_reg, total_match = scrape_schools(
        team_ids, group_ids, conn, client, resolver,
        dry_run=args.dry_run, state=state
    )

    if not args.dry_run:
        save_scrape_state(state)

    print("\n=== Scrape Summary ===")
    print("Schools scraped: {}".format(len(team_ids)))
    print("Total event IDs found: {}".format(total_events))
    print("New events registered: {}".format(total_reg))
    print("Matched to games: {}".format(total_match))

    # Show coverage stats
    total_sb = conn.execute("SELECT COUNT(*) FROM statbroadcast_events").fetchone()[0]
    matched_sb = conn.execute(
        "SELECT COUNT(*) FROM statbroadcast_events WHERE game_id IS NOT NULL"
    ).fetchone()[0]
    print("\nDB totals: {} SB events, {} matched to games".format(total_sb, matched_sb))


if __name__ == '__main__':
    main()
