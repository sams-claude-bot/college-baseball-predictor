#!/usr/bin/env python3
"""
StatBroadcast Live Poller — poll SB feeds and update game situation data.

Runs as a daemon, polling every 20 seconds for active games with known
StatBroadcast event IDs. Updates the games.situation_json column and
inserts events into live_events.

Usage:
    # Daemon mode (runs until SIGTERM/SIGINT)
    python3 scripts/statbroadcast_poller.py

    # Single pass (for testing / cron)
    python3 scripts/statbroadcast_poller.py --once

    # Custom interval
    python3 scripts/statbroadcast_poller.py --interval 30
"""

import argparse
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from statbroadcast_client import StatBroadcastClient, parse_situation
from statbroadcast_discovery import ensure_table, get_active_events, mark_completed

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "baseball.db"
DEFAULT_INTERVAL = 20  # seconds


# ---------------------------------------------------------------------------
# Live events table
# ---------------------------------------------------------------------------

ENSURE_LIVE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS live_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    data_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (game_id) REFERENCES games(id)
);
"""


def ensure_live_events_table(conn):
    """Create live_events table if it doesn't exist."""
    conn.execute(ENSURE_LIVE_EVENTS_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Situation JSON merge
# ---------------------------------------------------------------------------

def merge_situation(existing_json, sb_situation):
    """
    Merge StatBroadcast situation data into existing situation_json.

    Adds SB-specific fields prefixed with 'sb_' without overwriting
    ESPN data. Returns the merged JSON string.
    """
    existing = {}
    if existing_json:
        try:
            existing = json.loads(existing_json)
        except (json.JSONDecodeError, TypeError):
            existing = {}

    # Map SB situation fields to sb_-prefixed keys
    sb_fields = {
        'sb_outs': sb_situation.get('outs'),
        'sb_count': sb_situation.get('count'),
        'sb_balls': sb_situation.get('balls'),
        'sb_strikes': sb_situation.get('strikes'),
        'sb_batter': sb_situation.get('batter_name'),
        'sb_batter_number': sb_situation.get('batter_number'),
        'sb_batter_position': sb_situation.get('batter_position'),
        'sb_pitcher': sb_situation.get('pitcher_name'),
        'sb_pitcher_number': sb_situation.get('pitcher_number'),
        'sb_inning': sb_situation.get('inning'),
        'sb_inning_half': sb_situation.get('inning_half'),
        'sb_inning_display': sb_situation.get('inning_display'),
        'sb_visitor_score': sb_situation.get('visitor_score'),
        'sb_home_score': sb_situation.get('home_score'),
        'sb_title': sb_situation.get('title'),
        'sb_on_first': sb_situation.get('on_first'),
        'sb_on_second': sb_situation.get('on_second'),
        'sb_on_third': sb_situation.get('on_third'),
        'sb_runner_first': sb_situation.get('runner_first'),
        'sb_runner_second': sb_situation.get('runner_second'),
        'sb_runner_third': sb_situation.get('runner_third'),
        'sb_updated_at': datetime.utcnow().isoformat(),
    }

    # Collect batter stats if present
    batter_stats = {}
    for key in ('batter_avg', 'batter_hits', 'batter_abs'):
        if sb_situation.get(key) is not None:
            batter_stats[key] = sb_situation[key]
    if batter_stats:
        sb_fields['sb_batter_stats'] = json.dumps(batter_stats)

    # Collect pitcher stats if present
    pitcher_stats = {}
    for key in ('pitcher_era', 'pitcher_ip', 'pitcher_k', 'pitcher_bb'):
        if sb_situation.get(key) is not None:
            pitcher_stats[key] = sb_situation[key]
    if pitcher_stats:
        sb_fields['sb_pitcher_stats'] = json.dumps(pitcher_stats)

    # Merge: SB fields override previous SB fields, but don't touch non-sb_ keys
    for key, val in sb_fields.items():
        if val is not None:
            existing[key] = val

    return json.dumps(existing)


# ---------------------------------------------------------------------------
# Poller core
# ---------------------------------------------------------------------------

class StatBroadcastPoller:
    """Polls StatBroadcast for live game updates."""

    def __init__(self, conn, client=None, interval=DEFAULT_INTERVAL):
        self.conn = conn
        self.client = client or StatBroadcastClient()
        self.interval = interval
        self._filetimes = {}  # sb_event_id -> last filetime
        self._running = True

    def stop(self):
        """Signal the poller to stop."""
        self._running = False

    def poll_once(self):
        """
        Single polling pass: fetch active events, poll each, update DB.

        Returns number of games updated.
        """
        events = get_active_events(self.conn)
        if not events:
            logger.debug("No active SB events to poll")
            return 0

        updated = 0
        for ev in events:
            try:
                did_update = self._poll_event(ev)
                if did_update:
                    updated += 1
            except Exception as e:
                logger.error(
                    "Error polling SB event %s (game %s): %s",
                    ev['sb_event_id'], ev['game_id'], e
                )
        return updated

    def _poll_event(self, event):
        """Poll a single SB event and update the DB. Returns True if updated."""
        sb_id = event['sb_event_id']
        game_id = event['game_id']
        xml_file = event.get('xml_file', '')

        if not xml_file:
            logger.warning("No xml_file for SB event %s", sb_id)
            return False

        filetime = self._filetimes.get(sb_id, 0)

        try:
            html, new_ft = self.client.get_live_stats(
                sb_id, xml_file, filetime=filetime
            )
        except Exception as e:
            logger.error("HTTP error polling SB event %s: %s", sb_id, e)
            return False

        # Update stored filetime
        self._filetimes[sb_id] = new_ft

        if html is None:
            # 304 / no change
            logger.debug("No change for SB event %s (filetime=%d)", sb_id, new_ft)
            return False

        # Parse situation from HTML
        situation = parse_situation(html)
        if not situation:
            logger.debug("Empty situation for SB event %s", sb_id)
            return False

        # Check if game completed — check title, inning_display, and game status
        title = situation.get('title', '')
        inning_disp = situation.get('inning_display', '')
        is_final = (
            'final' in title.lower()
            or 'final' in inning_disp.lower()
        )

        # Also check if our games table already says final
        if not is_final:
            row = self.conn.execute(
                "SELECT status FROM games WHERE id = ?", (game_id,)
            ).fetchone()
            if row:
                status = row[0] if isinstance(row, tuple) else row['status']
                if status == 'final':
                    is_final = True

        if is_final:
            mark_completed(self.conn, sb_id)
            logger.info("Game %s (SB %s) is Final", game_id, sb_id)
            return False  # Don't push stale situation data for completed games

        # Merge into situation_json
        self._update_situation(game_id, situation)

        # Insert live_event
        self._insert_live_event(game_id, situation)

        # Update scores in games table if we have them
        self._update_scores(game_id, situation)

        logger.info(
            "Updated game %s from SB %s: %s %s-%s %s (%s, %s outs)",
            game_id, sb_id,
            situation.get('visitor', '?'),
            situation.get('visitor_score', '?'),
            situation.get('home_score', '?'),
            situation.get('home', '?'),
            situation.get('inning_display', '?'),
            situation.get('outs', '?'),
        )
        return True

    def _update_situation(self, game_id, situation):
        """Merge SB situation into games.situation_json."""
        c = self.conn.cursor()
        row = c.execute(
            "SELECT situation_json FROM games WHERE id = ?", (game_id,)
        ).fetchone()

        existing_json = None
        if row:
            existing_json = row[0] if isinstance(row, tuple) else row['situation_json']

        merged = merge_situation(existing_json, situation)

        self.conn.execute(
            "UPDATE games SET situation_json = ? WHERE id = ?",
            (merged, game_id)
        )
        self.conn.commit()

    def _insert_live_event(self, game_id, situation):
        """Insert a live_event record for this SB update."""
        ensure_live_events_table(self.conn)

        data = {
            'source': 'statbroadcast',
            'game_id': game_id,
            'outs': situation.get('outs'),
            'count': situation.get('count'),
            'batter': situation.get('batter_name'),
            'pitcher': situation.get('pitcher_name'),
            'inning': situation.get('inning'),
            'inning_half': situation.get('inning_half'),
            'visitor_score': situation.get('visitor_score'),
            'home_score': situation.get('home_score'),
            'on_first': situation.get('on_first', False),
            'on_second': situation.get('on_second', False),
            'on_third': situation.get('on_third', False),
        }

        self.conn.execute(
            "INSERT INTO live_events (game_id, event_type, data_json) VALUES (?, ?, ?)",
            (game_id, 'sb_situation', json.dumps(data))
        )
        self.conn.commit()

    def _update_scores(self, game_id, situation):
        """Update games table scores if SB provides them."""
        home_score = situation.get('home_score')
        visitor_score = situation.get('visitor_score')
        inning_display = situation.get('inning_display', '')

        if home_score is not None and visitor_score is not None:
            self.conn.execute("""
                UPDATE games
                SET home_score = ?, away_score = ?,
                    inning_text = COALESCE(?, inning_text),
                    status = 'in-progress',
                    updated_at = ?
                WHERE id = ?
            """, (home_score, visitor_score, inning_display,
                  datetime.utcnow().isoformat(), game_id))
            self.conn.commit()

    def run(self):
        """Run the polling loop until stopped."""
        logger.info("StatBroadcast poller starting (interval=%ds)", self.interval)

        while self._running:
            try:
                n = self.poll_once()
                if n > 0:
                    logger.info("Updated %d games", n)
            except Exception as e:
                logger.error("Polling loop error: %s", e)

            # Sleep in short intervals so we can respond to stop signals
            for _ in range(self.interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("StatBroadcast poller stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Poll StatBroadcast for live game updates"
    )
    parser.add_argument('--once', action='store_true',
                        help='Single poll pass then exit')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                        help='Poll interval in seconds (default: %d)' % DEFAULT_INTERVAL)
    parser.add_argument('--db', default=str(DB_PATH),
                        help='Database path')
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
    ensure_live_events_table(conn)

    client = StatBroadcastClient()
    poller = StatBroadcastPoller(conn, client=client, interval=args.interval)

    # Signal handling for clean shutdown
    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        poller.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if args.once:
        n = poller.poll_once()
        logger.info("Single pass complete: %d games updated", n)
    else:
        poller.run()

    conn.close()


if __name__ == '__main__':
    main()
