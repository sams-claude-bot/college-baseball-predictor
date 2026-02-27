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
from team_resolver import TeamResolver

logger = logging.getLogger(__name__)

_resolver = None

def _get_resolver():
    """Lazy-init a shared TeamResolver."""
    global _resolver
    if _resolver is None:
        _resolver = TeamResolver()
    return _resolver


def validate_sb_teams(situation, event, conn):
    """
    Cross-validate SB situation team names against our expected matchup.

    Returns True if the teams match (or can't be validated), False if there's
    a confirmed mismatch (SB is serving stale/wrong game data).
    """
    sb_home = situation.get('home', '')
    sb_visitor = situation.get('visitor', '')

    if not sb_home and not sb_visitor:
        # No team names in situation — can't validate, allow through
        return True

    game_id = event.get('game_id', '')
    if not game_id:
        return True

    # Get expected teams from our games table
    row = conn.execute(
        "SELECT home_team_id, away_team_id FROM games WHERE id = ?",
        (game_id,)
    ).fetchone()
    if not row:
        return True

    expected_home = row[0] if isinstance(row, tuple) else row['home_team_id']
    expected_away = row[1] if isinstance(row, tuple) else row['away_team_id']

    resolver = _get_resolver()

    # Resolve SB team names to our IDs
    from html import unescape
    sb_home_id = resolver.resolve(unescape(sb_home)) if sb_home else None
    sb_visitor_id = resolver.resolve(unescape(sb_visitor)) if sb_visitor else None

    # If we can't resolve either SB team, allow through (don't block on resolver gaps)
    if not sb_home_id and not sb_visitor_id:
        return True

    # Check if teams match (in either direction — SB sometimes swaps home/away)
    expected = {expected_home, expected_away}
    resolved = set()
    if sb_home_id:
        resolved.add(sb_home_id)
    if sb_visitor_id:
        resolved.add(sb_visitor_id)

    # At least one resolved team must be in the expected set
    if resolved & expected:
        return True

    logger.warning(
        "SB TEAM MISMATCH for game %s (SB event %s): "
        "SB says %s vs %s (resolved: %s vs %s), "
        "expected %s vs %s — skipping update (stale SB data?)",
        game_id, event.get('sb_event_id'),
        sb_visitor, sb_home, sb_visitor_id, sb_home_id,
        expected_away, expected_home,
    )
    return False

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
        'sb_visitor_innings': sb_situation.get('visitor_innings'),
        'sb_home_innings': sb_situation.get('home_innings'),
        'sb_visitor_hits': sb_situation.get('visitor_hits'),
        'sb_home_hits': sb_situation.get('home_hits'),
        'sb_visitor_errors': sb_situation.get('visitor_errors'),
        'sb_home_errors': sb_situation.get('home_errors'),
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

    # Store current inning plays (PXP view — current half-inning only)
    if sb_situation.get('current_plays'):
        existing['sb_plays'] = sb_situation['current_plays']

    # Store all scoring plays for the game
    if sb_situation.get('scoring_plays'):
        existing['sb_scoring_plays'] = sb_situation['scoring_plays']

    return json.dumps(existing)


# ---------------------------------------------------------------------------
# Poller core
# ---------------------------------------------------------------------------

class StatBroadcastPoller:
    """Polls StatBroadcast for live game updates."""

    # After this many consecutive 404s, mark event completed (feed not active)
    MAX_CONSECUTIVE_404 = 10

    def __init__(self, conn, client=None, interval=DEFAULT_INTERVAL):
        self.conn = conn
        self.client = client or StatBroadcastClient()
        self.interval = interval
        self._filetimes = {}  # sb_event_id -> last filetime
        self._poll_counts = {}  # sb_event_id -> poll cycle counter
        self._404_counts = {}  # sb_event_id -> consecutive 404 count
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

        # Use threading for parallel HTTP fetches when many games are live
        if len(events) > 5:
            return self._poll_parallel(events)

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

    def _poll_parallel(self, events):
        """Poll events in parallel using a thread pool.
        
        HTTP fetches happen in parallel, DB writes are serialized via lock.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not hasattr(self, '_db_lock'):
            self._db_lock = threading.Lock()

        updated = 0
        with ThreadPoolExecutor(max_workers=min(10, len(events))) as pool:
            futures = {
                pool.submit(self._poll_event_threadsafe, ev): ev
                for ev in events
            }
            for f in as_completed(futures):
                ev = futures[f]
                try:
                    if f.result():
                        updated += 1
                except Exception as e:
                    logger.error(
                        "Error polling SB event %s (game %s): %s",
                        ev['sb_event_id'], ev['game_id'], e
                    )
        return updated

    def _poll_event_threadsafe(self, event):
        """Thread-safe wrapper: HTTP fetch without lock, DB write with lock."""
        sb_id = event['sb_event_id']
        game_id = event['game_id']
        xml_file = event.get('xml_file', '')

        if not xml_file:
            return False

        filetime = self._filetimes.get(sb_id, 0)

        # HTTP fetch (no lock needed)
        try:
            html, new_ft = self.client.get_live_stats(
                sb_id, xml_file, filetime=filetime
            )
        except Exception as e:
            is_404 = '404' in str(e)
            if is_404:
                cnt = self._404_counts.get(sb_id, 0) + 1
                self._404_counts[sb_id] = cnt
                if cnt >= self.MAX_CONSECUTIVE_404:
                    logger.warning(
                        "SB event %s (game %s): %d consecutive 404s — marking completed",
                        sb_id, game_id, cnt,
                    )
                    with self._db_lock:
                        mark_completed(self.conn, sb_id)
                    self._404_counts.pop(sb_id, None)
                    return False
            logger.error("HTTP error polling SB event %s: %s", sb_id, e)
            return False

        # Successful fetch — reset 404 counter
        self._404_counts.pop(sb_id, None)
        self._filetimes[sb_id] = new_ft

        if html is None:
            return False

        situation = parse_situation(html)
        if not situation:
            return False

        # Cross-validate: ensure SB is serving data for the RIGHT game
        # (DB read needs lock)
        with self._db_lock:
            if not validate_sb_teams(situation, event, self.conn):
                logger.warning(
                    "Skipping SB event %s — team mismatch (stale data from previous game)",
                    sb_id,
                )
                return False

        # Check completion
        title = situation.get('title', '')
        inning_disp = situation.get('inning_display', '')
        is_final = 'final' in title.lower() or 'final' in inning_disp.lower()

        # PXP + scoring (every 3rd cycle)
        poll_count = self._poll_counts.get(sb_id, 0)
        self._poll_counts[sb_id] = poll_count + 1

        if poll_count % 3 == 0:
            try:
                plays = self.client.get_play_by_play(sb_id, xml_file)
                if plays:
                    situation['current_plays'] = plays
            except Exception:
                pass
            try:
                scoring = self.client.get_scoring_plays(sb_id, xml_file)
                if scoring:
                    situation['scoring_plays'] = scoring
            except Exception:
                pass

        # DB writes (serialized)
        with self._db_lock:
            if is_final:
                # Write final scores and mark game as final
                self._finalize_game(game_id, sb_id, situation)
                mark_completed(self.conn, sb_id)
                logger.info("Game %s (SB %s) is Final", game_id, sb_id)
                return False

            self._update_situation(game_id, situation)
            self._insert_live_event(game_id, situation)
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
            is_404 = '404' in str(e)
            if is_404:
                cnt = self._404_counts.get(sb_id, 0) + 1
                self._404_counts[sb_id] = cnt
                if cnt >= self.MAX_CONSECUTIVE_404:
                    logger.warning(
                        "SB event %s (game %s): %d consecutive 404s — marking completed",
                        sb_id, game_id, cnt,
                    )
                    mark_completed(self.conn, sb_id)
                    self._404_counts.pop(sb_id, None)
                    return False
            logger.error("HTTP error polling SB event %s: %s", sb_id, e)
            return False

        # Successful fetch — reset 404 counter
        self._404_counts.pop(sb_id, None)

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

        # Cross-validate: ensure SB is serving data for the RIGHT game
        if not validate_sb_teams(situation, event, self.conn):
            logger.warning(
                "Skipping SB event %s — team mismatch (stale data from previous game)",
                sb_id,
            )
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

        # Fetch PXP and scoring less frequently (every 3rd cycle ≈ 60s)
        # to reduce load when many games are live
        poll_count = self._poll_counts.get(sb_id, 0)
        self._poll_counts[sb_id] = poll_count + 1

        if poll_count % 3 == 0:
            # Fetch current inning play-by-play
            try:
                plays = self.client.get_play_by_play(sb_id, xml_file)
                if plays:
                    situation['current_plays'] = plays
            except Exception as e:
                logger.debug("PXP fetch failed for SB %s: %s", sb_id, e)

            # Fetch all scoring plays for the game
            try:
                scoring = self.client.get_scoring_plays(sb_id, xml_file)
                if scoring:
                    situation['scoring_plays'] = scoring
            except Exception as e:
                logger.debug("Scoring plays fetch failed for SB %s: %s", sb_id, e)

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

    def _finalize_game(self, game_id, sb_id, situation):
        """Set game status to 'final', write final scores, and determine winner.

        Computes actual innings from SB linescore data.  Only stores
        ``innings`` when the game went to extras (>9); clears it for
        regulation games so the UI doesn't show a stale mid-game count.
        """
        home_score = situation.get('home_score')
        visitor_score = situation.get('visitor_score')

        # Compute actual innings from linescore (most accurate)
        innings = None
        home_innings = situation.get('home_innings', [])
        visitor_innings = situation.get('visitor_innings', [])
        if home_innings or visitor_innings:
            innings = max(len(home_innings), len(visitor_innings))

        # Only store innings for extra-inning games
        final_innings = innings if innings and innings > 9 else None
        inning_display = 'Final' if not final_innings else f'Final/{final_innings}'

        if home_score is not None and visitor_score is not None:
            # Determine winner (need team IDs from games table)
            row = self.conn.execute(
                "SELECT home_team_id, away_team_id FROM games WHERE id = ?",
                (game_id,)
            ).fetchone()
            winner_id = None
            if row:
                home_tid = row[0] if isinstance(row, tuple) else row['home_team_id']
                away_tid = row[0] if isinstance(row, tuple) else row['away_team_id']
                if isinstance(row, tuple):
                    away_tid = row[1]
                if int(home_score) > int(visitor_score):
                    winner_id = home_tid
                elif int(visitor_score) > int(home_score):
                    winner_id = away_tid

            self.conn.execute("""
                UPDATE games
                SET home_score = ?, away_score = ?,
                    inning_text = ?,
                    innings = ?,
                    status = 'final',
                    winner_id = COALESCE(?, winner_id),
                    updated_at = ?
                WHERE id = ?
            """, (home_score, visitor_score, inning_display,
                  final_innings, winner_id,
                  datetime.utcnow().isoformat(), game_id))
            self.conn.commit()
            logger.info("Finalized game %s: %s-%s, winner=%s, innings=%s",
                       game_id, visitor_score, home_score, winner_id,
                       innings or '9 (regulation)')
        else:
            # No scores but game marked final — just update status, clear stale innings
            self.conn.execute("""
                UPDATE games SET status = 'final', innings = NULL, updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), game_id))
            self.conn.commit()
            logger.info("Finalized game %s (no scores available)", game_id)

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

    conn = sqlite3.connect(args.db, timeout=30, check_same_thread=False)
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
