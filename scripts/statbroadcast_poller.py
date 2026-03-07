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
import os
import signal
import sqlite3
import sys
import time

# Ensure scripts/ is on sys.path for local imports (notifications.py, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
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
from database import configure_connection

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
                    # Before marking completed, check if xml_file is wrong
                    # (tournament games often have wrong group prefix)
                    try:
                        info = self.client.get_event_info(sb_id)
                        if info and info.get('xml_file') and info['xml_file'] != xml_file:
                            real_xml = info['xml_file']
                            real_group = info.get('group_id', '')
                            logger.warning(
                                "SB event %s: xml_file mismatch! DB=%s API=%s — fixing",
                                sb_id, xml_file, real_xml,
                            )
                            with self._db_lock:
                                self.conn.execute(
                                    "UPDATE statbroadcast_events SET xml_file = ?, group_id = ? "
                                    "WHERE sb_event_id = ?",
                                    (real_xml, real_group, sb_id),
                                )
                                self.conn.commit()
                            self._404_counts[sb_id] = 0  # reset, retry with correct xml
                            return False
                    except Exception as fix_err:
                        logger.debug("Could not re-check event info for %s: %s", sb_id, fix_err)

                    # Before marking completed, check if the event is actually done.
                    # Some events return 404 on the data feed but are still live
                    # (broadcast page works, API says not completed).
                    try:
                        info2 = self.client.get_event_info(sb_id)
                        if info2 and not info2.get('completed'):
                            # Event is still live per SB — don't mark completed.
                            # Just stop polling this cycle (d1b_live_check will reactivate).
                            logger.info(
                                "SB event %s (game %s): %d 404s but event NOT completed per API — skipping",
                                sb_id, game_id, cnt,
                            )
                            self._404_counts[sb_id] = 0  # reset counter
                            return False
                    except Exception:
                        pass

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
            self._check_notifications(game_id, situation)

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

        # Every 6th poll (~2 min), do a fresh fetch ignoring filetime cache
        # to catch games that went Final without updating the XML timestamp
        poll_count = self._poll_counts.get(sb_id, 0)
        if poll_count > 0 and poll_count % 6 == 0:
            filetime = 0

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
                    # Before marking completed, check if xml_file is wrong
                    try:
                        info = self.client.get_event_info(sb_id)
                        if info and info.get('xml_file') and info['xml_file'] != xml_file:
                            real_xml = info['xml_file']
                            real_group = info.get('group_id', '')
                            logger.warning(
                                "SB event %s: xml_file mismatch! DB=%s API=%s — fixing",
                                sb_id, xml_file, real_xml,
                            )
                            self.conn.execute(
                                "UPDATE statbroadcast_events SET xml_file = ?, group_id = ? "
                                "WHERE sb_event_id = ?",
                                (real_xml, real_group, sb_id),
                            )
                            self.conn.commit()
                            self._404_counts[sb_id] = 0
                            return False
                    except Exception:
                        pass

                    # Verify event is actually done before marking completed
                    try:
                        info2 = self.client.get_event_info(sb_id)
                        if info2 and not info2.get('completed'):
                            logger.info(
                                "SB event %s (game %s): %d 404s but NOT completed per API — skipping",
                                sb_id, game_id, cnt,
                            )
                            self._404_counts[sb_id] = 0
                            return False
                    except Exception:
                        pass

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

        # Check for notification triggers (half-inning change, upsets, finals)
        self._check_notifications(game_id, situation)

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

            # Get hits/errors from the last known situation
            sit_row = self.conn.execute(
                "SELECT situation_json FROM games WHERE id = ?", (game_id,)
            ).fetchone()
            h_hits = a_hits = h_err = a_err = None
            if sit_row:
                sit_json = sit_row[0] if isinstance(sit_row, tuple) else sit_row['situation_json']
                if sit_json:
                    import json as _json
                    try:
                        sit = _json.loads(sit_json)
                        h_hits = sit.get('sb_home_hits')
                        a_hits = sit.get('sb_away_hits')
                        h_err = sit.get('sb_home_errors')
                        a_err = sit.get('sb_away_errors')
                    except Exception:
                        pass

            self.conn.execute("""
                UPDATE games
                SET home_score = ?, away_score = ?,
                    inning_text = ?,
                    innings = ?,
                    status = 'final',
                    winner_id = COALESCE(?, winner_id),
                    home_hits = COALESCE(?, home_hits),
                    away_hits = COALESCE(?, away_hits),
                    home_errors = COALESCE(?, home_errors),
                    away_errors = COALESCE(?, away_errors),
                    updated_at = ?
                WHERE id = ?
            """, (home_score, visitor_score, inning_display,
                  final_innings, winner_id,
                  h_hits, a_hits, h_err, a_err,
                  datetime.utcnow().isoformat(), game_id))
            self.conn.commit()
            logger.info("Finalized game %s: %s-%s, winner=%s, innings=%s",
                       game_id, visitor_score, home_score, winner_id,
                       innings or '9 (regulation)')
            # Send final score notifications
            self._check_final_notifications(game_id)
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
            # DH guard: never mark a game in-progress if its counterpart already is
            partner_id = game_id[:-4] if game_id.endswith('_gm2') else game_id + '_gm2'
            partner = self.conn.execute(
                "SELECT status FROM games WHERE id = ?", (partner_id,)
            ).fetchone()
            if partner and (partner[0] if isinstance(partner, tuple) else partner['status']) == 'in-progress':
                logger.debug("DH guard: skipping %s, partner %s is already in-progress", game_id, partner_id)
                return

            # Also propagate hits/errors from SB situation to games table
            # SB uses 'visitor_*' naming, games table uses 'away_*'
            home_hits = situation.get('home_hits')
            away_hits = situation.get('visitor_hits')
            home_errors = situation.get('home_errors')
            away_errors = situation.get('visitor_errors')

            self.conn.execute("""
                UPDATE games
                SET home_score = ?, away_score = ?,
                    inning_text = COALESCE(?, inning_text),
                    home_hits = COALESCE(?, home_hits),
                    away_hits = COALESCE(?, away_hits),
                    home_errors = COALESCE(?, home_errors),
                    away_errors = COALESCE(?, away_errors),
                    status = 'in-progress',
                    updated_at = ?
                WHERE id = ?
            """, (home_score, visitor_score, inning_display,
                  home_hits, away_hits, home_errors, away_errors,
                  datetime.utcnow().isoformat(), game_id))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Staleness Detection
    # ------------------------------------------------------------------

    def _check_stale_games(self):
        """Detect in-progress games that stopped updating — likely canceled/postponed.

        Checks every ~5 min. If a game has been 'in-progress' but not updated
        in 30+ minutes:
        1. Cross-check ESPN API for postponed/canceled status
        2. If ESPN confirms postponement → update status
        3. If no ESPN data but stale 45+ min in early innings with 0-0 → mark canceled
        """
        try:
            rows = self.conn.execute("""
                SELECT id, home_team_id, away_team_id, home_score, away_score,
                       inning_text, updated_at, date
                FROM games
                WHERE status = 'in-progress'
                  AND updated_at < datetime('now', '-30 minutes')
            """).fetchall()

            if not rows:
                return

            # Fetch ESPN scores for today to cross-reference
            from espn_live_scores import fetch_espn_scores, build_espn_id_mapping, espn_status_to_db
            today = datetime.now().strftime('%Y-%m-%d')
            espn_data = fetch_espn_scores(today)

            # Build ESPN game lookup by team ids
            espn_statuses = {}
            if espn_data and 'events' in espn_data:
                for event in espn_data['events']:
                    status_type = event.get('status', {}).get('type', {})
                    comps = event.get('competitions', [{}])[0]
                    teams = comps.get('competitors', [])
                    if len(teams) == 2:
                        team_ids = set()
                        for t in teams:
                            slug = t.get('team', {}).get('slug', '')
                            if slug:
                                team_ids.add(slug)
                        espn_statuses[frozenset(team_ids)] = status_type

            for row in rows:
                game_id = row[0] if isinstance(row, tuple) else row['id']
                home_id = row[1] if isinstance(row, tuple) else row['home_team_id']
                away_id = row[2] if isinstance(row, tuple) else row['away_team_id']
                home_score = row[3] if isinstance(row, tuple) else row['home_score']
                away_score = row[4] if isinstance(row, tuple) else row['away_score']
                inning = row[5] if isinstance(row, tuple) else row['inning_text']
                updated = row[6] if isinstance(row, tuple) else row['updated_at']

                # Check ESPN for this matchup
                team_key = frozenset({home_id, away_id})
                espn_status = espn_statuses.get(team_key)

                new_status = None
                if espn_status:
                    db_status = espn_status_to_db(espn_status)
                    if db_status in ('postponed', 'canceled'):
                        new_status = db_status
                        logger.info(
                            "Stale game %s: ESPN says %s — updating",
                            game_id, db_status
                        )

                # No ESPN match — if very stale + early innings + scoreless, assume canceled
                if not new_status and not espn_status:
                    is_early = inning and ('1st' in str(inning) or '2nd' in str(inning))
                    is_scoreless = (home_score or 0) == 0 and (away_score or 0) == 0
                    # Check if updated_at is >45 minutes old
                    try:
                        from datetime import datetime as dt2
                        updated_dt = dt2.fromisoformat(updated.replace('Z', '+00:00'))
                        age_min = (dt2.utcnow() - updated_dt.replace(tzinfo=None)).total_seconds() / 60
                    except Exception:
                        age_min = 999

                    if is_early and is_scoreless and age_min > 45:
                        new_status = 'canceled'
                        logger.info(
                            "Stale game %s: no ESPN data, %s 0-0, %.0f min stale — marking canceled",
                            game_id, inning, age_min
                        )

                if new_status:
                    self.conn.execute("""
                        UPDATE games
                        SET status = ?, home_score = NULL, away_score = NULL,
                            inning_text = NULL, innings = NULL,
                            home_hits = NULL, away_hits = NULL,
                            home_errors = NULL, away_errors = NULL,
                            situation_json = NULL,
                            updated_at = ?
                        WHERE id = ?
                    """, (new_status, datetime.utcnow().isoformat(), game_id))
                    self.conn.commit()

                    # Also mark the SB event as completed so we stop polling it
                    self.conn.execute("""
                        UPDATE statbroadcast_events
                        SET status = 'completed'
                        WHERE game_id = ?
                    """, (game_id,))
                    self.conn.commit()

                    logger.info("Game %s set to '%s', SB event marked completed", game_id, new_status)

        except Exception as e:
            logger.error("Staleness check error: %s", e, exc_info=True)

    def run(self):
        """Run the polling loop until stopped."""
        logger.info("StatBroadcast poller starting (interval=%ds)", self.interval)
        stale_check_counter = 0

        while self._running:
            try:
                n = self.poll_once()
                if n > 0:
                    logger.info("Updated %d games", n)
            except Exception as e:
                logger.error("Polling loop error: %s", e)

            # Staleness check every ~5 min (15 cycles × 20s interval)
            stale_check_counter += 1
            if stale_check_counter >= 15:
                stale_check_counter = 0
                try:
                    self._check_stale_games()
                except Exception as e:
                    logger.error("Staleness check failed: %s", e)

            # Sleep in short intervals so we can respond to stop signals
            for _ in range(self.interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("StatBroadcast poller stopped")

    # ------------------------------------------------------------------
    # Push Notification Hooks
    # ------------------------------------------------------------------

    def _check_notifications(self, game_id, situation):
        """Check whether game state changes should trigger push notifications.

        Triggers:
        1. Legacy half-inning transition (all transitions) → game_update
        2. Half-inning transition where runs scored in that half → game_update_scoring
        3. Any scoring change as it happens → score_change
        4. SEC team WP drops below 25% (checked on half transitions) → upset_watch
        5. Game goes final → final_score (handled in _check_final_notifications)
        """
        try:
            def _to_int(value):
                try:
                    if value is None:
                        return None
                    return int(value)
                except (TypeError, ValueError):
                    return None

            def _norm_half(value):
                if value is None:
                    return None
                v = str(value).strip().lower()
                if v.startswith('top') or v == 't':
                    return 'top'
                if v.startswith('bot') or v.startswith('bottom') or v == 'b':
                    return 'bottom'
                return v or None

            def _ordinal(n):
                n = _to_int(n)
                if n is None:
                    return '?'
                if 10 <= (n % 100) <= 20:
                    suffix = 'th'
                else:
                    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
                return f"{n}{suffix}"

            def _half_label(inning_num, half_val):
                h = _norm_half(half_val)
                prefix = 'Top' if h == 'top' else 'Bot'
                return f"{prefix} {_ordinal(inning_num)}"

            inning = _to_int(situation.get('inning'))
            half = _norm_half(situation.get('inning_half'))
            home_score = _to_int(situation.get('home_score'))
            visitor_score = _to_int(situation.get('visitor_score'))

            if inning is None or half is None or home_score is None or visitor_score is None:
                return

            current_half_state = (inning, half)
            current_score = (visitor_score, home_score)

            if not hasattr(self, '_notif_last_half_state'):
                self._notif_last_half_state = {}
            if not hasattr(self, '_notif_half_start_score'):
                self._notif_half_start_score = {}
            if not hasattr(self, '_notif_last_score_state'):
                self._notif_last_score_state = {}

            prev_half_state = self._notif_last_half_state.get(game_id)
            prev_score_state = self._notif_last_score_state.get(game_id)

            # First sighting: seed state to avoid startup spam.
            if prev_half_state is None or prev_score_state is None:
                self._notif_last_half_state[game_id] = current_half_state
                self._notif_last_score_state[game_id] = current_score
                self._notif_half_start_score[game_id] = current_score
                return

            half_transition = current_half_state != prev_half_state

            completed_half_label = None
            scored_in_completed_half = False
            half_delta_away = 0
            half_delta_home = 0
            if half_transition:
                start_score = self._notif_half_start_score.get(game_id, prev_score_state)
                start_away = _to_int(start_score[0]) if start_score else 0
                start_home = _to_int(start_score[1]) if start_score else 0
                if start_away is None:
                    start_away = 0
                if start_home is None:
                    start_home = 0

                half_delta_away = current_score[0] - start_away
                half_delta_home = current_score[1] - start_home
                scored_in_completed_half = half_delta_away > 0 or half_delta_home > 0
                completed_half_label = _half_label(prev_half_state[0], prev_half_state[1])

            score_changed = current_score != prev_score_state
            away_delta = current_score[0] - prev_score_state[0]
            home_delta = current_score[1] - prev_score_state[1]
            runs_scored_now = away_delta > 0 or home_delta > 0

            # Update state caches before any early returns.
            self._notif_last_half_state[game_id] = current_half_state
            self._notif_last_score_state[game_id] = current_score
            if half_transition:
                self._notif_half_start_score[game_id] = current_score
            elif game_id not in self._notif_half_start_score:
                self._notif_half_start_score[game_id] = current_score

            if not half_transition and not (score_changed and runs_scored_now):
                return

            # Get team IDs and conferences
            row = self.conn.execute(
                """
                SELECT g.home_team_id, g.away_team_id,
                       h.name as home_name, a.name as away_name,
                       h.conference as home_conf, a.conference as away_conf
                FROM games g
                JOIN teams h ON g.home_team_id = h.id
                JOIN teams a ON g.away_team_id = a.id
                WHERE g.id = ?
                """,
                (game_id,),
            ).fetchone()

            if not row:
                return

            home_tid = row[0] if isinstance(row, tuple) else row['home_team_id']
            away_tid = row[1] if isinstance(row, tuple) else row['away_team_id']
            home_name = row[2] if isinstance(row, tuple) else row['home_name']
            away_name = row[3] if isinstance(row, tuple) else row['away_name']
            home_conf = row[4] if isinstance(row, tuple) else row['home_conf']
            away_conf = row[5] if isinstance(row, tuple) else row['away_conf']

            inning_label = situation.get('inning_display', _half_label(inning, half))

            from notifications import send_team_notification, send_game_notification, ensure_tables
            ensure_tables(self.conn)

            score_line = f"{away_name} {visitor_score}, {home_name} {home_score}"

            # --- 1. Legacy half-inning updates (all transitions) ---
            if half_transition:
                for team_id in (home_tid, away_tid):
                    dedup = f"game_update:{game_id}:{team_id}:{inning}:{half}"
                    send_team_notification(
                        team_id,
                        'game_update',
                        {
                            'title': f"⚾ {score_line}",
                            'body': inning_label,
                            'url': f"/game/{game_id}",
                            'tag': f"game-{game_id}",
                            'game_id': game_id,
                        },
                        dedup_key=dedup,
                        conn=self.conn,
                    )

            # --- 2. Half-inning scoring recaps only ---
            if half_transition and scored_in_completed_half:
                delta_bits = []
                if half_delta_away > 0:
                    delta_bits.append(f"{away_name} +{half_delta_away}")
                if half_delta_home > 0:
                    delta_bits.append(f"{home_name} +{half_delta_home}")
                delta_text = ', '.join(delta_bits) if delta_bits else 'Runs scored'

                recap_payload = {
                    'title': f"📌 Inning Recap: {score_line}",
                    'body': f"{completed_half_label} • {delta_text}",
                    'url': f"/game/{game_id}",
                    'tag': f"inning-score-{game_id}",
                    'game_id': game_id,
                }

                for team_id in (home_tid, away_tid):
                    dedup = (
                        f"game_update_scoring:{game_id}:{team_id}:"
                        f"{prev_half_state[0]}:{prev_half_state[1]}:{visitor_score}-{home_score}"
                    )
                    send_team_notification(
                        team_id,
                        'game_update_scoring',
                        recap_payload,
                        dedup_key=dedup,
                        conn=self.conn,
                    )

                # Also deliver recap alerts to explicit followed-game subscribers.
                send_game_notification(
                    game_id,
                    'game_update_scoring',
                    recap_payload,
                    dedup_key=(
                        f"game_update_scoring:{game_id}:game:"
                        f"{prev_half_state[0]}:{prev_half_state[1]}:{visitor_score}-{home_score}"
                    ),
                    conn=self.conn,
                )

            # --- 3. Instant scoring alerts ---
            if score_changed and runs_scored_now:
                delta_bits = []
                if away_delta > 0:
                    delta_bits.append(f"{away_name} +{away_delta}")
                if home_delta > 0:
                    delta_bits.append(f"{home_name} +{home_delta}")
                delta_text = ', '.join(delta_bits) if delta_bits else 'Score changed'

                score_payload = {
                    'title': f"🚨 Score Change: {score_line}",
                    'body': f"{delta_text} • {inning_label}",
                    'url': f"/game/{game_id}",
                    'tag': f"score-change-{game_id}",
                    'game_id': game_id,
                }

                for team_id in (home_tid, away_tid):
                    dedup = f"score_change:{game_id}:{team_id}:{visitor_score}-{home_score}"
                    send_team_notification(
                        team_id,
                        'score_change',
                        score_payload,
                        dedup_key=dedup,
                        conn=self.conn,
                    )

                # Optional game-follow mode support if enabled in future.
                send_game_notification(
                    game_id,
                    'score_change',
                    score_payload,
                    dedup_key=f"score_change:{game_id}:game:{visitor_score}-{home_score}",
                    conn=self.conn,
                )

            # --- 4. SEC Upset Watch (only on half transitions) ---
            if half_transition:
                sec_team = None
                if home_conf == 'SEC' and away_conf != 'SEC':
                    sec_team = 'home'
                elif away_conf == 'SEC' and home_conf != 'SEC':
                    sec_team = 'away'

                if sec_team:
                    try:
                        from models.win_probability import WinProbabilityModel
                        wp_model = WinProbabilityModel()
                        home_wp = wp_model.calculate(
                            home_score=int(home_score),
                            away_score=int(visitor_score),
                            inning=int(inning),
                            inning_half=half,
                            outs=int(situation.get('outs', 0)),
                            on_first=situation.get('on_first', False),
                            on_second=situation.get('on_second', False),
                            on_third=situation.get('on_third', False),
                            game_id=game_id,
                        )

                        # SEC team losing probability > 75%?
                        sec_losing = (sec_team == 'home' and home_wp < 0.25) or \
                                     (sec_team == 'away' and home_wp > 0.75)

                        if sec_losing:
                            sec_name = home_name if sec_team == 'home' else away_name
                            opp_name = away_name if sec_team == 'home' else home_name
                            sec_wp = home_wp if sec_team == 'home' else (1 - home_wp)
                            lose_pct = (1 - sec_wp) * 100

                            from notifications import send_conference_notification
                            send_conference_notification(
                                'SEC',
                                'upset_watch',
                                {
                                    'title': f"⚠️ SEC Upset Watch: {sec_name}",
                                    'body': f"{sec_name} has {lose_pct:.0f}% chance to lose vs {opp_name} | {score_line} ({inning_label})",
                                    'url': f"/game/{game_id}",
                                    'tag': f"upset-{game_id}",
                                    'game_id': game_id,
                                },
                                dedup_key=f"upset:{game_id}",
                                conn=self.conn,
                            )
                    except Exception as e:
                        logger.warning("WP calculation failed for upset check: %s", e)

        except Exception as e:
            logger.warning("Notification check error for %s: %s", game_id, e, exc_info=True)

    def _check_final_notifications(self, game_id):
        """Send final score notifications when a game completes."""
        try:
            from notifications import send_team_notification, send_game_notification, ensure_tables
            ensure_tables(self.conn)

            row = self.conn.execute("""
                SELECT g.home_team_id, g.away_team_id,
                       h.name as home_name, a.name as away_name,
                       g.home_score, g.away_score, g.innings
                FROM games g
                JOIN teams h ON g.home_team_id = h.id
                JOIN teams a ON g.away_team_id = a.id
                WHERE g.id = ?
            """, (game_id,)).fetchone()

            if not row:
                return

            home_tid, away_tid = row[0], row[1]
            home_name, away_name = row[2], row[3]
            h_score, a_score = row[4], row[5]
            innings = row[6]

            extra = f" ({innings})" if innings and innings > 9 else ""
            winner = home_name if h_score > a_score else away_name
            score_line = f"{away_name} {a_score}, {home_name} {h_score}"

            final_payload = {
                'title': f"🏁 Final{extra}: {score_line}",
                'body': f"{winner} wins!",
                'url': f"/game/{game_id}",
                'tag': f"final-{game_id}",
                'game_id': game_id,
            }

            for team_id in (home_tid, away_tid):
                send_team_notification(
                    team_id, 'final_score',
                    final_payload,
                    dedup_key=f"final:{game_id}:{team_id}",
                    conn=self.conn,
                )

            # Also send to explicit game-follow subscribers.
            send_game_notification(
                game_id,
                'final_score',
                final_payload,
                dedup_key=f"final:{game_id}:game",
                conn=self.conn,
            )
        except Exception as e:
            logger.warning("Final notification error for %s: %s", game_id, e)


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

    conn = sqlite3.connect(args.db, timeout=60, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    configure_connection(conn, busy_timeout_ms=60000)
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
