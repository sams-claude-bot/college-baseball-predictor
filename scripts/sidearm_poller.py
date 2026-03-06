#!/usr/bin/env python3
"""
SIDEARM Live Stats Poller — poll SIDEARM feeds and update game situation data.

Reads today's SIDEARM links from the sidearm_links table, resolves school codes,
polls game.json for live data, and writes sa_-prefixed situation fields into
games.situation_json (merged with existing sb_ data).

Usage:
    # Daemon mode
    python3 scripts/sidearm_poller.py

    # Single pass
    python3 scripts/sidearm_poller.py --once

    # Custom interval
    python3 scripts/sidearm_poller.py --interval 45
"""

import argparse
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

DB_PATH = PROJECT_ROOT / "data" / "baseball.db"
DEFAULT_INTERVAL = 30
REQUEST_TIMEOUT = 15

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ordinal(n: int) -> str:
    """Return ordinal string for an integer (1st, 2nd, 3rd, ...)."""
    if 10 <= (n % 100) <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def player_name(player: Optional[dict]) -> Optional[str]:
    """Extract 'First Last' from a SIDEARM player dict."""
    if not player:
        return None
    first = player.get('FirstName', '')
    last = player.get('LastName', '')
    return f"{first} {last}".strip() or None


def extract_school_code(link: str) -> Optional[str]:
    """Extract school code from a sidearmstats.com link.

    E.g. 'https://sidearmstats.com/dixie/baseball/' → 'dixie'
    """
    m = re.search(r'sidearmstats\.com/([^/]+)/baseball', link)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# SIDEARM API
# ---------------------------------------------------------------------------

def resolve_school_code(domain: str) -> Optional[str]:
    """Resolve a SIDEARM domain to its sidearmstats.com school code.

    GET https://{domain}/api/livestats/baseball
    """
    url = f"https://{domain}/api/livestats/baseball"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            'User-Agent': 'CollegeBaseballPredictor/1.0'
        })
        resp.raise_for_status()
        data = resp.json()
        games = data.get('Games', [])
        for game in games:
            link = game.get('Link', '')
            code = extract_school_code(link)
            if code:
                return code
    except Exception as e:
        logger.warning("Failed to resolve school code for %s: %s", domain, e)
    return None


def fetch_game_data(school_code: str) -> Optional[dict]:
    """Fetch full game data from sidearmstats.com.

    GET https://sidearmstats.com/{code}/baseball/game.json?detail=full
    """
    url = f"https://sidearmstats.com/{school_code}/baseball/game.json?detail=full"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            'User-Agent': 'CollegeBaseballPredictor/1.0'
        })
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch game data for %s: %s", school_code, e)
    return None


# ---------------------------------------------------------------------------
# Situation extraction
# ---------------------------------------------------------------------------

def extract_situation(data: dict) -> Optional[Dict[str, Any]]:
    """Extract sa_-prefixed situation fields from SIDEARM game.json response."""
    game = data.get('Game')
    if not game:
        return None

    if not game.get('HasStarted', False):
        return None

    situation = game.get('Situation') or {}
    home = game.get('HomeTeam', {})
    visitor = game.get('VisitingTeam', {})

    batter = situation.get('Batter')
    pitcher = situation.get('Pitcher')
    on_deck = situation.get('OnDeck')
    on_first = situation.get('OnFirst')
    on_second = situation.get('OnSecond')
    on_third = situation.get('OnThird')

    inning_float = situation.get('Inning', 1.0)
    inning_int = int(inning_float)
    is_top = (inning_float == int(inning_float))
    inning_half = 'top' if is_top else 'bottom'
    inning_prefix = 'Top' if is_top else 'Bot'

    sa_fields: Dict[str, Any] = {
        'sa_outs': situation.get('Outs'),
        'sa_balls': situation.get('Balls'),
        'sa_strikes': situation.get('Strikes'),
        'sa_count': f"{situation.get('Balls', 0)}-{situation.get('Strikes', 0)}",
        'sa_batter': player_name(batter),
        'sa_batter_number': batter.get('UniformNumber') if batter else None,
        'sa_pitcher': player_name(pitcher),
        'sa_pitcher_number': pitcher.get('UniformNumber') if pitcher else None,
        'sa_pitcher_pitch_count': situation.get('PitcherPitchCount'),
        'sa_pitcher_hand': situation.get('PitcherHandedness'),
        'sa_batter_hand': situation.get('BatterHandedness'),
        'sa_inning': inning_int,
        'sa_inning_half': inning_half,
        'sa_inning_display': f"{inning_prefix} {ordinal(inning_int)}",
        'sa_on_first': on_first is not None,
        'sa_on_second': on_second is not None,
        'sa_on_third': on_third is not None,
        'sa_runner_first': player_name(on_first),
        'sa_runner_second': player_name(on_second),
        'sa_runner_third': player_name(on_third),
        'sa_on_deck': player_name(on_deck),
        'sa_home_score': home.get('Score'),
        'sa_visitor_score': visitor.get('Score'),
        'sa_home_innings': home.get('PeriodScores', []),
        'sa_visitor_innings': visitor.get('PeriodScores', []),
        'sa_home_name': home.get('Name'),
        'sa_visitor_name': visitor.get('Name'),
        'sa_updated_at': datetime.utcnow().isoformat(),
    }

    # Extract recent plays (last 10, most recent first)
    plays = data.get('Plays', [])
    last_10 = plays[-10:] if plays else []
    last_10.reverse()
    sa_fields['sa_plays'] = [{
        'inning': p.get('Period'),
        'half': 'top' if p.get('Team') == 'VisitingTeam' else 'bottom',
        'text': p.get('Narrative'),
        'type': p.get('Type'),
        'player': player_name(p.get('Player')),
        'context': p.get('Context'),
    } for p in last_10]

    # Extract scoring plays (plays with Score field set)
    scoring_plays = []
    for p in plays:
        score = p.get('Score')
        if not score:
            continue
        inning_num = p.get('Period', 0)
        team_key = p.get('Team', '')
        is_top = team_key == 'VisitingTeam'
        half_label = 'Top' if is_top else 'Bot'
        team_name = visitor.get('Name', 'Away') if is_top else home.get('Name', 'Home')
        scoring_plays.append({
            'inning': f'{half_label} {ordinal(inning_num)}',
            'team': team_name,
            'text': p.get('Narrative', ''),
            'scoring': p.get('Type', ''),
            'score_after': f"{score.get('VisitingTeam', 0)}-{score.get('HomeTeam', 0)}",
        })
    if scoring_plays:
        sa_fields['sa_scoring_plays'] = scoring_plays

    return sa_fields


def is_game_complete(data: dict) -> bool:
    """Check if the SIDEARM game data indicates the game is complete."""
    game = data.get('Game', {})
    return game.get('IsComplete', False)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_sidearm_situation(existing_json: Optional[str], sa_data: Dict[str, Any]) -> str:
    """Merge sa_-prefixed SIDEARM data into existing situation_json.

    Only updates sa_-prefixed keys; sb_ and other keys are preserved.
    """
    existing: Dict[str, Any] = {}
    if existing_json:
        try:
            existing = json.loads(existing_json)
        except (json.JSONDecodeError, TypeError):
            existing = {}

    for key, val in sa_data.items():
        if val is not None:
            existing[key] = val

    return json.dumps(existing)


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

class SidearmPoller:
    """Polls SIDEARM live stats for game updates."""

    def __init__(self, conn: sqlite3.Connection, interval: int = DEFAULT_INTERVAL):
        self.conn = conn
        self.interval = interval
        self._running = True
        self._school_codes: Dict[str, Optional[str]] = {}  # domain → code
        self._prev_scores: Dict[str, Tuple[int, int]] = {}  # game_id → (visitor, home)
        self._notif_dispatcher = None  # Lazy-init shared notification dispatcher

    def stop(self):
        self._running = False

    def _get_notif_dispatcher(self):
        """Lazy-init the shared notification dispatcher."""
        if self._notif_dispatcher is None:
            try:
                from game_notifications import GameNotificationDispatcher
                self._notif_dispatcher = GameNotificationDispatcher(self.conn)
            except ImportError:
                logger.debug("game_notifications module not available")
                self._notif_dispatcher = False  # Sentinel: don't retry
        return self._notif_dispatcher if self._notif_dispatcher else None

    def _get_school_code(self, domain: str, link_id: int) -> Optional[str]:
        """Get school code for a domain, caching in memory and DB."""
        if domain in self._school_codes:
            return self._school_codes[domain]

        # Check DB cache
        row = self.conn.execute(
            "SELECT school_code FROM sidearm_links WHERE domain = ? AND school_code IS NOT NULL LIMIT 1",
            (domain,)
        ).fetchone()
        if row:
            code = row[0] if isinstance(row, tuple) else row['school_code']
            if code:
                self._school_codes[domain] = code
                return code

        # Resolve from API
        code = resolve_school_code(domain)
        self._school_codes[domain] = code

        if code:
            self.conn.execute(
                "UPDATE sidearm_links SET school_code = ? WHERE domain = ?",
                (code, domain)
            )
            self.conn.commit()
            logger.info("Resolved school code for %s: %s", domain, code)

        return code

    def _get_todays_links(self) -> List[dict]:
        """Get today's SIDEARM links from the DB."""
        today = date.today().isoformat()
        rows = self.conn.execute(
            "SELECT id, game_id, domain, url, school_code FROM sidearm_links WHERE game_date = ?",
            (today,)
        ).fetchall()
        return [dict(r) for r in rows]

    def poll_once(self) -> int:
        """Single polling pass. Returns number of games updated."""
        links = self._get_todays_links()
        if not links:
            logger.debug("No SIDEARM links for today")
            return 0

        updated = 0
        # Group by domain to avoid duplicate fetches (same school code)
        seen_codes: Dict[str, dict] = {}  # code → game data

        for link in links:
            try:
                game_id = link['game_id']
                domain = link['domain']
                link_id = link['id']

                code = self._get_school_code(domain, link_id)
                if not code:
                    logger.debug("No school code for domain %s", domain)
                    continue

                # Fetch game data (cache per code per poll cycle)
                if code not in seen_codes:
                    data = fetch_game_data(code)
                    seen_codes[code] = data
                else:
                    data = seen_codes[code]

                if not data:
                    continue

                # Check completion
                if is_game_complete(data):
                    self._handle_complete(game_id, data)
                    continue

                # Extract situation
                sa_fields = extract_situation(data)
                if not sa_fields:
                    continue

                # Merge into DB
                self._update_situation(game_id, sa_fields)

                # Score change detection
                self._check_score_change(game_id, sa_fields)

                # Push notifications (half-inning transitions, recaps, upsets)
                dispatcher = self._get_notif_dispatcher()
                if dispatcher:
                    try:
                        dispatcher.check(game_id, {
                            'inning': sa_fields.get('sa_inning'),
                            'inning_half': sa_fields.get('sa_inning_half'),
                            'home_score': sa_fields.get('sa_home_score'),
                            'visitor_score': sa_fields.get('sa_visitor_score'),
                            'outs': sa_fields.get('sa_outs', 0),
                            'on_first': sa_fields.get('sa_on_first', False),
                            'on_second': sa_fields.get('sa_on_second', False),
                            'on_third': sa_fields.get('sa_on_third', False),
                            'inning_display': sa_fields.get('sa_inning_display'),
                        })
                    except Exception as e:
                        logger.debug("Notification check failed (non-fatal): %s", e)

                # Update game scores
                self._update_scores(game_id, sa_fields)

                logger.info(
                    "Updated game %s from SIDEARM %s: %s %s-%s %s (%s, %s outs)",
                    game_id, code,
                    sa_fields.get('sa_visitor_name', '?'),
                    sa_fields.get('sa_visitor_score', '?'),
                    sa_fields.get('sa_home_score', '?'),
                    sa_fields.get('sa_home_name', '?'),
                    sa_fields.get('sa_inning_display', '?'),
                    sa_fields.get('sa_outs', '?'),
                )
                updated += 1

            except Exception as e:
                logger.error("Error polling SIDEARM link %s: %s", link.get('game_id'), e)

        return updated

    def _update_situation(self, game_id: str, sa_fields: Dict[str, Any]):
        """Merge sa_ fields into games.situation_json and log situation event."""
        row = self.conn.execute(
            "SELECT situation_json FROM games WHERE id = ?", (game_id,)
        ).fetchone()
        existing = None
        if row:
            existing = row[0] if isinstance(row, tuple) else row['situation_json']

        merged = merge_sidearm_situation(existing, sa_fields)
        self.conn.execute(
            "UPDATE games SET situation_json = ? WHERE id = ?",
            (merged, game_id)
        )

        # Insert sa_situation event for win probability timeline
        sit_event = {
            'source': 'sidearm',
            'inning': sa_fields.get('sa_inning'),
            'inning_half': sa_fields.get('sa_inning_half'),
            'outs': sa_fields.get('sa_outs'),
            'home_score': sa_fields.get('sa_home_score'),
            'visitor_score': sa_fields.get('sa_visitor_score'),
            'on_first': sa_fields.get('sa_on_first'),
            'on_second': sa_fields.get('sa_on_second'),
            'on_third': sa_fields.get('sa_on_third'),
        }
        self.conn.execute(
            "INSERT INTO live_events (game_id, event_type, data_json) VALUES (?, ?, ?)",
            (game_id, 'sa_situation', json.dumps(sit_event))
        )
        self.conn.commit()

    def _check_score_change(self, game_id: str, sa_fields: Dict[str, Any]):
        """Detect score changes and insert live_events."""
        home = sa_fields.get('sa_home_score')
        visitor = sa_fields.get('sa_visitor_score')
        if home is None or visitor is None:
            return

        current = (int(visitor), int(home))
        prev = self._prev_scores.get(game_id)
        self._prev_scores[game_id] = current

        if prev is None:
            return  # First sighting, seed state

        if current != prev:
            event_data = {
                'source': 'sidearm',
                'game_id': game_id,
                'prev_visitor_score': prev[0],
                'prev_home_score': prev[1],
                'visitor_score': current[0],
                'home_score': current[1],
                'visitor_name': sa_fields.get('sa_visitor_name'),
                'home_name': sa_fields.get('sa_home_name'),
                'inning_display': sa_fields.get('sa_inning_display'),
            }
            self.conn.execute(
                "INSERT INTO live_events (game_id, event_type, data_json) VALUES (?, ?, ?)",
                (game_id, 'score_change', json.dumps(event_data))
            )
            self.conn.commit()
            logger.info(
                "Score change for %s: %d-%d → %d-%d",
                game_id, prev[0], prev[1], current[0], current[1]
            )

            # Note: push notifications are handled by the dispatcher.check()
            # call in poll_once after _check_score_change.

    def _update_scores(self, game_id: str, sa_fields: Dict[str, Any]):
        """Update games table scores from SIDEARM data."""
        home = sa_fields.get('sa_home_score')
        visitor = sa_fields.get('sa_visitor_score')
        inning_display = sa_fields.get('sa_inning_display', '')

        if home is not None and visitor is not None:
            self.conn.execute("""
                UPDATE games
                SET home_score = ?, away_score = ?,
                    inning_text = COALESCE(?, inning_text),
                    status = 'in-progress',
                    updated_at = ?
                WHERE id = ?
            """, (home, visitor, inning_display,
                  datetime.utcnow().isoformat(), game_id))
            self.conn.commit()

    def _handle_complete(self, game_id: str, data: dict):
        """Handle a completed game."""
        game = data.get('Game', {})
        home = game.get('HomeTeam', {})
        visitor = game.get('VisitingTeam', {})
        home_score = home.get('Score')
        visitor_score = visitor.get('Score')

        # Write final situation
        sa_fields = extract_situation(data)
        if sa_fields:
            self._update_situation(game_id, sa_fields)

        if home_score is not None and visitor_score is not None:
            # Determine innings from PeriodScores
            h_innings = home.get('PeriodScores', [])
            v_innings = visitor.get('PeriodScores', [])
            innings = max(len(h_innings), len(v_innings)) if (h_innings or v_innings) else None
            final_innings = innings if innings and innings > 9 else None

            row = self.conn.execute(
                "SELECT home_team_id, away_team_id FROM games WHERE id = ?",
                (game_id,)
            ).fetchone()
            winner_id = None
            if row:
                home_tid = row[0] if isinstance(row, tuple) else row['home_team_id']
                away_tid = row[1] if isinstance(row, tuple) else row['away_team_id']
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
            """, (home_score, visitor_score,
                  'Final' if not final_innings else f'Final/{final_innings}',
                  final_innings, winner_id,
                  datetime.utcnow().isoformat(), game_id))
            self.conn.commit()

        # Send final score notifications
        dispatcher = self._get_notif_dispatcher()
        if dispatcher:
            try:
                dispatcher.check_final(game_id)
                dispatcher.cleanup_game(game_id)
            except Exception as e:
                logger.debug("Final notification failed (non-fatal): %s", e)

        logger.info("Game %s marked complete from SIDEARM", game_id)

    def run(self):
        """Run the polling loop until stopped."""
        logger.info("SIDEARM poller starting (interval=%ds)", self.interval)

        while self._running:
            try:
                n = self.poll_once()
                if n > 0:
                    logger.info("Updated %d games", n)
            except Exception as e:
                logger.error("Polling loop error: %s", e)

            for _ in range(self.interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("SIDEARM poller stopped")


# ---------------------------------------------------------------------------
# DB migrations
# ---------------------------------------------------------------------------

def ensure_school_code_column(conn: sqlite3.Connection):
    """Add school_code column to sidearm_links if it doesn't exist."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(sidearm_links)").fetchall()]
    if 'school_code' not in cols:
        conn.execute("ALTER TABLE sidearm_links ADD COLUMN school_code TEXT")
        conn.commit()
        logger.info("Added school_code column to sidearm_links")


def ensure_live_events_table(conn: sqlite3.Connection):
    """Create live_events table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            data_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Poll SIDEARM for live game updates")
    parser.add_argument('--once', action='store_true', help='Single poll pass then exit')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                        help='Poll interval in seconds (default: %d)' % DEFAULT_INTERVAL)
    parser.add_argument('--db', default=str(DB_PATH), help='Database path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    ensure_school_code_column(conn)
    ensure_live_events_table(conn)

    poller = SidearmPoller(conn, interval=args.interval)

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
