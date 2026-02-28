#!/usr/bin/env python3
"""
Tests for StatBroadcast event discovery — table creation, game matching,
group ID loading, and event scanning.
"""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from statbroadcast_discovery import (
    ensure_table,
    match_game,
    load_group_ids,
    invert_group_ids,
    discover_events_by_scan,
    _upsert_sb_event,
    get_active_events,
    mark_completed,
    discover_events_for_games,
    CREATE_TABLE_SQL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory DB with games table and statbroadcast_events."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            status TEXT DEFAULT 'scheduled',
            home_score INTEGER,
            away_score INTEGER,
            situation_json TEXT,
            linescore_json TEXT,
            inning_text TEXT,
            time TEXT,
            winner_id TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE team_aliases (
            alias TEXT PRIMARY KEY,
            team_id TEXT,
            source TEXT DEFAULT 'manual'
        )
    """)
    conn.commit()
    return conn


def _seed_games(conn):
    """Insert sample games for matching tests."""
    conn.execute(
        "INSERT INTO games (id, date, home_team_id, away_team_id) VALUES (?, ?, ?, ?)",
        ('2026-02-26_byu_washington-state', '2026-02-26', 'washington-state', 'byu')
    )
    conn.execute(
        "INSERT INTO games (id, date, home_team_id, away_team_id) VALUES (?, ?, ?, ?)",
        ('2026-02-26_texas_lsu', '2026-02-26', 'lsu', 'texas')
    )
    conn.execute(
        "INSERT INTO games (id, date, home_team_id, away_team_id) VALUES (?, ?, ?, ?)",
        ('2026-02-27_florida_georgia', '2026-02-27', 'georgia', 'florida')
    )
    conn.commit()


def _seed_aliases(conn):
    """Insert team aliases for resolver tests."""
    aliases = [
        ('washington state', 'washington-state'),
        ('wsu', 'washington-state'),
        ('byu', 'byu'),
        ('brigham young', 'byu'),
        ('lsu', 'lsu'),
        ('texas', 'texas'),
        ('florida', 'florida'),
        ('georgia', 'georgia'),
    ]
    for alias, team_id in aliases:
        conn.execute(
            "INSERT INTO team_aliases (alias, team_id) VALUES (?, ?)",
            (alias, team_id)
        )
    conn.commit()


def _make_resolver(conn):
    """Create a TeamResolver backed by in-memory DB."""
    from team_resolver import TeamResolver
    resolver = TeamResolver.__new__(TeamResolver)
    resolver.db_path = ':memory:'
    resolver._cache = {}
    c = conn.cursor()
    c.execute("SELECT alias, team_id FROM team_aliases")
    for alias, team_id in c.fetchall():
        resolver._cache[alias.lower()] = team_id
    return resolver


# ---------------------------------------------------------------------------
# Table creation tests
# ---------------------------------------------------------------------------

class TestTableCreation:
    def test_ensure_table_creates_table(self):
        """ensure_table creates statbroadcast_events table."""
        conn = sqlite3.connect(':memory:')
        ensure_table(conn)
        # Verify table exists
        c = conn.cursor()
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='statbroadcast_events'"
        )
        assert c.fetchone() is not None

    def test_ensure_table_idempotent(self):
        """ensure_table can be called multiple times without error."""
        conn = sqlite3.connect(':memory:')
        ensure_table(conn)
        ensure_table(conn)  # Should not raise
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM statbroadcast_events")
        assert c.fetchone()[0] == 0

    def test_table_has_expected_columns(self):
        """statbroadcast_events has all required columns."""
        conn = sqlite3.connect(':memory:')
        ensure_table(conn)
        c = conn.cursor()
        c.execute("PRAGMA table_info(statbroadcast_events)")
        cols = {row[1] for row in c.fetchall()}
        expected = {
            'sb_event_id', 'game_id', 'home_team', 'visitor_team',
            'home_team_id', 'visitor_team_id', 'game_date', 'group_id',
            'xml_file', 'completed', 'discovered_at',
        }
        assert expected.issubset(cols)


# ---------------------------------------------------------------------------
# Game matching tests
# ---------------------------------------------------------------------------

class TestMatchGame:
    def test_match_exact(self):
        """Matches game by date + home/visitor team names."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        resolver = _make_resolver(conn)

        sb_info = {
            'event_id': 652739,
            'home': 'Washington State',
            'visitor': 'BYU',
            'date': '2026-02-26',
            'sport': 'bsgame',
            'completed': False,
        }
        game_id = match_game(sb_info, conn, resolver)
        assert game_id == '2026-02-26_byu_washington-state'

    def test_match_reversed_teams(self):
        """Matches even if SB has different home/away from our DB."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        resolver = _make_resolver(conn)

        # SB says BYU is home, but our DB has WSU as home
        sb_info = {
            'event_id': 652740,
            'home': 'BYU',
            'visitor': 'Washington State',
            'date': '2026-02-26',
            'sport': 'bsgame',
        }
        game_id = match_game(sb_info, conn, resolver)
        assert game_id == '2026-02-26_byu_washington-state'

    def test_match_wrong_date(self):
        """No match when date differs."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        resolver = _make_resolver(conn)

        sb_info = {
            'event_id': 652741,
            'home': 'Washington State',
            'visitor': 'BYU',
            'date': '2026-03-01',  # Wrong date
            'sport': 'bsgame',
        }
        assert match_game(sb_info, conn, resolver) is None

    def test_match_unknown_team_fallback(self):
        """Falls back to single-team match when one team can't be resolved."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        resolver = _make_resolver(conn)

        # BYU resolves but "Unknown University" doesn't — fallback finds
        # BYU's only game on 2026-02-26
        sb_info = {
            'event_id': 652742,
            'home': 'Unknown University',
            'visitor': 'BYU',
            'date': '2026-02-26',
            'sport': 'bsgame',
        }
        assert match_game(sb_info, conn, resolver) == '2026-02-26_byu_washington-state'

    def test_match_neither_team_resolves(self):
        """Returns None when neither team can be resolved."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        resolver = _make_resolver(conn)

        sb_info = {
            'event_id': 652742,
            'home': 'Unknown University',
            'visitor': 'Mystery College',
            'date': '2026-02-26',
            'sport': 'bsgame',
        }
        assert match_game(sb_info, conn, resolver) is None

    def test_match_non_baseball(self):
        """Returns None for non-baseball sport."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        resolver = _make_resolver(conn)

        sb_info = {
            'event_id': 652743,
            'home': 'Washington State',
            'visitor': 'BYU',
            'date': '2026-02-26',
            'sport': 'sbgame',  # softball, not baseball
        }
        assert match_game(sb_info, conn, resolver) is None

    def test_match_empty_info(self):
        """Returns None for empty/None input."""
        conn = _make_db()
        assert match_game(None, conn) is None
        assert match_game({}, conn) is None


# ---------------------------------------------------------------------------
# Group ID tests
# ---------------------------------------------------------------------------

class TestGroupIds:
    def test_load_group_ids(self):
        """Loads group IDs from the actual JSON file."""
        gids = load_group_ids(PROJECT_ROOT / 'scripts' / 'sb_group_ids.json')
        assert isinstance(gids, dict)
        assert len(gids) >= 30  # At least 30 schools
        assert gids.get('washington-state') == 'wast'
        assert gids.get('byu') == 'byu'

    def test_load_group_ids_missing_file(self):
        """Returns empty dict for missing file."""
        gids = load_group_ids('/tmp/nonexistent_file.json')
        assert gids == {}

    def test_invert_group_ids(self):
        """Inverts team_id -> group_id to group_id -> [team_id, ...]."""
        gids = {'washington-state': 'wsu', 'byu': 'byu', 'weber-state': 'wsu'}
        inverted = invert_group_ids(gids)
        assert 'wsu' in inverted
        assert 'washington-state' in inverted['wsu']
        assert 'weber-state' in inverted['wsu']
        assert inverted['byu'] == ['byu']


# ---------------------------------------------------------------------------
# DB helper tests
# ---------------------------------------------------------------------------

class TestDbHelpers:
    def test_upsert_sb_event(self):
        """Inserts a new SB event record."""
        conn = sqlite3.connect(':memory:')
        ensure_table(conn)

        record = {
            'sb_event_id': 652739,
            'game_id': '2026-02-26_byu_washington-state',
            'home_team': 'Washington State',
            'visitor_team': 'BYU',
            'home_team_id': 'washington-state',
            'visitor_team_id': 'byu',
            'game_date': '2026-02-26',
            'group_id': 'wsu',
            'xml_file': 'wsu/652739.xml',
            'completed': 0,
        }
        _upsert_sb_event(conn, record)

        c = conn.cursor()
        row = c.execute(
            "SELECT * FROM statbroadcast_events WHERE sb_event_id = 652739"
        ).fetchone()
        assert row is not None

    def test_upsert_sb_event_update(self):
        """Upsert overwrites existing record."""
        conn = sqlite3.connect(':memory:')
        ensure_table(conn)

        record = {
            'sb_event_id': 652739,
            'game_id': None,
            'home_team': 'WSU',
            'visitor_team': 'BYU',
            'game_date': '2026-02-26',
            'group_id': 'wsu',
            'xml_file': 'wsu/652739.xml',
            'completed': 0,
        }
        _upsert_sb_event(conn, record)

        # Update with game_id
        record['game_id'] = '2026-02-26_byu_washington-state'
        _upsert_sb_event(conn, record)

        c = conn.cursor()
        row = c.execute(
            "SELECT game_id FROM statbroadcast_events WHERE sb_event_id = 652739"
        ).fetchone()
        assert row[0] == '2026-02-26_byu_washington-state'

    def test_get_active_events(self):
        """Returns non-completed events with a game_id."""
        import pytz
        from datetime import datetime

        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        ensure_table(conn)

        # get_active_events JOINs on games, so we need that table too
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY,
                date TEXT,
                time TEXT,
                status TEXT DEFAULT 'scheduled',
                home_team_id TEXT,
                away_team_id TEXT
            )
        """)

        # Use today's date so the time-gate logic includes these games
        ct = pytz.timezone('America/Chicago')
        today = datetime.now(ct).strftime('%Y-%m-%d')

        conn.execute(
            "INSERT INTO games (id, date, time, status) VALUES (?, ?, ?, ?)",
            ('game-a', today, '12:00 PM', 'in-progress')
        )
        conn.execute(
            "INSERT INTO games (id, date, time, status) VALUES (?, ?, ?, ?)",
            ('game-b', today, '12:00 PM', 'final')
        )
        conn.commit()

        # Insert one active, one completed, one unmatched
        for eid, gid, completed in [(100, 'game-a', 0), (101, 'game-b', 1), (102, None, 0)]:
            conn.execute(
                "INSERT INTO statbroadcast_events (sb_event_id, game_id, completed, xml_file, game_date) VALUES (?, ?, ?, ?, ?)",
                (eid, gid, completed, 'test/%d.xml' % eid, today)
            )
        conn.commit()

        active = get_active_events(conn)
        # Should include game-a (in-progress, not completed)
        active_ids = [e['sb_event_id'] for e in active]
        assert 100 in active_ids
        # game-b is completed=1, should NOT appear
        assert 101 not in active_ids

    def test_mark_completed(self):
        """Marks an event as completed."""
        conn = sqlite3.connect(':memory:')
        ensure_table(conn)
        conn.execute(
            "INSERT INTO statbroadcast_events (sb_event_id, game_id, completed, xml_file) VALUES (?, ?, ?, ?)",
            (200, 'game-x', 0, 'test/200.xml')
        )
        conn.commit()

        mark_completed(conn, 200)

        c = conn.cursor()
        row = c.execute(
            "SELECT completed FROM statbroadcast_events WHERE sb_event_id = 200"
        ).fetchone()
        assert row[0] == 1

    def test_discover_events_by_scan_mocked(self):
        """Scan range with mocked client finds and inserts events."""
        conn = _make_db()
        _seed_games(conn)
        _seed_aliases(conn)
        ensure_table(conn)
        resolver = _make_resolver(conn)

        mock_client = MagicMock()

        # Event 100 matches our game, event 101 is non-baseball
        def fake_event_info(eid):
            if eid == 100:
                return {
                    'event_id': 100,
                    'home': 'Washington State',
                    'visitor': 'BYU',
                    'date': '2026-02-26',
                    'sport': 'bsgame',
                    'completed': False,
                    'group_id': 'wsu',
                    'xml_file': 'wsu/100.xml',
                }
            if eid == 101:
                return {
                    'event_id': 101,
                    'home': 'WSU',
                    'visitor': 'Oregon',
                    'date': '2026-02-26',
                    'sport': 'sbgame',  # softball
                    'completed': False,
                    'group_id': 'wsu',
                    'xml_file': 'wsu/101.xml',
                }
            return None

        mock_client.get_event_info.side_effect = fake_event_info

        results = discover_events_by_scan(
            conn, 99, 103, target_date='2026-02-26',
            client=mock_client, resolver=resolver
        )

        # Only the baseball game should be discovered
        assert len(results) == 1
        assert results[0]['sb_event_id'] == 100
        assert results[0]['game_id'] == '2026-02-26_byu_washington-state'

        # Verify it was persisted
        c = conn.cursor()
        row = c.execute(
            "SELECT game_id FROM statbroadcast_events WHERE sb_event_id = 100"
        ).fetchone()
        assert row[0] == '2026-02-26_byu_washington-state'
