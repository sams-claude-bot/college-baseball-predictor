"""Tests for scripts/line_tracker.py"""

import sqlite3
from datetime import datetime
from unittest.mock import patch

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.line_tracker import (
    american_to_prob,
    get_line_movement,
    snapshot_all_lines,
)


@pytest.fixture
def mem_db():
    """In-memory SQLite with required tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE teams (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE games (
            id TEXT PRIMARY KEY, date TEXT, time TEXT,
            home_team_id TEXT, away_team_id TEXT, status TEXT DEFAULT 'scheduled'
        );
        CREATE TABLE betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, date TEXT, home_team_id TEXT, away_team_id TEXT,
            book TEXT DEFAULT 'draftkings',
            home_ml INTEGER, away_ml INTEGER,
            home_spread REAL, home_spread_odds INTEGER,
            away_spread REAL, away_spread_odds INTEGER,
            over_under REAL, over_odds INTEGER, under_odds INTEGER,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, home_team_id, away_team_id, book)
        );
        CREATE TABLE betting_line_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, date TEXT, home_team_id TEXT, away_team_id TEXT,
            book TEXT DEFAULT 'draftkings',
            home_ml INTEGER, away_ml INTEGER,
            over_under REAL, over_odds INTEGER, under_odds INTEGER,
            snapshot_type TEXT, captured_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO teams VALUES ('team_a', 'Alabama');
        INSERT INTO teams VALUES ('team_b', 'Auburn');
        INSERT INTO teams VALUES ('team_c', 'LSU');
        INSERT INTO teams VALUES ('team_d', 'Ole Miss');
    """)
    return conn


def _add_game(conn, gid, date, home='team_a', away='team_b', status='scheduled', time='7:00 PM'):
    conn.execute(
        "INSERT INTO games (id, date, time, home_team_id, away_team_id, status) VALUES (?,?,?,?,?,?)",
        (gid, date, time, home, away, status))


def _add_line(conn, gid, date, home='team_a', away='team_b', home_ml=-150, away_ml=130, ou=8.5):
    conn.execute(
        "INSERT OR REPLACE INTO betting_lines (game_id, date, home_team_id, away_team_id, home_ml, away_ml, over_under) VALUES (?,?,?,?,?,?,?)",
        (gid, date, home, away, home_ml, away_ml, ou))


def _add_history(conn, gid, date, home='team_a', away='team_b', home_ml=-150, away_ml=130,
                 ou=8.5, snap_type='periodic', captured_at=None):
    conn.execute(
        "INSERT INTO betting_line_history (game_id, date, home_team_id, away_team_id, home_ml, away_ml, over_under, snapshot_type, captured_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (gid, date, home, away, home_ml, away_ml, ou, snap_type, captured_at or '2026-03-06T10:00:00'))
    conn.commit()


class TestAmericanToProb:
    def test_favorite(self):
        assert american_to_prob(-150) == pytest.approx(0.6, abs=0.001)

    def test_underdog(self):
        assert american_to_prob(150) == pytest.approx(0.4, abs=0.001)

    def test_even(self):
        assert american_to_prob(100) == pytest.approx(0.5, abs=0.001)

    def test_heavy_favorite(self):
        assert american_to_prob(-300) == pytest.approx(0.75, abs=0.001)

    def test_none(self):
        assert american_to_prob(None) is None


class TestSnapshotAllLines:
    def test_captures_new_lines(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_line(mem_db, 'g1', '2026-03-06')
        mem_db.commit()

        with patch('scripts.line_tracker.get_ct_now',
                   return_value=datetime(2026, 3, 6, 12, 0)):
            captured, unchanged, recent = snapshot_all_lines(mem_db)

        assert captured == 1
        snaps = mem_db.execute(
            "SELECT * FROM betting_line_history WHERE game_id='g1' AND snapshot_type='periodic'"
        ).fetchall()
        assert len(snaps) == 1

    def test_skips_unchanged_within_interval(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_line(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', captured_at='2026-03-06T11:58:00')
        mem_db.commit()

        with patch('scripts.line_tracker.get_ct_now',
                   return_value=datetime(2026, 3, 6, 12, 0)):
            captured, unchanged, recent = snapshot_all_lines(mem_db)

        assert captured == 0
        assert recent == 1

    def test_captures_when_line_moved(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_line(mem_db, 'g1', '2026-03-06', home_ml=-180, away_ml=155)
        # Previous snapshot had different line
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, away_ml=130,
                     captured_at='2026-03-06T11:58:00')
        mem_db.commit()

        with patch('scripts.line_tracker.get_ct_now',
                   return_value=datetime(2026, 3, 6, 12, 0)):
            captured, unchanged, recent = snapshot_all_lines(mem_db)

        assert captured == 1

    def test_skips_final_games(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06', status='final')
        _add_line(mem_db, 'g1', '2026-03-06')
        mem_db.commit()

        with patch('scripts.line_tracker.get_ct_now',
                   return_value=datetime(2026, 3, 6, 12, 0)):
            captured, _, _ = snapshot_all_lines(mem_db)

        assert captured == 0

    def test_skips_no_ml_data(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        mem_db.execute(
            "INSERT INTO betting_lines (game_id, date, home_team_id, away_team_id, home_ml, away_ml) VALUES (?,?,?,?,NULL,NULL)",
            ('g1', '2026-03-06', 'team_a', 'team_b'))
        mem_db.commit()

        with patch('scripts.line_tracker.get_ct_now',
                   return_value=datetime(2026, 3, 6, 12, 0)):
            captured, _, _ = snapshot_all_lines(mem_db)

        assert captured == 0

    def test_multiple_games(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_game(mem_db, 'g2', '2026-03-06', home='team_c', away='team_d')
        _add_line(mem_db, 'g1', '2026-03-06')
        _add_line(mem_db, 'g2', '2026-03-06', home='team_c', away='team_d',
                  home_ml=-200, away_ml=170)
        mem_db.commit()

        with patch('scripts.line_tracker.get_ct_now',
                   return_value=datetime(2026, 3, 6, 12, 0)):
            captured, _, _ = snapshot_all_lines(mem_db)

        assert captured == 2


class TestGetLineMovement:
    def test_calculates_movement(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, away_ml=130,
                     snap_type='opening', captured_at='2026-03-06T08:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-180, away_ml=155,
                     snap_type='periodic', captured_at='2026-03-06T15:00:00')
        mem_db.commit()

        moves = get_line_movement(mem_db, '2026-03-06')
        assert len(moves) == 1
        assert moves[0]['home_ml_open'] == -150
        assert moves[0]['home_ml_latest'] == -180
        assert moves[0]['home_prob_move'] is not None
        # -180 is higher implied prob than -150, so positive move toward home
        assert moves[0]['home_prob_move'] > 0

    def test_no_movement(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, away_ml=130,
                     snap_type='opening', captured_at='2026-03-06T08:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, away_ml=130,
                     snap_type='periodic', captured_at='2026-03-06T15:00:00')
        mem_db.commit()

        moves = get_line_movement(mem_db, '2026-03-06')
        assert len(moves) == 1
        assert moves[0]['home_prob_move'] == pytest.approx(0.0)

    def test_ou_movement(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', ou=8.5,
                     snap_type='opening', captured_at='2026-03-06T08:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', ou=9.0,
                     snap_type='periodic', captured_at='2026-03-06T15:00:00')
        mem_db.commit()

        moves = get_line_movement(mem_db, '2026-03-06')
        assert moves[0]['ou_move'] == pytest.approx(0.5)

    def test_empty_date_returns_empty(self, mem_db):
        moves = get_line_movement(mem_db, '2026-03-06')
        assert moves == []

    def test_snapshot_count(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        for i, hour in enumerate([8, 10, 12, 14, 16]):
            _add_history(mem_db, 'g1', '2026-03-06',
                         home_ml=-150 - i * 10, away_ml=130 + i * 10,
                         snap_type='periodic', captured_at=f'2026-03-06T{hour:02d}:00:00')
        mem_db.commit()

        moves = get_line_movement(mem_db, '2026-03-06')
        assert moves[0]['snapshots'] == 5
