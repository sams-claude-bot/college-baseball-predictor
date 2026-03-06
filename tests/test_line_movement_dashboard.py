"""Tests for line movement dashboard: chart data, top movers, steam detection."""

import sqlite3
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.line_tracker import (
    american_to_prob,
    get_game_line_history,
    get_line_movement,
    detect_steam_moves,
)


@pytest.fixture
def mem_db():
    """In-memory SQLite with required tables and test data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE teams (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE games (
            id TEXT PRIMARY KEY, date TEXT, time TEXT,
            home_team_id TEXT, away_team_id TEXT, status TEXT DEFAULT 'scheduled'
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


def _add_history(conn, gid, date, home='team_a', away='team_b', home_ml=-150, away_ml=130,
                 ou=8.5, snap_type='periodic', captured_at=None):
    conn.execute(
        "INSERT INTO betting_line_history (game_id, date, home_team_id, away_team_id, home_ml, away_ml, over_under, snapshot_type, captured_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (gid, date, home, away, home_ml, away_ml, ou, snap_type, captured_at or '2026-03-06T10:00:00'))
    conn.commit()


# ── Chart data (get_game_line_history) ──

class TestGameLineHistory:
    def test_returns_chronological_snapshots(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-160, captured_at='2026-03-06T10:15:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-170, captured_at='2026-03-06T10:30:00')

        snaps = get_game_line_history(mem_db, 'g1')
        assert len(snaps) == 3
        # Chronological order
        assert snaps[0]['captured_at'] == '2026-03-06T10:00:00'
        assert snaps[1]['captured_at'] == '2026-03-06T10:15:00'
        assert snaps[2]['captured_at'] == '2026-03-06T10:30:00'

    def test_home_prob_calculated(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')

        snaps = get_game_line_history(mem_db, 'g1')
        assert snaps[0]['home_prob'] == pytest.approx(60.0, abs=0.1)

    def test_includes_snapshot_type(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', snap_type='opening', captured_at='2026-03-06T09:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', snap_type='periodic', captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', snap_type='closing', captured_at='2026-03-06T18:00:00')

        snaps = get_game_line_history(mem_db, 'g1')
        assert snaps[0]['snapshot_type'] == 'opening'
        assert snaps[1]['snapshot_type'] == 'periodic'
        assert snaps[2]['snapshot_type'] == 'closing'

    def test_includes_over_under(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', ou=8.5, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', ou=9.0, captured_at='2026-03-06T10:30:00')

        snaps = get_game_line_history(mem_db, 'g1')
        assert snaps[0]['over_under'] == 8.5
        assert snaps[1]['over_under'] == 9.0

    def test_empty_history(self, mem_db):
        snaps = get_game_line_history(mem_db, 'nonexistent')
        assert snaps == []

    def test_null_ml_handled(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        mem_db.execute(
            "INSERT INTO betting_line_history (game_id, date, home_team_id, away_team_id, home_ml, away_ml, over_under, snapshot_type, captured_at) VALUES (?,?,?,?,?,?,?,?,?)",
            ('g1', '2026-03-06', 'team_a', 'team_b', None, None, 8.5, 'periodic', '2026-03-06T10:00:00'))
        mem_db.commit()

        snaps = get_game_line_history(mem_db, 'g1')
        assert len(snaps) == 1
        assert snaps[0]['home_prob'] is None


# ── Top movers (get_line_movement) ──

class TestTopMovers:
    def test_sorted_by_absolute_movement(self, mem_db):
        # Game 1: small movement (-150 -> -160 = ~1.5pp)
        _add_game(mem_db, 'g1', '2026-03-06', home='team_a', away='team_b')
        _add_history(mem_db, 'g1', '2026-03-06', home='team_a', away='team_b',
                     home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home='team_a', away='team_b',
                     home_ml=-160, captured_at='2026-03-06T12:00:00')

        # Game 2: large movement (-150 -> -250 = ~10pp)
        _add_game(mem_db, 'g2', '2026-03-06', home='team_c', away='team_d')
        _add_history(mem_db, 'g2', '2026-03-06', home='team_c', away='team_d',
                     home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g2', '2026-03-06', home='team_c', away='team_d',
                     home_ml=-250, captured_at='2026-03-06T12:00:00')

        movements = get_line_movement(mem_db, '2026-03-06')
        # Sort by absolute prob movement (mimicking how betting route does it)
        with_moves = [m for m in movements if m['home_prob_move'] is not None]
        with_moves.sort(key=lambda m: abs(m['home_prob_move']), reverse=True)

        assert len(with_moves) == 2
        # Bigger mover first
        assert with_moves[0]['game_id'] == 'g2'
        assert abs(with_moves[0]['home_prob_move']) > abs(with_moves[1]['home_prob_move'])

    def test_no_movement_returns_zero(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T12:00:00')

        movements = get_line_movement(mem_db, '2026-03-06')
        assert len(movements) == 1
        assert movements[0]['home_prob_move'] == 0.0

    def test_empty_date(self, mem_db):
        movements = get_line_movement(mem_db, '2099-01-01')
        assert movements == []

    def test_direction_home(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=130, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-130, captured_at='2026-03-06T12:00:00')

        movements = get_line_movement(mem_db, '2026-03-06')
        assert movements[0]['home_prob_move'] > 0  # moved toward home


# ── Steam move detection ──

class TestSteamDetection:
    def test_fires_on_large_quick_move(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        # 5pp move in 10 minutes -> steam
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-250, captured_at='2026-03-06T10:10:00')

        steam = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=30)
        assert len(steam) == 1
        assert steam[0]['game_id'] == 'g1'
        assert steam[0]['direction'] == 'home'
        assert abs(steam[0]['move_pp']) >= 3.0

    def test_ignores_gradual_moves(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        # 5pp total but spread over 3+ hours (each step < 3pp)
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-160, captured_at='2026-03-06T11:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-170, captured_at='2026-03-06T12:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-180, captured_at='2026-03-06T13:00:00')

        steam = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=30)
        assert len(steam) == 0

    def test_away_direction(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        # Move toward away: home ML goes from -200 to +120
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-200, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=120, captured_at='2026-03-06T10:15:00')

        steam = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=30)
        assert len(steam) == 1
        assert steam[0]['direction'] == 'away'
        assert steam[0]['move_pp'] < 0

    def test_empty_history(self, mem_db):
        steam = detect_steam_moves(mem_db, '2099-01-01', threshold_pp=3.0, window_minutes=30)
        assert steam == []

    def test_single_snapshot_no_steam(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')

        steam = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=30)
        assert steam == []

    def test_custom_threshold(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        # ~1.5pp move in 5 min: steam with 1pp threshold, not with 3pp
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-160, captured_at='2026-03-06T10:05:00')

        steam_low = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=1.0, window_minutes=30)
        steam_high = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=30)

        assert len(steam_low) == 1
        assert len(steam_high) == 0

    def test_window_minutes_respected(self, mem_db):
        _add_game(mem_db, 'g1', '2026-03-06')
        # Large move but over 45 min (outside 30-min window)
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-150, captured_at='2026-03-06T10:00:00')
        _add_history(mem_db, 'g1', '2026-03-06', home_ml=-300, captured_at='2026-03-06T10:45:00')

        steam_30 = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=30)
        steam_60 = detect_steam_moves(mem_db, '2026-03-06', threshold_pp=3.0, window_minutes=60)

        assert len(steam_30) == 0
        assert len(steam_60) == 1
