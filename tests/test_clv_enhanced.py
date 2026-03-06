#!/usr/bin/env python3
"""Tests for enhanced CLV: get_best_closing_line, retroactive update, correlation, spread."""

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from capture_closing_lines import get_best_closing_line, compute_clv
from clv_enhanced import (
    compute_enhanced_clv,
    retroactive_clv_update,
    clv_vs_result_correlation,
    opening_vs_closing_spread,
)
from datetime import datetime


# ────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────

def _create_schema(db):
    """Create minimal schema for testing."""
    db.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT,
            time TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            status TEXT DEFAULT 'final'
        )
    """)
    db.execute("""
        CREATE TABLE betting_line_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            date TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            book TEXT DEFAULT 'draftkings',
            home_ml INTEGER,
            away_ml INTEGER,
            over_under REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            snapshot_type TEXT NOT NULL,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE tracked_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            pick_team_id TEXT,
            pick_team_name TEXT,
            opponent_name TEXT,
            is_home INTEGER,
            moneyline INTEGER,
            model_prob REAL,
            dk_implied REAL,
            edge REAL,
            bet_amount REAL DEFAULT 100,
            won INTEGER,
            profit REAL,
            closing_ml REAL,
            clv_implied REAL,
            clv_cents REAL
        )
    """)
    db.execute("""
        CREATE TABLE tracked_confident_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            pick_team_id TEXT,
            pick_team_name TEXT,
            opponent_name TEXT,
            is_home INTEGER,
            moneyline INTEGER,
            models_agree INTEGER,
            models_total INTEGER,
            avg_prob REAL,
            confidence REAL,
            bet_amount REAL DEFAULT 100,
            won INTEGER,
            profit REAL,
            closing_ml REAL,
            clv_implied REAL,
            clv_cents REAL
        )
    """)
    db.commit()


@pytest.fixture
def db():
    """In-memory SQLite database with test schema."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    yield conn
    conn.close()


def _seed_game(db, game_id='game1', date='2026-03-01', time='7:00 PM',
               home='team_a', away='team_b'):
    db.execute(
        "INSERT INTO games (id, date, time, home_team_id, away_team_id) VALUES (?,?,?,?,?)",
        (game_id, date, time, home, away)
    )
    db.commit()


def _seed_line(db, game_id='game1', date='2026-03-01', home='team_a', away='team_b',
               home_ml=-150, away_ml=130, snap_type='closing', captured_at='2026-03-01 18:30:00'):
    db.execute("""
        INSERT INTO betting_line_history
            (game_id, date, home_team_id, away_team_id, home_ml, away_ml, snapshot_type, captured_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (game_id, date, home, away, home_ml, away_ml, snap_type, captured_at))
    db.commit()


def _seed_bet(db, table='tracked_bets', game_id='game1', date='2026-03-01',
              pick_team_id='team_a', moneyline=-130, is_home=1, won=1, profit=76.92):
    if table == 'tracked_bets':
        db.execute(f"""
            INSERT INTO {table}
                (game_id, date, pick_team_id, pick_team_name, is_home, moneyline,
                 model_prob, dk_implied, edge, won, profit)
            VALUES (?,?,?,?,?,?,0.6,0.55,0.05,?,?)
        """, (game_id, date, pick_team_id, 'Team A', is_home, moneyline, won, profit))
    else:
        db.execute(f"""
            INSERT INTO {table}
                (game_id, date, pick_team_id, pick_team_name, is_home, moneyline,
                 models_agree, models_total, avg_prob, confidence, won, profit)
            VALUES (?,?,?,?,?,?,3,4,0.6,0.75,?,?)
        """, (game_id, date, pick_team_id, 'Team A', is_home, moneyline, won, profit))
    db.commit()


# ────────────────────────────────────────────────
# get_best_closing_line tests
# ────────────────────────────────────────────────

class TestGetBestClosingLine:

    def test_prefers_recent_periodic_over_older_closing(self, db):
        """A periodic snapshot closer to game time should win over an older closing."""
        _seed_game(db)
        # Closing captured at 18:30 (30 min before 19:00 game)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)
        # Periodic captured at 18:50 (10 min before game, within 15 min window)
        _seed_line(db, snap_type='periodic', captured_at='2026-03-01 18:50:00',
                   home_ml=-160, away_ml=140)

        game_dt = datetime(2026, 3, 1, 19, 0)
        result = get_best_closing_line(db, 'game1', game_dt)

        assert result is not None
        assert result['home_ml'] == -160
        assert result['away_ml'] == 140
        assert result['source'] == 'periodic'

    def test_falls_back_to_closing_if_no_periodic(self, db):
        """If no periodic snapshot exists within window, use closing."""
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)

        game_dt = datetime(2026, 3, 1, 19, 0)
        result = get_best_closing_line(db, 'game1', game_dt)

        assert result is not None
        assert result['home_ml'] == -150
        assert result['source'] == 'closing'

    def test_closing_wins_when_more_recent_than_periodic(self, db):
        """If closing is more recent than periodic, closing wins."""
        _seed_game(db)
        _seed_line(db, snap_type='periodic', captured_at='2026-03-01 18:46:00',
                   home_ml=-140, away_ml=120)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:55:00',
                   home_ml=-155, away_ml=135)

        game_dt = datetime(2026, 3, 1, 19, 0)
        result = get_best_closing_line(db, 'game1', game_dt)

        assert result is not None
        assert result['home_ml'] == -155
        assert result['source'] == 'closing'

    def test_no_line_history_returns_none(self, db):
        """No snapshots at all should return None."""
        _seed_game(db)
        game_dt = datetime(2026, 3, 1, 19, 0)
        result = get_best_closing_line(db, 'game1', game_dt)
        assert result is None

    def test_none_game_datetime_returns_none(self, db):
        """None game_datetime should return None gracefully."""
        _seed_game(db)
        result = get_best_closing_line(db, 'game1', None)
        assert result is None

    def test_periodic_outside_15min_window_ignored(self, db):
        """Periodic snapshots older than 15 min before game time are not considered."""
        _seed_game(db)
        # Periodic at 18:40 — 20 min before game, outside 15 min window
        _seed_line(db, snap_type='periodic', captured_at='2026-03-01 18:40:00',
                   home_ml=-160, away_ml=140)
        # Closing at 18:30
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)

        game_dt = datetime(2026, 3, 1, 19, 0)
        result = get_best_closing_line(db, 'game1', game_dt)

        assert result is not None
        assert result['home_ml'] == -150
        assert result['source'] == 'closing'

    def test_null_mls_returns_none(self, db):
        """Snapshot with NULL home_ml and away_ml returns None."""
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=None, away_ml=None)

        game_dt = datetime(2026, 3, 1, 19, 0)
        result = get_best_closing_line(db, 'game1', game_dt)
        assert result is None


# ────────────────────────────────────────────────
# retroactive_clv_update tests
# ────────────────────────────────────────────────

class TestRetroactiveCLVUpdate:

    def test_updates_graded_bets(self, db):
        """Graded bets should get CLV recalculated."""
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)
        _seed_bet(db, moneyline=-130, won=1, profit=76.92)

        count = retroactive_clv_update(db=db)
        assert count == 1

        row = db.execute("SELECT closing_ml, clv_implied, clv_cents FROM tracked_bets WHERE id = 1").fetchone()
        assert row['closing_ml'] == -150
        assert row['clv_implied'] is not None
        assert row['clv_implied'] > 0  # -130 -> -150 means +CLV

    def test_updates_confident_bets(self, db):
        """tracked_confident_bets should also be updated."""
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)
        _seed_bet(db, table='tracked_confident_bets', moneyline=-130, won=1, profit=76.92)

        count = retroactive_clv_update(db=db)
        assert count == 1

        row = db.execute(
            "SELECT closing_ml, clv_implied FROM tracked_confident_bets WHERE id = 1"
        ).fetchone()
        assert row['closing_ml'] == -150
        assert row['clv_implied'] > 0

    def test_skips_ungraded_bets(self, db):
        """Bets without won = NULL should not be updated."""
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)
        _seed_bet(db, moneyline=-130, won=None, profit=None)

        count = retroactive_clv_update(db=db)
        assert count == 0

    def test_uses_best_closing_line(self, db):
        """Should use periodic if more recent than closing."""
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)
        _seed_line(db, snap_type='periodic', captured_at='2026-03-01 18:50:00',
                   home_ml=-170, away_ml=150)
        _seed_bet(db, moneyline=-130, won=1, profit=76.92)

        retroactive_clv_update(db=db)

        row = db.execute("SELECT closing_ml FROM tracked_bets WHERE id = 1").fetchone()
        assert row['closing_ml'] == -170  # periodic was used

    def test_no_closing_data_returns_zero(self, db):
        """No closing data means nothing to update."""
        _seed_game(db)
        _seed_bet(db, moneyline=-130, won=1, profit=76.92)

        count = retroactive_clv_update(db=db)
        assert count == 0


# ────────────────────────────────────────────────
# clv_vs_result_correlation tests
# ────────────────────────────────────────────────

class TestCLVCorrelation:

    def test_returns_meaningful_data(self, db):
        """With +CLV and -CLV bets, should return grouped stats."""
        _seed_game(db)
        # Bet 1: +CLV winner
        db.execute("""
            INSERT INTO tracked_bets
                (game_id, date, pick_team_id, is_home, moneyline, won, profit,
                 closing_ml, clv_implied, clv_cents, model_prob, dk_implied, edge)
            VALUES ('game1','2026-03-01','team_a',1,-130,1,76.92,-150,0.035,3.5,0.6,0.55,0.05)
        """)
        # Bet 2: -CLV loser
        _seed_game(db, game_id='game2', home='team_c', away='team_d')
        db.execute("""
            INSERT INTO tracked_bets
                (game_id, date, pick_team_id, is_home, moneyline, won, profit,
                 closing_ml, clv_implied, clv_cents, model_prob, dk_implied, edge)
            VALUES ('game2','2026-03-01','team_c',1,-150,0,-100,-130,-0.035,-3.5,0.6,0.55,0.05)
        """)
        db.commit()

        result = clv_vs_result_correlation(db=db)
        assert result['total'] == 2
        assert result['positive_clv']['count'] == 1
        assert result['positive_clv']['win_rate'] == 1.0
        assert result['negative_clv']['count'] == 1
        assert result['negative_clv']['win_rate'] == 0.0

    def test_no_clv_data(self, db):
        """No bets with CLV should return total=0."""
        result = clv_vs_result_correlation(db=db)
        assert result['total'] == 0
        assert result['positive_clv'] is None

    def test_date_filtering(self, db):
        """Date range should filter properly."""
        _seed_game(db)
        db.execute("""
            INSERT INTO tracked_bets
                (game_id, date, pick_team_id, is_home, moneyline, won, profit,
                 closing_ml, clv_implied, clv_cents, model_prob, dk_implied, edge)
            VALUES ('game1','2026-03-01','team_a',1,-130,1,76.92,-150,0.035,3.5,0.6,0.55,0.05)
        """)
        db.commit()

        result = clv_vs_result_correlation(start_date='2026-04-01', db=db)
        assert result['total'] == 0

        result = clv_vs_result_correlation(start_date='2026-02-01', end_date='2026-04-01', db=db)
        assert result['total'] == 1


# ────────────────────────────────────────────────
# opening_vs_closing_spread tests
# ────────────────────────────────────────────────

class TestOpeningVsClosingSpread:

    def test_correct_spread_calculation(self, db):
        """Opening -130 -> closing -150 should show positive home prob movement."""
        _seed_game(db)
        _seed_line(db, snap_type='opening', captured_at='2026-03-01 10:00:00',
                   home_ml=-130, away_ml=110)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)

        result = opening_vs_closing_spread(db=db)
        assert result['game_count'] == 1
        assert result['avg_home_move'] is not None
        # Home went from -130 (56.5%) to -150 (60.0%), move ~ +3.5%
        assert result['avg_home_move'] > 0

    def test_no_opening_lines(self, db):
        """No opening lines should return game_count=0."""
        result = opening_vs_closing_spread(db=db)
        assert result['game_count'] == 0
        assert result['avg_home_move'] is None

    def test_no_closing_for_game(self, db):
        """Opening exists but no closing should skip that game."""
        _seed_game(db)
        _seed_line(db, snap_type='opening', captured_at='2026-03-01 10:00:00',
                   home_ml=-130, away_ml=110)

        result = opening_vs_closing_spread(db=db)
        assert result['game_count'] == 0

    def test_date_filtering(self, db):
        """Date range filter should work."""
        _seed_game(db)
        _seed_line(db, snap_type='opening', captured_at='2026-03-01 10:00:00',
                   home_ml=-130, away_ml=110)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)

        result = opening_vs_closing_spread(start_date='2026-04-01', db=db)
        assert result['game_count'] == 0

        result = opening_vs_closing_spread(start_date='2026-02-01', db=db)
        assert result['game_count'] == 1


# ────────────────────────────────────────────────
# compute_enhanced_clv tests
# ────────────────────────────────────────────────

class TestComputeEnhancedCLV:

    def test_returns_results_for_game_with_bets(self, db):
        _seed_game(db)
        _seed_line(db, snap_type='closing', captured_at='2026-03-01 18:30:00',
                   home_ml=-150, away_ml=130)
        _seed_bet(db, moneyline=-130, won=1, profit=76.92)

        results = compute_enhanced_clv('game1', db=db)
        assert len(results) == 1
        assert results[0]['closing_ml'] == -150
        assert results[0]['clv_implied'] > 0

    def test_no_game_returns_empty(self, db):
        results = compute_enhanced_clv('nonexistent', db=db)
        assert results == []

    def test_no_closing_line_returns_empty(self, db):
        _seed_game(db)
        _seed_bet(db, moneyline=-130, won=1, profit=76.92)
        results = compute_enhanced_clv('game1', db=db)
        assert results == []
