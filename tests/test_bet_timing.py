#!/usr/bin/env python3
"""Tests for Bet Timing & Edge Analysis.

Uses in-memory SQLite to verify:
- edge_timeline returns correct values at each snapshot
- edge_timeline with no history returns empty
- optimal_timing computes edge at open vs close correctly
- line_confirms_model identifies confirmation/non-confirmation
- Game with bet but no history handled gracefully
- Line moving away from pick = not confirmed
"""

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# Schema helpers
# ============================================

def _create_tables(conn):
    """Create minimal tables needed for bet timing analysis."""
    conn.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            status TEXT DEFAULT 'scheduled',
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE model_predictions (
            game_id TEXT,
            model_name TEXT,
            predicted_home_prob REAL,
            prediction_source TEXT,
            predicted_at TEXT,
            was_correct INTEGER,
            UNIQUE(game_id, model_name)
        )
    """)
    conn.execute("""
        CREATE TABLE betting_line_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            book TEXT,
            home_ml REAL,
            away_ml REAL,
            over_under REAL,
            snapshot_type TEXT,
            captured_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE tracked_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            pick_team_id TEXT,
            is_home INTEGER,
            moneyline INTEGER,
            model_prob REAL,
            edge REAL,
            won INTEGER,
            profit REAL,
            closing_ml REAL,
            clv_implied REAL,
            clv_cents REAL
        )
    """)
    conn.commit()


def _make_conn():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def conn():
    """In-memory DB with empty tables."""
    c = _make_conn()
    yield c
    c.close()


@pytest.fixture
def conn_with_timeline():
    """DB with a game, meta_ensemble prediction, and periodic snapshots."""
    c = _make_conn()

    c.execute(
        "INSERT INTO games (id, date, time, status, home_team_id, away_team_id) "
        "VALUES ('g1', '2026-03-01', '7:00 PM', 'final', 'team_h', 'team_a')"
    )
    # meta_ensemble predicts 60% home
    c.execute(
        "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at) "
        "VALUES ('g1', 'meta_ensemble', 0.60, '2026-03-01T09:00:00')"
    )
    # Three periodic snapshots with moving lines
    # -150 => implied = 150/250 = 0.6000
    # -130 => implied = 130/230 = 0.5652
    # -170 => implied = 170/270 = 0.6296
    c.executemany(
        "INSERT INTO betting_line_history "
        "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
        "VALUES (?, ?, ?, ?, 'periodic', ?)",
        [
            ('g1', '2026-03-01', -150, 130, '2026-03-01T08:00:00'),
            ('g1', '2026-03-01', -130, 110, '2026-03-01T10:00:00'),
            ('g1', '2026-03-01', -170, 150, '2026-03-01T12:00:00'),
        ]
    )
    c.commit()
    yield c
    c.close()


# ============================================
# edge_timeline tests
# ============================================

class TestEdgeTimeline:

    def test_correct_values_at_each_snapshot(self, conn_with_timeline):
        """Edge timeline returns correct edge at each periodic snapshot."""
        from scripts.bet_timing_analysis import edge_timeline

        tl = edge_timeline('g1', conn=conn_with_timeline)
        assert len(tl) == 3

        # Snapshot 1: ML=-150, implied=0.6000, model=0.60, edge=0.00pp
        assert tl[0]['ml'] == -150
        assert abs(tl[0]['market_implied'] - 0.6000) < 0.001
        assert abs(tl[0]['edge_pp'] - 0.0) < 0.1

        # Snapshot 2: ML=-130, implied=0.5652, model=0.60, edge=+3.48pp
        assert tl[1]['ml'] == -130
        assert abs(tl[1]['market_implied'] - 0.5652) < 0.001
        assert abs(tl[1]['edge_pp'] - 3.48) < 0.1

        # Snapshot 3: ML=-170, implied=0.6296, model=0.60, edge=-2.96pp
        assert tl[2]['ml'] == -170
        assert abs(tl[2]['market_implied'] - 0.6296) < 0.001
        assert abs(tl[2]['edge_pp'] - (-2.96)) < 0.1

    def test_no_history_returns_empty(self, conn):
        """Game with no line history returns empty list."""
        from scripts.bet_timing_analysis import edge_timeline

        # Add game and prediction but no snapshots
        conn.execute(
            "INSERT INTO games (id, date, time, status, home_team_id, away_team_id) "
            "VALUES ('g_none', '2026-03-01', '7:00 PM', 'final', 'h', 'a')"
        )
        conn.execute(
            "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at) "
            "VALUES ('g_none', 'meta_ensemble', 0.55, '2026-03-01T09:00:00')"
        )
        conn.commit()

        tl = edge_timeline('g_none', conn=conn)
        assert tl == []

    def test_no_prediction_returns_empty(self, conn):
        """Game with no meta_ensemble prediction returns empty list."""
        from scripts.bet_timing_analysis import edge_timeline

        conn.execute(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES ('g_nopred', '2026-03-01', -130, 110, 'periodic', '2026-03-01T10:00:00')"
        )
        conn.commit()

        tl = edge_timeline('g_nopred', conn=conn)
        assert tl == []

    def test_model_prob_consistent(self, conn_with_timeline):
        """Model prob is the same across all snapshots (single prediction)."""
        from scripts.bet_timing_analysis import edge_timeline

        tl = edge_timeline('g1', conn=conn_with_timeline)
        probs = {pt['model_prob'] for pt in tl}
        assert len(probs) == 1
        assert abs(probs.pop() - 0.60) < 0.001


# ============================================
# optimal_timing_report tests
# ============================================

class TestOptimalTimingReport:

    def test_edge_at_open_vs_close(self, conn):
        """Computes edge at open and close correctly."""
        from scripts.bet_timing_analysis import optimal_timing_report

        conn.execute(
            "INSERT INTO games (id, date, time, status, home_team_id, away_team_id) "
            "VALUES ('g2', '2026-03-01', '7:00 PM', 'final', 'h', 'a')"
        )
        # Bet on home side, model_prob=0.65
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('g2', '2026-03-01', 'h', 1, -130, 0.65, 8.0, 1, 76.92)"
        )
        # Opening line: -130 => implied=0.5652, edge = (0.65-0.5652)*100 = 8.48
        # Closing line: -180 => implied=0.6429, edge = (0.65-0.6429)*100 = 0.71
        conn.execute(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES ('g2', '2026-03-01', -130, 110, 'periodic', '2026-03-01T08:00:00')"
        )
        conn.execute(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES ('g2', '2026-03-01', -180, 160, 'closing', '2026-03-01T18:45:00')"
        )
        conn.commit()

        report = optimal_timing_report('2026-03-01', '2026-03-01', conn=conn)
        assert report['count'] == 1
        assert abs(report['avg_edge_at_open'] - 8.48) < 0.1
        assert abs(report['avg_edge_at_close'] - 0.71) < 0.1

    def test_no_bets_returns_zero_count(self, conn):
        """No tracked bets returns count=0."""
        from scripts.bet_timing_analysis import optimal_timing_report

        report = optimal_timing_report('2026-01-01', '2026-01-31', conn=conn)
        assert report['count'] == 0
        assert report['avg_edge_at_open'] is None

    def test_bet_no_history_skipped(self, conn):
        """Bet with no line history is gracefully skipped."""
        from scripts.bet_timing_analysis import optimal_timing_report

        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('g_orphan', '2026-03-01', 'h', 1, -130, 0.65, 8.0, 1, 76.92)"
        )
        conn.commit()

        report = optimal_timing_report('2026-03-01', '2026-03-01', conn=conn)
        assert report['count'] == 0

    def test_won_lost_edge_breakdown(self, conn):
        """Won vs lost games have separate edge-at-close averages."""
        from scripts.bet_timing_analysis import optimal_timing_report

        # Game won: open=-130(0.5652), close=-160(0.6154), model=0.65
        # edge_close = (0.65-0.6154)*100 = 3.46
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('gw', '2026-03-01', 'h', 1, -130, 0.65, 8.0, 1, 76.92)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('gw', '2026-03-01', -130, 110, 'periodic', '2026-03-01T08:00:00'),
                ('gw', '2026-03-01', -160, 140, 'closing', '2026-03-01T18:45:00'),
            ]
        )

        # Game lost: open=-110(0.5238), close=-105(0.5122), model=0.55
        # edge_close = (0.55-0.5122)*100 = 3.78
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('gl', '2026-03-01', 'h', 1, -110, 0.55, 2.6, 0, -100)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('gl', '2026-03-01', -110, 100, 'periodic', '2026-03-01T08:00:00'),
                ('gl', '2026-03-01', -105, 95, 'closing', '2026-03-01T18:45:00'),
            ]
        )
        conn.commit()

        report = optimal_timing_report('2026-03-01', '2026-03-01', conn=conn)
        assert report['count'] == 2
        assert report['won_avg_edge_at_close'] is not None
        assert report['lost_avg_edge_at_close'] is not None


# ============================================
# line_confirms_model tests
# ============================================

class TestLineConfirmsModel:

    def test_confirms_when_line_moves_toward_pick(self, conn):
        """Line moving toward pick (higher implied for picked side) = confirmed."""
        from scripts.bet_timing_analysis import line_confirms_model

        # Bet on home. Open -120 (impl=0.5455), Close -150 (impl=0.6000)
        # Line moved toward home => confirmed
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('gc', '2026-03-01', 'h', 1, -120, 0.62, 7.5, 1, 83.33)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('gc', '2026-03-01', -120, 100, 'periodic', '2026-03-01T08:00:00'),
                ('gc', '2026-03-01', -150, 130, 'closing', '2026-03-01T18:45:00'),
            ]
        )
        conn.commit()

        result = line_confirms_model('2026-03-01', '2026-03-01', conn=conn)
        assert result['count'] == 1
        assert result['pct_confirmed'] == 100.0
        assert result['avg_confirmation_size'] > 0

    def test_not_confirmed_when_line_moves_away(self, conn):
        """Line moving away from pick = not confirmed."""
        from scripts.bet_timing_analysis import line_confirms_model

        # Bet on home. Open -150 (impl=0.6000), Close -120 (impl=0.5455)
        # Line moved away from home => not confirmed
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('gnc', '2026-03-01', 'h', 1, -150, 0.65, 5.0, 0, -100)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('gnc', '2026-03-01', -150, 130, 'periodic', '2026-03-01T08:00:00'),
                ('gnc', '2026-03-01', -120, 100, 'closing', '2026-03-01T18:45:00'),
            ]
        )
        conn.commit()

        result = line_confirms_model('2026-03-01', '2026-03-01', conn=conn)
        assert result['count'] == 1
        assert result['pct_confirmed'] == 0.0

    def test_mixed_confirmation(self, conn):
        """Mix of confirmed and not-confirmed gives correct percentages."""
        from scripts.bet_timing_analysis import line_confirms_model

        # Confirmed: home open=-120, close=-150 (moved toward)
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('gm1', '2026-03-01', 'h', 1, -120, 0.62, 7.5, 1, 83.33)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('gm1', '2026-03-01', -120, 100, 'periodic', '2026-03-01T08:00:00'),
                ('gm1', '2026-03-01', -150, 130, 'closing', '2026-03-01T18:45:00'),
            ]
        )

        # Not confirmed: home open=-150, close=-120 (moved away)
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('gm2', '2026-03-01', 'h', 1, -150, 0.65, 5.0, 0, -100)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('gm2', '2026-03-01', -150, 130, 'periodic', '2026-03-01T08:00:00'),
                ('gm2', '2026-03-01', -120, 100, 'closing', '2026-03-01T18:45:00'),
            ]
        )
        conn.commit()

        result = line_confirms_model('2026-03-01', '2026-03-01', conn=conn)
        assert result['count'] == 2
        assert result['pct_confirmed'] == 50.0
        assert result['win_rate_confirmed'] == 100.0
        assert result['win_rate_not_confirmed'] == 0.0

    def test_no_history_skipped(self, conn):
        """Bet with no line history is gracefully skipped (needs >=2 snapshots)."""
        from scripts.bet_timing_analysis import line_confirms_model

        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('g_no_hist', '2026-03-01', 'h', 1, -130, 0.60, 3.5, 1, 76.92)"
        )
        conn.commit()

        result = line_confirms_model('2026-03-01', '2026-03-01', conn=conn)
        assert result['count'] == 0
        assert result['pct_confirmed'] is None

    def test_away_pick_confirmation(self, conn):
        """Confirmation works for away-side picks too."""
        from scripts.bet_timing_analysis import line_confirms_model

        # Bet on away. Open away_ml=+130 (impl=0.4348), Close away_ml=+110 (impl=0.4762)
        # Away implied went up => confirmed
        conn.execute(
            "INSERT INTO tracked_bets "
            "(game_id, date, pick_team_id, is_home, moneyline, model_prob, edge, won, profit) "
            "VALUES ('ga', '2026-03-01', 'a', 0, 130, 0.50, 6.5, 1, 130)"
        )
        conn.executemany(
            "INSERT INTO betting_line_history "
            "(game_id, date, home_ml, away_ml, snapshot_type, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ('ga', '2026-03-01', -150, 130, 'periodic', '2026-03-01T08:00:00'),
                ('ga', '2026-03-01', -130, 110, 'closing', '2026-03-01T18:45:00'),
            ]
        )
        conn.commit()

        result = line_confirms_model('2026-03-01', '2026-03-01', conn=conn)
        assert result['count'] == 1
        assert result['pct_confirmed'] == 100.0
