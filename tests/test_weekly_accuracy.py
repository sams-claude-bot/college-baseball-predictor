"""Tests for weekly (7-day) accuracy queries on the models page."""
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest


def _make_db():
    """Create in-memory DB with model_predictions, totals_predictions, games."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT,
            status TEXT DEFAULT 'final',
            home_team_id TEXT,
            away_team_id TEXT,
            home_score INTEGER,
            away_score INTEGER
        );
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            model_name TEXT,
            predicted_home_prob REAL,
            was_correct INTEGER,
            predicted_at TEXT
        );
        CREATE TABLE totals_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            model_name TEXT,
            over_under_line REAL,
            projected_total REAL,
            prediction TEXT,
            was_correct INTEGER,
            predicted_at TEXT,
            actual_total INTEGER,
            edge_pct REAL,
            confidence REAL,
            betting_line_id TEXT
        );
    ''')

    today = datetime.now().strftime('%Y-%m-%d')
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    # Games: 2 recent, 1 old
    c.execute("INSERT INTO games VALUES ('g1', ?, 'final', 't1', 't2', 5, 3)", (today,))
    c.execute("INSERT INTO games VALUES ('g2', ?, 'final', 't1', 't2', 7, 6)", (three_days_ago,))
    c.execute("INSERT INTO games VALUES ('g3', ?, 'final', 't1', 't2', 2, 1)", (thirty_days_ago,))

    # model_predictions: 2 recent correct, 1 old incorrect for 'elo'
    c.execute("INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, was_correct) VALUES ('g1', 'elo', 0.6, 1)")
    c.execute("INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, was_correct) VALUES ('g2', 'elo', 0.55, 1)")
    c.execute("INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, was_correct) VALUES ('g3', 'elo', 0.7, 0)")

    # totals_predictions: 1 recent correct, 1 recent wrong, 1 old correct
    c.execute("INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line) VALUES ('g1', 'runs_ensemble', 9.0, 1, 'OVER', 7.5)")
    c.execute("INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line) VALUES ('g2', 'runs_ensemble', 10.0, 0, 'UNDER', 14.5)")
    c.execute("INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line) VALUES ('g3', 'runs_ensemble', 5.0, 1, 'OVER', 2.5)")

    conn.commit()
    return conn


class TestWeeklySidesAccuracy:
    """Test 7-day accuracy query for sides (model_predictions)."""

    def test_weekly_sides_query_structure(self):
        conn = _make_db()
        c = conn.cursor()
        c.execute('''
            SELECT mp.model_name,
                   SUM(mp.was_correct) as correct,
                   COUNT(*) as total
            FROM model_predictions mp
            JOIN games g ON mp.game_id = g.id
            WHERE mp.was_correct IS NOT NULL
            AND g.date >= date('now', '-7 days')
            GROUP BY mp.model_name
        ''')
        rows = c.fetchall()
        result = {row['model_name']: {'correct': row['correct'], 'total': row['total']} for row in rows}

        assert 'elo' in result
        assert result['elo']['correct'] == 2
        assert result['elo']['total'] == 2
        conn.close()

    def test_weekly_excludes_old_games(self):
        conn = _make_db()
        c = conn.cursor()
        # All-time should include the old game
        c.execute('''
            SELECT COUNT(*) as total, SUM(was_correct) as correct
            FROM model_predictions
            WHERE was_correct IS NOT NULL AND model_name = 'elo'
        ''')
        all_time = dict(c.fetchone())
        assert all_time['total'] == 3
        assert all_time['correct'] == 2

        # Weekly should exclude the 30-day-old game
        c.execute('''
            SELECT COUNT(*) as total, SUM(was_correct) as correct
            FROM model_predictions mp
            JOIN games g ON mp.game_id = g.id
            WHERE mp.was_correct IS NOT NULL AND mp.model_name = 'elo'
            AND g.date >= date('now', '-7 days')
        ''')
        weekly = dict(c.fetchone())
        assert weekly['total'] == 2
        assert weekly['correct'] == 2
        conn.close()


class TestWeeklyTotalsAccuracy:
    """Test 7-day accuracy queries for totals models."""

    def test_weekly_totals_mae_query(self):
        conn = _make_db()
        c = conn.cursor()
        c.execute('''
            SELECT
                tp.model_name,
                COUNT(*) as n,
                ROUND(AVG(ABS(tp.projected_total - (g.home_score + g.away_score))), 2) as mae
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            WHERE g.status = 'final'
              AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
              AND g.date >= date('now', '-7 days')
            GROUP BY tp.model_name
        ''')
        rows = c.fetchall()
        result = {row['model_name']: {'n': row['n'], 'mae': row['mae']} for row in rows}

        assert 'runs_ensemble' in result
        assert result['runs_ensemble']['n'] == 2  # Only 2 recent games
        conn.close()

    def test_weekly_totals_ou_query(self):
        conn = _make_db()
        c = conn.cursor()
        c.execute('''
            SELECT
                model_name,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as hit_rate
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            WHERE tp.was_correct IS NOT NULL
              AND tp.prediction IN ('OVER', 'UNDER')
              AND g.date >= date('now', '-7 days')
            GROUP BY model_name
        ''')
        rows = c.fetchall()
        result = {row['model_name']: {'total': row['total'], 'correct': row['correct'], 'hit_rate': row['hit_rate']} for row in rows}

        assert 'runs_ensemble' in result
        assert result['runs_ensemble']['total'] == 2
        assert result['runs_ensemble']['correct'] == 1
        assert result['runs_ensemble']['hit_rate'] == 50.0
        conn.close()

    def test_no_recent_returns_empty(self):
        """If all games are old, weekly queries return no rows."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.executescript('''
            CREATE TABLE games (id TEXT PRIMARY KEY, date TEXT, status TEXT, home_team_id TEXT, away_team_id TEXT, home_score INTEGER, away_score INTEGER);
            CREATE TABLE totals_predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, model_name TEXT, projected_total REAL, was_correct INTEGER, prediction TEXT, over_under_line REAL, predicted_at TEXT, actual_total INTEGER, edge_pct REAL, confidence REAL, betting_line_id TEXT);
        ''')
        old_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        c.execute("INSERT INTO games VALUES ('g1', ?, 'final', 't1', 't2', 5, 3)", (old_date,))
        c.execute("INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line) VALUES ('g1', 'runs_ensemble', 9.0, 1, 'OVER', 7.5)")
        conn.commit()

        c.execute('''
            SELECT tp.model_name, COUNT(*) as n
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            WHERE g.date >= date('now', '-7 days')
            GROUP BY tp.model_name
        ''')
        assert len(c.fetchall()) == 0
        conn.close()
