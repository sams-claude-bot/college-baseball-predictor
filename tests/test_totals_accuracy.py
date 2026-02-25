"""Tests for show_totals_accuracy() in predict_and_track.py"""
import sqlite3
import pytest
from unittest.mock import patch, MagicMock


def _setup_test_db(conn):
    """Create minimal schema and test data."""
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            home_score INTEGER,
            away_score INTEGER,
            status TEXT DEFAULT 'final',
            home_team_id TEXT,
            away_team_id TEXT,
            date TEXT
        );
        CREATE TABLE IF NOT EXISTS totals_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            model_name TEXT DEFAULT 'runs_ensemble',
            projected_total REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            actual_total INTEGER,
            was_correct INTEGER,
            prediction TEXT,
            over_under_line REAL,
            betting_line_id TEXT,
            edge_pct REAL,
            confidence REAL
        );

        INSERT INTO games VALUES ('g1', 5, 3, 'final', 't1', 't2', '2025-03-01');
        INSERT INTO games VALUES ('g2', 7, 6, 'final', 't1', 't2', '2025-03-02');
        INSERT INTO games VALUES ('g3', 2, 1, 'final', 't1', 't2', '2025-03-03');

        INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line)
            VALUES ('g1', 'runs_ensemble', 9.0, 1, 'OVER', 7.5);
        INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line)
            VALUES ('g2', 'runs_ensemble', 10.0, 0, 'UNDER', 12.5);
        INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line)
            VALUES ('g3', 'runs_ensemble', 5.0, NULL, NULL, NULL);
        INSERT INTO totals_predictions (game_id, model_name, projected_total, was_correct, prediction, over_under_line)
            VALUES ('g1', 'runs_poisson', 10.0, 1, 'OVER', 7.5);
    ''')
    conn.commit()


def test_show_totals_accuracy(capsys):
    """Test that show_totals_accuracy prints MAE and O/U tables."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _setup_test_db(conn)

    with patch('scripts.predict_and_track.get_connection', return_value=conn):
        from scripts.predict_and_track import show_totals_accuracy
        show_totals_accuracy()

    output = capsys.readouterr().out
    assert 'MAE' in output
    assert 'O/U Record' in output
    assert 'Over/Under Split' in output
    assert 'runs_ensemble' in output
    assert 'runs_poisson' in output


def test_show_totals_accuracy_mae_values(capsys):
    """Verify MAE calculation correctness."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _setup_test_db(conn)

    with patch('scripts.predict_and_track.get_connection', return_value=conn):
        from scripts.predict_and_track import show_totals_accuracy
        show_totals_accuracy()

    output = capsys.readouterr().out
    # runs_ensemble: |9-8| + |10-13| + |5-3| = 1+3+2 = 6, avg = 2.0
    # runs_poisson: |10-8| = 2.0
    assert '2.0' in output  # Both models have MAE 2.0
