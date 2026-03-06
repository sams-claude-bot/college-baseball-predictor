#!/usr/bin/env python3
"""Tests for scripts/coverage_check.py using in-memory SQLite."""

import sqlite3
from unittest.mock import patch

import pytest

from scripts.coverage_check import MODEL_NAMES, check_coverage, main


@pytest.fixture
def mem_conn():
    """In-memory SQLite with games, teams, and model_predictions tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE teams (
            id TEXT PRIMARY KEY,
            name TEXT
        );
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            status TEXT DEFAULT 'scheduled',
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL
        );
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_home_prob REAL,
            prediction_source TEXT NOT NULL DEFAULT 'live',
            predicted_at TEXT,
            was_correct INTEGER,
            UNIQUE(game_id, model_name)
        );
        INSERT INTO teams VALUES ('team_a', 'Alabama');
        INSERT INTO teams VALUES ('team_b', 'Auburn');
        INSERT INTO teams VALUES ('team_c', 'LSU');
        INSERT INTO teams VALUES ('team_d', 'Ole Miss');
    """)
    return conn


def _insert_game(conn, game_id, date, home="team_a", away="team_b"):
    conn.execute(
        "INSERT INTO games (id, date, home_team_id, away_team_id, status) VALUES (?,?,?,?,'scheduled')",
        (game_id, date, home, away),
    )


def _insert_predictions(conn, game_id, models):
    for m in models:
        conn.execute(
            "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob) VALUES (?,?,0.55)",
            (game_id, m),
        )
    conn.commit()


class TestCheckCoverage:
    """Unit tests for check_coverage()."""

    def test_all_covered_returns_empty(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        _insert_predictions(mem_conn, "g1", MODEL_NAMES)
        gaps = check_coverage(mem_conn, "2026-03-06")
        assert gaps == []

    def test_partial_coverage_lists_missing(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        present = MODEL_NAMES[:10]
        _insert_predictions(mem_conn, "g1", present)

        gaps = check_coverage(mem_conn, "2026-03-06")
        assert len(gaps) == 1
        assert gaps[0]['game_id'] == "g1"
        assert gaps[0]['missing_count'] == 3
        assert set(gaps[0]['missing_models']) == set(MODEL_NAMES) - set(present)

    def test_zero_predictions_shows_all_missing(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        mem_conn.commit()

        gaps = check_coverage(mem_conn, "2026-03-06")
        assert len(gaps) == 1
        assert gaps[0]['missing_count'] == len(MODEL_NAMES)
        assert set(gaps[0]['missing_models']) == set(MODEL_NAMES)

    def test_no_scheduled_games_returns_empty(self, mem_conn):
        # A final game should not appear
        mem_conn.execute(
            "INSERT INTO games (id, date, home_team_id, away_team_id, status) VALUES ('g1','2026-03-06','team_a','team_b','final')"
        )
        mem_conn.commit()
        gaps = check_coverage(mem_conn, "2026-03-06")
        assert gaps == []

    def test_multiple_games_mixed_coverage(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        _insert_game(mem_conn, "g2", "2026-03-06", home="team_c", away="team_d")
        _insert_predictions(mem_conn, "g1", MODEL_NAMES)  # fully covered
        _insert_predictions(mem_conn, "g2", MODEL_NAMES[:5])  # partial
        mem_conn.commit()

        gaps = check_coverage(mem_conn, "2026-03-06")
        assert len(gaps) == 1
        assert gaps[0]['game_id'] == "g2"
        assert gaps[0]['missing_count'] == len(MODEL_NAMES) - 5

    def test_different_date_not_included(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        _insert_game(mem_conn, "g2", "2026-03-07")
        mem_conn.commit()

        gaps = check_coverage(mem_conn, "2026-03-06")
        # g1 has no predictions, g2 is a different date
        assert len(gaps) == 1
        assert gaps[0]['game_id'] == "g1"

    def test_team_names_resolved(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        mem_conn.commit()

        gaps = check_coverage(mem_conn, "2026-03-06")
        assert gaps[0]['home_team'] == "Alabama"
        assert gaps[0]['away_team'] == "Auburn"


class TestMainExitCodes:
    """Integration tests for main() exit codes and --quiet."""

    def test_full_coverage_exits_0(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        _insert_predictions(mem_conn, "g1", MODEL_NAMES)

        with patch("scripts.coverage_check.get_connection", return_value=mem_conn):
            code = main(["--date", "2026-03-06"])
        assert code == 0

    def test_gaps_exits_1(self, mem_conn):
        _insert_game(mem_conn, "g1", "2026-03-06")
        mem_conn.commit()

        with patch("scripts.coverage_check.get_connection", return_value=mem_conn):
            code = main(["--date", "2026-03-06"])
        assert code == 1

    def test_quiet_suppresses_when_covered(self, mem_conn, capsys):
        _insert_game(mem_conn, "g1", "2026-03-06")
        _insert_predictions(mem_conn, "g1", MODEL_NAMES)

        with patch("scripts.coverage_check.get_connection", return_value=mem_conn):
            code = main(["--date", "2026-03-06", "--quiet"])
        assert code == 0
        assert capsys.readouterr().out == ""

    def test_quiet_still_outputs_when_gaps(self, mem_conn, capsys):
        _insert_game(mem_conn, "g1", "2026-03-06")
        mem_conn.commit()

        with patch("scripts.coverage_check.get_connection", return_value=mem_conn):
            code = main(["--date", "2026-03-06", "--quiet"])
        assert code == 1
        out = capsys.readouterr().out
        assert "GAPS" in out
        assert "g1" in out

    def test_no_games_exits_0(self, mem_conn):
        with patch("scripts.coverage_check.get_connection", return_value=mem_conn):
            code = main(["--date", "2026-03-06"])
        assert code == 0
