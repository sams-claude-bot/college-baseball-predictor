#!/usr/bin/env python3
"""
Tests for scripts/finalize_games.py — Phase 3 merge_rescheduled.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.finalize_games import merge_rescheduled


def _make_db():
    """Create an in-memory SQLite connection with the games table + FK tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            winner_id TEXT,
            innings INTEGER DEFAULT 9,
            is_conference_game INTEGER DEFAULT 0,
            is_neutral_site INTEGER DEFAULT 0,
            tournament_id TEXT,
            venue TEXT,
            attendance INTEGER,
            notes TEXT,
            status TEXT DEFAULT 'scheduled',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            inning_text TEXT
        )
    """)

    c.execute("""
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_home_prob REAL,
            UNIQUE(game_id, model_name)
        )
    """)
    c.execute("""
        CREATE TABLE betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            book TEXT DEFAULT 'draftkings',
            home_ml INTEGER,
            away_ml INTEGER
        )
    """)

    conn.commit()
    return conn


class TestPhase3MergesRescheduled:
    """test_phase3_merges_rescheduled — postponed + replacement → merged."""

    @patch('scripts.finalize_games.ScheduleGateway')
    def test_merges(self, mock_gw_class):
        db = _make_db()

        # Insert postponed game on Mar 3
        db.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status)
            VALUES ('2025-03-03_auburn_mississippi-state', '2025-03-03',
                    'mississippi-state', 'auburn', 'postponed')
        """)
        # Insert replacement game on Mar 4
        db.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status,
                               home_score, away_score, winner_id)
            VALUES ('2025-03-04_auburn_mississippi-state', '2025-03-04',
                    'mississippi-state', 'auburn', 'final', 5, 3, 'mississippi-state')
        """)
        db.commit()

        mock_gw = MagicMock()
        mock_gw.migrate_fk_rows.return_value = {'migrated': 2, 'deleted': 0}
        mock_gw_class.return_value = mock_gw

        merged = merge_rescheduled(db, '2025-03-03')

        assert merged == 1
        # Ghost should be deleted
        ghost = db.execute("SELECT * FROM games WHERE id = '2025-03-03_auburn_mississippi-state'").fetchone()
        assert ghost is None
        # Replacement should still exist
        replacement = db.execute("SELECT * FROM games WHERE id = '2025-03-04_auburn_mississippi-state'").fetchone()
        assert replacement is not None
        # FK migration should have been called
        mock_gw.migrate_fk_rows.assert_called_once_with(
            '2025-03-03_auburn_mississippi-state',
            '2025-03-04_auburn_mississippi-state'
        )


class TestPhase3NoReplacement:
    """test_phase3_no_replacement — postponed with no nearby game → left alone."""

    @patch('scripts.finalize_games.ScheduleGateway')
    def test_no_merge(self, mock_gw_class):
        db = _make_db()

        # Insert postponed game with no replacement nearby
        db.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status)
            VALUES ('2025-03-03_auburn_mississippi-state', '2025-03-03',
                    'mississippi-state', 'auburn', 'postponed')
        """)
        db.commit()

        merged = merge_rescheduled(db, '2025-03-03')

        assert merged == 0
        # Game should still exist
        ghost = db.execute("SELECT * FROM games WHERE id = '2025-03-03_auburn_mississippi-state'").fetchone()
        assert ghost is not None
