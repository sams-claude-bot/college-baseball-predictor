"""Tests for calibration fitting (update_model_calibration.py)."""
import sqlite3
import random
import sys
import os
import json

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.update_model_calibration import fit_for_model, ensure_table, ensure_raw_prob_column, MIN_SAMPLES


def _create_test_db(tmp_path, n=200):
    """Create an in-memory-like test DB with games and model_predictions."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Create games table
    cur.execute("""
        CREATE TABLE games (
            id INTEGER PRIMARY KEY,
            home_team_id INTEGER,
            away_team_id INTEGER,
            winner_id INTEGER,
            status TEXT
        )
    """)

    # Create model_predictions table
    cur.execute("""
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY,
            game_id INTEGER,
            model_name TEXT,
            predicted_home_prob REAL,
            raw_home_prob REAL,
            was_correct INTEGER,
            predicted_at TEXT
        )
    """)

    ensure_table(cur)
    conn.commit()

    random.seed(42)
    np.random.seed(42)

    for i in range(n):
        home_id = 100 + i
        away_id = 200 + i
        # Home team wins ~60% of the time
        home_won = 1 if random.random() < 0.6 else 0
        winner = home_id if home_won else away_id

        cur.execute(
            "INSERT INTO games (id, home_team_id, away_team_id, winner_id, status) VALUES (?,?,?,?,?)",
            (i + 1, home_id, away_id, winner, "final"),
        )

        # Generate a prob that's correlated with outcome
        prob = np.clip(0.5 + (home_won - 0.5) * 0.3 + random.gauss(0, 0.15), 0.05, 0.95)
        was_correct = int((prob > 0.5 and home_won) or (prob < 0.5 and not home_won))

        cur.execute(
            "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, raw_home_prob, was_correct, predicted_at) VALUES (?,?,?,?,?,?)",
            (i + 1, "test_model", prob, None, was_correct, "2026-02-27T00:00:00"),
        )

    conn.commit()
    return conn


class TestFitForModel:
    """Tests for fit_for_model calibration fitting."""

    def test_uses_home_won_not_was_correct(self, tmp_path):
        """Verify calibration uses actual home_won from games, not was_correct."""
        conn = _create_test_db(tmp_path, n=200)
        cur = conn.cursor()

        # Corrupt was_correct to all 1s — if the code used was_correct,
        # it would get degenerate labels and return None
        cur.execute("UPDATE model_predictions SET was_correct = 1")
        conn.commit()

        result = fit_for_model(cur, "test_model")
        # Should still work because it uses home_won from games table
        assert result is not None, "fit_for_model should use home_won from games, not was_correct"
        conn.close()

    def test_isotonic_method(self, tmp_path):
        """Test that isotonic calibration produces valid output."""
        conn = _create_test_db(tmp_path, n=250)
        cur = conn.cursor()
        result = fit_for_model(cur, "test_model")
        assert result is not None
        # Check structure
        assert result["method"] in ("platt", "isotonic")
        assert result["n"] >= MIN_SAMPLES
        assert 0 <= result["brier_cal"] <= 1
        assert 0 <= result["brier_raw"] <= 1
        conn.close()

    def test_platt_method(self, tmp_path):
        """Test that Platt scaling produces valid coefficients."""
        conn = _create_test_db(tmp_path, n=200)
        cur = conn.cursor()
        result = fit_for_model(cur, "test_model")
        assert result is not None
        if result["method"] == "platt":
            assert isinstance(result["a"], float)
            assert isinstance(result["b"], float)
        conn.close()

    def test_calibrated_probs_in_range(self, tmp_path):
        """Calibrated probabilities must be between 0 and 1."""
        from scripts.update_model_calibration import logit, sigmoid

        conn = _create_test_db(tmp_path, n=200)
        cur = conn.cursor()
        result = fit_for_model(cur, "test_model")
        assert result is not None

        if result["method"] == "platt":
            a, b = result["a"], result["b"]
            for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
                cal = sigmoid(a * logit(p) + b)
                assert 0 < cal < 1, "Platt calibrated prob out of range"
        else:
            iso_data = json.loads(result["isotonic_json"])
            for y_val in iso_data["y"]:
                assert 0 <= y_val <= 1, "Isotonic calibrated prob out of range"
        conn.close()

    def test_insufficient_samples(self, tmp_path):
        """Should return None with too few samples."""
        conn = _create_test_db(tmp_path, n=50)
        cur = conn.cursor()
        result = fit_for_model(cur, "test_model")
        assert result is None
        conn.close()

    def test_query_joins_games_table(self, tmp_path):
        """Verify the query joins games — if games rows are missing, no results."""
        conn = _create_test_db(tmp_path, n=200)
        cur = conn.cursor()
        # Delete all games — should get no rows
        cur.execute("DELETE FROM games")
        conn.commit()
        result = fit_for_model(cur, "test_model")
        assert result is None, "Should return None when games table has no matching rows"
        conn.close()
