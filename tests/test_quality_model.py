#!/usr/bin/env python3
"""Tests for the Quality Metrics model."""

import math
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.quality_model import QualityModel

_SHARED_URI = "file:test_quality?mode=memory&cache=shared"


def _create_tables(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS team_batting_quality (
            team_id TEXT PRIMARY KEY,
            lineup_ops REAL, lineup_woba REAL, lineup_wrc_plus REAL,
            lineup_iso REAL, lineup_k_pct REAL, lineup_bb_pct REAL,
            runs_per_game REAL, last_updated TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS team_pitching_quality (
            team_id TEXT PRIMARY KEY,
            staff_era REAL, staff_whip REAL, staff_k_per_9 REAL,
            staff_fip REAL, quality_arms INTEGER, shutdown_arms INTEGER,
            last_updated TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS team_aggregate_stats (
            team_id TEXT PRIMARY KEY,
            games INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
            win_pct REAL DEFAULT 0, pythagorean_pct REAL DEFAULT 0,
            blowout_wins INTEGER DEFAULT 0, blowout_losses INTEGER DEFAULT 0,
            close_games_wins INTEGER DEFAULT 0, close_games_losses INTEGER DEFAULT 0,
            last_updated TEXT
        )
    ''')


# --- Test teams ---
# Strong team
STRONG_BAT = {"team_id": "strong", "lineup_ops": 0.900, "lineup_woba": 0.400, "lineup_wrc_plus": 130.0, "runs_per_game": 8.0}
STRONG_PITCH = {"team_id": "strong", "staff_era": 2.50, "staff_whip": 1.00, "staff_k_per_9": 11.0, "staff_fip": 2.80, "quality_arms": 6, "shutdown_arms": 3}
STRONG_AGG = {"team_id": "strong", "games": 30, "wins": 25, "losses": 5, "win_pct": 0.833, "pythagorean_pct": 0.830, "blowout_wins": 10, "blowout_losses": 1, "close_games_wins": 5, "close_games_losses": 2}

# Weak team
WEAK_BAT = {"team_id": "weak", "lineup_ops": 0.600, "lineup_woba": 0.280, "lineup_wrc_plus": 70.0, "runs_per_game": 3.5}
WEAK_PITCH = {"team_id": "weak", "staff_era": 6.50, "staff_whip": 1.70, "staff_k_per_9": 5.0, "staff_fip": 6.00, "quality_arms": 1, "shutdown_arms": 0}
WEAK_AGG = {"team_id": "weak", "games": 30, "wins": 8, "losses": 22, "win_pct": 0.267, "pythagorean_pct": 0.260, "blowout_wins": 1, "blowout_losses": 8, "close_games_wins": 3, "close_games_losses": 6}

# Clone of strong for identical-team tests
CLONE_BAT = {**STRONG_BAT, "team_id": "clone"}
CLONE_PITCH = {**STRONG_PITCH, "team_id": "clone"}
CLONE_AGG = {**STRONG_AGG, "team_id": "clone"}

# Average team (for 3-team z-score spread)
AVG_BAT = {"team_id": "average", "lineup_ops": 0.750, "lineup_woba": 0.340, "lineup_wrc_plus": 100.0, "runs_per_game": 5.5}
AVG_PITCH = {"team_id": "average", "staff_era": 4.50, "staff_whip": 1.35, "staff_k_per_9": 8.0, "staff_fip": 4.40, "quality_arms": 3, "shutdown_arms": 1}
AVG_AGG = {"team_id": "average", "games": 30, "wins": 15, "losses": 15, "win_pct": 0.500, "pythagorean_pct": 0.500, "blowout_wins": 4, "blowout_losses": 4, "close_games_wins": 4, "close_games_losses": 4}


def _insert(conn, table, data):
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})", list(data.values()))


def _setup_shared_db():
    anchor = sqlite3.connect(_SHARED_URI, uri=True)
    anchor.row_factory = sqlite3.Row
    _create_tables(anchor)
    # Clear tables
    for t in ("team_batting_quality", "team_pitching_quality", "team_aggregate_stats"):
        anchor.execute(f"DELETE FROM {t}")

    for bat, pit, agg in [
        (STRONG_BAT, STRONG_PITCH, STRONG_AGG),
        (WEAK_BAT, WEAK_PITCH, WEAK_AGG),
        (CLONE_BAT, CLONE_PITCH, CLONE_AGG),
        (AVG_BAT, AVG_PITCH, AVG_AGG),
    ]:
        _insert(anchor, "team_batting_quality", bat)
        _insert(anchor, "team_pitching_quality", pit)
        _insert(anchor, "team_aggregate_stats", agg)

    anchor.commit()
    return anchor


def _shared_get_connection():
    conn = sqlite3.connect(_SHARED_URI, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture(autouse=True)
def shared_db():
    anchor = _setup_shared_db()
    with patch("models.quality_model.get_connection", side_effect=_shared_get_connection):
        yield
    anchor.close()


def _fresh_model():
    return QualityModel()


class TestQualityModel:
    def test_strong_vs_weak(self):
        """Strong team should be heavily favored over weak team."""
        result = _fresh_model().predict_game("strong", "weak")
        assert result is not None
        assert result["home_win_probability"] > 0.65

    def test_symmetry(self):
        """Flipping home/away on neutral site should give complementary probs."""
        m1 = _fresh_model()
        result_ab = m1.predict_game("strong", "weak", neutral_site=True)
        m2 = _fresh_model()
        result_ba = m2.predict_game("weak", "strong", neutral_site=True)
        assert result_ab is not None and result_ba is not None
        total = result_ab["home_win_probability"] + result_ba["home_win_probability"]
        assert abs(total - 1.0) < 0.02, f"Symmetry violated: {total}"

    def test_identical_teams_near_50(self):
        """Two identical teams on neutral site should be ~50%."""
        result = _fresh_model().predict_game("strong", "clone", neutral_site=True)
        assert result is not None
        assert abs(result["home_win_probability"] - 0.5) < 0.05, \
            f"Expected ~0.50 for identical teams, got {result['home_win_probability']}"

    def test_missing_team_returns_none(self):
        """Missing team data should return None."""
        result = _fresh_model().predict_game("strong", "nonexistent")
        assert result is None

    def test_output_keys(self):
        """All required output keys should be present."""
        result = _fresh_model().predict_game("strong", "weak")
        for key in ["home_win_probability", "away_win_probability",
                     "projected_home_runs", "projected_away_runs",
                     "projected_total", "model", "run_line"]:
            assert key in result, f"Missing key: {key}"

    def test_probability_bounds(self):
        """Probabilities should be clamped to [0.05, 0.95]."""
        result = _fresh_model().predict_game("strong", "weak")
        assert 0.05 <= result["home_win_probability"] <= 0.95
        assert 0.05 <= result["away_win_probability"] <= 0.95

    def test_home_advantage(self):
        """Non-neutral game should favor home team more than neutral."""
        neutral = _fresh_model().predict_game("strong", "weak", neutral_site=True)
        home = _fresh_model().predict_game("strong", "weak", neutral_site=False)
        assert home["home_win_probability"] > neutral["home_win_probability"]

    def test_model_info(self):
        """Model metadata should be correct."""
        model = QualityModel()
        info = model.get_info()
        assert info["name"] == "quality"
        assert info["version"] == "1.0"

    def test_probabilities_sum_to_one(self):
        """Home + away probabilities should sum to 1."""
        result = _fresh_model().predict_game("strong", "weak")
        total = result["home_win_probability"] + result["away_win_probability"]
        assert abs(total - 1.0) < 0.01

    def test_projected_runs_positive(self):
        """Projected runs should be positive."""
        result = _fresh_model().predict_game("strong", "weak")
        assert result["projected_home_runs"] > 0
        assert result["projected_away_runs"] > 0

    def test_z_score_ordering(self):
        """Strong team should have higher quality than weak team."""
        model = _fresh_model()
        model._load_all()
        strong = model._team_scores["strong"]
        weak = model._team_scores["weak"]
        assert strong["team_quality"] > weak["team_quality"]
        assert strong["offense_z"] > weak["offense_z"]
        assert strong["pitching_z"] > weak["pitching_z"]

    def test_caching(self):
        """Second call should use cached z-scores (no re-computation)."""
        model = _fresh_model()
        model.predict_game("strong", "weak")
        scores_ref = model._team_scores
        model.predict_game("strong", "average")
        assert model._team_scores is scores_ref  # same object, not reloaded
