#!/usr/bin/env python3
"""Tests for the PEAR Ratings model."""

import sys
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pear_model import PearModel

# Use a named shared in-memory DB so multiple get_connection() calls
# within one test all see the same data (each opens a new connection).
_SHARED_URI = "file:test_pear?mode=memory&cache=shared"


def _setup_shared_db(teams):
    """Populate the shared in-memory DB. Returns the 'anchor' connection
    that must stay open for the DB to persist."""
    anchor = sqlite3.connect(_SHARED_URI, uri=True)
    anchor.row_factory = sqlite3.Row
    anchor.execute('''
        CREATE TABLE IF NOT EXISTS pear_ratings (
            team_name TEXT NOT NULL,
            team_id TEXT,
            season INTEGER NOT NULL,
            rating REAL, net_score REAL, net_rank INTEGER,
            rqi REAL, rqi_rank INTEGER, sos_rank INTEGER, sor_rank INTEGER,
            elo REAL, elo_rank INTEGER, rpi_rank INTEGER,
            resume_quality REAL, avg_expected_wins REAL,
            fwar REAL, owar_z REAL, pwar_z REAL, wpoe_pct REAL,
            pythag REAL, killshots REAL, conceded REAL, kshot_ratio REAL,
            era REAL, whip REAL, kp9 REAL, rpg REAL,
            ba REAL, obp REAL, slg REAL, ops REAL, woba REAL, iso REAL, pct REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (team_id, season)
        )
    ''')
    anchor.execute("DELETE FROM pear_ratings")
    for t in teams:
        cols = ', '.join(t.keys())
        placeholders = ', '.join(['?'] * len(t))
        anchor.execute(f"INSERT INTO pear_ratings ({cols}) VALUES ({placeholders})", list(t.values()))
    anchor.commit()
    return anchor


def _shared_get_connection():
    """Return a new connection to the shared in-memory DB."""
    conn = sqlite3.connect(_SHARED_URI, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


TEAM_A = {
    "team_name": "Alpha", "team_id": "alpha", "season": 2026,
    "rating": 6.0, "net_score": 0.9, "net_rank": 3,
    "rqi": 3.0, "rqi_rank": 3, "sos_rank": 10, "sor_rank": 5,
    "elo": 1700.0, "elo_rank": 2, "rpi_rank": 3,
    "resume_quality": 0.12, "avg_expected_wins": 0.4,
    "fwar": 2.0, "owar_z": 1.2, "pwar_z": 0.8, "wpoe_pct": 0.09,
    "pythag": 0.85, "killshots": 12, "conceded": 3, "kshot_ratio": 4.0,
    "era": 2.5, "whip": 1.0, "kp9": 10.0, "rpg": 7.5,
    "ba": 0.300, "obp": 0.400, "slg": 0.500, "ops": 0.900,
    "woba": 0.380, "iso": 0.200, "pct": 0.850,
    "fetched_at": "2026-02-25T00:00:00",
}

TEAM_B = {
    "team_name": "Bravo", "team_id": "bravo", "season": 2026,
    "rating": 3.0, "net_score": 0.5, "net_rank": 100,
    "rqi": 5.0, "rqi_rank": 80, "sos_rank": 150, "sor_rank": 120,
    "elo": 1400.0, "elo_rank": 100, "rpi_rank": 110,
    "resume_quality": 0.05, "avg_expected_wins": 0.25,
    "fwar": 0.5, "owar_z": 0.3, "pwar_z": 0.2, "wpoe_pct": -0.02,
    "pythag": 0.45, "killshots": 3, "conceded": 10, "kshot_ratio": 0.3,
    "era": 5.5, "whip": 1.6, "kp9": 6.0, "rpg": 4.0,
    "ba": 0.240, "obp": 0.310, "slg": 0.360, "ops": 0.670,
    "woba": 0.290, "iso": 0.120, "pct": 0.400,
    "fetched_at": "2026-02-25T00:00:00",
}

# Identical stats to TEAM_A for symmetry tests
TEAM_C = {
    **TEAM_A,
    "team_name": "Charlie", "team_id": "charlie",
}

ALL_TEAMS = [TEAM_A, TEAM_B, TEAM_C]


@pytest.fixture(autouse=True)
def shared_db():
    """Set up and tear down the shared in-memory DB for each test."""
    anchor = _setup_shared_db(ALL_TEAMS)
    with patch("models.pear_model.get_connection", side_effect=_shared_get_connection):
        yield
    anchor.close()


def _fresh_model():
    model = PearModel()
    model._season = 2026
    return model


class TestPearModel:
    def test_predict_strong_vs_weak(self):
        """Strong team should be favored over weak team."""
        result = _fresh_model().predict_game("alpha", "bravo")
        assert result is not None
        assert result["home_win_probability"] > 0.5
        assert result["away_win_probability"] < 0.5
        assert abs(result["home_win_probability"] + result["away_win_probability"] - 1.0) < 0.01
        assert result["projected_home_runs"] > 0
        assert result["projected_away_runs"] > 0

    def test_missing_team_returns_none(self):
        """If a team has no PEAR data, model returns None."""
        result = _fresh_model().predict_game("alpha", "nonexistent")
        assert result is None

    def test_symmetry(self):
        """Flipping home/away should give complementary probabilities on neutral site."""
        model = _fresh_model()
        result_ab = model.predict_game("alpha", "bravo", neutral_site=True)

        model2 = _fresh_model()
        result_ba = model2.predict_game("bravo", "alpha", neutral_site=True)

        assert result_ab is not None
        assert result_ba is not None
        total = result_ab["home_win_probability"] + result_ba["home_win_probability"]
        assert abs(total - 1.0) < 0.02, \
            f"Symmetry violated: {result_ab['home_win_probability']} + {result_ba['home_win_probability']} = {total}"

    def test_identical_teams_near_50(self):
        """Two identical teams on neutral site should be ~50%."""
        result = _fresh_model().predict_game("alpha", "charlie", neutral_site=True)
        assert result is not None
        assert abs(result["home_win_probability"] - 0.5) < 0.05, \
            f"Expected ~0.50 for identical teams, got {result['home_win_probability']}"

    def test_home_advantage(self):
        """Non-neutral game should favor home team more than neutral."""
        neutral = _fresh_model().predict_game("alpha", "bravo", neutral_site=True)
        home = _fresh_model().predict_game("alpha", "bravo", neutral_site=False)
        assert home["home_win_probability"] > neutral["home_win_probability"]

    def test_output_keys(self):
        """Verify all required output keys are present."""
        result = _fresh_model().predict_game("alpha", "bravo")
        for key in ["home_win_probability", "projected_home_runs", "projected_away_runs"]:
            assert key in result, f"Missing key: {key}"

    def test_probability_bounds(self):
        """Probabilities should be clamped to [0.02, 0.98]."""
        result = _fresh_model().predict_game("alpha", "bravo")
        assert 0.02 <= result["home_win_probability"] <= 0.98
        assert 0.02 <= result["away_win_probability"] <= 0.98

    def test_log5_static(self):
        """Test the static Log5 formula directly."""
        assert PearModel._log5(0.5, 0.5) == 0.5
        p = PearModel._log5(0.7, 0.3)
        assert p > 0.7
        p_flip = PearModel._log5(0.3, 0.7)
        assert abs(p + p_flip - 1.0) < 1e-10

    def test_model_info(self):
        """Model metadata should be correct."""
        model = PearModel()
        info = model.get_info()
        assert info["name"] == "pear"
        assert info["version"] == "1.0"
