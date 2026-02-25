#!/usr/bin/env python3
"""
Tests for home bias reduction changes.

Validates:
  1. HFA constants are within reasonable bounds
  2. Poisson home advantage is reduced
  3. Slim features include 3 new strength-differential features
  4. Early season feature behaves correctly by date
  5. Strength diff magnitude is symmetric (same regardless of home/away)
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import HOME_ADVANTAGE_PROB, HOME_ADVANTAGE_ELO
from models.nn_features_slim import (
    NUM_FEATURES,
    FEATURE_NAMES,
    SlimHistoricalFeatureComputer,
    ELO_HOME_ADV as SLIM_ELO_HOME_ADV,
)
from models.nn_features import ELO_HOME_ADV as FULL_ELO_HOME_ADV


# ============================================================
# 1. Home advantage constants are reasonable
# ============================================================

class TestHomeAdvantageConstantsReasonable:

    def test_home_advantage_prob(self):
        assert HOME_ADVANTAGE_PROB <= 0.05, (
            f"HOME_ADVANTAGE_PROB={HOME_ADVANTAGE_PROB} exceeds 0.05"
        )

    def test_home_advantage_elo(self):
        assert HOME_ADVANTAGE_ELO <= 60, (
            f"HOME_ADVANTAGE_ELO={HOME_ADVANTAGE_ELO} exceeds 60"
        )

    def test_slim_elo_home_adv(self):
        assert SLIM_ELO_HOME_ADV <= 60, (
            f"nn_features_slim.ELO_HOME_ADV={SLIM_ELO_HOME_ADV} exceeds 60"
        )

    def test_full_elo_home_adv(self):
        assert FULL_ELO_HOME_ADV <= 60, (
            f"nn_features.ELO_HOME_ADV={FULL_ELO_HOME_ADV} exceeds 60"
        )


# ============================================================
# 2. Poisson home advantage reduced
# ============================================================

class TestPoissonHomeAdvantageReduced:

    def test_non_neutral_home_adv_reduced(self):
        """Poisson home_adv for non-neutral games should be <= 0.35."""
        from models.poisson.predict import predict

        # We can't easily call predict without DB, so inspect the source directly.
        import inspect
        source = inspect.getsource(predict)
        # Find the home_adv assignment for non-neutral
        # The line is: home_adv = 0.0 if neutral_site else X
        assert "else 0.3" in source or "else 0.2" in source or "else 0.25" in source, (
            "Poisson home_adv for non-neutral games should be <= 0.35; "
            "expected 'else 0.3' (or lower) in predict source"
        )


# ============================================================
# 3. Slim features include strength-differential features
# ============================================================

class TestSlimFeaturesIncludeStrengthDiff:

    def test_feature_count_increased(self):
        """Feature count should be 61 (was 58, +3 new features)."""
        assert NUM_FEATURES == 61, f"Expected 61 features, got {NUM_FEATURES}"

    def test_new_feature_names_present(self):
        assert 'strength_diff_magnitude' in FEATURE_NAMES
        assert 'is_home_int' in FEATURE_NAMES
        assert 'is_early_season' in FEATURE_NAMES

    def test_historical_computer_produces_61_features(self):
        hfc = SlimHistoricalFeatureComputer()
        game_row = {
            'home_team': 'Team A', 'away_team': 'Team B',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-20', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hfc.compute_game_features(game_row)
        assert features.shape[0] == 61, (
            f"Historical features: {features.shape[0]}, expected 61"
        )


# ============================================================
# 4. Early season feature
# ============================================================

class TestEarlySeasonFeature:

    @pytest.fixture
    def hfc(self):
        return SlimHistoricalFeatureComputer()

    def _get_early_season_value(self, hfc, date_str):
        game_row = {
            'home_team': 'Team A', 'away_team': 'Team B',
            'home_score': 5, 'away_score': 3,
            'date': date_str, 'neutral_site': 0, 'season': int(date_str[:4]),
        }
        features, _ = hfc.compute_game_features(game_row)
        idx = FEATURE_NAMES.index('is_early_season')
        return features[idx]

    def test_february_is_early_season(self, hfc):
        val = self._get_early_season_value(hfc, '2025-02-20')
        assert val == 1.0, f"Feb 20 should be early season, got {val}"

    def test_march_10_is_early_season(self, hfc):
        val = self._get_early_season_value(hfc, '2025-03-10')
        assert val == 1.0, f"Mar 10 should be early season, got {val}"

    def test_april_is_not_early_season(self, hfc):
        val = self._get_early_season_value(hfc, '2025-04-15')
        assert val == 0.0, f"Apr 15 should NOT be early season, got {val}"

    def test_march_20_is_not_early_season(self, hfc):
        val = self._get_early_season_value(hfc, '2025-03-20')
        assert val == 0.0, f"Mar 20 should NOT be early season, got {val}"


# ============================================================
# 5. Strength diff magnitude is symmetric
# ============================================================

class TestStrengthDiffSymmetric:

    def test_symmetric_strength_diff(self):
        """strength_diff_magnitude should be the same regardless of which team is home."""
        hfc = SlimHistoricalFeatureComputer()

        # Set different Elos for the two teams
        hfc.elo['Team A'] = 1600
        hfc.elo['Team B'] = 1400

        game_ab = {
            'home_team': 'Team A', 'away_team': 'Team B',
            'home_score': 5, 'away_score': 3,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        game_ba = {
            'home_team': 'Team B', 'away_team': 'Team A',
            'home_score': 5, 'away_score': 3,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }

        features_ab, _ = hfc.compute_game_features(game_ab)
        features_ba, _ = hfc.compute_game_features(game_ba)

        idx = FEATURE_NAMES.index('strength_diff_magnitude')
        assert abs(features_ab[idx] - features_ba[idx]) < 1e-6, (
            f"strength_diff_magnitude should be symmetric: "
            f"A-home={features_ab[idx]:.4f}, B-home={features_ba[idx]:.4f}"
        )
