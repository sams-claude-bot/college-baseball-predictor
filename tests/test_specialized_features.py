#!/usr/bin/env python3
"""
Tests for specialized feature computers (BattingFeatureComputer, PitchingFeatureComputer).

Verifies:
- Feature count in expected range (30-40)
- No NaN/Inf values
- Correct return types (float32 numpy arrays)
- Historical and live feature dimension match
- Feature name consistency
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.features_batting import BattingFeatureComputer, HistoricalBattingFeatureComputer
from models.features_pitching import PitchingFeatureComputer, HistoricalPitchingFeatureComputer


# --- BattingFeatureComputer Tests ---

class TestBattingFeatureComputer:

    @pytest.fixture
    def fc(self):
        return BattingFeatureComputer()

    def test_feature_count_in_range(self, fc):
        """Batting features should be 30-40."""
        n = fc.get_num_features()
        assert 30 <= n <= 40, f"Expected 30-40 batting features, got {n}"

    def test_feature_names_match_count(self, fc):
        names = fc.get_feature_names()
        assert len(names) == fc.get_num_features()

    def test_feature_names_unique(self, fc):
        names = fc.get_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names"

    def test_compute_features_dimension(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert features.shape[0] == fc.get_num_features()

    def test_features_no_nan(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert not np.any(np.isnan(features)), "NaN in batting features"

    def test_features_no_inf(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert not np.any(np.isinf(features)), "Inf in batting features"

    def test_features_dtype(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert features.dtype == np.float32

    def test_features_contain_batting_names(self, fc):
        """Feature names should be batting-related."""
        names = fc.get_feature_names()
        batting_keywords = ['ops', 'woba', 'iso', 'babip', 'hr', 'rpg', 'runs',
                           'scoring', 'obp', 'bench', 'elite', 'k_pct', 'bb_pct']
        # At least half should contain batting keywords
        batting_count = sum(1 for n in names if any(k in n for k in batting_keywords))
        assert batting_count >= len(names) * 0.5, \
            f"Only {batting_count}/{len(names)} features are batting-related"


class TestHistoricalBattingFeatureComputer:

    @pytest.fixture
    def hist_fc(self):
        return HistoricalBattingFeatureComputer()

    def test_dimension_matches_live(self, hist_fc):
        """Historical must match live feature count."""
        live_fc = BattingFeatureComputer()
        game_row = {
            'home_team': 'team-a', 'away_team': 'team-b',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        features, label = hist_fc.compute_game_features(game_row)
        assert features.shape[0] == live_fc.get_num_features(), \
            f"Historical ({features.shape[0]}) != Live ({live_fc.get_num_features()})"

    def test_no_nan(self, hist_fc):
        game_row = {
            'home_team': 'a', 'away_team': 'b',
            'home_score': 7, 'away_score': 2,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hist_fc.compute_game_features(game_row)
        assert not np.any(np.isnan(features))

    def test_no_inf(self, hist_fc):
        game_row = {
            'home_team': 'a', 'away_team': 'b',
            'home_score': 10, 'away_score': 0,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hist_fc.compute_game_features(game_row)
        assert not np.any(np.isinf(features))

    def test_label_binary(self, hist_fc):
        _, label1 = hist_fc.compute_game_features({
            'home_team': 'a', 'away_team': 'b',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        })
        assert label1 == 1.0

        _, label2 = hist_fc.compute_game_features({
            'home_team': 'a', 'away_team': 'b',
            'home_score': 2, 'away_score': 7,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        })
        assert label2 == 0.0

    def test_state_updates(self, hist_fc):
        game_row = {
            'home_team': 'x', 'away_team': 'y',
            'home_score': 6, 'away_score': 4,
            'date': '2025-05-01', 'neutral_site': 0, 'season': 2025,
        }
        hist_fc.update_state(game_row)
        assert hist_fc.team_stats['x']['games'] == 1
        assert hist_fc.team_stats['x']['runs_scored'] == 6

    def test_reset(self, hist_fc):
        hist_fc.team_stats['z']['games'] = 50
        hist_fc.reset()
        assert hist_fc.team_stats['z']['games'] == 0


# --- PitchingFeatureComputer Tests ---

class TestPitchingFeatureComputer:

    @pytest.fixture
    def fc(self):
        return PitchingFeatureComputer()

    def test_feature_count_in_range(self, fc):
        """Pitching features should be 30-40."""
        n = fc.get_num_features()
        assert 30 <= n <= 40, f"Expected 30-40 pitching features, got {n}"

    def test_feature_names_match_count(self, fc):
        names = fc.get_feature_names()
        assert len(names) == fc.get_num_features()

    def test_feature_names_unique(self, fc):
        names = fc.get_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names"

    def test_compute_features_dimension(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert features.shape[0] == fc.get_num_features()

    def test_features_no_nan(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert not np.any(np.isnan(features)), "NaN in pitching features"

    def test_features_no_inf(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert not np.any(np.isinf(features)), "Inf in pitching features"

    def test_features_dtype(self, fc, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams")
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15')
        assert features.dtype == np.float32

    def test_features_contain_pitching_names(self, fc):
        """Feature names should be pitching-related."""
        names = fc.get_feature_names()
        pitching_keywords = ['era', 'whip', 'fip', 'k_per_9', 'bb_per_9',
                            'rotation', 'bullpen', 'starter', 'rapg',
                            'fielding', 'quality_arms', 'innings']
        pitching_count = sum(1 for n in names if any(k in n for k in pitching_keywords))
        assert pitching_count >= len(names) * 0.5, \
            f"Only {pitching_count}/{len(names)} features are pitching-related"


class TestHistoricalPitchingFeatureComputer:

    @pytest.fixture
    def hist_fc(self):
        return HistoricalPitchingFeatureComputer()

    def test_dimension_matches_live(self, hist_fc):
        """Historical must match live feature count."""
        live_fc = PitchingFeatureComputer()
        game_row = {
            'home_team': 'team-a', 'away_team': 'team-b',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        features, label = hist_fc.compute_game_features(game_row)
        assert features.shape[0] == live_fc.get_num_features(), \
            f"Historical ({features.shape[0]}) != Live ({live_fc.get_num_features()})"

    def test_no_nan(self, hist_fc):
        game_row = {
            'home_team': 'a', 'away_team': 'b',
            'home_score': 7, 'away_score': 2,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hist_fc.compute_game_features(game_row)
        assert not np.any(np.isnan(features))

    def test_no_inf(self, hist_fc):
        game_row = {
            'home_team': 'a', 'away_team': 'b',
            'home_score': 10, 'away_score': 0,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hist_fc.compute_game_features(game_row)
        assert not np.any(np.isinf(features))

    def test_label_binary(self, hist_fc):
        _, label1 = hist_fc.compute_game_features({
            'home_team': 'a', 'away_team': 'b',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        })
        assert label1 == 1.0

    def test_state_updates(self, hist_fc):
        game_row = {
            'home_team': 'x', 'away_team': 'y',
            'home_score': 6, 'away_score': 4,
            'date': '2025-05-01', 'neutral_site': 0, 'season': 2025,
        }
        hist_fc.update_state(game_row)
        assert hist_fc.team_stats['x']['games'] == 1
        assert hist_fc.team_stats['x']['runs_allowed'] == 4

    def test_reset(self, hist_fc):
        hist_fc.team_stats['z']['games'] = 50
        hist_fc.reset()
        assert hist_fc.team_stats['z']['games'] == 0
