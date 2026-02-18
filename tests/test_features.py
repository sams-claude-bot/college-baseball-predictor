#!/usr/bin/env python3
"""
Feature Pipeline Integrity Tests

Tests for FeatureComputer and HistoricalFeatureComputer to catch:
- Feature dimension mismatches (the bug that broke XGB/LGB on Feb 17)
- NaN/Inf values in feature vectors
- Feature name/count consistency

Reference: CONTEXT.md says feature dimension is 81 with use_model_predictions=False
"""

import sys
import math
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nn_features import FeatureComputer, HistoricalFeatureComputer


# Expected feature count per CONTEXT.md
EXPECTED_FEATURE_COUNT = 81


class TestFeatureComputer:
    """Tests for live prediction feature computation."""
    
    @pytest.fixture
    def feature_computer(self):
        """Create FeatureComputer without model predictions (training mode)."""
        return FeatureComputer(use_model_predictions=False)
    
    @pytest.fixture
    def feature_computer_with_meta(self):
        """Create FeatureComputer with model predictions (live mode)."""
        return FeatureComputer(use_model_predictions=True)
    
    def test_feature_count_matches_expected(self, feature_computer):
        """FeatureComputer should produce exactly 81 features (no model predictions)."""
        num_features = feature_computer.get_num_features()
        assert num_features == EXPECTED_FEATURE_COUNT, (
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {num_features}. "
            f"This mismatch will break XGB/LGB models trained on historical data."
        )
    
    def test_feature_names_match_count(self, feature_computer):
        """Feature names list should match feature count."""
        names = feature_computer.get_feature_names()
        num_features = feature_computer.get_num_features()
        assert len(names) == num_features, (
            f"Feature names count ({len(names)}) != num_features ({num_features}). "
            f"This indicates a bug in get_feature_names() or get_num_features()."
        )
    
    def test_feature_names_unique(self, feature_computer):
        """All feature names should be unique."""
        names = feature_computer.get_feature_names()
        duplicates = [n for n in names if names.count(n) > 1]
        assert len(duplicates) == 0, f"Duplicate feature names: {set(duplicates)}"
    
    def test_compute_features_returns_correct_dimension(self, feature_computer, sample_team_ids):
        """Computed features should match expected dimension."""
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        features = feature_computer.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15'
        )
        
        assert features.shape[0] == EXPECTED_FEATURE_COUNT, (
            f"Computed features have {features.shape[0]} dimensions, "
            f"expected {EXPECTED_FEATURE_COUNT}"
        )
    
    def test_features_no_nan(self, feature_computer, sample_team_ids):
        """Features should not contain NaN values."""
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        features = feature_computer.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15'
        )
        
        nan_indices = np.where(np.isnan(features))[0]
        assert len(nan_indices) == 0, (
            f"Features contain NaN at indices: {nan_indices.tolist()}"
        )
    
    def test_features_no_inf(self, feature_computer, sample_team_ids):
        """Features should not contain Inf values."""
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        features = feature_computer.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15'
        )
        
        inf_indices = np.where(np.isinf(features))[0]
        assert len(inf_indices) == 0, (
            f"Features contain Inf at indices: {inf_indices.tolist()}"
        )
    
    def test_features_dtype(self, feature_computer, sample_team_ids):
        """Features should be float32."""
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        features = feature_computer.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15'
        )
        
        assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    
    def test_with_model_predictions_adds_features(self, feature_computer, feature_computer_with_meta):
        """use_model_predictions=True should add 5 meta features."""
        base_count = feature_computer.get_num_features()
        meta_count = feature_computer_with_meta.get_num_features()
        
        # Should add 5 meta features (elo, advanced, pitching, pythagorean, log5)
        assert meta_count == base_count + 5, (
            f"Expected meta features to add 5, but got {meta_count - base_count}"
        )


class TestHistoricalFeatureComputer:
    """Tests for training data feature computation."""
    
    @pytest.fixture
    def hist_computer(self):
        """Create HistoricalFeatureComputer."""
        return HistoricalFeatureComputer()
    
    def test_historical_matches_live_dimension(self, hist_computer):
        """
        CRITICAL: HistoricalFeatureComputer must produce same dimension as 
        FeatureComputer with use_model_predictions=False.
        
        This was the bug on Feb 17 that broke XGB/LGB (trained on 77, predicted with 81).
        """
        live_computer = FeatureComputer(use_model_predictions=False)
        
        # Compute features for a dummy historical game
        game_row = {
            'home_team': 'team-a',
            'away_team': 'team-b',
            'home_score': 5,
            'away_score': 3,
            'date': '2025-03-15',
            'neutral_site': 0,
            'season': 2025,
        }
        
        features, label = hist_computer.compute_game_features(game_row)
        live_count = live_computer.get_num_features()
        
        assert features.shape[0] == live_count, (
            f"DIMENSION MISMATCH! Historical features: {features.shape[0]}, "
            f"Live features: {live_count}. This will break XGB/LGB models!"
        )
    
    def test_historical_features_no_nan(self, hist_computer):
        """Historical features should not contain NaN."""
        game_row = {
            'home_team': 'team-a',
            'away_team': 'team-b', 
            'home_score': 7,
            'away_score': 2,
            'date': '2025-04-01',
            'neutral_site': 0,
            'season': 2025,
        }
        
        features, label = hist_computer.compute_game_features(game_row)
        
        nan_indices = np.where(np.isnan(features))[0]
        assert len(nan_indices) == 0, (
            f"Historical features contain NaN at indices: {nan_indices.tolist()}"
        )
    
    def test_historical_features_no_inf(self, hist_computer):
        """Historical features should not contain Inf."""
        game_row = {
            'home_team': 'team-a',
            'away_team': 'team-b',
            'home_score': 10,
            'away_score': 0,  # Edge case: shutout
            'date': '2025-04-01',
            'neutral_site': 0,
            'season': 2025,
        }
        
        features, label = hist_computer.compute_game_features(game_row)
        
        inf_indices = np.where(np.isinf(features))[0]
        assert len(inf_indices) == 0, (
            f"Historical features contain Inf at indices: {inf_indices.tolist()}"
        )
    
    def test_label_is_binary(self, hist_computer):
        """Label should be 0.0 or 1.0."""
        # Home win
        features1, label1 = hist_computer.compute_game_features({
            'home_team': 'a', 'away_team': 'b',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        })
        assert label1 == 1.0, f"Home win should have label 1.0, got {label1}"
        
        # Away win
        features2, label2 = hist_computer.compute_game_features({
            'home_team': 'a', 'away_team': 'b',
            'home_score': 2, 'away_score': 7,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        })
        assert label2 == 0.0, f"Away win should have label 0.0, got {label2}"
    
    def test_state_updates_after_game(self, hist_computer):
        """update_state should modify internal state."""
        game_row = {
            'home_team': 'test-team-x',
            'away_team': 'test-team-y',
            'home_score': 6,
            'away_score': 4,
            'date': '2025-05-01',
            'neutral_site': 0,
            'season': 2025,
        }
        
        # Before update
        stats_before = hist_computer.team_stats['test-team-x']['games']
        
        # Update state
        hist_computer.update_state(game_row)
        
        # After update
        stats_after = hist_computer.team_stats['test-team-x']['games']
        
        assert stats_after == stats_before + 1, "Team games count should increment"
        assert hist_computer.team_stats['test-team-x']['wins'] == 1
        assert hist_computer.team_stats['test-team-y']['losses'] == 1
    
    def test_reset_clears_state(self, hist_computer):
        """reset() should clear all accumulated state."""
        # Accumulate some state
        hist_computer.team_stats['some-team']['games'] = 50
        hist_computer.elo['some-team'] = 1600
        
        hist_computer.reset()
        
        # Should be back to defaults
        assert hist_computer.team_stats['some-team']['games'] == 0
        assert hist_computer.elo['some-team'] == 1500  # DEFAULT_ELO
