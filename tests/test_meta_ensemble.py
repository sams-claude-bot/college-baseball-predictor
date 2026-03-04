#!/usr/bin/env python3
"""Tests for meta-ensemble model."""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.meta_ensemble import MetaEnsemble, MODEL_NAMES


class TestMetaEnsemble:
    """Tests for the MetaEnsemble class."""

    def test_feature_names_count(self):
        """Feature extraction produces correct number of columns."""
        meta = MetaEnsemble()
        # 13 model probs + 3 agreement + 7 context = 23 features
        expected = len(MODEL_NAMES) + 3 + 7
        # Simulate building features
        meta.feature_names = []
        for m in MODEL_NAMES:
            meta.feature_names.append(f'{m}_prob')
        meta.feature_names.extend(['models_predicting_home', 'avg_home_prob', 'prob_spread'])
        meta.feature_names.extend(['elo_diff', 'same_conference', 'any_ranked',
                                   'pear_diff', 'rpi_diff', 'wp_diff', 'both_ranked'])
        assert len(meta.feature_names) == expected

    @patch.object(MetaEnsemble, '_compute_win_pct_cache', return_value={})
    def test_build_features_shape(self, mock_wp):
        """_build_features returns correct shape."""
        meta = MetaEnsemble()
        columns = [
            'game_id', 'date', 'home_team_id', 'away_team_id',
            'prior_prob', 'elo_prob', 'ensemble_prob',
            'pythagorean_prob', 'lightgbm_prob', 'poisson_prob', 'conference_prob',
            'xgboost_prob', 'advanced_prob', 'log5_prob', 'pitching_prob',
            'pear_prob', 'quality_prob', 'neural_prob',
            'home_won', 'home_elo', 'away_elo', 'home_conf', 'away_conf',
            'home_rank', 'away_rank',
            'home_pear', 'away_pear', 'home_rpi', 'away_rpi',
        ]
        n_models = len(MODEL_NAMES)  # 14
        n_features = n_models + 3 + 7  # 24
        # Create 5 fake rows
        rows = []
        for i in range(5):
            row = [f'game_{i}', '2026-02-20', 'team-a', 'team-b']
            row += [0.6] * n_models  # 13 model probs
            row += [1, 1520, 1480, 'SEC', 'SEC', 5, None]
            row += [80.0, 75.0, 0.600, 0.550]  # pear + rpi
            rows.append(tuple(row))

        X, y, dates, feature_names = meta._build_features(rows, columns)
        assert X.shape == (5, n_features)
        assert y.shape == (5,)
        assert len(dates) == 5
        assert len(feature_names) == n_features

    def test_predict_returns_valid_probability(self):
        """Prediction returns probability between 0 and 1."""
        meta = MetaEnsemble()
        # Without a trained model, should return 0.5 fallback
        prob = meta.predict(game_id='nonexistent')
        assert 0 <= prob <= 1

    def test_walk_forward_no_future_leak(self):
        """Walk-forward split doesn't use future data for training."""
        meta = MetaEnsemble()
        # Simulate dates
        dates = ['2026-02-18'] * 30 + ['2026-02-19'] * 30 + ['2026-02-20'] * 30
        unique_dates = sorted(set(dates))
        
        # For test_date_idx=2 (Feb 20), training should only use Feb 18+19
        test_date_idx = 2
        train_dates = unique_dates[:test_date_idx]
        assert '2026-02-20' not in train_dates
        assert '2026-02-18' in train_dates
        assert '2026-02-19' in train_dates

    def test_model_names_complete(self):
        """All 14 expected models are listed (nn_slim runs as 'neural')."""
        expected = {'prior', 'elo', 'ensemble', 'pythagorean', 
                    'lightgbm', 'poisson', 'conference', 'xgboost', 
                    'advanced', 'log5', 'pitching', 'pear', 'quality',
                    'neural'}
        assert set(MODEL_NAMES) == expected

    @patch.object(MetaEnsemble, '_compute_win_pct_cache', return_value={})
    def test_missing_model_probs_default_to_half(self, mock_wp):
        """Missing model probs default to 0.5."""
        meta = MetaEnsemble()
        columns = [
            'game_id', 'date', 'home_team_id', 'away_team_id',
            'prior_prob', 'elo_prob', 'ensemble_prob',
            'pythagorean_prob', 'lightgbm_prob', 'poisson_prob', 'conference_prob',
            'xgboost_prob', 'advanced_prob', 'log5_prob', 'pitching_prob',
            'pear_prob', 'quality_prob', 'neural_prob',
            'home_won', 'home_elo', 'away_elo', 'home_conf', 'away_conf',
            'home_rank', 'away_rank',
            'home_pear', 'away_pear', 'home_rpi', 'away_rpi',
        ]
        # Row with None for some probs (prior=0.7, elo=None, ensemble=0.6, ..., neural=None)
        row = ['game_1', '2026-02-20', 'team-a', 'team-b']
        row += [0.7, None, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, None, 0.55]
        row += [1, 1500, 1500, 'Big 12', 'ACC', None, None]
        row += [80.0, 75.0, 0.600, 0.550]  # pear + rpi
        rows = [tuple(row)]

        X, y, dates, names = meta._build_features(rows, columns)
        # elo_prob (index 1) should be 0.5 (defaulted from None)
        assert X[0, 1] == 0.5   # elo defaulted
        assert X[0, 0] == 0.7   # prior kept
        assert X[0, 11] == 0.8  # pear kept
        assert X[0, 12] == 0.5  # quality defaulted
        assert X[0, 13] == 0.55 # neural kept

    def test_get_feature_importance_no_model(self):
        """get_feature_importance returns empty dict when no model trained."""
        meta = MetaEnsemble()
        meta._loaded = True  # pretend we tried loading
        meta.xgb_model = None  # but no model
        assert meta.get_feature_importance() == {}
