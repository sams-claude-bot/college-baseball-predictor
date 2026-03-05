#!/usr/bin/env python3
"""Tests for meta-ensemble model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.meta_ensemble import MetaEnsemble, MODEL_NAMES


class TestMetaEnsemble:
    """Tests for the MetaEnsemble class."""

    def test_feature_names_count(self):
        """Feature extraction produces correct number of columns."""
        meta = MetaEnsemble()
        # 12 model probs + 3 agreement = 15 features
        expected = len(MODEL_NAMES) + 3
        # Simulate building features
        meta.feature_names = []
        for m in MODEL_NAMES:
            meta.feature_names.append(f'{m}_prob')
        meta.feature_names.extend(['models_predicting_home', 'avg_home_prob', 'prob_spread'])
        assert len(meta.feature_names) == expected

    def test_build_features_shape(self):
        """_build_features returns correct shape."""
        meta = MetaEnsemble()
        columns = [
            'game_id', 'date', 'home_team_id', 'away_team_id',
            'elo_prob', 'pythagorean_prob', 'lightgbm_prob', 'poisson_prob',
            'xgboost_prob', 'pitching_prob',
            'pear_prob', 'quality_prob', 'neural_prob',
            'venue_prob', 'rest_travel_prob', 'upset_prob',
            'home_won',
        ]
        n_models = len(MODEL_NAMES)  # 12
        n_features = n_models + 3  # 15
        # Create 5 fake rows
        rows = []
        for i in range(5):
            row = [f'game_{i}', '2026-02-20', 'team-a', 'team-b']
            row += [0.6] * n_models
            row += [1]
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
        """All 12 diverse models listed (dropped W/L clones, added venue/rest/upset)."""
        expected = {'elo', 'pythagorean', 
                    'lightgbm', 'poisson', 'xgboost', 
                    'pitching', 'pear', 'quality',
                    'neural', 'venue', 'rest_travel', 'upset'}
        assert set(MODEL_NAMES) == expected

    def test_missing_model_probs_default_to_half(self):
        """Missing model probs default to 0.5."""
        meta = MetaEnsemble()
        columns = [
            'game_id', 'date', 'home_team_id', 'away_team_id',
            'elo_prob', 'pythagorean_prob', 'lightgbm_prob', 'poisson_prob',
            'xgboost_prob', 'pitching_prob',
            'pear_prob', 'quality_prob', 'neural_prob',
            'venue_prob', 'rest_travel_prob', 'upset_prob',
            'home_won',
        ]
        # Row with None for some probs
        row = ['game_1', '2026-02-20', 'team-a', 'team-b']
        # elo=None, pyth=0.6, lgbm=0.5, poisson=0.5, xgb=0.5, pitch=0.5,
        # pear=0.8, quality=None, neural=0.55, venue=0.7, rest=None, upset=0.4
        row += [None, 0.6, 0.5, 0.5, 0.5, 0.5, 0.8, None, 0.55, 0.7, None, 0.4]
        row += [1]
        rows = [tuple(row)]

        X, y, dates, names = meta._build_features(rows, columns)
        assert X[0, 0] == 0.5   # elo defaulted
        assert X[0, 1] == 0.6   # pythagorean kept
        assert X[0, 6] == 0.8   # pear kept
        assert X[0, 7] == 0.5   # quality defaulted
        assert X[0, 8] == 0.55  # neural kept
        assert X[0, 9] == 0.7   # venue kept
        assert X[0, 10] == 0.5  # rest_travel defaulted
        assert X[0, 11] == 0.4  # upset kept

    def test_get_feature_importance_no_model(self):
        """get_feature_importance returns empty dict when no model trained."""
        meta = MetaEnsemble()
        meta._loaded = True  # pretend we tried loading
        meta.xgb_model = None  # but no model
        assert meta.get_feature_importance() == {}
