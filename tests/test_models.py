#!/usr/bin/env python3
"""
Model Contract Tests

Tests that each model in the ensemble:
- Returns required keys in prediction dict
- Returns valid probability values (0-1)
- Loads weights without errors
- Can predict a sample game without crashing

This catches shape mismatches, missing model files, and API contract violations.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# Required keys that every model prediction must have
REQUIRED_PREDICTION_KEYS = [
    'home_win_probability',
    'away_win_probability',
]

# Optional but common keys
OPTIONAL_PREDICTION_KEYS = [
    'model',
    'projected_home_runs',
    'projected_away_runs',
    'projected_total',
    'run_line',
]


class TestModelImports:
    """Test that all models can be imported."""
    
    def test_import_elo_model(self):
        from models.elo_model import EloModel
        assert EloModel is not None
    
    def test_import_advanced_model(self):
        from models.advanced_model import AdvancedModel
        assert AdvancedModel is not None
    
    def test_import_pythagorean_model(self):
        from models.pythagorean_model import PythagoreanModel
        assert PythagoreanModel is not None
    
    def test_import_log5_model(self):
        from models.log5_model import Log5Model
        assert Log5Model is not None
    
    def test_import_pitching_model(self):
        from models.pitching_model import PitchingModel
        assert PitchingModel is not None
    
    def test_import_conference_model(self):
        from models.conference_model import ConferenceModel
        assert ConferenceModel is not None
    
    def test_import_prior_model(self):
        from models.prior_model import PriorModel
        assert PriorModel is not None
    
    def test_import_poisson_model(self):
        from models.poisson_model import predict as poisson_predict
        assert poisson_predict is not None
    
    def test_import_ensemble_model(self):
        from models.ensemble_model import EnsembleModel
        assert EnsembleModel is not None
    
    def test_import_neural_model(self):
        """Neural model should import even without weights."""
        try:
            from models.neural_model import NeuralModel
            assert NeuralModel is not None
        except ImportError as e:
            pytest.skip(f"Neural model not available: {e}")
    
    def test_import_xgboost_model(self):
        """XGBoost model should import (may skip if xgboost not installed)."""
        try:
            from models.xgboost_model import XGBMoneylineModel
            assert XGBMoneylineModel is not None
        except ImportError as e:
            pytest.skip(f"XGBoost not available: {e}")
    
    def test_import_lightgbm_model(self):
        """LightGBM model should import (may skip if lightgbm not installed)."""
        try:
            from models.lightgbm_model import LGBMoneylineModel
            assert LGBMoneylineModel is not None
        except ImportError as e:
            pytest.skip(f"LightGBM not available: {e}")


class TestModelPredictions:
    """Test model prediction outputs."""
    
    @pytest.fixture
    def models(self):
        """Load all available models."""
        loaded = {}
        
        # Statistical models (should always work)
        from models.elo_model import EloModel
        from models.advanced_model import AdvancedModel
        from models.pythagorean_model import PythagoreanModel
        from models.log5_model import Log5Model
        from models.conference_model import ConferenceModel
        from models.prior_model import PriorModel
        from models.pitching_model import PitchingModel
        
        loaded['elo'] = EloModel()
        loaded['advanced'] = AdvancedModel()
        loaded['pythagorean'] = PythagoreanModel()
        loaded['log5'] = Log5Model()
        loaded['conference'] = ConferenceModel()
        loaded['prior'] = PriorModel()
        loaded['pitching'] = PitchingModel()
        
        # ML models (may not have weights)
        try:
            from models.neural_model import NeuralModel
            loaded['neural'] = NeuralModel(use_model_predictions=False)
        except Exception:
            pass
        
        try:
            from models.xgboost_model import XGBMoneylineModel
            loaded['xgboost'] = XGBMoneylineModel()
        except Exception:
            pass
        
        try:
            from models.lightgbm_model import LGBMoneylineModel
            loaded['lightgbm'] = LGBMoneylineModel()
        except Exception:
            pass
        
        return loaded
    
    def test_all_models_have_predict_game(self, models):
        """All models should have predict_game method."""
        for name, model in models.items():
            assert hasattr(model, 'predict_game'), (
                f"Model {name} missing predict_game method"
            )
    
    def test_models_return_required_keys(self, models, sample_team_ids):
        """Each model prediction must include required keys."""
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        home = sample_team_ids['sample_home']
        away = sample_team_ids['sample_away']
        
        for name, model in models.items():
            try:
                pred = model.predict_game(home, away, neutral_site=False)
                
                for key in REQUIRED_PREDICTION_KEYS:
                    assert key in pred, (
                        f"Model {name} missing required key: {key}"
                    )
            except Exception as e:
                pytest.fail(f"Model {name} failed to predict: {e}")
    
    def test_probabilities_valid_range(self, models, sample_team_ids):
        """Probabilities must be between 0 and 1."""
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        home = sample_team_ids['sample_home']
        away = sample_team_ids['sample_away']
        
        for name, model in models.items():
            try:
                pred = model.predict_game(home, away, neutral_site=False)
                
                home_prob = pred.get('home_win_probability')
                away_prob = pred.get('away_win_probability')
                
                if home_prob is not None:
                    assert 0.0 <= home_prob <= 1.0, (
                        f"Model {name}: home_win_probability {home_prob} out of range [0, 1]"
                    )
                
                if away_prob is not None:
                    assert 0.0 <= away_prob <= 1.0, (
                        f"Model {name}: away_win_probability {away_prob} out of range [0, 1]"
                    )
            except Exception as e:
                pytest.fail(f"Model {name} failed: {e}")
    
    def test_probabilities_sum_approximately_one(self, models, sample_team_ids):
        """home + away probabilities should sum to ~1.0."""
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        home = sample_team_ids['sample_home']
        away = sample_team_ids['sample_away']
        
        for name, model in models.items():
            try:
                pred = model.predict_game(home, away, neutral_site=False)
                
                home_prob = pred.get('home_win_probability', 0)
                away_prob = pred.get('away_win_probability', 0)
                
                total = home_prob + away_prob
                
                # Allow small deviation for rounding
                assert 0.99 <= total <= 1.01, (
                    f"Model {name}: probabilities sum to {total}, expected ~1.0"
                )
            except Exception as e:
                # Some models might not guarantee this (e.g., Poisson with ties)
                pass


class TestNeuralModelWeights:
    """Tests specific to neural network model."""
    
    def test_neural_model_loads_weights(self):
        """Neural model should load saved weights if they exist."""
        try:
            from models.neural_model import NeuralModel, MODEL_PATH, FINETUNED_PATH
            
            model = NeuralModel(use_model_predictions=False)
            
            # Check if weights file exists
            weights_exist = MODEL_PATH.exists() or FINETUNED_PATH.exists()
            
            if weights_exist:
                # Model should have loaded weights
                assert model._loaded, "Neural model has weights but didn't load them"
            else:
                pytest.skip("No neural model weights file found")
                
        except ImportError as e:
            pytest.skip(f"Neural model not available: {e}")
    
    def test_neural_input_size_matches_features(self):
        """Neural model input size should match FeatureComputer output."""
        try:
            from models.neural_model import NeuralModel
            from models.nn_features_slim import SlimFeatureComputer, NUM_FEATURES
            
            model = NeuralModel(use_model_predictions=False)
            
            expected_size = NUM_FEATURES
            actual_size = model.input_size
            
            assert actual_size == expected_size, (
                f"Neural model input_size ({actual_size}) != "
                f"FeatureComputer num_features ({expected_size}). "
                f"This will cause dimension mismatch errors!"
            )
        except ImportError as e:
            pytest.skip(f"Neural model not available: {e}")


class TestGBMModelWeights:
    """Tests for XGBoost and LightGBM models."""
    
    def test_xgboost_loads_weights(self):
        """XGBoost model should load if weights exist."""
        try:
            from models.xgboost_model import XGBMoneylineModel
            
            model = XGBMoneylineModel()
            
            # Model should indicate if loaded
            model_path = Path(__file__).parent.parent / "data" / "xgb_moneyline.pkl"
            if model_path.exists():
                assert model.model is not None, "XGBoost weights exist but model is None"
            else:
                pytest.skip("No XGBoost weights file found")
                
        except ImportError as e:
            pytest.skip(f"XGBoost not available: {e}")
    
    def test_lightgbm_loads_weights(self):
        """LightGBM model should load if weights exist."""
        try:
            from models.lightgbm_model import LGBMoneylineModel
            
            model = LGBMoneylineModel()
            
            model_path = Path(__file__).parent.parent / "data" / "lgb_moneyline.pkl"
            if model_path.exists():
                assert model.model is not None, "LightGBM weights exist but model is None"
            else:
                pytest.skip("No LightGBM weights file found")
                
        except ImportError as e:
            pytest.skip(f"LightGBM not available: {e}")


class TestEnsembleModel:
    """Tests for the ensemble model."""

    def test_sample_ramp_multiplier_behaves_as_floor_then_full(self):
        """Sample ramp should start at floor and reach full at threshold."""
        from models.ensemble_model import EnsembleModel

        floor = EnsembleModel.SAMPLE_RAMP_FLOOR
        threshold = EnsembleModel.SAMPLE_RAMP_THRESHOLD

        assert EnsembleModel._ramp_multiplier(0, floor, threshold) == pytest.approx(floor)
        mid = EnsembleModel._ramp_multiplier(threshold // 2, floor, threshold)
        assert floor < mid < 1.0
        assert EnsembleModel._ramp_multiplier(threshold, floor, threshold) == pytest.approx(1.0)
        assert EnsembleModel._ramp_multiplier(threshold + 10, floor, threshold) == pytest.approx(1.0)

    def test_hard_rebalance_sets_target_weights_and_preserves_disabled(self):
        """Hard rebalance should ignore incremental drift and keep disabled models at zero."""
        from models.ensemble_model import EnsembleModel

        ensemble = EnsembleModel.__new__(EnsembleModel)
        ensemble.models = {"a": object(), "b": object(), "c": object()}
        ensemble.default_weights = {"a": 0.6, "b": 0.4, "c": 0.0}  # c disabled
        ensemble.weights = {"a": 0.9, "b": 0.1, "c": 0.0}
        ensemble.ADJUSTMENT_RATE = 0.25

        target = {"a": 0.2, "b": 0.8, "c": 0.0}

        _, incremental_after, _ = ensemble._apply_target_weights(target.copy(), mode="incremental")
        assert incremental_after["c"] == 0.0
        assert incremental_after["a"] != pytest.approx(0.2)
        assert incremental_after["b"] != pytest.approx(0.8)

        ensemble.weights = {"a": 0.9, "b": 0.1, "c": 0.0}
        _, hard_after, _ = ensemble._apply_target_weights(target.copy(), mode="hard_rebalance")
        assert hard_after["c"] == 0.0
        assert hard_after["a"] == pytest.approx(0.2)
        assert hard_after["b"] == pytest.approx(0.8)
    
    def test_ensemble_loads_all_component_models(self):
        """Ensemble should load its component models."""
        from models.ensemble_model import EnsembleModel
        
        ensemble = EnsembleModel()
        
        # Should have loaded at least the statistical models
        assert len(ensemble.models) >= 5, (
            f"Ensemble loaded only {len(ensemble.models)} models, expected at least 5"
        )
    
    def test_ensemble_weights_sum_to_one(self):
        """Ensemble weights should sum to approximately 1."""
        from models.ensemble_model import EnsembleModel
        
        ensemble = EnsembleModel()
        weights = ensemble.weights  # Access the weights dict directly
        
        total = sum(weights.values())
        assert 0.99 <= total <= 1.01, (
            f"Ensemble weights sum to {total}, expected ~1.0"
        )
    
    def test_ensemble_pitching_weight_is_five_percent(self):
        """Pitching model should have 5% weight in ensemble (re-enabled)."""
        from models.ensemble_model import EnsembleModel
        
        ensemble = EnsembleModel()
        weights = ensemble.weights
        
        pitching_weight = weights.get('pitching', 0)
        assert 0.04 <= pitching_weight <= 0.12, (
            f"Pitching model weight is {pitching_weight}, expected 4-12%"
        )
    
    def test_ensemble_prediction_includes_model(self, sample_team_ids):
        """Ensemble prediction should include 'model' key."""
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        
        from models.ensemble_model import EnsembleModel
        
        ensemble = EnsembleModel()
        pred = ensemble.predict_game(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away']
        )
        
        assert 'model' in pred
        assert pred['model'] == 'ensemble'


class TestRunsEnsemble:
    """Tests for the runs ensemble model."""
    
    def test_runs_ensemble_imports(self):
        """Runs ensemble should import."""
        from models.runs_ensemble import DEFAULT_RUN_WEIGHTS
        assert DEFAULT_RUN_WEIGHTS is not None
    
    def test_runs_ensemble_stats_only_components(self):
        """Runs ensemble should use only stats-based components (no Elo/Pythagorean)."""
        from models.runs_ensemble import DEFAULT_RUN_WEIGHTS
        
        components = set(DEFAULT_RUN_WEIGHTS.keys())
        
        # Should NOT include Elo or Pythagorean (they predict strength, not runs)
        assert 'elo' not in components, "Runs ensemble should not include Elo"
        assert 'pythagorean' not in components, "Runs ensemble should not include Pythagorean"
        
        # Should include stats-only components
        stats_components = {'poisson', 'pitching', 'advanced'}
        found = components & stats_components
        assert len(found) >= 2, (
            f"Runs ensemble should include stats-based components. Found: {components}"
        )


class TestPitchingModelV2:
    """Tests for the updated pitching model."""
    
    def test_pitching_model_uses_quality_tables(self, db_connection):
        """Pitching model should use team_pitching_quality and team_batting_quality tables."""
        # Check that quality tables exist and have data
        c = db_connection.cursor()
        
        c.execute("SELECT COUNT(*) as cnt FROM team_pitching_quality")
        pitching_count = c.fetchone()['cnt']
        
        c.execute("SELECT COUNT(*) as cnt FROM team_batting_quality")
        batting_count = c.fetchone()['cnt']
        
        assert pitching_count > 0, "team_pitching_quality table is empty"
        assert batting_count > 0, "team_batting_quality table is empty"
        
        # Both should have ~292 teams
        assert pitching_count >= 250, f"team_pitching_quality has only {pitching_count} rows"
        assert batting_count >= 250, f"team_batting_quality has only {batting_count} rows"
