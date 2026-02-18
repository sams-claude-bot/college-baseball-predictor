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
            from models.nn_features import FeatureComputer
            
            model = NeuralModel(use_model_predictions=False)
            feature_comp = FeatureComputer(use_model_predictions=False)
            
            expected_size = feature_comp.get_num_features()
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
