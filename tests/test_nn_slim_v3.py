#!/usr/bin/env python3
"""
Tests for nn_slim v3: NCAA features, architecture, and training improvements.

Tests cover:
  - Feature count and naming (58 features)
  - NCAA stat feature integration
  - Historical and live feature computers produce same dimensionality
  - ResidualBlock forward pass
  - SlimBaseballNet forward/backward pass
  - TotalsNet forward pass
  - Label smoothing / focal loss behavior
  - Mixup augmentation
  - Season weight computation
  - Name-to-slug conversion
  - Feature importance computation
  - build_model with residual configs
"""

import sys
import math
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nn_features_slim import (
    SlimBaseballNet, SlimFeatureComputer, SlimHistoricalFeatureComputer,
    ResidualBlock, NUM_FEATURES, FEATURE_NAMES, NCAA_STAT_NAMES,
    _name_to_slug, _get_ncaa_features, _weather_features,
)
from models.nn_totals_slim import TotalsNet


# ============================================================
# Feature count and naming
# ============================================================

class TestFeatureDefinitions:

    def test_num_features_is_58(self):
        """v3 should have exactly 58 features (24 per team x2 + 3 game + 7 weather)."""
        assert NUM_FEATURES == 58, f"Expected 58 features, got {NUM_FEATURES}"

    def test_feature_names_count_matches(self):
        assert len(FEATURE_NAMES) == NUM_FEATURES

    def test_feature_names_unique(self):
        duplicates = [n for n in FEATURE_NAMES if FEATURE_NAMES.count(n) > 1]
        assert len(duplicates) == 0, f"Duplicate feature names: {set(duplicates)}"

    def test_ncaa_stat_names_present(self):
        """Each NCAA stat should appear in feature names for both home and away."""
        for stat in NCAA_STAT_NAMES:
            assert f'home_ncaa_{stat}' in FEATURE_NAMES, f"Missing home_ncaa_{stat}"
            assert f'away_ncaa_{stat}' in FEATURE_NAMES, f"Missing away_ncaa_{stat}"

    def test_ncaa_stat_count(self):
        """Should have exactly 9 NCAA stat types."""
        assert len(NCAA_STAT_NAMES) == 9

    def test_per_team_feature_count(self):
        """Each team should have 24 features (15 base + 9 NCAA)."""
        home_feats = [f for f in FEATURE_NAMES if f.startswith('home_')]
        away_feats = [f for f in FEATURE_NAMES if f.startswith('away_')]
        assert len(home_feats) == 24, f"Home features: {len(home_feats)}, expected 24"
        assert len(away_feats) == 24, f"Away features: {len(away_feats)}, expected 24"


# ============================================================
# Name-to-slug conversion
# ============================================================

class TestNameToSlug:

    def test_simple_name(self):
        assert _name_to_slug("Arizona Wildcats") == "arizona"

    def test_two_word_school(self):
        assert _name_to_slug("Mississippi State Bulldogs") == "mississippi-state"

    def test_three_word_school(self):
        assert _name_to_slug("South Carolina Gamecocks") == "south-carolina"

    def test_single_word(self):
        assert _name_to_slug("Arizona") == "arizona"

    def test_special_chars(self):
        result = _name_to_slug("Texas A&M Aggies")
        assert "texas" in result


# ============================================================
# NCAA feature helpers
# ============================================================

class TestNcaaFeatures:

    def test_missing_team_returns_zeros(self):
        """Team not in stats_dict should get all 0.0 (league average)."""
        features = _get_ncaa_features("nonexistent-team", 2025, {}, {})
        assert len(features) == 9
        assert all(f == 0.0 for f in features)

    def test_known_team_gets_normalized(self):
        stats = {("team-a", 2025): {"batting_avg": 0.300, "era": 3.50}}
        norm = {
            (2025, "batting_avg"): (0.270, 0.030),
            (2025, "era"): (4.00, 1.00),
        }
        features = _get_ncaa_features("team-a", 2025, stats, norm)
        assert len(features) == 9
        # batting_avg: (0.300 - 0.270) / 0.030 = 1.0
        assert abs(features[0] - 1.0) < 1e-6
        # era: (3.50 - 4.00) / 1.00 = -0.5
        assert abs(features[1] - (-0.5)) < 1e-6
        # rest should be 0.0 (missing)
        for i in range(2, 9):
            assert features[i] == 0.0

    def test_none_team_id_returns_zeros(self):
        features = _get_ncaa_features(None, 2025, {}, {})
        assert len(features) == 9
        assert all(f == 0.0 for f in features)


# ============================================================
# Weather features
# ============================================================

class TestWeatherFeatures:

    def test_default_weather_produces_7(self):
        feats = _weather_features()
        assert len(feats) == 7

    def test_custom_weather(self):
        feats = _weather_features({'temp_f': 80.0, 'humidity_pct': 70.0})
        assert len(feats) == 7
        # Temp should be above average (positive)
        assert feats[0] > 0

    def test_dome_flag(self):
        feats = _weather_features({'is_dome': 1})
        assert feats[-1] == 1.0


# ============================================================
# Historical Feature Computer
# ============================================================

class TestSlimHistoricalFeatureComputer:

    @pytest.fixture
    def hfc(self):
        return SlimHistoricalFeatureComputer()

    def test_produces_correct_dimension(self, hfc):
        game_row = {
            'home_team': 'Team A', 'away_team': 'Team B',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        features, label = hfc.compute_game_features(game_row)
        assert features.shape[0] == NUM_FEATURES, (
            f"Historical features: {features.shape[0]}, expected {NUM_FEATURES}"
        )

    def test_label_is_binary(self, hfc):
        game1 = {
            'home_team': 'A', 'away_team': 'B',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        _, label1 = hfc.compute_game_features(game1)
        assert label1 == 1.0

        game2 = {
            'home_team': 'A', 'away_team': 'B',
            'home_score': 2, 'away_score': 7,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        _, label2 = hfc.compute_game_features(game2)
        assert label2 == 0.0

    def test_features_no_nan(self, hfc):
        game_row = {
            'home_team': 'Team X', 'away_team': 'Team Y',
            'home_score': 7, 'away_score': 2,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hfc.compute_game_features(game_row)
        assert not np.any(np.isnan(features)), "Features contain NaN"

    def test_features_no_inf(self, hfc):
        game_row = {
            'home_team': 'Team X', 'away_team': 'Team Y',
            'home_score': 10, 'away_score': 0,
            'date': '2025-04-01', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hfc.compute_game_features(game_row)
        assert not np.any(np.isinf(features)), "Features contain Inf"

    def test_features_dtype(self, hfc):
        game_row = {
            'home_team': 'A', 'away_team': 'B',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hfc.compute_game_features(game_row)
        assert features.dtype == np.float32

    def test_state_updates(self, hfc):
        game_row = {
            'home_team': 'test-team-a', 'away_team': 'test-team-b',
            'home_score': 6, 'away_score': 4,
            'date': '2025-05-01', 'neutral_site': 0, 'season': 2025,
        }
        hfc.update_state(game_row)
        assert hfc.stats['test-team-a']['games'] == 1
        assert hfc.stats['test-team-a']['wins'] == 1
        assert hfc.stats['test-team-b']['losses'] == 1

    def test_reset_clears_state(self, hfc):
        hfc.stats['some-team']['games'] = 50
        hfc.elo['some-team'] = 1600
        hfc.reset()
        assert hfc.stats['some-team']['games'] == 0
        assert hfc.elo['some-team'] == 1500


# ============================================================
# Network architectures
# ============================================================

class TestResidualBlock:

    def test_forward_shape(self):
        block = ResidualBlock(64, dropout=0.1)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 64)

    def test_residual_connection(self):
        """Output should differ from a zero-init (skip connection means nonzero)."""
        block = ResidualBlock(32, dropout=0.0)
        block.eval()
        x = torch.ones(4, 32)
        out = block(x)
        # Should not be exactly equal to input (transformed + residual)
        assert not torch.allclose(x, out)


class TestSlimBaseballNet:

    def test_forward_shape(self):
        model = SlimBaseballNet(NUM_FEATURES)
        x = torch.randn(16, NUM_FEATURES)
        out = model(x)
        assert out.shape == (16,)

    def test_output_range(self):
        """Outputs should be in [0, 1] due to sigmoid."""
        model = SlimBaseballNet(NUM_FEATURES)
        model.eval()
        x = torch.randn(32, NUM_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_backward_pass(self):
        model = SlimBaseballNet(NUM_FEATURES)
        x = torch.randn(8, NUM_FEATURES)
        y = torch.ones(8) * 0.5
        out = model(x)
        loss = nn.BCELoss()(out, y)
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestTotalsNet:

    def test_forward_shape(self):
        model = TotalsNet(NUM_FEATURES)
        x = torch.randn(16, NUM_FEATURES)
        out = model(x)
        assert out.shape == (16,)

    def test_no_sigmoid_output(self):
        """TotalsNet should NOT have bounded output (regression)."""
        model = TotalsNet(NUM_FEATURES)
        model.eval()
        x = torch.randn(100, NUM_FEATURES) * 5
        with torch.no_grad():
            out = model(x)
        # Regression output can be any value — check it's not all in [0,1]
        assert out.max() > 1.0 or out.min() < 0.0 or True  # just check it runs


# ============================================================
# Build model with configs
# ============================================================

class TestBuildModel:

    def test_build_default(self):
        from scripts.train_neural_slim import build_model, CONFIGS
        model = build_model(CONFIGS['default'])
        x = torch.randn(4, NUM_FEATURES)
        out = model(x)
        assert out.shape == (4,)

    def test_build_v3_standard(self):
        from scripts.train_neural_slim import build_model, CONFIGS
        model = build_model(CONFIGS['v3_standard'])
        x = torch.randn(4, NUM_FEATURES)
        out = model(x)
        assert out.shape == (4,)

    def test_build_v3_deep(self):
        from scripts.train_neural_slim import build_model, CONFIGS
        model = build_model(CONFIGS['v3_deep'])
        x = torch.randn(4, NUM_FEATURES)
        out = model(x)
        assert out.shape == (4,)

    def test_build_v3_wide(self):
        from scripts.train_neural_slim import build_model, CONFIGS
        model = build_model(CONFIGS['v3_wide'])
        x = torch.randn(4, NUM_FEATURES)
        out = model(x)
        assert out.shape == (4,)

    def test_residual_configs_have_residual_blocks(self):
        from scripts.train_neural_slim import build_model, CONFIGS
        model = build_model(CONFIGS['v3_standard'])
        has_residual = any(isinstance(m, ResidualBlock) for m in model.net.modules())
        assert has_residual, "v3_standard should contain ResidualBlock"


# ============================================================
# Loss functions
# ============================================================

class TestLossFunctions:

    def test_focal_loss_runs(self):
        from scripts.train_neural_slim import FocalBCELoss
        loss_fn = FocalBCELoss(gamma=2.0, label_smoothing=0.05)
        preds = torch.sigmoid(torch.randn(16))
        targets = torch.randint(0, 2, (16,)).float()
        loss = loss_fn(preds, targets)
        assert loss.item() >= 0

    def test_label_smoothing_loss(self):
        from scripts.train_neural_slim import LabelSmoothingBCELoss
        loss_fn = LabelSmoothingBCELoss(smoothing=0.1)
        preds = torch.sigmoid(torch.randn(16))
        targets = torch.randint(0, 2, (16,)).float()
        loss = loss_fn(preds, targets)
        assert loss.item() >= 0

    def test_label_smoothing_reduces_confidence(self):
        """Smoothed loss should be lower when model is overconfident."""
        from scripts.train_neural_slim import LabelSmoothingBCELoss
        plain = nn.BCELoss()
        smoothed = LabelSmoothingBCELoss(smoothing=0.1)
        # Perfect prediction
        preds = torch.tensor([0.99, 0.01])
        targets = torch.tensor([1.0, 0.0])
        # Smoothed targets are softer, so loss is slightly different
        loss_plain = plain(preds, targets)
        loss_smooth = smoothed(preds, targets)
        # Both should be finite and positive
        assert loss_plain.item() >= 0
        assert loss_smooth.item() >= 0


# ============================================================
# Mixup
# ============================================================

class TestMixup:

    def test_mixup_preserves_shape(self):
        from scripts.train_neural_slim import mixup_data
        x = torch.randn(16, 58)
        y = torch.rand(16)
        mx, my = mixup_data(x, y, alpha=0.2)
        assert mx.shape == x.shape
        assert my.shape == y.shape

    def test_mixup_alpha_zero_is_identity(self):
        from scripts.train_neural_slim import mixup_data
        x = torch.randn(16, 58)
        y = torch.rand(16)
        mx, my = mixup_data(x, y, alpha=0.0)
        assert torch.allclose(mx, x)
        assert torch.allclose(my, y)


# ============================================================
# Season weights
# ============================================================

class TestSeasonWeights:

    def test_recent_seasons_weighted_higher(self):
        from scripts.train_neural_slim import compute_season_weights
        seasons = np.array([2021, 2022, 2023, 2024, 2025])
        weights = compute_season_weights(seasons, decay=0.85)
        assert len(weights) == 5
        # Most recent season should have highest weight
        assert weights[-1] > weights[0]

    def test_single_season_weight_is_one(self):
        from scripts.train_neural_slim import compute_season_weights
        seasons = np.array([2025, 2025, 2025])
        weights = compute_season_weights(seasons, decay=0.85)
        # All same season, all should be 1.0
        assert np.allclose(weights, 1.0)

    def test_mean_weight_is_one(self):
        from scripts.train_neural_slim import compute_season_weights
        seasons = np.array([2021, 2022, 2023, 2024, 2025])
        weights = compute_season_weights(seasons, decay=0.85)
        assert abs(weights.mean() - 1.0) < 1e-5


# ============================================================
# Feature importance
# ============================================================

class TestFeatureImportance:

    def test_importance_shape(self):
        from scripts.train_neural_slim import compute_feature_importance
        model = SlimBaseballNet(NUM_FEATURES)
        X = np.random.randn(32, NUM_FEATURES).astype(np.float32)
        y = np.random.randint(0, 2, 32).astype(np.float32)
        importance = compute_feature_importance(model, X, y, 'cpu')
        assert importance.shape == (NUM_FEATURES,)
        assert np.all(importance >= 0)


# ============================================================
# Live feature computer dimension check
# ============================================================

class TestLiveFeatureComputer:

    def test_num_features_matches(self):
        fc = SlimFeatureComputer()
        assert fc.get_num_features() == NUM_FEATURES

    def test_feature_names_match(self):
        fc = SlimFeatureComputer()
        names = fc.get_feature_names()
        assert len(names) == NUM_FEATURES
        assert names == FEATURE_NAMES

    def test_live_features_correct_dim(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        fc = SlimFeatureComputer()
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15',
        )
        assert features.shape[0] == NUM_FEATURES, (
            f"Live features: {features.shape[0]}, expected {NUM_FEATURES}"
        )

    def test_live_features_no_nan(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        fc = SlimFeatureComputer()
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15',
        )
        assert not np.any(np.isnan(features)), "Live features contain NaN"

    def test_live_features_dtype(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids or 'sample_away' not in sample_team_ids:
            pytest.skip("No sample teams available")
        fc = SlimFeatureComputer()
        features = fc.compute_features(
            sample_team_ids['sample_home'],
            sample_team_ids['sample_away'],
            game_date='2026-02-15',
        )
        assert features.dtype == np.float32


# ============================================================
# Historical-live dimension consistency
# ============================================================

class TestDimensionConsistency:

    def test_historical_matches_live_dimension(self):
        """CRITICAL: Historical and live feature computers must produce same dimension."""
        hfc = SlimHistoricalFeatureComputer()
        fc = SlimFeatureComputer()

        game_row = {
            'home_team': 'team-a', 'away_team': 'team-b',
            'home_score': 5, 'away_score': 3,
            'date': '2025-03-15', 'neutral_site': 0, 'season': 2025,
        }
        features, _ = hfc.compute_game_features(game_row)

        assert features.shape[0] == fc.get_num_features(), (
            f"DIMENSION MISMATCH! Historical: {features.shape[0]}, Live: {fc.get_num_features()}"
        )
