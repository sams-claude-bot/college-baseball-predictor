"""Tests for the 3-game series probability calculator."""

import pytest
from web.services.series_probability import compute_series_probs


class TestCoinFlip:
    """p = 0.5: perfectly symmetric."""

    def test_coin_flip_symmetry(self):
        result = compute_series_probs(0.5)
        assert result['home_win_1plus'] == pytest.approx(result['away_win_1plus'])
        assert result['home_win_2plus'] == pytest.approx(result['away_win_2plus'])
        assert result['home_sweep'] == pytest.approx(result['away_sweep'])

    def test_coin_flip_values(self):
        result = compute_series_probs(0.5)
        assert result['home_win_1plus'] == pytest.approx(0.875)
        assert result['home_win_2plus'] == pytest.approx(0.5)
        assert result['home_sweep'] == pytest.approx(0.125)


class TestEdgeCases:
    """p = 0 and p = 1: deterministic outcomes."""

    def test_p_zero(self):
        result = compute_series_probs(0.0)
        assert result['home_win_1plus'] == pytest.approx(0.0)
        assert result['home_win_2plus'] == pytest.approx(0.0)
        assert result['home_sweep'] == pytest.approx(0.0)
        assert result['away_win_1plus'] == pytest.approx(1.0)
        assert result['away_win_2plus'] == pytest.approx(1.0)
        assert result['away_sweep'] == pytest.approx(1.0)

    def test_p_one(self):
        result = compute_series_probs(1.0)
        assert result['home_win_1plus'] == pytest.approx(1.0)
        assert result['home_win_2plus'] == pytest.approx(1.0)
        assert result['home_sweep'] == pytest.approx(1.0)
        assert result['away_win_1plus'] == pytest.approx(0.0)
        assert result['away_win_2plus'] == pytest.approx(0.0)
        assert result['away_sweep'] == pytest.approx(0.0)


class TestKnownValues:
    """p = 0.6: hand-calculated reference values."""

    def test_p_06_sweep(self):
        result = compute_series_probs(0.6)
        assert result['home_sweep'] == pytest.approx(0.216)

    def test_p_06_win_2plus(self):
        result = compute_series_probs(0.6)
        assert result['home_win_2plus'] == pytest.approx(0.648)

    def test_p_06_win_1plus(self):
        result = compute_series_probs(0.6)
        assert result['home_win_1plus'] == pytest.approx(0.936)


class TestSymmetry:
    """Verify home + away probabilities are consistent."""

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0])
    def test_sweep_sums_to_less_than_one(self, p):
        result = compute_series_probs(p)
        # Sweeps can't sum to > 1 (only one team can sweep)
        assert result['home_sweep'] + result['away_sweep'] <= 1.0 + 1e-9

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0])
    def test_win_1plus_relationship(self, p):
        result = compute_series_probs(p)
        # P(home wins 1+) = 1 - P(away sweep)
        assert result['home_win_1plus'] == pytest.approx(1.0 - result['away_sweep'])
        assert result['away_win_1plus'] == pytest.approx(1.0 - result['home_sweep'])

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0])
    def test_ordering(self, p):
        result = compute_series_probs(p)
        # sweep <= win 2+ <= win 1+ (always)
        assert result['home_sweep'] <= result['home_win_2plus'] + 1e-9
        assert result['home_win_2plus'] <= result['home_win_1plus'] + 1e-9
        assert result['away_sweep'] <= result['away_win_2plus'] + 1e-9
        assert result['away_win_2plus'] <= result['away_win_1plus'] + 1e-9


class TestClamping:
    """Probabilities outside [0, 1] should be clamped."""

    def test_negative_prob(self):
        result = compute_series_probs(-0.5)
        # Should behave like p=0
        assert result['home_sweep'] == pytest.approx(0.0)
        assert result['away_sweep'] == pytest.approx(1.0)

    def test_over_one_prob(self):
        result = compute_series_probs(1.5)
        # Should behave like p=1
        assert result['home_sweep'] == pytest.approx(1.0)
        assert result['away_sweep'] == pytest.approx(0.0)
