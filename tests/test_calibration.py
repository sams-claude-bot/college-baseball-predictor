#!/usr/bin/env python3
"""Tests for probability calibration module."""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.calibration import Calibrator, PROB_FLOOR, PROB_CEILING


class TestCalibrator:
    """Tests for the Calibrator class."""

    def test_calibrate_returns_valid_range(self):
        """Calibrated probabilities stay within [PROB_FLOOR, PROB_CEILING]."""
        cal = Calibrator()
        # Without a fitted model, should return raw prob
        assert PROB_FLOOR <= cal.calibrate(0.5) <= PROB_CEILING
        assert PROB_FLOOR <= cal.calibrate(0.99) <= PROB_CEILING
        assert PROB_FLOOR <= cal.calibrate(0.01) <= PROB_CEILING

    def test_calibrate_unfitted_returns_raw(self):
        """Without fitting and no saved model, calibrate returns the input probability."""
        cal = Calibrator()
        cal._loaded = True  # Pretend we tried loading but found nothing
        cal.global_calibrator = None
        cal.calibrators = {}
        assert cal.calibrate(0.75) == 0.75
        assert cal.calibrate(0.60) == 0.60

    def test_kelly_calibration_is_more_conservative(self):
        """Kelly calibration should shrink probabilities toward 0.5."""
        cal = Calibrator()
        # Even without fitting, the shrinkage should apply
        raw = 0.80
        kelly = cal.calibrate_for_kelly(raw)
        assert kelly < raw  # Shrunk toward 0.5
        assert kelly > 0.5  # But still above 0.5

    def test_kelly_shrinkage_direction(self):
        """Kelly shrinkage should always move probability toward 0.5."""
        cal = Calibrator()
        cal._loaded = True
        cal.global_calibrator = None
        cal.calibrators = {}
        # With no calibrator, only shrinkage applies
        high = cal.calibrate_for_kelly(0.80)
        low = cal.calibrate_for_kelly(0.20)
        assert high < 0.80  # Shrunk down toward 0.5
        assert low > 0.20   # Shrunk up toward 0.5
        assert high > 0.5   # Still on the right side
        assert low < 0.5    # Still on the right side

    def test_fit_with_synthetic_data(self):
        """Calibrator can fit on synthetic overconfident data."""
        from sklearn.isotonic import IsotonicRegression

        cal = Calibrator()

        # Simulate overconfident model: predicts 80% but wins only 60%
        np.random.seed(42)
        n = 200
        raw_probs = np.random.uniform(0.55, 0.95, n)
        # Actual outcomes are less extreme than predicted
        shrunk = 0.5 + (raw_probs - 0.5) * 0.6  # Shrink toward 0.5
        outcomes = (np.random.random(n) < shrunk).astype(int)

        iso = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEILING,
                                  out_of_bounds='clip')
        iso.fit(raw_probs, outcomes)
        iso._method = 'isotonic'
        cal.global_calibrator = iso
        cal._loaded = True

        # 85% raw should calibrate to something lower
        calibrated = cal.calibrate(0.85)
        assert calibrated < 0.85
        assert calibrated > 0.5

    def test_calibration_table_format(self):
        """calibration_table returns expected format."""
        cal = Calibrator()
        table = cal.calibration_table()
        assert len(table) > 0
        for row in table:
            assert 'raw' in row
            assert 'calibrated' in row
            assert 'kelly' in row
            assert 0 <= row['raw'] <= 1
            assert 0 <= row['calibrated'] <= 1
            assert 0 <= row['kelly'] <= 1

    def test_calibration_stats_keys(self):
        """_calibration_stats returns expected keys."""
        cal = Calibrator()
        probs = np.array([0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([1, 1, 0, 1, 0, 1, 1, 0])

        # Use identity calibrator (no-op)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEILING,
                                  out_of_bounds='clip')
        iso.fit(probs, outcomes)
        iso._method = 'isotonic'

        stats = cal._calibration_stats(probs, outcomes, iso)
        assert 'brier_raw' in stats
        assert 'brier_calibrated' in stats
        assert 'brier_improvement' in stats
        assert 'ece_raw' in stats
        assert 'ece_calibrated' in stats
        assert 'bins' in stats


class TestKellyWithCalibration:
    """Test that calibrated probs produce smaller Kelly fractions."""

    def test_calibrated_kelly_bets_less(self):
        """With overconfident probs, calibrated Kelly should suggest smaller bets."""
        from scripts.betting.risk import raw_kelly_fraction

        # Scenario: model says 82%, true calibrated is ~73%
        raw_prob = 0.82
        calibrated_prob = 0.73
        odds = -175  # Moderate favorite

        kelly_raw = raw_kelly_fraction(raw_prob, odds)
        kelly_cal = raw_kelly_fraction(calibrated_prob, odds)

        assert kelly_cal < kelly_raw
        assert kelly_cal > 0  # Should still want to bet

    def test_calibration_eliminates_phantom_edge(self):
        """When model prob barely exceeds implied, calibration may eliminate the bet."""
        from scripts.betting.risk import raw_kelly_fraction

        # Scenario: model says 62%, implied is 60%, but calibrated is 58.8%
        raw_prob = 0.62
        calibrated_prob = 0.588  # Below implied
        odds = -150  # Implied ~60%

        kelly_raw = raw_kelly_fraction(raw_prob, odds)
        kelly_cal = raw_kelly_fraction(calibrated_prob, odds)

        assert kelly_raw > 0  # Raw thinks there's edge
        assert kelly_cal == 0  # Calibrated says no edge

    def test_high_confidence_still_bets(self):
        """Even with calibration, very strong edges should still bet."""
        from scripts.betting.risk import raw_kelly_fraction

        # Scenario: model says 90%, calibrated 75%, odds +120 (implied 45%)
        calibrated_prob = 0.75
        odds = 120

        kelly = raw_kelly_fraction(calibrated_prob, odds)
        assert kelly > 0.15  # Should be a substantial bet
