#!/usr/bin/env python3
"""Tests for calibration: Platt scaling, isotonic regression, and raw prob separation."""

import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.update_model_calibration import (
    logit,
    sigmoid,
    fit_for_model,
    ensure_table,
    ensure_raw_prob_column,
    _brier_score,
)
from scripts.predict_and_track import (
    _apply_calibration,
    _apply_platt,
    _apply_isotonic,
    _load_calibration_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Create a temporary in-memory–like SQLite database with the needed schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_home_prob REAL,
            raw_home_prob REAL,
            predicted_home_runs REAL,
            predicted_away_runs REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            was_correct INTEGER,
            UNIQUE(game_id, model_name)
        )
    """)
    ensure_table(cur)
    conn.commit()
    return conn, cur


def _insert_predictions(cur, model_name, probs, labels, use_raw=True):
    """Insert synthetic predictions into the test database."""
    for i, (p, y) in enumerate(zip(probs, labels)):
        raw = p if use_raw else None
        cur.execute(
            """INSERT INTO model_predictions
               (game_id, model_name, predicted_home_prob, raw_home_prob, was_correct, predicted_at)
               VALUES (?, ?, ?, ?, ?, datetime('now', ?))""",
            (f"game_{i}", model_name, p, raw, int(y), f"-{i} seconds"),
        )


# ---------------------------------------------------------------------------
# Platt roundtrip
# ---------------------------------------------------------------------------

class TestPlattRoundtrip:
    def test_platt_fit_and_apply(self, tmp_path):
        """Fit Platt on synthetic data, then apply — output should be closer to actual rate."""
        conn, cur = _make_db(tmp_path)

        np.random.seed(42)
        n = 300
        # Generate predictions that are systematically overconfident
        true_probs = np.random.uniform(0.3, 0.7, n)
        # Stretch them to be overconfident
        raw_probs = 0.5 + (true_probs - 0.5) * 2.0
        raw_probs = np.clip(raw_probs, 0.05, 0.95)
        labels = (np.random.rand(n) < true_probs).astype(int)

        _insert_predictions(cur, "test_platt", raw_probs.tolist(), labels.tolist())
        conn.commit()

        result = fit_for_model(cur, "test_platt")
        assert result is not None
        assert result["method"] in ("platt", "isotonic")
        assert result["n"] == n

        # Apply calibration
        if result["method"] == "platt":
            params = {"method": "platt", "a": result["a"], "b": result["b"], "n": n}
        else:
            iso_data = json.loads(result["isotonic_json"])
            params = {"method": "isotonic", "x": iso_data["x"], "y": iso_data["y"], "n": n}

        calibrated = [_apply_calibration(p, params) for p in raw_probs]

        # Brier score should improve
        raw_brier = float(np.mean((raw_probs - labels) ** 2))
        cal_brier = float(np.mean((np.array(calibrated) - labels) ** 2))
        assert cal_brier <= raw_brier, f"Calibrated Brier {cal_brier:.4f} > raw {raw_brier:.4f}"

        conn.close()

    def test_platt_apply_identity(self):
        """With a=1.0, b=0.0, Platt should be near identity."""
        params = {"method": "platt", "a": 1.0, "b": 0.0, "n": 200}
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = _apply_calibration(p, params)
            assert abs(result - p) < 0.01, f"p={p}, result={result}"

    def test_platt_clamps_output(self):
        """Output should be clamped to [0.001, 0.999]."""
        # Very extreme a pushing toward 0 or 1
        params = {"method": "platt", "a": 10.0, "b": 0.0, "n": 200}
        assert _apply_calibration(0.001, params) >= 0.001
        assert _apply_calibration(0.999, params) <= 0.999


# ---------------------------------------------------------------------------
# Isotonic roundtrip
# ---------------------------------------------------------------------------

class TestIsotonicRoundtrip:
    def test_isotonic_fit_and_apply(self, tmp_path):
        """Fit isotonic on data with non-monotone raw probs, verify improvement."""
        conn, cur = _make_db(tmp_path)

        np.random.seed(123)
        n = 300
        # Probabilities that are poorly calibrated
        raw_probs = np.random.uniform(0.1, 0.9, n)
        # Actual outcomes: home wins more often than predicted in low range
        true_probs = 0.3 + 0.4 * raw_probs  # compressed toward 0.5
        labels = (np.random.rand(n) < true_probs).astype(int)

        _insert_predictions(cur, "test_iso", raw_probs.tolist(), labels.tolist())
        conn.commit()

        result = fit_for_model(cur, "test_iso")
        assert result is not None
        assert result["n"] == n

        # Apply whichever method was chosen
        if result["method"] == "isotonic":
            iso_data = json.loads(result["isotonic_json"])
            params = {"method": "isotonic", "x": iso_data["x"], "y": iso_data["y"], "n": n}
        else:
            params = {"method": "platt", "a": result["a"], "b": result["b"], "n": n}

        calibrated = [_apply_calibration(p, params) for p in raw_probs]

        raw_brier = float(np.mean((raw_probs - labels) ** 2))
        cal_brier = float(np.mean((np.array(calibrated) - labels) ** 2))
        assert cal_brier <= raw_brier

        conn.close()

    def test_isotonic_apply_interpolates(self):
        """Isotonic apply should linearly interpolate between breakpoints."""
        params = {
            "method": "isotonic",
            "x": [0.1, 0.3, 0.5, 0.7, 0.9],
            "y": [0.2, 0.35, 0.5, 0.65, 0.8],
            "n": 200,
        }
        # At breakpoints
        assert abs(_apply_isotonic(0.1, params) - 0.2) < 1e-6
        assert abs(_apply_isotonic(0.9, params) - 0.8) < 1e-6

        # Between breakpoints: linear interpolation
        result = _apply_isotonic(0.2, params)
        expected = 0.2 + (0.35 - 0.2) * (0.2 - 0.1) / (0.3 - 0.1)
        assert abs(result - expected) < 1e-6

    def test_isotonic_clamps_edges(self):
        """Values outside the breakpoint range should clamp."""
        params = {
            "method": "isotonic",
            "x": [0.2, 0.8],
            "y": [0.3, 0.7],
            "n": 200,
        }
        # Below min
        assert _apply_isotonic(0.05, params) >= 0.001
        assert abs(_apply_isotonic(0.05, params) - 0.3) < 1e-6
        # Above max
        assert _apply_isotonic(0.95, params) <= 0.999
        assert abs(_apply_isotonic(0.95, params) - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Calibrated probs are closer to actual win rates
# ---------------------------------------------------------------------------

class TestCalibrationImprovesFit:
    def test_calibrated_closer_to_actual(self, tmp_path):
        """Across reliability bins, calibrated probs should be closer to actual win rates."""
        conn, cur = _make_db(tmp_path)

        np.random.seed(77)
        n = 500
        # Systematically biased predictions
        true_probs = np.random.uniform(0.2, 0.8, n)
        # Raw predictions are biased high
        raw_probs = np.clip(true_probs + 0.15, 0.05, 0.95)
        labels = (np.random.rand(n) < true_probs).astype(int)

        _insert_predictions(cur, "test_biased", raw_probs.tolist(), labels.tolist())
        conn.commit()

        result = fit_for_model(cur, "test_biased")
        assert result is not None

        if result["method"] == "isotonic":
            iso_data = json.loads(result["isotonic_json"])
            params = {"method": "isotonic", "x": iso_data["x"], "y": iso_data["y"], "n": n}
        else:
            params = {"method": "platt", "a": result["a"], "b": result["b"], "n": n}

        calibrated = np.array([_apply_calibration(p, params) for p in raw_probs])

        # Compare bin-level error
        n_bins = 5
        raw_bin_error = 0.0
        cal_bin_error = 0.0
        for i in range(n_bins):
            lo = i / n_bins
            hi = (i + 1) / n_bins
            mask = (raw_probs >= lo) & (raw_probs < hi)
            if mask.sum() == 0:
                continue
            actual_rate = labels[mask].mean()
            raw_bin_error += abs(raw_probs[mask].mean() - actual_rate) * mask.sum()
            cal_bin_error += abs(calibrated[mask].mean() - actual_rate) * mask.sum()

        assert cal_bin_error < raw_bin_error, \
            f"Calibrated bin error {cal_bin_error:.4f} >= raw {raw_bin_error:.4f}"

        conn.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_same_prediction(self, tmp_path):
        """If all predictions are the same value, calibration still doesn't crash."""
        conn, cur = _make_db(tmp_path)
        n = 200
        probs = [0.6] * n
        labels = [1] * 120 + [0] * 80

        _insert_predictions(cur, "test_same", probs, labels)
        conn.commit()

        # logit(0.6) is the same for all — LogisticRegression should still fit
        # (single feature value is degenerate for LR but sklearn handles it)
        result = fit_for_model(cur, "test_same")
        # May return None or a result; either is acceptable
        if result is not None:
            assert result["n"] == n
        conn.close()

    def test_degenerate_labels_all_wins(self, tmp_path):
        """All labels are 1 — should return None (degenerate)."""
        conn, cur = _make_db(tmp_path)
        n = 200
        probs = list(np.random.uniform(0.3, 0.9, n))
        labels = [1] * n

        _insert_predictions(cur, "test_all_wins", probs, labels)
        conn.commit()

        result = fit_for_model(cur, "test_all_wins")
        assert result is None
        conn.close()

    def test_degenerate_labels_all_losses(self, tmp_path):
        """All labels are 0 — should return None (degenerate)."""
        conn, cur = _make_db(tmp_path)
        n = 200
        probs = list(np.random.uniform(0.1, 0.7, n))
        labels = [0] * n

        _insert_predictions(cur, "test_all_losses", probs, labels)
        conn.commit()

        result = fit_for_model(cur, "test_all_losses")
        assert result is None
        conn.close()

    def test_too_few_samples(self, tmp_path):
        """Below MIN_SAMPLES, should return None."""
        conn, cur = _make_db(tmp_path)
        n = 50  # below MIN_SAMPLES=120
        probs = list(np.random.uniform(0.2, 0.8, n))
        labels = [1 if p > 0.5 else 0 for p in probs]

        _insert_predictions(cur, "test_few", probs, labels)
        conn.commit()

        result = fit_for_model(cur, "test_few")
        assert result is None
        conn.close()

    def test_apply_calibration_none_params(self):
        """If params is None, return original probability."""
        assert _apply_calibration(0.65, None) == 0.65

    def test_apply_calibration_low_n(self):
        """If n < 120, return original probability."""
        params = {"method": "platt", "a": 2.0, "b": 1.0, "n": 50}
        assert _apply_calibration(0.65, params) == 0.65

    def test_isotonic_empty_breakpoints(self):
        """Empty isotonic breakpoints should return input."""
        params = {"method": "isotonic", "x": [], "y": [], "n": 200}
        assert _apply_isotonic(0.5, params) == 0.5


# ---------------------------------------------------------------------------
# Calibration uses raw probs (not re-calibrated)
# ---------------------------------------------------------------------------

class TestRawProbSeparation:
    def test_fit_uses_raw_prob_when_available(self, tmp_path):
        """Calibration fitter should use raw_home_prob, not predicted_home_prob."""
        conn, cur = _make_db(tmp_path)

        np.random.seed(99)
        n = 200

        # raw probs are reasonable
        raw_probs = np.random.uniform(0.3, 0.7, n)
        labels = (np.random.rand(n) < raw_probs).astype(int)

        # predicted probs are all squished to 0.5 (as if previously calibrated badly)
        cal_probs = [0.5] * n

        for i in range(n):
            cur.execute(
                """INSERT INTO model_predictions
                   (game_id, model_name, predicted_home_prob, raw_home_prob, was_correct, predicted_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now', ?))""",
                (f"game_{i}", "test_raw", cal_probs[i], float(raw_probs[i]), int(labels[i]), f"-{i} seconds"),
            )
        conn.commit()

        result = fit_for_model(cur, "test_raw")
        assert result is not None

        # If it used predicted_home_prob (all 0.5), the model would have no signal.
        # Since it should use raw_home_prob, it should find a meaningful fit.
        # For Platt: a should be meaningfully different from 0
        if result["method"] == "platt":
            assert abs(result["a"]) > 0.1, f"a={result['a']}, expected meaningful coefficient"

        conn.close()


# ---------------------------------------------------------------------------
# Load calibration params
# ---------------------------------------------------------------------------

class TestLoadCalibrationParams:
    def test_load_platt(self, tmp_path):
        """Load Platt params from database."""
        conn, cur = _make_db(tmp_path)
        cur.execute(
            """INSERT INTO model_calibration (model_name, a, b, n_samples, updated_at, method)
               VALUES ('test', 1.5, -0.3, 300, '2025-01-01', 'platt')"""
        )
        conn.commit()

        params = _load_calibration_params(cur)
        assert "test" in params
        assert params["test"]["method"] == "platt"
        assert abs(params["test"]["a"] - 1.5) < 1e-6
        assert abs(params["test"]["b"] - (-0.3)) < 1e-6
        conn.close()

    def test_load_isotonic(self, tmp_path):
        """Load isotonic params from database."""
        conn, cur = _make_db(tmp_path)
        iso_json = json.dumps({"x": [0.1, 0.5, 0.9], "y": [0.2, 0.5, 0.8]})
        cur.execute(
            """INSERT INTO model_calibration (model_name, a, b, n_samples, updated_at, method, isotonic_json)
               VALUES ('test_iso', 1.0, 0.0, 300, '2025-01-01', 'isotonic', ?)""",
            (iso_json,),
        )
        conn.commit()

        params = _load_calibration_params(cur)
        assert "test_iso" in params
        assert params["test_iso"]["method"] == "isotonic"
        assert params["test_iso"]["x"] == [0.1, 0.5, 0.9]
        assert params["test_iso"]["y"] == [0.2, 0.5, 0.8]
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_logit_sigmoid_inverse(self):
        """logit and sigmoid should be inverses."""
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            assert abs(sigmoid(logit(p)) - p) < 1e-6

    def test_logit_clamps(self):
        """logit should clamp extreme inputs."""
        # Should not raise for 0.0 or 1.0
        logit(0.0)
        logit(1.0)

    def test_brier_score_perfect(self):
        """Perfect predictions should have Brier score of 0."""
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        labels = np.array([1, 0, 1, 0])
        assert _brier_score(probs, labels) == 0.0

    def test_brier_score_worst(self):
        """Worst predictions should have Brier score of 1."""
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        labels = np.array([1, 0, 1, 0])
        assert abs(_brier_score(probs, labels) - 1.0) < 1e-6
