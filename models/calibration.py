#!/usr/bin/env python3
"""
Probability Calibration Module (Experimental)

Provides isotonic regression calibration for model probabilities.
Calibrated probabilities map model confidence to actual win rates,
fixing overconfidence that causes Kelly sizing to overbet.

Usage:
    from models.calibration import Calibrator

    cal = Calibrator()
    cal.fit()  # Train on all graded predictions
    
    # Calibrate a single probability
    raw_prob = 0.82
    calibrated = cal.calibrate(raw_prob, model='meta_ensemble')
    
    # Calibrate for Kelly sizing specifically
    kelly_prob = cal.calibrate_for_kelly(raw_prob, model='meta_ensemble')

Design principles:
    - Separate from production prediction pipeline (experimental)
    - Uses isotonic regression (monotonic, non-parametric)
    - Falls back to Platt scaling when sample size < 50
    - Never returns probabilities outside [0.05, 0.95] (avoid extremes)
    - Saves calibration curves to disk for inspection
"""

import json
import pickle
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
CALIBRATION_PATH = DATA_DIR / 'calibration_models.pkl'
CALIBRATION_REPORT_PATH = DATA_DIR / 'calibration_report.json'

# Probability clamp — never go below 5% or above 95% even after calibration
PROB_FLOOR = 0.05
PROB_CEILING = 0.95

# Minimum graded predictions to fit a model-specific calibrator
MIN_SAMPLES = 30


class Calibrator:
    """Isotonic regression probability calibrator."""

    def __init__(self):
        self.calibrators: Dict[str, object] = {}
        self.global_calibrator = None
        self.report: Dict = {}
        self._loaded = False

    def _get_connection(self):
        conn = sqlite3.connect(str(DATA_DIR / 'baseball.db'), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def fit(self, min_samples: int = MIN_SAMPLES) -> dict:
        """Train calibration curves from graded predictions.
        
        Fits:
        1. Per-model calibrators (where sample size >= min_samples)
        2. Global calibrator (all models pooled)
        3. 'consensus' calibrator (for avg_prob from confident bets)
        
        Returns report dict with calibration stats.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        conn = self._get_connection()

        # Get all graded predictions with model probabilities
        rows = conn.execute("""
            SELECT model_name, predicted_home_prob, was_correct,
                   g.home_team_id, g.away_team_id
            FROM model_predictions mp
            JOIN games g ON mp.game_id = g.id
            WHERE mp.was_correct IS NOT NULL
              AND mp.predicted_home_prob IS NOT NULL
        """).fetchall()
        conn.close()

        if len(rows) < min_samples:
            print(f"[Calibrator] Only {len(rows)} graded predictions, need {min_samples}")
            return {}

        # Group by model
        model_data = defaultdict(lambda: {'probs': [], 'outcomes': []})
        all_probs = []
        all_outcomes = []

        for r in rows:
            model = r['model_name']
            prob = r['predicted_home_prob']
            outcome = r['was_correct']

            # For calibration, we need the probability of the *predicted* outcome
            # was_correct=1 means the prediction was right
            # predicted_home_prob > 0.5 means we predicted home win
            # So: prob_of_predicted_outcome = prob if prob > 0.5, else (1-prob)
            if prob >= 0.5:
                pred_prob = prob
            else:
                pred_prob = 1.0 - prob

            model_data[model]['probs'].append(pred_prob)
            model_data[model]['outcomes'].append(outcome)
            all_probs.append(pred_prob)
            all_outcomes.append(outcome)

        all_probs = np.array(all_probs)
        all_outcomes = np.array(all_outcomes)

        report = {
            'fit_date': datetime.utcnow().isoformat(),
            'total_predictions': len(all_probs),
            'models': {},
        }

        # Fit per-model calibrators
        for model, data in model_data.items():
            probs = np.array(data['probs'])
            outcomes = np.array(data['outcomes'])

            if len(probs) < min_samples:
                report['models'][model] = {
                    'status': 'skipped',
                    'n': len(probs),
                    'reason': f'< {min_samples} samples',
                }
                continue

            cal = self._fit_one(probs, outcomes, model)
            self.calibrators[model] = cal

            # Compute calibration stats
            stats = self._calibration_stats(probs, outcomes, cal)
            report['models'][model] = {
                'status': 'fitted',
                'n': len(probs),
                **stats,
            }

        # Fit global calibrator
        global_cal = self._fit_one(all_probs, all_outcomes, 'global')
        self.global_calibrator = global_cal
        global_stats = self._calibration_stats(all_probs, all_outcomes, global_cal)
        report['global'] = {
            'n': len(all_probs),
            **global_stats,
        }

        # Save
        self._save()
        self.report = report
        self._save_report(report)
        self._loaded = True

        return report

    def _fit_one(self, probs, outcomes, label):
        """Fit a single calibrator. Uses isotonic if enough data, else Platt."""
        from sklearn.isotonic import IsotonicRegression

        if len(probs) >= 50:
            cal = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEILING,
                                     out_of_bounds='clip')
            cal.fit(probs, outcomes)
            cal._method = 'isotonic'
        else:
            # Platt scaling fallback
            from sklearn.linear_model import LogisticRegression as LR
            lr = LR(C=1.0, max_iter=1000)
            lr.fit(probs.reshape(-1, 1), outcomes)
            cal = lr
            cal._method = 'platt'

        return cal

    def _calibrate_one(self, prob: float, calibrator) -> float:
        """Apply a fitted calibrator to a single probability."""
        if calibrator is None:
            return prob

        if getattr(calibrator, '_method', '') == 'isotonic':
            result = float(calibrator.predict([prob])[0])
        else:
            # Platt
            result = float(calibrator.predict_proba([[prob]])[0, 1])

        return max(PROB_FLOOR, min(PROB_CEILING, result))

    def calibrate(self, prob: float, model: str = None) -> float:
        """Calibrate a raw model probability.
        
        Uses model-specific calibrator if available, else global.
        Input prob should be the probability of the predicted outcome
        (i.e., > 0.5 for the favored side).
        """
        if not self._loaded:
            self._load()

        if model and model in self.calibrators:
            return self._calibrate_one(prob, self.calibrators[model])
        elif self.global_calibrator is not None:
            return self._calibrate_one(prob, self.global_calibrator)
        else:
            return prob

    def calibrate_for_kelly(self, prob: float, model: str = None) -> float:
        """Calibrate probability specifically for Kelly sizing.
        
        More conservative than regular calibration:
        - Uses calibrated probability
        - Applies an additional shrinkage toward 0.5 (regularization)
        - This prevents Kelly from ever seeing extreme edges
        """
        cal_prob = self.calibrate(prob, model)
        # Shrink 10% toward 0.5 — extra conservatism for bankroll management
        shrinkage = 0.10
        return cal_prob * (1 - shrinkage) + 0.5 * shrinkage

    def calibration_table(self, model: str = None) -> list:
        """Return a table of raw prob → calibrated prob for inspection."""
        table = []
        for raw in np.arange(0.50, 0.96, 0.05):
            cal = self.calibrate(float(raw), model)
            kelly_cal = self.calibrate_for_kelly(float(raw), model)
            table.append({
                'raw': round(float(raw), 2),
                'calibrated': round(cal, 3),
                'kelly': round(kelly_cal, 3),
            })
        return table

    def _calibration_stats(self, probs, outcomes, calibrator) -> dict:
        """Compute calibration metrics."""
        cal_probs = np.array([self._calibrate_one(p, calibrator) for p in probs])

        # Brier score (lower is better)
        brier_raw = np.mean((probs - outcomes) ** 2)
        brier_cal = np.mean((cal_probs - outcomes) ** 2)

        # Bin-level calibration error
        bins = np.arange(0.5, 1.0, 0.1)
        ece_raw = 0
        ece_cal = 0
        bin_details = []

        for i in range(len(bins)):
            lo = bins[i]
            hi = bins[i + 1] if i + 1 < len(bins) else 1.01
            mask = (probs >= lo) & (probs < hi)
            n = mask.sum()
            if n == 0:
                continue

            raw_mean = probs[mask].mean()
            cal_mean = cal_probs[mask].mean()
            actual_mean = outcomes[mask].mean()

            ece_raw += abs(raw_mean - actual_mean) * n
            ece_cal += abs(cal_mean - actual_mean) * n

            bin_details.append({
                'range': f'{lo:.0%}-{hi:.0%}',
                'n': int(n),
                'raw_pred': round(float(raw_mean), 3),
                'cal_pred': round(float(cal_mean), 3),
                'actual': round(float(actual_mean), 3),
            })

        total = len(probs)
        ece_raw /= total
        ece_cal /= total

        return {
            'brier_raw': round(float(brier_raw), 4),
            'brier_calibrated': round(float(brier_cal), 4),
            'brier_improvement': round(float(brier_raw - brier_cal), 4),
            'ece_raw': round(float(ece_raw), 4),
            'ece_calibrated': round(float(ece_cal), 4),
            'bins': bin_details,
        }

    def _save(self):
        """Save calibration models to disk."""
        with open(CALIBRATION_PATH, 'wb') as f:
            pickle.dump({
                'calibrators': self.calibrators,
                'global_calibrator': self.global_calibrator,
            }, f)

    def _save_report(self, report):
        """Save calibration report as JSON."""
        with open(CALIBRATION_REPORT_PATH, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def _load(self):
        """Load saved calibration models."""
        if self._loaded:
            return True
        if not CALIBRATION_PATH.exists():
            return False
        try:
            with open(CALIBRATION_PATH, 'rb') as f:
                data = pickle.load(f)
            self.calibrators = data['calibrators']
            self.global_calibrator = data['global_calibrator']
            self._loaded = True
            return True
        except Exception as e:
            print(f"[Calibrator] Failed to load: {e}")
            return False


if __name__ == '__main__':
    cal = Calibrator()
    report = cal.fit()

    if report:
        print(f"\n{'='*60}")
        print(f"CALIBRATION REPORT — {report['total_predictions']} predictions")
        print(f"{'='*60}")

        g = report.get('global', {})
        print(f"\nGlobal ({g.get('n', 0)} predictions):")
        print(f"  Brier score: {g.get('brier_raw', 0):.4f} → {g.get('brier_calibrated', 0):.4f} "
              f"(improvement: {g.get('brier_improvement', 0):+.4f})")
        print(f"  ECE:         {g.get('ece_raw', 0):.4f} → {g.get('ece_calibrated', 0):.4f}")

        print(f"\n{'Model':>20} {'N':>6} {'Brier':>8} {'→':>2} {'Cal':>8} {'Δ':>8}")
        print('-' * 55)
        for model, info in sorted(report.get('models', {}).items(),
                                   key=lambda x: x[1].get('brier_improvement', 0),
                                   reverse=True):
            if info['status'] != 'fitted':
                continue
            print(f"{model:>20} {info['n']:>6} "
                  f"{info['brier_raw']:>8.4f} → {info['brier_calibrated']:>8.4f} "
                  f"{info['brier_improvement']:>+8.4f}")

        print(f"\n{'='*60}")
        print("Calibration Table (global):")
        print(f"{'Raw':>8} {'Calibrated':>12} {'Kelly':>8}")
        for row in cal.calibration_table():
            print(f"{row['raw']:>7.0%} {row['calibrated']:>11.1%} {row['kelly']:>7.1%}")
