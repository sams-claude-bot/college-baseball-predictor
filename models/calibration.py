#!/usr/bin/env python3
"""
Probability Calibration Module

Maps raw model probabilities P(home_win) to calibrated P(home_win) using
isotonic regression or Platt scaling, fitted against actual game outcomes.

Industry-standard approach:
  - Calibrate P(home_win) directly on the full [0, 1] range
  - Label = 1 if home team won, 0 if away team won (from games.winner_id)
  - Isotonic regression (non-parametric, monotonic) when n >= 50
  - Platt scaling (logistic) as fallback for small samples
  - Per-model calibrators + global fallback
  - Brier score and ECE for evaluation

For Kelly sizing, calibrate_for_kelly() returns the TRUE calibrated probability.
Bankroll conservatism is handled by fractional Kelly in risk.py (default 25%),
NOT by distorting the probability signal.

Usage:
    from models.calibration import Calibrator

    cal = Calibrator()
    cal.fit()  # Train on all graded predictions

    # Calibrate a single probability
    raw_prob = 0.82  # P(home_win) from model
    calibrated = cal.calibrate(raw_prob, model='meta_ensemble')

    # For Kelly sizing — same calibrated prob, no distortion
    kelly_prob = cal.calibrate_for_kelly(raw_prob, model='meta_ensemble')
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
    """Isotonic regression probability calibrator.

    Calibrates raw P(home_win) → true P(home_win) using actual game outcomes.
    """

    def __init__(self):
        self.calibrators = {}  # type: Dict[str, object]
        self.global_calibrator = None
        self.report = {}  # type: Dict
        self._loaded = False

    def _get_connection(self):
        conn = sqlite3.connect(str(DATA_DIR / 'baseball.db'), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def fit(self, min_samples=MIN_SAMPLES):
        # type: (int) -> dict
        """Train calibration curves from graded predictions.

        Fits against actual game outcomes (home_won), NOT was_correct.
        Uses predicted_home_prob on full [0, 1] range — no reflection.

        Fits:
        1. Per-model calibrators (where sample size >= min_samples)
        2. Global calibrator (all models pooled)

        Returns report dict with calibration stats.
        """
        from sklearn.isotonic import IsotonicRegression

        conn = self._get_connection()

        # Get predictions with ACTUAL home win outcome from games table
        rows = conn.execute("""
            SELECT mp.model_name,
                   COALESCE(mp.raw_home_prob, mp.predicted_home_prob) as prob,
                   CASE WHEN g.winner_id = g.home_team_id THEN 1
                        WHEN g.winner_id = g.away_team_id THEN 0
                        ELSE NULL END as home_won
            FROM model_predictions mp
            JOIN games g ON mp.game_id = g.id
            WHERE g.status = 'final'
              AND g.winner_id IS NOT NULL
              AND mp.predicted_home_prob IS NOT NULL
        """).fetchall()
        conn.close()

        # Filter nulls
        rows = [r for r in rows if r['home_won'] is not None]

        if len(rows) < min_samples:
            print("[Calibrator] Only {} graded predictions, need {}".format(
                len(rows), min_samples))
            return {}

        # Group by model — use P(home_win) directly, full [0, 1] range
        model_data = defaultdict(lambda: {'probs': [], 'outcomes': []})
        all_probs = []
        all_outcomes = []

        for r in rows:
            model = r['model_name']
            prob = float(r['prob'])
            outcome = int(r['home_won'])

            model_data[model]['probs'].append(prob)
            model_data[model]['outcomes'].append(outcome)
            all_probs.append(prob)
            all_outcomes.append(outcome)

        all_probs = np.array(all_probs)
        all_outcomes = np.array(all_outcomes)

        report = {
            'fit_date': datetime.utcnow().isoformat(),
            'total_predictions': len(all_probs),
            'home_win_rate': float(all_outcomes.mean()),
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
                    'reason': '< {} samples'.format(min_samples),
                }
                continue

            cal = self._fit_one(probs, outcomes, model)
            self.calibrators[model] = cal

            stats = self._calibration_stats(probs, outcomes, cal)
            report['models'][model] = {
                'status': 'fitted',
                'n': len(probs),
            }
            report['models'][model].update(stats)

        # Fit global calibrator
        global_cal = self._fit_one(all_probs, all_outcomes, 'global')
        self.global_calibrator = global_cal
        global_stats = self._calibration_stats(all_probs, all_outcomes, global_cal)
        report['global'] = {'n': len(all_probs)}
        report['global'].update(global_stats)

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

    def _calibrate_one(self, prob, calibrator):
        # type: (float, object) -> float
        """Apply a fitted calibrator to a single probability."""
        if calibrator is None:
            return prob

        if getattr(calibrator, '_method', '') == 'isotonic':
            result = float(calibrator.predict([prob])[0])
        else:
            # Platt
            result = float(calibrator.predict_proba([[prob]])[0, 1])

        return max(PROB_FLOOR, min(PROB_CEILING, result))

    def calibrate(self, prob, model=None):
        # type: (float, Optional[str]) -> float
        """Calibrate a raw model probability P(home_win) → calibrated P(home_win).

        Uses model-specific calibrator if available, else global.
        Input: raw P(home_win) on [0, 1].
        Output: calibrated P(home_win) on [0.05, 0.95].
        """
        if not self._loaded:
            self._load()

        if model and model in self.calibrators:
            return self._calibrate_one(prob, self.calibrators[model])
        elif self.global_calibrator is not None:
            return self._calibrate_one(prob, self.global_calibrator)
        else:
            return prob

    def calibrate_for_kelly(self, prob, model=None):
        # type: (float, Optional[str]) -> float
        """Calibrate probability for Kelly criterion sizing.

        Returns the TRUE calibrated probability — no artificial shrinkage.
        Bankroll conservatism is handled by fractional Kelly in the risk
        engine (RISK_KELLY_FRACTION, default 0.25 = quarter Kelly), not
        by distorting the probability signal.

        Shrinking P toward 0.5 is not standard practice because it
        non-linearly distorts the edge calculation in ways that don't
        correspond to any coherent bankroll management strategy.
        """
        return self.calibrate(prob, model)

    def calibration_table(self, model=None):
        # type: (Optional[str]) -> list
        """Return a table of raw prob → calibrated prob for inspection.

        Covers full [0.05, 0.95] range since we calibrate P(home_win) directly.
        """
        table = []
        for raw_int in range(5, 100, 5):
            raw = raw_int / 100.0
            cal = self.calibrate(float(raw), model)
            table.append({
                'raw': round(float(raw), 2),
                'calibrated': round(cal, 3),
            })
        return table

    def _calibration_stats(self, probs, outcomes, calibrator):
        # type: (np.ndarray, np.ndarray, object) -> dict
        """Compute calibration metrics over full [0, 1] range."""
        cal_probs = np.array([self._calibrate_one(p, calibrator) for p in probs])

        # Brier score (lower is better)
        brier_raw = float(np.mean((probs - outcomes) ** 2))
        brier_cal = float(np.mean((cal_probs - outcomes) ** 2))

        # Expected Calibration Error — 10 bins across [0, 1]
        n_bins = 10
        ece_raw = 0.0
        ece_cal = 0.0
        bin_details = []

        for i in range(n_bins):
            lo = i / n_bins
            hi = (i + 1) / n_bins
            if i == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            n = int(mask.sum())
            if n == 0:
                continue

            raw_mean = float(probs[mask].mean())
            cal_mean = float(cal_probs[mask].mean())
            actual_mean = float(outcomes[mask].mean())

            ece_raw += abs(raw_mean - actual_mean) * n
            ece_cal += abs(cal_mean - actual_mean) * n

            bin_details.append({
                'range': '[{:.0%}, {:.0%}{}'.format(lo, hi, ']' if i == n_bins - 1 else ')'),
                'n': n,
                'raw_pred': round(raw_mean, 3),
                'cal_pred': round(cal_mean, 3),
                'actual': round(actual_mean, 3),
            })

        total = len(probs)
        if total > 0:
            ece_raw /= total
            ece_cal /= total

        return {
            'brier_raw': round(brier_raw, 4),
            'brier_calibrated': round(brier_cal, 4),
            'brier_improvement': round(brier_raw - brier_cal, 4),
            'ece_raw': round(float(ece_raw), 4),
            'ece_calibrated': round(float(ece_cal), 4),
            'bins': bin_details,
        }

    def _save(self):
        """Save calibration models to disk."""
        with open(str(CALIBRATION_PATH), 'wb') as f:
            pickle.dump({
                'calibrators': self.calibrators,
                'global_calibrator': self.global_calibrator,
            }, f)

    def _save_report(self, report):
        """Save calibration report as JSON."""
        with open(str(CALIBRATION_REPORT_PATH), 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def _load(self):
        """Load saved calibration models."""
        if self._loaded:
            return True
        if not CALIBRATION_PATH.exists():
            return False
        try:
            with open(str(CALIBRATION_PATH), 'rb') as f:
                data = pickle.load(f)
            self.calibrators = data['calibrators']
            self.global_calibrator = data['global_calibrator']
            self._loaded = True
            return True
        except Exception as e:
            print("[Calibrator] Failed to load: {}".format(e))
            return False


if __name__ == '__main__':
    cal = Calibrator()
    report = cal.fit()

    if report:
        print("\n" + "=" * 60)
        print("CALIBRATION REPORT — {} predictions".format(report['total_predictions']))
        print("Home win rate: {:.1%}".format(report.get('home_win_rate', 0)))
        print("=" * 60)

        g = report.get('global', {})
        print("\nGlobal ({} predictions):".format(g.get('n', 0)))
        print("  Brier score: {:.4f} -> {:.4f} (improvement: {:+.4f})".format(
            g.get('brier_raw', 0), g.get('brier_calibrated', 0),
            g.get('brier_improvement', 0)))
        print("  ECE:         {:.4f} -> {:.4f}".format(
            g.get('ece_raw', 0), g.get('ece_calibrated', 0)))

        print("\n{:>20} {:>6} {:>8} {:>2} {:>8} {:>8}".format(
            'Model', 'N', 'Brier', '', 'Cal', 'Delta'))
        print('-' * 55)
        for model, info in sorted(report.get('models', {}).items(),
                                   key=lambda x: x[1].get('brier_improvement', 0),
                                   reverse=True):
            if info['status'] != 'fitted':
                continue
            print("{:>20} {:>6} {:>8.4f} -> {:>8.4f} {:>+8.4f}".format(
                model, info['n'],
                info['brier_raw'], info['brier_calibrated'],
                info['brier_improvement']))

        print("\n" + "=" * 60)
        print("Reliability Table (global):")
        print("{:>12} {:>6} {:>10} {:>10} {:>10}".format(
            'Bin', 'N', 'Raw Pred', 'Cal Pred', 'Actual'))
        for b in g.get('bins', []):
            print("{:>12} {:>6} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                b['range'], b['n'], b['raw_pred'], b['cal_pred'], b['actual']))

        print("\nCalibration Table (global):")
        print("{:>8} {:>12}".format('Raw', 'Calibrated'))
        for row in cal.calibration_table():
            print("{:>7.0%} {:>11.1%}".format(row['raw'], row['calibrated']))
