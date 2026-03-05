#!/usr/bin/env python3
"""
Meta-Ensemble: A trained XGBoost meta-learner that combines 12 base model
probabilities plus leak-safe agreement features to produce a calibrated win
probability.

Unlike the static ensemble which uses fixed/slowly-adapting weights, this model
learns optimal blending from graded game data using walk-forward validation.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "data" / "meta_ensemble_xgb.pkl"

# Removed 'ensemble' (double-counts individual models) and 'prior' (preseason
# priors no longer needed mid-season). nn_slim runs as 'neural'.
MODEL_NAMES = [
    'elo', 'pythagorean', 'lightgbm',
    'poisson', 'xgboost', 'pitching',
    'pear', 'quality', 'neural',
    'venue', 'rest_travel', 'upset'
]

AGREEMENT_FEATURES = ['models_predicting_home', 'avg_home_prob', 'prob_spread']

# 12 diverse models (replacing 12 correlated ones):
#   Dropped: conference/advanced/log5 (r=0.94-0.96 W/L clones)
#   Added: venue (park factors, r=0.34), rest_travel (fatigue, r=0.09),
#          upset (contrarian signal, r=0.46)
#   Specialized: lightgbm=batting features, xgboost=pitching features


class MetaEnsemble:
    def __init__(self):
        self.xgb_model = None
        self.lr_model = None
        self.calibrator = None
        self.feature_names = []
        self._loaded = False

    @staticmethod
    def _expected_feature_names():
        return [f'{m}_prob' for m in MODEL_NAMES] + AGREEMENT_FEATURES

    @staticmethod
    def _build_meta_features(probs):
        home_votes = sum(1 for p in probs if p > 0.5)
        avg_prob = float(np.mean(probs))
        spread = max(probs) - min(probs)
        return probs + [home_votes, avg_prob, spread]

    def _get_connection(self):
        from scripts.database import get_connection
        return get_connection()

    def _extract_training_data(self):
        """Extract leak-safe feature matrix from graded games in the database.

        Rules:
        - Exclude rows tagged as prediction_source='backfill'.
        - Exclude any row where predicted_at is after pregame cutoff.
        - Pregame cutoff uses game start when time is available, else end-of-date fallback
          (23:59:59 local) minus 5 minutes.
        """
        conn = self._get_connection()
        c = conn.cursor()

        c.execute("PRAGMA table_info(model_predictions)")
        mp_cols = {row[1] for row in c.fetchall()}
        has_prediction_source = 'prediction_source' in mp_cols

        source_filter = "COALESCE(mp.prediction_source, 'live') != 'backfill' AND" if has_prediction_source else ""

        query = f"""
        SELECT 
            mp.game_id,
            g.date,
            g.home_team_id, g.away_team_id,
            MAX(CASE WHEN mp.model_name='elo' THEN mp.predicted_home_prob END) as elo_prob,
            MAX(CASE WHEN mp.model_name='pythagorean' THEN mp.predicted_home_prob END) as pythagorean_prob,
            MAX(CASE WHEN mp.model_name='lightgbm' THEN mp.predicted_home_prob END) as lightgbm_prob,
            MAX(CASE WHEN mp.model_name='poisson' THEN mp.predicted_home_prob END) as poisson_prob,
            MAX(CASE WHEN mp.model_name='xgboost' THEN mp.predicted_home_prob END) as xgboost_prob,
            MAX(CASE WHEN mp.model_name='pitching' THEN mp.predicted_home_prob END) as pitching_prob,
            MAX(CASE WHEN mp.model_name='pear' THEN mp.predicted_home_prob END) as pear_prob,
            MAX(CASE WHEN mp.model_name='quality' THEN mp.predicted_home_prob END) as quality_prob,
            MAX(CASE WHEN mp.model_name='neural' THEN mp.predicted_home_prob END) as neural_prob,
            MAX(CASE WHEN mp.model_name='venue' THEN mp.predicted_home_prob END) as venue_prob,
            MAX(CASE WHEN mp.model_name='rest_travel' THEN mp.predicted_home_prob END) as rest_travel_prob,
            MAX(CASE WHEN mp.model_name='upset' THEN mp.predicted_home_prob END) as upset_prob,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_won
        FROM model_predictions mp
        JOIN games g ON mp.game_id = g.id
        WHERE mp.was_correct IS NOT NULL
          AND {source_filter}
          datetime(mp.predicted_at) <= datetime(
              COALESCE(
                datetime(g.date || ' ' || g.time),
                datetime(g.date || ' 23:59:59')
              ),
              '-5 minutes'
          )
        GROUP BY mp.game_id
        HAVING COUNT(DISTINCT mp.model_name) >= 7
        ORDER BY g.date
        """
        c.execute(query)
        rows = c.fetchall()
        columns = [desc[0] for desc in c.description]
        conn.close()

        return rows, columns

    def cohort_integrity_report(self):
        """Return source/timestamp exclusion counts + final cohort size for training."""
        conn = self._get_connection()
        c = conn.cursor()

        c.execute("PRAGMA table_info(model_predictions)")
        mp_cols = {row[1] for row in c.fetchall()}
        has_prediction_source = 'prediction_source' in mp_cols

        source_expr = "COALESCE(mp.prediction_source, 'live')" if has_prediction_source else "'live'"

        c.execute(f"""
            WITH evaluated AS (
                SELECT mp.game_id, mp.model_name, mp.predicted_at, {source_expr} as prediction_source,
                       g.date, g.time
                FROM model_predictions mp
                JOIN games g ON g.id = mp.game_id
                WHERE mp.was_correct IS NOT NULL
            ), flags AS (
                SELECT *,
                       CASE WHEN prediction_source = 'backfill' THEN 1 ELSE 0 END as by_source,
                       CASE WHEN datetime(predicted_at) > datetime(
                            COALESCE(
                                datetime(date || ' ' || time),
                                datetime(date || ' 23:59:59')
                            ),
                            '-5 minutes'
                       ) THEN 1 ELSE 0 END as by_timestamp
                FROM evaluated
            )
            SELECT
                SUM(by_source) as excluded_by_source,
                SUM(CASE WHEN by_source = 0 AND by_timestamp = 1 THEN 1 ELSE 0 END) as excluded_by_timestamp,
                SUM(CASE WHEN by_source = 0 AND by_timestamp = 0 THEN 1 ELSE 0 END) as included_rows,
                COUNT(*) as evaluated_rows
            FROM flags
        """)
        row = c.fetchone()

        c.execute(f"""
            WITH filtered AS (
                SELECT mp.game_id
                FROM model_predictions mp
                JOIN games g ON g.id = mp.game_id
                WHERE mp.was_correct IS NOT NULL
                  AND {source_expr} != 'backfill'
                  AND datetime(mp.predicted_at) <= datetime(
                        COALESCE(
                            datetime(g.date || ' ' || g.time),
                            datetime(g.date || ' 23:59:59')
                        ),
                        '-5 minutes'
                  )
                GROUP BY mp.game_id
                HAVING COUNT(DISTINCT mp.model_name) >= 7
            )
            SELECT COUNT(*) FROM filtered
        """)
        final_games = c.fetchone()[0]
        conn.close()

        return {
            'excluded_by_source': int(row[0] or 0),
            'excluded_by_timestamp': int(row[1] or 0),
            'included_rows': int(row[2] or 0),
            'evaluated_rows': int(row[3] or 0),
            'final_training_games': int(final_games or 0),
        }

    def _build_features(self, rows, columns):
        """Build feature matrix from raw query results.
        
        Returns: (X as numpy array, y as numpy array, dates list, feature_names list)
        """
        col_idx = {name: i for i, name in enumerate(columns)}

        feature_names = self._expected_feature_names()

        X = []
        y = []
        dates = []

        for row in rows:
            probs = []
            for m in MODEL_NAMES:
                p = row[col_idx[f'{m}_prob']]
                probs.append(p if p is not None else 0.5)
            features = self._build_meta_features(probs)
            X.append(features)
            y.append(row[col_idx['home_won']])
            dates.append(row[col_idx['date']])

        self.feature_names = feature_names
        return np.array(X), np.array(y), dates, feature_names

    def train(self, retrain=False):
        """Train on all graded games with walk-forward validation.
        
        Returns dict with walk-forward results.
        """
        if not retrain and self._loaded and self.xgb_model is not None:
            return None

        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        rows, columns = self._extract_training_data()
        if len(rows) < 50:
            print(f"Only {len(rows)} graded games, need at least 50 for meta-ensemble training")
            return None

        X, y, dates, feature_names = self._build_features(rows, columns)
        self.feature_names = feature_names

        # Walk-forward validation
        unique_dates = sorted(set(dates))
        date_to_idx = defaultdict(list)
        for i, d in enumerate(dates):
            date_to_idx[d].append(i)

        wf_preds_xgb = {}
        wf_preds_lr = {}
        wf_actuals = {}

        # Start from date index 2 (need at least 2 dates for training)
        for test_date_idx in range(2, len(unique_dates)):
            test_date = unique_dates[test_date_idx]
            train_dates = unique_dates[:test_date_idx]

            train_idx = [i for d in train_dates for i in date_to_idx[d]]
            test_idx = date_to_idx[test_date]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # XGBoost — tuned params (sweep 2026-03-02: +2.6% acc, -0.07 ll)
            model_xgb = xgb.XGBClassifier(
                max_depth=3, n_estimators=100, learning_rate=0.05,
                min_child_weight=5, subsample=0.7, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False, eval_metric='logloss',
                verbosity=0
            )
            model_xgb.fit(X_train, y_train)
            xgb_probs = model_xgb.predict_proba(X_test)[:, 1]

            # Logistic Regression
            model_lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            model_lr.fit(X_train, y_train)
            lr_probs = model_lr.predict_proba(X_test)[:, 1]

            for i, idx in enumerate(test_idx):
                wf_preds_xgb[idx] = xgb_probs[i]
                wf_preds_lr[idx] = lr_probs[i]
                wf_actuals[idx] = y_test[i]

        # Compute walk-forward accuracy
        indices = sorted(wf_actuals.keys())
        wf_y = np.array([wf_actuals[i] for i in indices])
        wf_xgb = np.array([wf_preds_xgb[i] for i in indices])
        wf_lr = np.array([wf_preds_lr[i] for i in indices])

        xgb_acc = np.mean((wf_xgb > 0.5) == wf_y) * 100
        lr_acc = np.mean((wf_lr > 0.5) == wf_y) * 100

        # Per-model accuracy from the feature matrix
        model_accs = {}
        for j, m in enumerate(MODEL_NAMES):
            m_probs = np.array([X[i, j] for i in indices])
            m_correct = (m_probs > 0.5) == wf_y
            model_accs[m] = np.mean(m_correct) * 100

        # Now train final models on ALL data — tuned params
        self.xgb_model = xgb.XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.05,
            min_child_weight=5, subsample=0.7, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False, eval_metric='logloss',
            verbosity=0
        )
        self.xgb_model.fit(X, y)

        self.lr_model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        self.lr_model.fit(X, y)

        # Save
        self._save()
        self._loaded = True

        # Per-date accuracy
        per_date = {}
        for test_date_idx in range(2, len(unique_dates)):
            test_date = unique_dates[test_date_idx]
            test_idx = date_to_idx[test_date]
            valid_idx = [i for i in test_idx if i in wf_actuals]
            if valid_idx:
                date_y = np.array([wf_actuals[i] for i in valid_idx])
                date_xgb = np.array([wf_preds_xgb[i] for i in valid_idx])
                per_date[test_date] = {
                    'total': len(valid_idx),
                    'xgb_correct': int(np.sum((date_xgb > 0.5) == date_y)),
                    'xgb_acc': np.mean((date_xgb > 0.5) == date_y) * 100
                }

        return {
            'xgb_accuracy': xgb_acc,
            'lr_accuracy': lr_acc,
            'model_accuracies': model_accs,
            'n_games': len(indices),
            'n_total': len(X),
            'per_date': per_date,
            'feature_importance': self.get_feature_importance()
        }

    def _save(self):
        """Save trained models to disk."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'xgb_model': self.xgb_model,
                'lr_model': self.lr_model,
                'feature_names': self.feature_names,
            }, f)

    def _load(self):
        """Load trained models from disk."""
        if self._loaded:
            return True
        if not MODEL_PATH.exists():
            return False
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            self.xgb_model = data['xgb_model']
            self.lr_model = data['lr_model']
            expected = self._expected_feature_names()
            loaded_feature_names = data.get('feature_names') or []
            if loaded_feature_names and loaded_feature_names != expected:
                print(
                    f"Meta-ensemble schema mismatch (saved={len(loaded_feature_names)}, expected={len(expected)}). "
                    "Retrain required."
                )
                return False

            model_feature_count = None
            if self.xgb_model is not None:
                model_feature_count = getattr(self.xgb_model, "n_features_in_", None)
            if model_feature_count is None and self.lr_model is not None:
                model_feature_count = getattr(self.lr_model, "n_features_in_", None)
            if model_feature_count is not None and model_feature_count != len(expected):
                print(
                    f"Meta-ensemble model feature count mismatch (saved={model_feature_count}, expected={len(expected)}). "
                    "Retrain required."
                )
                return False

            self.feature_names = loaded_feature_names or expected
            self._loaded = True
            return True
        except Exception as e:
            print(f"Failed to load meta-ensemble: {e}")
            return False

    def predict(self, game_id=None, home_team_id=None, away_team_id=None):
        """Return meta-ensemble home win probability.
        
        Can predict by game_id (looks up existing model predictions) or by
        home_team_id/away_team_id (runs all models fresh — slower).
        """
        if not self._load():
            return 0.5  # fallback if no trained model

        conn = self._get_connection()
        c = conn.cursor()

        if game_id:
            # Get existing model predictions for this game
            c.execute("""
                SELECT mp.model_name, mp.predicted_home_prob,
                       g.date, g.home_team_id, g.away_team_id
                FROM model_predictions mp
                JOIN games g ON mp.game_id = g.id
                WHERE mp.game_id = ? AND mp.model_name != 'meta_ensemble'
            """, (game_id,))
            rows = c.fetchall()
        elif home_team_id and away_team_id:
            # Get latest predictions for this matchup
            c.execute("""
                SELECT mp.model_name, mp.predicted_home_prob,
                       g.date, g.home_team_id, g.away_team_id
                FROM model_predictions mp
                JOIN games g ON mp.game_id = g.id
                WHERE g.home_team_id = ? AND g.away_team_id = ?
                  AND mp.model_name != 'meta_ensemble'
                ORDER BY mp.predicted_at DESC
                LIMIT 13
            """, (home_team_id, away_team_id))
            rows = c.fetchall()
        else:
            conn.close()
            return 0.5

        if not rows:
            conn.close()
            return 0.5

        # Build feature vector
        model_probs = {}

        for row in rows:
            model_probs[row['model_name']] = row['predicted_home_prob']

        conn.close()

        probs = [model_probs.get(m, 0.5) for m in MODEL_NAMES]
        features = np.array([self._build_meta_features(probs)])

        # Use XGBoost as primary — LR has pathological coefficients due to
        # multicollinearity (7/12 model-prob weights inverted, away picks near-random).
        # XGB handles correlated features naturally via tree splits.
        if self.xgb_model is not None:
            prob = float(self.xgb_model.predict_proba(features)[:, 1][0])
        else:
            prob = float(self.lr_model.predict_proba(features)[:, 1][0])
        return min(max(prob, 0.001), 0.999)

    def get_feature_importance(self):
        """Return feature importances for debugging."""
        if not self._load():
            return {}
        if self.xgb_model is None:
            return {}
        importances = self.xgb_model.feature_importances_
        return dict(zip(self.feature_names, [float(x) for x in importances]))
