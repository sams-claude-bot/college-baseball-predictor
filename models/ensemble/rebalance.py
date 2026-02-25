#!/usr/bin/env python3
"""Rebalancing and weight logging helpers for EnsembleModel."""

import json


class EnsembleRebalanceMixin:
    def _apply_target_weights(self, target_weights, mode="incremental"):
        """
        Apply target weights.

        mode='incremental' uses ADJUSTMENT_RATE drift.
        mode='hard_rebalance' sets weights directly to target (disabled models preserved).
        """
        adjustment_rate = 1.0 if mode == "hard_rebalance" else self.ADJUSTMENT_RATE
        before = self.weights.copy()

        for name in self.models:
            # Respect explicitly disabled models (default_weight == 0)
            if self.default_weights.get(name, 0) == 0:
                self.weights[name] = 0.0
                continue

            current = self.weights.get(name, self.MIN_WEIGHT)
            target = target_weights.get(name, self.MIN_WEIGHT)
            if mode == "hard_rebalance":
                new_weight = target
            else:
                new_weight = current + (target - current) * adjustment_rate
            self.weights[name] = max(new_weight, self.MIN_WEIGHT)

        self.weights = self._normalize_weights(self.weights)
        return before, self.weights.copy(), adjustment_rate

    def _recalculate_weights(self, mode="incremental"):
        """
        Recompute ensemble weights from evaluated predictions.

        Args:
            mode: 'incremental' or 'hard_rebalance'
        """
        if mode not in {"incremental", "hard_rebalance"}:
            raise ValueError(f"Unsupported rebalance mode: {mode}")

        try:
            from scripts.database import get_connection
            conn = get_connection()
            c = conn.cursor()

            # Get per-prediction accuracy with recency weighting
            c.execute('''
                SELECT model_name, was_correct, game_id, predicted_at,
                       ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY predicted_at DESC) as recency_rank,
                       (julianday('now') - julianday(predicted_at)) as age_days
                FROM model_predictions
                WHERE was_correct IS NOT NULL
                AND model_name IN ({})
            '''.format(','.join('?' * len(self.models))), list(self.models.keys()))

            model_scores = {
                name: {
                    'weighted_correct': 0.0,
                    'weighted_total': 0.0,
                    'count': 0,
                    'latest_predicted_at': None,
                }
                for name in self.models
            }

            for row in c.fetchall():
                model_name = row['model_name']
                if model_name not in self.models:
                    continue
                rank = row['recency_rank']
                rank_weight = self.RECENCY_RANK_DECAY_BASE ** (rank - 1)
                day_weight = self._day_recency_weight(row['age_days'])
                weight = rank_weight * day_weight

                model_scores[model_name]['weighted_correct'] += row['was_correct'] * weight
                model_scores[model_name]['weighted_total'] += weight
                model_scores[model_name]['count'] += 1
                if model_scores[model_name]['latest_predicted_at'] is None:
                    model_scores[model_name]['latest_predicted_at'] = row['predicted_at']

            total_evaluated = max(s['count'] for s in model_scores.values()) if model_scores else 0
            conn.close()

        except Exception:
            return False

        if total_evaluated < 20:
            return False

        recent_accuracy = {}
        for name in self.models:
            s = model_scores[name]
            if s['weighted_total'] > 0:
                recent_accuracy[name] = s['weighted_correct'] / s['weighted_total']
            else:
                recent_accuracy[name] = 0.5

        decay_factor = max(0.0, 1.0 - (total_evaluated / self.PRESEASON_DECAY_GAMES))
        target_weights, accuracy_scores = self._compute_target_weights(
            model_scores, recent_accuracy, decay_factor
        )
        if not target_weights:
            return False

        before_weights, after_weights, adjustment_rate = self._apply_target_weights(
            target_weights, mode=mode
        )

        self._log_weight_update(
            recent_accuracy,
            decay_factor,
            total_evaluated,
            update_mode=mode,
            adjustment_rate=adjustment_rate,
            target_weights=target_weights,
            previous_weights=before_weights,
            metadata={
                "recency_rank_decay_base": self.RECENCY_RANK_DECAY_BASE,
                "recency_day_halflife_days": self.RECENCY_DAY_HALFLIFE_DAYS,
                "sample_ramp_floor": self.SAMPLE_RAMP_FLOOR,
                "sample_ramp_threshold": self.SAMPLE_RAMP_THRESHOLD,
                "accuracy_scores": accuracy_scores,
                "model_eval_counts": {k: v["count"] for k, v in model_scores.items()},
                "sample_multipliers": {
                    k: self._sample_influence_multiplier(v["count"])
                    for k, v in model_scores.items()
                },
            },
        )
        return before_weights != after_weights

    def _update_weights_from_history(self):
        """
        Update weights based on recency-weighted model accuracy.

        Key features:
        - Recent predictions count more than old ones (exponential decay)
        - Preseason models (prior, conference) decay as real data accumulates
        - Accuracy^3 amplification rewards consistently good models
        - Aggressive adjustment rate for fast convergence
        """
        self._recalculate_weights(mode="incremental")

    def hard_rebalance_weights(self, save_accuracy_history=False):
        """
        Weekly hard rebalance: set weights directly to computed targets.

        Disabled models (default weight = 0) remain disabled.
        """
        changed = self._recalculate_weights(mode="hard_rebalance")
        if save_accuracy_history:
            self._save_accuracy_history()
        return changed

    def _ensure_weight_log_columns(self, conn):
        """Backfill logging columns on existing databases."""
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(ensemble_weight_log)").fetchall()
        }
        if "update_mode" not in cols:
            conn.execute("ALTER TABLE ensemble_weight_log ADD COLUMN update_mode TEXT")
        if "adjustment_rate" not in cols:
            conn.execute("ALTER TABLE ensemble_weight_log ADD COLUMN adjustment_rate REAL")
        if "target_weights_json" not in cols:
            conn.execute("ALTER TABLE ensemble_weight_log ADD COLUMN target_weights_json TEXT")
        if "previous_weights_json" not in cols:
            conn.execute("ALTER TABLE ensemble_weight_log ADD COLUMN previous_weights_json TEXT")
        if "metadata_json" not in cols:
            conn.execute("ALTER TABLE ensemble_weight_log ADD COLUMN metadata_json TEXT")

    def _log_weight_update(
        self,
        accuracy,
        decay_factor,
        total_games,
        update_mode="incremental",
        adjustment_rate=None,
        target_weights=None,
        previous_weights=None,
        metadata=None,
    ):
        """Log weight updates to ensemble_weights_history table.
        Only logs when weights actually change (compares to last entry)."""
        try:
            from scripts.database import get_connection
            conn = get_connection()
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ensemble_weight_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    total_games INTEGER,
                    decay_factor REAL,
                    weights_json TEXT,
                    accuracy_json TEXT,
                    update_mode TEXT,
                    adjustment_rate REAL,
                    target_weights_json TEXT,
                    previous_weights_json TEXT,
                    metadata_json TEXT
                )
            ''')
            self._ensure_weight_log_columns(conn)
            # Only log if weights changed from last entry
            last = conn.execute(
                'SELECT weights_json, update_mode FROM ensemble_weight_log ORDER BY id DESC LIMIT 1'
            ).fetchone()
            new_weights_json = json.dumps(self.weights)
            if last and last[0] == new_weights_json and (last[1] or "incremental") == update_mode:
                conn.close()
                return  # No change, skip logging
            conn.execute('''
                INSERT INTO ensemble_weight_log (
                    total_games, decay_factor, weights_json, accuracy_json,
                    update_mode, adjustment_rate, target_weights_json,
                    previous_weights_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                total_games,
                decay_factor,
                new_weights_json,
                json.dumps(accuracy),
                update_mode,
                adjustment_rate,
                json.dumps(target_weights) if target_weights is not None else None,
                json.dumps(previous_weights) if previous_weights is not None else None,
                json.dumps(metadata) if metadata is not None else None,
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass  # Non-critical

    def record_prediction(self, game_id, predictions, actual_winner_id):
        """
        Record a prediction outcome for accuracy tracking.

        Args:
            game_id: Unique game identifier
            predictions: Dict of {model_name: predicted_home_prob}
            actual_winner_id: The team that actually won ('home' or 'away')

        Call this after each game result is known.
        """
        home_won = actual_winner_id == 'home'

        # Record for each model
        for name, home_prob in predictions.items():
            if name not in self.models:
                continue

            # Model was correct if:
            # - home_prob > 0.5 and home team won, OR
            # - home_prob < 0.5 and away team won
            model_correct = (home_prob > 0.5) == home_won

            stats = self.accuracy_history["model_stats"].setdefault(
                name, {"correct": 0, "total": 0, "recent": []}
            )

            stats["total"] += 1
            if model_correct:
                stats["correct"] += 1

            # Add to recent (keep rolling window)
            stats["recent"].append(1 if model_correct else 0)
            if len(stats["recent"]) > self.ROLLING_WINDOW * 2:
                stats["recent"] = stats["recent"][-self.ROLLING_WINDOW:]

        # Store full prediction record
        self.accuracy_history["predictions"].append({
            "game_id": game_id,
            "predictions": predictions,
            "actual_winner": actual_winner_id,
            "timestamp": datetime.now().isoformat()
        })

        # Trim old predictions
        if len(self.accuracy_history["predictions"]) > 500:
            self.accuracy_history["predictions"] = \
                self.accuracy_history["predictions"][-500:]

        # Update weights based on new data
        self._update_weights_from_history()

        # Save to file
        self._save_accuracy_history()
