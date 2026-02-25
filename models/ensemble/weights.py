#!/usr/bin/env python3
"""Weight normalization/storage/report helpers for EnsembleModel."""

import json
from datetime import datetime
from pathlib import Path

ACCURACY_FILE = Path(__file__).resolve().parents[2] / "data" / "model_accuracy.json"


class EnsembleWeightsMixin:
    def _normalize_weights(self, weights):
        """Normalize weights to sum to 1.0, respecting minimum"""
        # First, apply minimum floor (skip floor for explicitly zeroed models)
        normalized = {}
        for name in self.models:
            w = weights.get(name, self.MIN_WEIGHT)
            if w == 0 and self.default_weights.get(name, 0) == 0:
                normalized[name] = 0.0
            else:
                normalized[name] = max(w, self.MIN_WEIGHT)

        # Then normalize to sum to 1.0
        total = sum(normalized.values())
        return {k: v / total for k, v in normalized.items()}

    def _load_accuracy_history(self):
        """Load accuracy tracking from file"""
        if ACCURACY_FILE.exists():
            try:
                with open(ACCURACY_FILE) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass

        # Initialize empty history
        return {
            "predictions": [],  # List of {game_id, model_predictions, actual_winner}
            "model_stats": {name: {"correct": 0, "total": 0, "recent": []} 
                          for name in self.models},
            "last_updated": None
        }

    def _save_accuracy_history(self):
        """Save accuracy tracking to file"""
        ACCURACY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.accuracy_history["last_updated"] = datetime.now().isoformat()
        with open(ACCURACY_FILE, 'w') as f:
            json.dump(self.accuracy_history, f, indent=2)

    @classmethod
    def _ramp_multiplier(cls, count, floor, threshold):
        """Linear ramp multiplier from floor to 1.0 by threshold count."""
        if threshold is None or threshold <= 0:
            return 1.0
        count = max(0, int(count or 0))
        floor = max(0.0, min(1.0, float(floor)))
        if count >= threshold:
            return 1.0
        return floor + (count / float(threshold)) * (1.0 - floor)

    def _sample_influence_multiplier(self, evaluated_count):
        """Guard against overreacting to tiny evaluated samples for any model."""
        return self._ramp_multiplier(
            evaluated_count,
            self.SAMPLE_RAMP_FLOOR,
            self.SAMPLE_RAMP_THRESHOLD,
        )

    def _day_recency_weight(self, age_days):
        """Optional age-based recency decay multiplier."""
        half_life = self.RECENCY_DAY_HALFLIFE_DAYS
        if half_life is None or half_life <= 0:
            return 1.0
        if age_days is None:
            return 1.0
        try:
            age = max(0.0, float(age_days))
        except (TypeError, ValueError):
            return 1.0
        return 0.5 ** (age / float(half_life))

    def _compute_target_weights(self, model_scores, recent_accuracy, decay_factor):
        """Compute target normalized weights from accuracy scores and guards."""
        accuracy_scores = {}
        for name in self.models:
            acc = max(recent_accuracy[name], 0.3)
            score = acc ** 3

            # Global sample ramp: low-sample models should not fully dominate.
            score *= self._sample_influence_multiplier(model_scores[name]['count'])

            # Preseason decay: prior/conference fade as real data accumulates
            if name in self.PRESEASON_MODELS and model_scores[name]['count'] < 50:
                score *= decay_factor
                score = max(score, self.MIN_WEIGHT)

            # Early-season dampening: stats-dependent models get reduced weight
            # until they have enough evaluated predictions to be trustworthy.
            # Weight ramps linearly from EARLY_SEASON_FLOOR to full over STATS_MATURITY_GAMES.
            if name in self.STATS_DEPENDENT_MODELS:
                n_evaluated = model_scores[name]['count']
                if n_evaluated < self.STATS_MATURITY_GAMES:
                    maturity = n_evaluated / self.STATS_MATURITY_GAMES  # 0.0 -> 1.0
                    dampen = self.EARLY_SEASON_FLOOR + maturity * (1.0 - self.EARLY_SEASON_FLOOR)
                    score *= dampen

            accuracy_scores[name] = score

        total_score = sum(accuracy_scores.values())
        if total_score <= 0:
            return None, accuracy_scores
        return ({n: s / total_score for n, s in accuracy_scores.items()}, accuracy_scores)

    def get_model_accuracy(self):
        """
        Get accuracy statistics for each model.

        Returns dict of {model_name: {accuracy, recent_accuracy, predictions}}
        """
        result = {}

        for name in self.models:
            stats = self.accuracy_history.get("model_stats", {}).get(
                name, {"correct": 0, "total": 0, "recent": []}
            )

            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            recent = stats.get("recent", [])[-self.ROLLING_WINDOW:]

            result[name] = {
                "all_time_accuracy": correct / total if total > 0 else None,
                "all_time_predictions": total,
                "recent_accuracy": sum(recent) / len(recent) if recent else None,
                "recent_predictions": len(recent),
                "current_weight": round(self.weights.get(name, 0), 4)
            }

        return result

    def get_weights_report(self):
        """
        Get a formatted report of current weights and accuracy.
        """
        accuracy = self.get_model_accuracy()

        lines = [
            "=" * 60,
            "ENSEMBLE MODEL WEIGHTS REPORT",
            "=" * 60,
            "",
            f"{'Model':<15} {'Weight':>8} {'Recent Acc':>12} {'All-Time':>12} {'N':>6}",
            "-" * 60
        ]

        # Sort by weight descending
        sorted_models = sorted(
            accuracy.items(),
            key=lambda x: self.weights.get(x[0], 0),
            reverse=True
        )

        for name, stats in sorted_models:
            weight = self.weights.get(name, 0) * 100
            recent = stats['recent_accuracy']
            recent_str = f"{recent*100:.1f}%" if recent else "N/A"
            alltime = stats['all_time_accuracy']
            alltime_str = f"{alltime*100:.1f}%" if alltime else "N/A"
            n = stats['all_time_predictions']

            lines.append(f"{name:<15} {weight:>7.1f}% {recent_str:>12} {alltime_str:>12} {n:>6}")

        lines.append("-" * 60)
        lines.append(f"Total predictions tracked: {sum(s['all_time_predictions'] for s in accuracy.values())}")

        return "\n".join(lines)

    def set_weights(self, weights):
        """Manually set model weights"""
        self.weights = self._normalize_weights(weights)

    def reset_weights(self):
        """Reset to default weights"""
        self.weights = self.default_weights.copy()

    def reset_accuracy_history(self):
        """Clear all accuracy history and start fresh"""
        self.accuracy_history = {
            "predictions": [],
            "model_stats": {name: {"correct": 0, "total": 0, "recent": []} 
                          for name in self.models},
            "last_updated": None
        }
        self.weights = self.default_weights.copy()
        self._save_accuracy_history()
