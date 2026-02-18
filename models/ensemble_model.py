#!/usr/bin/env python3
"""
Dynamic Ensemble Model

Combines multiple prediction models with dynamic weights that
adjust based on each model's recent accuracy.

Features:
- Rolling accuracy tracking (last N predictions)
- Auto-adjusting weights based on performance
- Minimum weight floor (no model drops below 5%)
- Includes new pitching, conference, and prior models
- Reports current weights and model accuracy
"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_model import BaseModel
from models.pythagorean_model import PythagoreanModel
from models.elo_model import EloModel
from models.log5_model import Log5Model
from models.advanced_model import AdvancedModel
from models.pitching_model import PitchingModel
from models.conference_model import ConferenceModel
from models.prior_model import PriorModel
from models.poisson_model import predict as poisson_predict
from models.momentum_model import predict_with_momentum

# Try to import neural model (may not have weights yet)
try:
    from models.neural_model import NeuralModel
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

# Try to import XGBoost model
try:
    from models.xgboost_model import XGBMoneylineModel
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# Try to import LightGBM model
try:
    from models.lightgbm_model import LGBMoneylineModel
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

# File to store model accuracy history
ACCURACY_FILE = Path(__file__).parent.parent / "data" / "model_accuracy.json"


class PoissonModelWrapper(BaseModel):
    """Wrapper to make Poisson model compatible with ensemble interface."""
    
    name = "poisson"
    version = "1.1"
    description = "Poisson run distribution model (with weather)"
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_id=None, weather_data=None):
        result = poisson_predict(home_team_id, away_team_id, 
                                 neutral_site=neutral_site, 
                                 team_a_home=True,
                                 game_id=game_id,
                                 weather_data=weather_data)
        return {
            "model": self.name,
            "home_win_probability": result['win_prob_a'],
            "away_win_probability": result['win_prob_b'],
            "projected_home_runs": result['expected_runs_a'],
            "projected_away_runs": result['expected_runs_b'],
            "projected_total": result['expected_total'],
            "run_line": self.calculate_run_line(
                result['expected_runs_a'], result['expected_runs_b']
            ),
            "weather": result.get('weather')
        }


class EnsembleModel(BaseModel):
    """
    Dynamic ensemble that combines all prediction models.
    
    Weights adjust based on each model's recent prediction accuracy.
    """
    
    name = "ensemble"
    version = "3.0"
    description = "Dynamic weighted ensemble with accuracy-based weight adjustment"
    
    # Minimum weight any model can have
    MIN_WEIGHT = 0.03
    
    # Number of recent predictions to track for accuracy
    ROLLING_WINDOW = 100
    
    # How quickly weights adjust (0-1, lower = slower adjustment)
    ADJUSTMENT_RATE = 0.4
    
    # Models that should decay as season data accumulates
    PRESEASON_MODELS = {"prior", "conference"}
    
    # Games threshold where preseason models fully decay to MIN_WEIGHT
    PRESEASON_DECAY_GAMES = 200
    
    def __init__(self, weights=None):
        """
        Initialize with model weights.
        
        Args:
            weights: dict of {model_name: weight}. If None, uses default weights.
        """
        # Initialize all models
        self.models = {
            "pythagorean": PythagoreanModel(),
            "elo": EloModel(),
            "log5": Log5Model(),
            "advanced": AdvancedModel(),
            "pitching": PitchingModel(),
            "conference": ConferenceModel(),
            "prior": PriorModel(),
            "poisson": PoissonModelWrapper()
        }
        
        # Add XGBoost if available and trained
        if _XGB_AVAILABLE:
            xgb_model = XGBMoneylineModel()
            if xgb_model.is_trained():
                self.models["xgboost"] = xgb_model
        
        # Add LightGBM if available and trained
        if _LGB_AVAILABLE:
            lgb_model = LGBMoneylineModel()
            if lgb_model.is_trained():
                self.models["lightgbm"] = lgb_model
        
        # Neural model excluded from ensemble for now — tracking independently
        
        # Default weights (sum to 1.0, will be normalized if gradient boosting added)
        self.default_weights = {
            "pythagorean": 0.07,
            "elo": 0.12,
            "log5": 0.08,
            "advanced": 0.18,
            "pitching": 0.13,
            "conference": 0.07,
            "prior": 0.10,
            "poisson": 0.15,
            "xgboost": 0.05,   # Start low, will adjust with accuracy
            "lightgbm": 0.05,  # Start low, will adjust with accuracy
        }
        
        # Momentum adjustment settings
        self.use_momentum = True
        self.momentum_strength = 1.0  # Multiplier for momentum adjustment (0-2)
        
        # Use provided weights or defaults
        if weights is not None:
            self.weights = self._normalize_weights(weights)
        else:
            self.weights = self.default_weights.copy()
        
        # Load accuracy history
        self.accuracy_history = self._load_accuracy_history()
        
        # Apply any weight updates based on history
        self._update_weights_from_history()
    
    def _normalize_weights(self, weights):
        """Normalize weights to sum to 1.0, respecting minimum"""
        # First, apply minimum floor
        normalized = {}
        for name in self.models:
            w = weights.get(name, self.MIN_WEIGHT)
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
    
    def _update_weights_from_history(self):
        """
        Update weights based on recency-weighted model accuracy.
        
        Key features:
        - Recent predictions count more than old ones (exponential decay)
        - Preseason models (prior, conference) decay as real data accumulates
        - Accuracy^3 amplification rewards consistently good models
        - Aggressive adjustment rate for fast convergence
        """
        try:
            from scripts.database import get_connection
            conn = get_connection()
            c = conn.cursor()
            
            # Get per-prediction accuracy with recency weighting
            # More recent predictions get higher weight
            c.execute('''
                SELECT model_name, was_correct, game_id,
                       ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY predicted_at DESC) as recency_rank
                FROM model_predictions 
                WHERE was_correct IS NOT NULL
                AND model_name IN ({})
            '''.format(','.join('?' * len(self.models))), list(self.models.keys()))
            
            # Build recency-weighted accuracy per model
            model_scores = {name: {'weighted_correct': 0.0, 'weighted_total': 0.0, 'count': 0} 
                          for name in self.models}
            
            for row in c.fetchall():
                model_name = row['model_name']
                if model_name not in self.models:
                    continue
                rank = row['recency_rank']
                # Exponential decay: recent games weight ~3x more than old ones
                # Half-life of ~30 predictions
                weight = 0.977 ** (rank - 1)  # 0.977^30 ≈ 0.5
                
                model_scores[model_name]['weighted_correct'] += row['was_correct'] * weight
                model_scores[model_name]['weighted_total'] += weight
                model_scores[model_name]['count'] += 1
            
            # Total evaluated predictions (for preseason decay calculation)
            total_evaluated = max(s['count'] for s in model_scores.values()) if model_scores else 0
            
            conn.close()
            
        except Exception as e:
            return  # Can't update without data
        
        if total_evaluated < 20:
            return  # Not enough data to adjust yet
        
        # Calculate recency-weighted accuracy
        recent_accuracy = {}
        for name in self.models:
            s = model_scores[name]
            if s['weighted_total'] > 0:
                recent_accuracy[name] = s['weighted_correct'] / s['weighted_total']
            else:
                recent_accuracy[name] = 0.5
        
        # Apply preseason decay: prior and conference models fade as season data grows
        # At 0 games: full weight. At PRESEASON_DECAY_GAMES: drops to MIN_WEIGHT
        decay_factor = max(0.0, 1.0 - (total_evaluated / self.PRESEASON_DECAY_GAMES))
        
        # Calculate target weights using accuracy^3 (amplifies differences)
        accuracy_scores = {}
        for name in self.models:
            acc = max(recent_accuracy[name], 0.3)
            score = acc ** 3
            
            # Apply preseason decay to prior/conference models
            if name in self.PRESEASON_MODELS:
                score *= decay_factor
                score = max(score, self.MIN_WEIGHT)
            
            accuracy_scores[name] = score
        
        total_score = sum(accuracy_scores.values())
        
        if total_score > 0:
            target_weights = {n: s / total_score for n, s in accuracy_scores.items()}
            
            # Blend current weights toward target
            for name in self.models:
                current = self.weights.get(name, self.MIN_WEIGHT)
                target = target_weights.get(name, self.MIN_WEIGHT)
                new_weight = current + (target - current) * self.ADJUSTMENT_RATE
                self.weights[name] = max(new_weight, self.MIN_WEIGHT)
            
            # Re-normalize
            self.weights = self._normalize_weights(self.weights)
            
            # Log weight changes for tracking
            self._log_weight_update(recent_accuracy, decay_factor, total_evaluated)
    
    def _log_weight_update(self, accuracy, decay_factor, total_games):
        """Log weight updates to ensemble_weights_history table."""
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
                    accuracy_json TEXT
                )
            ''')
            conn.execute('''
                INSERT INTO ensemble_weight_log (total_games, decay_factor, weights_json, accuracy_json)
                VALUES (?, ?, ?, ?)
            ''', (total_games, decay_factor, json.dumps(self.weights), json.dumps(accuracy)))
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
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_id=None, weather_data=None):
        """
        Make a prediction using the weighted ensemble.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            neutral_site: If True, no home advantage
            game_id: Optional game ID for weather lookup (passed to weather-aware models)
            weather_data: Optional dict with weather (overrides database lookup)
        """
        predictions = {}
        weather_info = None
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                # Models that support weather get the weather parameters
                if name in ('poisson', 'pitching'):
                    pred = model.predict_game(home_team_id, away_team_id, neutral_site,
                                             game_id=game_id, weather_data=weather_data)
                    # Capture weather info from first model that has it
                    if weather_info is None and pred.get('weather'):
                        weather_info = pred['weather']
                else:
                    pred = model.predict_game(home_team_id, away_team_id, neutral_site)
                predictions[name] = pred
            except Exception as e:
                print(f"Warning: {name} model failed: {e}")
        
        if not predictions:
            return {
                "model": self.name,
                "error": "All models failed"
            }
        
        # Weighted average
        home_prob = 0
        home_runs = 0
        away_runs = 0
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                home_prob += pred['home_win_probability'] * weight
                home_runs += pred['projected_home_runs'] * weight
                away_runs += pred['projected_away_runs'] * weight
                total_weight += weight
        
        # Normalize in case some models failed
        if total_weight > 0 and total_weight < 1.0:
            home_prob /= total_weight
            home_runs /= total_weight
            away_runs /= total_weight
        
        home_prob = max(0.1, min(0.9, home_prob))
        run_line = self.calculate_run_line(home_runs, away_runs)
        
        # Apply momentum adjustment if enabled
        momentum_info = None
        if self.use_momentum:
            try:
                mom_result = predict_with_momentum(
                    home_team_id, away_team_id, 
                    base_prob_a=home_prob
                )
                # Apply momentum with configurable strength
                adjustment = mom_result['adjustment'] * self.momentum_strength
                home_prob_adjusted = home_prob + adjustment
                home_prob_adjusted = max(0.1, min(0.9, home_prob_adjusted))
                
                momentum_info = {
                    "base_prob": round(home_prob, 3),
                    "adjustment": round(adjustment, 3),
                    "adjusted_prob": round(home_prob_adjusted, 3),
                    "home_momentum": round(mom_result['team_a_momentum']['momentum_score'], 3),
                    "away_momentum": round(mom_result['team_b_momentum']['momentum_score'], 3),
                    "home_streak": mom_result['team_a_momentum']['components'].get('streak', 'N/A'),
                    "away_streak": mom_result['team_b_momentum']['components'].get('streak', 'N/A')
                }
                home_prob = home_prob_adjusted
            except Exception as e:
                # If momentum fails, continue without it
                momentum_info = {"error": str(e)}
        
        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(home_runs, 1),
            "projected_away_runs": round(away_runs, 1),
            "projected_total": round(home_runs + away_runs, 1),
            "run_line": run_line,
            "momentum": momentum_info,
            "weather": weather_info,
            "component_predictions": {
                name: {
                    "home_prob": pred['home_win_probability'],
                    "weight": round(self.weights.get(name, 0), 4)
                }
                for name, pred in predictions.items()
            },
            "weights": {k: round(v, 4) for k, v in self.weights.items()}
        }
    
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


# For testing
if __name__ == "__main__":
    import argparse
    
    # Check for special commands first
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        model = EnsembleModel()
        print(model.get_weights_report())
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "reset":
        model = EnsembleModel()
        model.reset_accuracy_history()
        print("Reset accuracy history and weights to defaults")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Ensemble Prediction Model')
    parser.add_argument('home_team', nargs='?', help='Home team ID')
    parser.add_argument('away_team', nargs='?', help='Away team ID')
    parser.add_argument('--neutral', action='store_true', help='Neutral site game')
    parser.add_argument('--game-id', type=str, help='Game ID for weather lookup')
    parser.add_argument('--temp', type=float, help='Temperature (°F)')
    parser.add_argument('--wind', type=float, help='Wind speed (mph)')
    parser.add_argument('--wind-dir', type=int, help='Wind direction (degrees)')
    parser.add_argument('--humidity', type=float, help='Humidity (%)')
    parser.add_argument('--dome', action='store_true', help='Indoor dome')
    args = parser.parse_args()
    
    model = EnsembleModel()
    
    if args.home_team and args.away_team:
        home = args.home_team.lower().replace(" ", "-")
        away = args.away_team.lower().replace(" ", "-")
        
        # Build weather data from CLI args if provided
        weather_data = None
        if any([args.temp is not None, args.wind is not None, args.wind_dir is not None, 
                args.humidity is not None, args.dome]):
            weather_data = {}
            if args.temp is not None:
                weather_data['temp_f'] = args.temp
            if args.wind is not None:
                weather_data['wind_speed_mph'] = args.wind
            if args.wind_dir is not None:
                weather_data['wind_direction_deg'] = args.wind_dir
            if args.humidity is not None:
                weather_data['humidity_pct'] = args.humidity
            if args.dome:
                weather_data['is_dome'] = 1
        
        pred = model.predict_game(home, away, args.neutral,
                                  game_id=args.game_id, weather_data=weather_data)
        
        print(f"\n{'='*55}")
        print(f"  ENSEMBLE MODEL: {away} @ {home}")
        print('='*55)
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        
        print(f"\nComponent Models:")
        for name, comp in sorted(pred['component_predictions'].items(), 
                                key=lambda x: -x[1]['weight']):
            print(f"  {name}: {comp['home_prob']*100:.1f}% (w={comp['weight']:.2f})")
        
        # Weather info
        weather = pred.get('weather')
        if weather and (weather.get('has_data') or weather_data):
            print(f"\n🌤️ Weather Impact:")
            comp = weather.get('components', {})
            if comp.get('dome'):
                print(f"  Dome: Indoor game, no weather adjustment")
            else:
                mult = weather.get('multiplier', 1.0)
                if mult != 1.0:
                    print(f"  Poisson run multiplier: {mult:.3f}x")
                if 'temp_f' in comp:
                    print(f"  Temperature: {comp['temp_f']:.0f}°F")
                if 'wind_effect' in comp:
                    print(f"  Wind: {comp.get('wind_speed_mph', 0):.0f} mph ({comp['wind_effect']})")
        
    else:
        print("Usage:")
        print("  python ensemble_model.py <home_team> <away_team> [--neutral] [--game-id ID] [--temp F] [--wind MPH] [--wind-dir DEG] [--humidity %] [--dome]")
        print("  python ensemble_model.py report   # Show weights and accuracy")
        print("  python ensemble_model.py reset    # Reset accuracy history")
        print("\nCurrent Weights:")
        for name, weight in sorted(model.weights.items(), key=lambda x: -x[1]):
            print(f"  {name}: {weight*100:.1f}%")
