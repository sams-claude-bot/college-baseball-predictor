#!/usr/bin/env python3
"""Prediction orchestration and main EnsembleModel class."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from .rebalance import EnsembleRebalanceMixin
from .weights import EnsembleWeightsMixin

try:
    from models.neural_model import NeuralModel
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

try:
    from models.xgboost_model import XGBMoneylineModel
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from models.lightgbm_model import LGBMoneylineModel
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


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

class EnsembleModel(EnsembleRebalanceMixin, EnsembleWeightsMixin, BaseModel):
    """
    Dynamic ensemble that combines all prediction models.

    Weights adjust based on each model's recent prediction accuracy.
    """
    name = "ensemble"
    version = "3.1"
    description = "Dynamic weighted ensemble with accuracy-based weight adjustment"
    MIN_WEIGHT = 0.03
    ROLLING_WINDOW = 100
    ADJUSTMENT_RATE = 1.0  # Jump directly to accuracy-based target weights
    SAMPLE_RAMP_FLOOR = 0.35
    SAMPLE_RAMP_THRESHOLD = 40
    PRESEASON_MODELS = {"prior", "conference"}
    PRESEASON_DECAY_GAMES = 200
    RECENCY_RANK_DECAY_BASE = 0.977      # ~30 predictions half-life
    RECENCY_DAY_HALFLIFE_DAYS = 21.0     # Set <=0/None to disable
    STATS_DEPENDENT_MODELS = {"log5", "poisson", "advanced", "pythagorean"}
    STATS_MATURITY_GAMES = 100  # predictions per model before full trust
    EARLY_SEASON_FLOOR = 0.02   # near-zero weight until stats mature

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

        # Default weights (updated 2026-02-18)
        # Early season: lean heavily on Elo, pitching, and ML models.
        # Stats-dependent models (log5, poisson, advanced, pythagorean) get
        # dampened via STATS_DEPENDENT_MODELS until enough games accumulate.
        self.default_weights = {
            "elo": 0.20,          # 77.4% accuracy, strong baseline
            "pitching": 0.15,     # 67.7% — decent but not top tier
            "lightgbm": 0.05,     # 71% — lowered, underperforming heuristics
            "xgboost": 0.05,      # 67.7% — lowered, underperforming heuristics
            "prior": 0.08,        # 80.6% — preseason priors, decays with data
            "conference": 0.08,   # 80.6% — conference strength
            "advanced": 0.12,     # 80.6% — stats-dependent, rising with data
            "pythagorean": 0.08,  # 80.6% — stats-dependent, rising with data
            "log5": 0.10,         # 80.6% — stats-dependent, rising with data
            "poisson": 0.09,      # 77.4% — stats-dependent
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
                total_weight += weight

        # Normalize probability
        if total_weight > 0 and total_weight < 1.0:
            home_prob /= total_weight

        # Use runs ensemble for run projections (designed for totals accuracy)
        # instead of averaging all models' crude run estimates
        try:
            from models.runs_ensemble import predict as runs_predict
            runs_result = runs_predict(home_team_id, away_team_id)
            home_runs = runs_result['projected_home_runs']
            away_runs = runs_result['projected_away_runs']
        except Exception:
            # Fallback: average from models that produce runs
            runs_weight = 0
            for name, pred in predictions.items():
                w = self.weights.get(name, 0)
                if w > 0 and pred.get('projected_home_runs') is not None and pred['projected_home_runs'] > 0:
                    home_runs += pred['projected_home_runs'] * w
                    away_runs += pred['projected_away_runs'] * w
                    runs_weight += w
            if runs_weight > 0:
                home_runs /= runs_weight
                away_runs /= runs_weight

        home_prob = max(0.02, min(0.98, home_prob))
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
                home_prob_adjusted = max(0.02, min(0.98, home_prob_adjusted))

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


def main():
    import argparse

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
        for name, comp in sorted(pred['component_predictions'].items(), key=lambda x: -x[1]['weight']):
            print(f"  {name}: {comp['home_prob']*100:.1f}% (w={comp['weight']:.2f})")

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


if __name__ == "__main__":
    main()
