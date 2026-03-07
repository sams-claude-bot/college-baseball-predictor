#!/usr/bin/env python3
"""
College Baseball Prediction Model (Database Version)

Main entry point for predictions. Uses all available models
through the dynamic ensemble.

Features:
- Multiple prediction models (Pythagorean, Elo, Log5, Advanced, Pitching, Conference, Prior)
- Dynamic ensemble weighting based on model accuracy
- Series predictions
- Tournament/neutral site handling
- Full model comparison mode

Usage:
    python predictor_db.py "Mississippi State" "Hofstra"
    python predictor_db.py "Mississippi State" "UCLA" --neutral
    python predictor_db.py "Mississippi State" "Arkansas" --compare
    python predictor_db.py --models  # List all models
"""

import sys
import math
from pathlib import Path

# Add scripts to path for database module
# sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))  # Removed by cleanup
# sys.path.insert(0, str(Path(__file__).parent))  # Removed by cleanup

from scripts.database import get_connection, get_team_record, get_team_runs, get_recent_games

# Import all models
from models.ensemble_model import EnsembleModel, PoissonModelWrapper
from models.pythagorean_model import PythagoreanModel
from models.elo_model import EloModel
from models.log5_model import Log5Model
from models.advanced_model import AdvancedModel
from models.pitching_model import PitchingModel
from models.conference_model import ConferenceModel
from models.prior_model import PriorModel
from models.xgboost_model import XGBMoneylineModel
from models.lightgbm_model import LGBMoneylineModel
from models.pear_model import PearModel
from models.quality_model import QualityModel
from models.venue_model import VenueModel
from models.rest_travel_model import RestTravelModel
from models.upset_model import UpsetModel


class TeamStats:
    """Container for team statistics loaded from database"""
    def __init__(self, team_id):
        self.team_id = team_id
        self._load_from_db()
    
    def _load_from_db(self):
        # Get record
        record = get_team_record(self.team_id)
        self.wins = record['wins']
        self.losses = record['losses']
        self.games_played = self.wins + self.losses
        
        # Get runs
        runs = get_team_runs(self.team_id)
        self.runs_scored = runs['runs_scored']
        self.runs_allowed = runs['runs_allowed']
        
        # Get recent games for form calculation
        recent = get_recent_games(self.team_id, limit=10)
        self.recent_games = []
        for g in recent:
            won = g['winner_id'] == self.team_id
            self.recent_games.append({'won': won, 'date': g['date']})
    
    @property
    def win_pct(self):
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played
    
    @property
    def runs_per_game(self):
        if self.games_played == 0:
            return 5.0
        return self.runs_scored / self.games_played
    
    @property
    def runs_allowed_per_game(self):
        if self.games_played == 0:
            return 5.0
        return self.runs_allowed / self.games_played
    
    @property
    def pythagorean_win_pct(self):
        """Bill James Pythagorean expectation"""
        if self.runs_scored == 0 or self.runs_allowed == 0:
            return self.win_pct
        exponent = 1.83
        rs_exp = self.runs_scored ** exponent
        ra_exp = self.runs_allowed ** exponent
        return rs_exp / (rs_exp + ra_exp)
    
    def recent_form(self, n=5):
        """Win rate in last n games"""
        if not self.recent_games:
            return 0.5
        recent = self.recent_games[:n]  # Already sorted by date desc
        return sum(1 for g in recent if g['won']) / len(recent) if recent else 0.5


class Predictor:
    """Main prediction engine using database and all models"""
    
    # Available models
    MODELS = {
        'pythagorean': PythagoreanModel,
        'elo': EloModel,
        'log5': Log5Model,
        'advanced': AdvancedModel,
        'pitching': PitchingModel,
        'conference': ConferenceModel,
        'prior': PriorModel,
        'poisson': PoissonModelWrapper,
        'xgboost': XGBMoneylineModel,
        'lightgbm': LGBMoneylineModel,
        'pear': PearModel,
        'quality': QualityModel,
        'venue': VenueModel,
        'rest_travel': RestTravelModel,
        'upset': UpsetModel,
        'ensemble': EnsembleModel,
        'neural': None  # Lazy-loaded due to PyTorch dependency
    }
    
    def __init__(self, model='ensemble'):
        """
        Initialize predictor with specified model.
        
        Args:
            model: Model name or 'ensemble' (default)
        """
        self.team_cache = {}
        
        if model == 'ensemble':
            self.model = EnsembleModel()
        elif model == 'neural':
            from models.neural_model import NeuralModel
            self.model = NeuralModel()
        elif model in self.MODELS and self.MODELS[model] is not None:
            self.model = self.MODELS[model]()
        else:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")
        
        self.model_name = model
    
    @classmethod
    def list_models(cls):
        """List available models with descriptions"""
        print("\nAvailable Models:")
        print("-" * 60)
        for name, model_cls in cls.MODELS.items():
            m = model_cls()
            print(f"  {name:<15} - {m.description}")
        print()
    
    def get_team_stats(self, team_id):
        """Get or load team stats"""
        # Normalize team ID
        team_id = team_id.lower().replace(" ", "-")
        
        if team_id not in self.team_cache:
            self.team_cache[team_id] = TeamStats(team_id)
        return self.team_cache[team_id]
    
    def predict_game(self, home_team, away_team, neutral_site=False):
        """Predict game outcome using configured model.

        Some specialty models (e.g. PEAR / quality) may return None when their
        source tables are missing a team. In that case, we gracefully fall back
        to a lightweight baseline so the pipeline stays complete.
        """
        home_id = home_team.lower().replace(" ", "-")
        away_id = away_team.lower().replace(" ", "-")

        # Get team stats for additional context
        home = self.get_team_stats(home_id)
        away = self.get_team_stats(away_id)

        # Get prediction from model
        pred = self.model.predict_game(home_id, away_id, neutral_site)
        fallback_used = False

        # Graceful fallback for models that return None for missing inputs.
        if pred is None:
            fallback_used = True
            base_prob = 0.5 + (home.win_pct - away.win_pct) * 0.35
            if not neutral_site:
                base_prob += 0.04
            base_prob = max(0.05, min(0.95, base_prob))

            # Simple run projection baseline from team scoring + opponent prevention.
            home_runs = 0.6 * home.runs_per_game + 0.4 * away.runs_allowed_per_game
            away_runs = 0.6 * away.runs_per_game + 0.4 * home.runs_allowed_per_game
            if not neutral_site:
                home_runs *= 1.03
                away_runs *= 0.97

            pred = {
                'model': self.model_name,
                'home_win_probability': round(base_prob, 3),
                'away_win_probability': round(1 - base_prob, 3),
                'projected_home_runs': round(home_runs, 1),
                'projected_away_runs': round(away_runs, 1),
                'projected_total': round(home_runs + away_runs, 1),
                'inputs': {'fallback_reason': 'model_returned_none'},
            }

        if not isinstance(pred, dict):
            raise TypeError(f"Model {self.model_name} returned {type(pred).__name__}, expected dict")

        # Defensive defaults if a model returns a partial dict.
        pred = dict(pred)
        if pred.get('home_win_probability') is None:
            pred['home_win_probability'] = 0.5
        pred['home_win_probability'] = float(pred['home_win_probability'])
        pred['away_win_probability'] = float(pred.get('away_win_probability', 1 - pred['home_win_probability']))
        if pred.get('projected_home_runs') is None:
            pred['projected_home_runs'] = round(home.runs_per_game, 1)
        if pred.get('projected_away_runs') is None:
            pred['projected_away_runs'] = round(away.runs_per_game, 1)
        if pred.get('projected_total') is None:
            pred['projected_total'] = round(pred['projected_home_runs'] + pred['projected_away_runs'], 1)

        # Add team info to prediction
        pred['home_team'] = home_team
        pred['away_team'] = away_team
        pred['neutral_site'] = neutral_site
        pred['predicted_winner'] = home_team if pred['home_win_probability'] > 0.5 else away_team
        pred['confidence'] = round(abs(pred['home_win_probability'] - 0.5) * 2, 3)

        # Add basic team stats
        if 'model_inputs' not in pred or not isinstance(pred.get('model_inputs'), dict):
            pred['model_inputs'] = {}

        pred['model_inputs']['home_record'] = f"{home.wins}-{home.losses}"
        pred['model_inputs']['away_record'] = f"{away.wins}-{away.losses}"
        if fallback_used:
            pred['model_inputs']['fallback_used'] = True

        return pred
    
    def predict_series(self, home_team, away_team, games=3, neutral_site=False):
        """Predict a series outcome"""
        game_pred = self.predict_game(home_team, away_team, neutral_site)
        p = game_pred['home_win_probability']
        
        if games == 3:
            series_prob = 3 * (p**2) * (1-p) + p**3
        else:
            wins_needed = (games // 2) + 1
            series_prob = 0
            for wins in range(wins_needed, games + 1):
                combinations = math.comb(games, wins)
                series_prob += combinations * (p ** wins) * ((1-p) ** (games - wins))
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "games": games,
            "neutral_site": neutral_site,
            "home_series_probability": round(series_prob, 3),
            "away_series_probability": round(1 - series_prob, 3),
            "predicted_series_winner": home_team if series_prob > 0.5 else away_team,
            "per_game_home_probability": round(p, 3)
        }
    
    def predict_tournament_matchup(self, team1, team2, venue_team=None):
        """Predict a tournament/neutral site game"""
        neutral = venue_team is None or venue_team not in [team1, team2]
        
        if neutral:
            return self.predict_game(team1, team2, neutral_site=True)
        elif venue_team == team1:
            return self.predict_game(team1, team2, neutral_site=False)
        else:
            return self.predict_game(team2, team1, neutral_site=False)
    
    def compare_models(self, home_team, away_team, neutral_site=False):
        """Get predictions from all models for comparison"""
        home_id = home_team.lower().replace(" ", "-")
        away_id = away_team.lower().replace(" ", "-")
        
        results = {}
        for name, model_cls in self.MODELS.items():
            if name == 'ensemble':
                continue  # Skip ensemble for comparison
            try:
                model = model_cls()
                pred = model.predict_game(home_id, away_id, neutral_site)
                results[name] = {
                    'home_prob': pred['home_win_probability'],
                    'home_runs': pred['projected_home_runs'],
                    'away_runs': pred['projected_away_runs'],
                    'inputs': pred.get('inputs', {})
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        # Add ensemble
        ensemble = EnsembleModel()
        ens_pred = ensemble.predict_game(home_id, away_id, neutral_site)
        results['ensemble'] = {
            'home_prob': ens_pred['home_win_probability'],
            'home_runs': ens_pred['projected_home_runs'],
            'away_runs': ens_pred['projected_away_runs'],
            'weights': ens_pred.get('weights', {})
        }
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'neutral_site': neutral_site,
            'models': results
        }


def print_comparison(comparison):
    """Print model comparison in a nice format"""
    home = comparison['home_team']
    away = comparison['away_team']
    neutral = " (Neutral Site)" if comparison['neutral_site'] else ""
    
    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON: {away} @ {home}{neutral}")
    print('='*65)
    
    print(f"\n{'Model':<15} {'Home Win':>10} {'Home Runs':>12} {'Away Runs':>12}")
    print('-'*65)
    
    # Sort by home probability descending
    sorted_models = sorted(
        comparison['models'].items(),
        key=lambda x: x[1].get('home_prob', 0) if 'error' not in x[1] else 0,
        reverse=True
    )
    
    for name, result in sorted_models:
        if 'error' in result:
            print(f"  {name:<13} ERROR: {result['error'][:40]}")
        else:
            prob = result['home_prob'] * 100
            hr = result['home_runs']
            ar = result['away_runs']
            marker = " ★" if name == 'ensemble' else ""
            print(f"  {name:<13} {prob:>9.1f}% {hr:>11.1f} {ar:>11.1f}{marker}")
    
    print('-'*65)
    
    # Average (excluding ensemble)
    valid = [r for n, r in comparison['models'].items() 
             if 'error' not in r and n != 'ensemble']
    if valid:
        avg_prob = sum(r['home_prob'] for r in valid) / len(valid)
        avg_hr = sum(r['home_runs'] for r in valid) / len(valid)
        avg_ar = sum(r['away_runs'] for r in valid) / len(valid)
        print(f"  {'Average':<13} {avg_prob*100:>9.1f}% {avg_hr:>11.1f} {avg_ar:>11.1f}")
    print()


def main():
    predictor = Predictor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--models":
        Predictor.list_models()
        return
    
    if len(sys.argv) > 2:
        home = sys.argv[1]
        away = sys.argv[2]
        neutral = "--neutral" in sys.argv
        compare = "--compare" in sys.argv
        
        if compare:
            comparison = predictor.compare_models(home, away, neutral)
            print_comparison(comparison)
            return
        
        site_label = " (Neutral Site)" if neutral else ""
        
        print(f"\n{'='*55}")
        print(f"  {away} @ {home}{site_label}")
        print('='*55)
        
        pred = predictor.predict_game(home, away, neutral_site=neutral)
        
        print(f"\n📊 MONEYLINE")
        print(f"   {home}: {pred['home_win_probability']*100:.1f}%")
        print(f"   {away}: {pred['away_win_probability']*100:.1f}%")
        print(f"   → Pick: {pred['predicted_winner']} ({pred['confidence']*100:.0f}% conf)")
        
        print(f"\n🎯 RUN LINE (-1.5)")
        rl = pred['run_line']
        print(f"   {home} -1.5: {rl['home_cover_prob']*100:.1f}%")
        print(f"   {away} +1.5: {rl['away_cover_prob']*100:.1f}%")
        
        print(f"\n📈 PROJECTED SCORE")
        print(f"   {away}: {pred['projected_away_runs']:.1f}")
        print(f"   {home}: {pred['projected_home_runs']:.1f}")
        print(f"   Total: {pred['projected_total']:.1f}")
        
        print(f"\n🏆 SERIES (Best of 3)")
        series = predictor.predict_series(home, away, neutral_site=neutral)
        print(f"   {home}: {series['home_series_probability']*100:.1f}%")
        print(f"   {away}: {series['away_series_probability']*100:.1f}%")
        print(f"   → Pick: {series['predicted_series_winner']}")
        
        # Show model details for ensemble
        if 'component_predictions' in pred:
            print(f"\n🔬 MODEL BREAKDOWN")
            for name, comp in sorted(pred['component_predictions'].items(),
                                    key=lambda x: -x[1]['weight']):
                prob = comp['home_prob'] * 100
                weight = comp['weight'] * 100
                print(f"   {name:<12}: {prob:>5.1f}% (w={weight:.0f}%)")
        
        print(f"\n📋 Model Inputs:")
        for k, v in pred.get('model_inputs', {}).items():
            print(f"   {k}: {v}")
        
        # Show inputs from specific models if available
        if 'inputs' in pred:
            print(f"\n📋 Additional Inputs:")
            for k, v in pred['inputs'].items():
                print(f"   {k}: {v}")
        
        print()
    else:
        print("Usage: python predictor_db.py <home_team> <away_team> [--neutral] [--compare]")
        print("\nExamples:")
        print("  python predictor_db.py 'Mississippi State' 'Hofstra'")
        print("  python predictor_db.py 'Mississippi State' 'UCLA' --neutral")
        print("  python predictor_db.py 'Mississippi State' 'Arkansas' --compare")
        print("\nOptions:")
        print("  --models   List all available models")
        print("  --neutral  Neutral site game")
        print("  --compare  Compare all models side by side")


if __name__ == "__main__":
    main()
