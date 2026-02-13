#!/usr/bin/env python3
"""
College Baseball Prediction Model

Predicts:
- Game winners
- Series winners (best of 3)
- Projected runs scored
- Run differential

Features used:
- Team batting average
- Team ERA
- Win-loss record
- Recent performance (last 10 games)
- Home/away factor
- Head-to-head history
- Preseason ranking (weighted low)
"""

import json
from pathlib import Path
from datetime import datetime
import math

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

class TeamStats:
    """Container for team statistics"""
    def __init__(self, team_id):
        self.team_id = team_id
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.runs_scored = 0
        self.runs_allowed = 0
        self.batting_avg = 0.0
        self.era = 0.0
        self.preseason_rank = None
        self.recent_games = []  # Last 10 results
        
    @property
    def win_pct(self):
        if self.games_played == 0:
            return 0.5  # Default for no games
        return self.wins / self.games_played
    
    @property
    def runs_per_game(self):
        if self.games_played == 0:
            return 5.0  # League average estimate
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
        exponent = 1.83  # Baseball standard
        rs_exp = self.runs_scored ** exponent
        ra_exp = self.runs_allowed ** exponent
        return rs_exp / (rs_exp + ra_exp)
    
    def recent_form(self, n=5):
        """Win rate in last n games"""
        if not self.recent_games:
            return 0.5
        recent = self.recent_games[-n:]
        return sum(1 for g in recent if g['won']) / len(recent)

class Predictor:
    """Main prediction engine"""
    
    # Feature weights (can be tuned)
    WEIGHTS = {
        'win_pct': 0.25,
        'pythagorean': 0.20,
        'recent_form': 0.20,
        'batting_avg': 0.10,
        'era': 0.10,
        'home_advantage': 0.10,
        'preseason_rank': 0.05
    }
    
    HOME_ADVANTAGE = 0.54  # Home teams win ~54% in college baseball
    
    def __init__(self):
        self.team_stats = {}
        self.load_team_data()
    
    def load_team_data(self):
        """Load all available team stats"""
        games_file = DATA_DIR / "games" / "all_games.json"
        if games_file.exists():
            with open(games_file) as f:
                data = json.load(f)
            
            for game in data.get("games", []):
                self._process_game(game)
    
    def _process_game(self, game):
        """Update team stats from a game result"""
        home = game['home_team']
        away = game['away_team']
        
        if home not in self.team_stats:
            self.team_stats[home] = TeamStats(home)
        if away not in self.team_stats:
            self.team_stats[away] = TeamStats(away)
        
        home_stats = self.team_stats[home]
        away_stats = self.team_stats[away]
        
        if game.get('home_score') is not None and game.get('away_score') is not None:
            home_score = game['home_score']
            away_score = game['away_score']
            
            home_stats.games_played += 1
            away_stats.games_played += 1
            
            home_stats.runs_scored += home_score
            home_stats.runs_allowed += away_score
            away_stats.runs_scored += away_score
            away_stats.runs_allowed += home_score
            
            home_won = home_score > away_score
            if home_won:
                home_stats.wins += 1
                away_stats.losses += 1
            else:
                away_stats.wins += 1
                home_stats.losses += 1
            
            home_stats.recent_games.append({'won': home_won, 'date': game['date']})
            away_stats.recent_games.append({'won': not home_won, 'date': game['date']})
    
    def get_team_stats(self, team_id):
        """Get or create team stats"""
        if team_id not in self.team_stats:
            self.team_stats[team_id] = TeamStats(team_id)
        return self.team_stats[team_id]
    
    def predict_game(self, home_team, away_team):
        """
        Predict game outcome
        Returns: dict with predictions
        """
        home = self.get_team_stats(home_team)
        away = self.get_team_stats(away_team)
        
        # Calculate feature values
        home_score = 0.5  # Start neutral
        
        # Win percentage differential
        win_diff = (home.win_pct - away.win_pct) / 2
        home_score += win_diff * self.WEIGHTS['win_pct']
        
        # Pythagorean differential  
        pyth_diff = (home.pythagorean_win_pct - away.pythagorean_win_pct) / 2
        home_score += pyth_diff * self.WEIGHTS['pythagorean']
        
        # Recent form differential
        form_diff = (home.recent_form() - away.recent_form()) / 2
        home_score += form_diff * self.WEIGHTS['recent_form']
        
        # Home advantage
        home_score += (self.HOME_ADVANTAGE - 0.5) * self.WEIGHTS['home_advantage']
        
        # Clamp to valid probability
        home_win_prob = max(0.1, min(0.9, home_score))
        
        # Project runs
        home_runs_proj = (home.runs_per_game + away.runs_allowed_per_game) / 2
        away_runs_proj = (away.runs_per_game + home.runs_allowed_per_game) / 2
        
        # Apply home advantage to runs
        home_runs_proj *= 1.02
        away_runs_proj *= 0.98
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": round(home_win_prob, 3),
            "away_win_probability": round(1 - home_win_prob, 3),
            "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
            "confidence": round(abs(home_win_prob - 0.5) * 2, 3),
            "projected_home_runs": round(home_runs_proj, 1),
            "projected_away_runs": round(away_runs_proj, 1),
            "projected_total": round(home_runs_proj + away_runs_proj, 1),
            "model_inputs": {
                "home_record": f"{home.wins}-{home.losses}",
                "away_record": f"{away.wins}-{away.losses}",
                "home_pythagorean": round(home.pythagorean_win_pct, 3),
                "away_pythagorean": round(away.pythagorean_win_pct, 3),
                "home_recent_form": round(home.recent_form(), 3),
                "away_recent_form": round(away.recent_form(), 3)
            }
        }
    
    def predict_series(self, home_team, away_team, games=3):
        """Predict a series (typically best of 3)"""
        game_pred = self.predict_game(home_team, away_team)
        p = game_pred['home_win_probability']
        
        # For 3-game series, home team wins if they win 2+ games
        # P(win series) = P(win 2) + P(win 3)
        # = C(3,2)*p^2*(1-p) + p^3
        if games == 3:
            series_prob = 3 * (p**2) * (1-p) + p**3
        else:
            # Generic calculation for odd-numbered series
            wins_needed = (games // 2) + 1
            series_prob = 0
            for wins in range(wins_needed, games + 1):
                combinations = math.comb(games, wins)
                series_prob += combinations * (p ** wins) * ((1-p) ** (games - wins))
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "games": games,
            "home_series_probability": round(series_prob, 3),
            "away_series_probability": round(1 - series_prob, 3),
            "predicted_series_winner": home_team if series_prob > 0.5 else away_team,
            "per_game_home_probability": round(p, 3)
        }

def main():
    import sys
    
    predictor = Predictor()
    
    if len(sys.argv) > 2:
        home = sys.argv[1]
        away = sys.argv[2]
        
        print(f"\n=== Game Prediction: {away} @ {home} ===")
        pred = predictor.predict_game(home, away)
        print(f"Predicted Winner: {pred['predicted_winner']} ({pred['home_win_probability']*100:.1f}% home)")
        print(f"Confidence: {pred['confidence']*100:.1f}%")
        print(f"Projected Score: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nModel Inputs:")
        for k, v in pred['model_inputs'].items():
            print(f"  {k}: {v}")
        
        print(f"\n=== Series Prediction (Best of 3) ===")
        series = predictor.predict_series(home, away)
        print(f"Predicted Series Winner: {series['predicted_series_winner']}")
        print(f"Home Series Win Probability: {series['home_series_probability']*100:.1f}%")
    else:
        print("Usage: python predictor.py <home_team> <away_team>")
        print("\nExample: python predictor.py 'Mississippi State' 'Hofstra'")

if __name__ == "__main__":
    main()
