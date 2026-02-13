#!/usr/bin/env python3
"""
College Baseball Prediction Model (Database Version)

Uses SQLite database for all team stats and game data.
"""

import sys
import math
from pathlib import Path

# Add scripts to path for database module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from database import get_connection, get_team_record, get_team_runs, get_recent_games

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
    """Main prediction engine using database"""
    
    WEIGHTS = {
        'win_pct': 0.25,
        'pythagorean': 0.20,
        'recent_form': 0.20,
        'batting_avg': 0.10,
        'era': 0.10,
        'home_advantage': 0.10,
        'preseason_rank': 0.05
    }
    
    HOME_ADVANTAGE = 0.54
    NEUTRAL_SITE_ADVANTAGE = 0.50  # No home advantage at neutral sites
    
    def __init__(self):
        self.team_cache = {}
    
    def get_team_stats(self, team_id):
        """Get or load team stats"""
        # Normalize team ID
        team_id = team_id.lower().replace(" ", "-")
        
        if team_id not in self.team_cache:
            self.team_cache[team_id] = TeamStats(team_id)
        return self.team_cache[team_id]
    
    def predict_game(self, home_team, away_team, neutral_site=False):
        """Predict game outcome"""
        home_id = home_team.lower().replace(" ", "-")
        away_id = away_team.lower().replace(" ", "-")
        
        home = self.get_team_stats(home_id)
        away = self.get_team_stats(away_id)
        
        # Calculate base probability
        home_score = 0.5
        
        # Win percentage differential
        win_diff = (home.win_pct - away.win_pct) / 2
        home_score += win_diff * self.WEIGHTS['win_pct']
        
        # Pythagorean differential
        pyth_diff = (home.pythagorean_win_pct - away.pythagorean_win_pct) / 2
        home_score += pyth_diff * self.WEIGHTS['pythagorean']
        
        # Recent form differential
        form_diff = (home.recent_form() - away.recent_form()) / 2
        home_score += form_diff * self.WEIGHTS['recent_form']
        
        # Home/neutral advantage
        if neutral_site:
            home_score += (self.NEUTRAL_SITE_ADVANTAGE - 0.5) * self.WEIGHTS['home_advantage']
        else:
            home_score += (self.HOME_ADVANTAGE - 0.5) * self.WEIGHTS['home_advantage']
        
        home_win_prob = max(0.1, min(0.9, home_score))
        
        # Project runs
        home_runs_proj = (home.runs_per_game + away.runs_allowed_per_game) / 2
        away_runs_proj = (away.runs_per_game + home.runs_allowed_per_game) / 2
        
        if not neutral_site:
            home_runs_proj *= 1.02
            away_runs_proj *= 0.98
        
        # Run line calculation
        projected_diff = home_runs_proj - away_runs_proj
        run_diff_std = 3.5
        
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        z_home_cover = (projected_diff - 1.5) / run_diff_std
        home_cover_prob = norm_cdf(z_home_cover)
        
        z_away_cover = (-projected_diff + 1.5) / run_diff_std
        away_cover_prob = norm_cdf(z_away_cover)
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral_site,
            "home_win_probability": round(home_win_prob, 3),
            "away_win_probability": round(1 - home_win_prob, 3),
            "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
            "confidence": round(abs(home_win_prob - 0.5) * 2, 3),
            "projected_home_runs": round(home_runs_proj, 1),
            "projected_away_runs": round(away_runs_proj, 1),
            "projected_total": round(home_runs_proj + away_runs_proj, 1),
            "projected_run_diff": round(projected_diff, 1),
            "run_line": {
                "home_minus_1_5": round(home_cover_prob, 3),
                "away_plus_1_5": round(away_cover_prob, 3),
                "pick": f"{home_team} -1.5" if home_cover_prob > 0.5 else f"{away_team} +1.5"
            },
            "model_inputs": {
                "home_record": f"{home.wins}-{home.losses}",
                "away_record": f"{away.wins}-{away.losses}",
                "home_pythagorean": round(home.pythagorean_win_pct, 3),
                "away_pythagorean": round(away.pythagorean_win_pct, 3),
                "home_recent_form": round(home.recent_form(), 3),
                "away_recent_form": round(away.recent_form(), 3)
            }
        }
    
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

def main():
    predictor = Predictor()
    
    if len(sys.argv) > 2:
        home = sys.argv[1]
        away = sys.argv[2]
        neutral = "--neutral" in sys.argv
        
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
        print(f"   {home} -1.5: {rl['home_minus_1_5']*100:.1f}%")
        print(f"   {away} +1.5: {rl['away_plus_1_5']*100:.1f}%")
        print(f"   → Pick: {rl['pick']}")
        
        print(f"\n📈 PROJECTED SCORE")
        print(f"   {away}: {pred['projected_away_runs']:.1f}")
        print(f"   {home}: {pred['projected_home_runs']:.1f}")
        print(f"   Total: {pred['projected_total']:.1f}")
        
        print(f"\n🏆 SERIES (Best of 3)")
        series = predictor.predict_series(home, away, neutral_site=neutral)
        print(f"   {home}: {series['home_series_probability']*100:.1f}%")
        print(f"   {away}: {series['away_series_probability']*100:.1f}%")
        print(f"   → Pick: {series['predicted_series_winner']}")
        
        print(f"\n📋 Model Inputs:")
        for k, v in pred['model_inputs'].items():
            print(f"   {k}: {v}")
        print()
    else:
        print("Usage: python predictor_db.py <home_team> <away_team> [--neutral]")
        print("\nExamples:")
        print("  python predictor_db.py 'Mississippi State' 'Hofstra'")
        print("  python predictor_db.py 'Mississippi State' 'UCLA' --neutral")

if __name__ == "__main__":
    main()
