#!/usr/bin/env python3
"""
Pythagorean Expectation Model

Uses Bill James' Pythagorean theorem to estimate win probability
based on runs scored vs runs allowed.
"""

import sys
from pathlib import Path

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
# sys.path.insert(0, str(_models_dir))  # Removed by cleanup
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from models.base_model import BaseModel
from scripts.database import get_team_record, get_team_runs, get_recent_games

class PythagoreanModel(BaseModel):
    name = "pythagorean"
    version = "1.0"
    description = "Bill James Pythagorean expectation based on runs scored/allowed"
    
    EXPONENT = 1.83  # Baseball standard
    HOME_ADVANTAGE = 0.03  # Additive home advantage (~3%)
    NEUTRAL_ADVANTAGE = 0.50
    
    def __init__(self):
        self.cache = {}
    
    def _get_team_data(self, team_id):
        if team_id not in self.cache:
            record = get_team_record(team_id)
            runs = get_team_runs(team_id)
            recent = get_recent_games(team_id, limit=10)
            
            self.cache[team_id] = {
                'wins': record['wins'],
                'losses': record['losses'],
                'runs_scored': runs['runs_scored'],
                'runs_allowed': runs['runs_allowed'],
                'games': runs['games'],
                'recent': recent
            }
        return self.cache[team_id]
    
    def _pythagorean_pct(self, runs_scored, runs_allowed):
        if runs_scored == 0 or runs_allowed == 0:
            return 0.5
        rs_exp = runs_scored ** self.EXPONENT
        ra_exp = runs_allowed ** self.EXPONENT
        return rs_exp / (rs_exp + ra_exp)
    
    def _recent_form(self, team_id, n=5):
        data = self._get_team_data(team_id)
        recent = data['recent'][:n]
        if not recent:
            return 0.5
        wins = sum(1 for g in recent if g.get('winner_id') == team_id)
        return wins / len(recent)
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        home = self._get_team_data(home_team_id)
        away = self._get_team_data(away_team_id)
        
        # Pythagorean win expectancy
        home_pyth = self._pythagorean_pct(home['runs_scored'], home['runs_allowed'])
        away_pyth = self._pythagorean_pct(away['runs_scored'], away['runs_allowed'])
        
        # Base probability from Pythagorean
        if home_pyth + away_pyth > 0:
            base_prob = home_pyth / (home_pyth + away_pyth)
        else:
            base_prob = 0.5
        
        # Adjust for home/neutral
        if neutral_site:
            home_prob = base_prob
        else:
            # Additive home advantage
            home_prob = base_prob + self.HOME_ADVANTAGE
        
        home_prob = max(0.1, min(0.9, home_prob))
        
        # Project runs
        home_rpg = home['runs_scored'] / home['games'] if home['games'] > 0 else 5.0
        away_rpg = away['runs_scored'] / away['games'] if away['games'] > 0 else 5.0
        home_rapg = home['runs_allowed'] / home['games'] if home['games'] > 0 else 5.0
        away_rapg = away['runs_allowed'] / away['games'] if away['games'] > 0 else 5.0
        
        home_runs = (home_rpg + away_rapg) / 2
        away_runs = (away_rpg + home_rapg) / 2
        
        if not neutral_site:
            home_runs *= 1.02
            away_runs *= 0.98
        
        run_line = self.calculate_run_line(home_runs, away_runs)
        
        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(home_runs, 1),
            "projected_away_runs": round(away_runs, 1),
            "projected_total": round(home_runs + away_runs, 1),
            "run_line": run_line,
            "inputs": {
                "home_pythagorean": round(home_pyth, 3),
                "away_pythagorean": round(away_pyth, 3)
            }
        }
