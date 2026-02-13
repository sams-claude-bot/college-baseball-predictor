#!/usr/bin/env python3
"""
Log5 Model

Bill James' Log5 formula for head-to-head probability
based on each team's winning percentage.

P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)

Where pA and pB are the teams' winning percentages.
"""

import sys
from pathlib import Path

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
sys.path.insert(0, str(_models_dir))
sys.path.insert(0, str(_scripts_dir))

from base_model import BaseModel
from database import get_team_record, get_team_runs

class Log5Model(BaseModel):
    name = "log5"
    version = "1.0"
    description = "Bill James Log5 formula based on winning percentage"
    
    HOME_ADVANTAGE = 0.04  # Add 4% to home team
    DEFAULT_WIN_PCT = 0.5
    
    def __init__(self):
        self.cache = {}
    
    def _get_team_data(self, team_id):
        if team_id not in self.cache:
            record = get_team_record(team_id)
            runs = get_team_runs(team_id)
            
            games = record['wins'] + record['losses']
            win_pct = record['wins'] / games if games > 0 else self.DEFAULT_WIN_PCT
            
            self.cache[team_id] = {
                'wins': record['wins'],
                'losses': record['losses'],
                'games': games,
                'win_pct': win_pct,
                'runs_scored': runs['runs_scored'],
                'runs_allowed': runs['runs_allowed']
            }
        return self.cache[team_id]
    
    def _log5(self, pA, pB):
        """
        Log5 formula
        P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
        """
        numerator = pA - (pA * pB)
        denominator = pA + pB - (2 * pA * pB)
        
        if denominator == 0:
            return 0.5
        
        return numerator / denominator
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        home = self._get_team_data(home_team_id)
        away = self._get_team_data(away_team_id)
        
        home_pct = home['win_pct']
        away_pct = away['win_pct']
        
        # Apply home advantage to home team's effective win pct
        if not neutral_site:
            home_pct = min(0.95, home_pct + self.HOME_ADVANTAGE)
        
        # Log5 calculation
        home_prob = self._log5(home_pct, away_pct)
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
                "home_win_pct": round(home['win_pct'], 3),
                "away_win_pct": round(away['win_pct'], 3),
                "home_record": f"{home['wins']}-{home['losses']}",
                "away_record": f"{away['wins']}-{away['losses']}"
            }
        }
