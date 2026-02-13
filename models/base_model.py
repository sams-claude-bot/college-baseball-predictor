#!/usr/bin/env python3
"""
Base class for prediction models

All models inherit from this and implement predict_game()
"""

from abc import ABC, abstractmethod
import math

class BaseModel(ABC):
    """Abstract base class for prediction models"""
    
    name = "base"
    version = "1.0"
    description = "Base model"
    
    @abstractmethod
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        """
        Predict a single game
        
        Returns dict with at minimum:
        - home_win_probability: float 0-1
        - projected_home_runs: float
        - projected_away_runs: float
        """
        pass
    
    def predict_series(self, home_team_id, away_team_id, games=3, neutral_site=False):
        """Predict a series based on single game probability"""
        pred = self.predict_game(home_team_id, away_team_id, neutral_site)
        p = pred['home_win_probability']
        
        if games == 3:
            series_prob = 3 * (p**2) * (1-p) + p**3
        else:
            wins_needed = (games // 2) + 1
            series_prob = 0
            for wins in range(wins_needed, games + 1):
                combinations = math.comb(games, wins)
                series_prob += combinations * (p ** wins) * ((1-p) ** (games - wins))
        
        return {
            "home_series_probability": round(series_prob, 3),
            "away_series_probability": round(1 - series_prob, 3),
            "per_game_probability": round(p, 3)
        }
    
    def calculate_run_line(self, home_runs, away_runs, spread=1.5):
        """Calculate run line probabilities"""
        projected_diff = home_runs - away_runs
        run_diff_std = 3.5  # Typical std dev for college baseball
        
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        z_home_cover = (projected_diff - spread) / run_diff_std
        home_cover_prob = norm_cdf(z_home_cover)
        
        return {
            "home_cover_prob": round(home_cover_prob, 3),
            "away_cover_prob": round(1 - home_cover_prob, 3),
            "spread": spread
        }
    
    def get_info(self):
        """Return model metadata"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description
        }
