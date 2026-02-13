#!/usr/bin/env python3
"""
Ensemble Model

Combines multiple models with configurable weights.
Default: Equal weight to all models.
Can be tuned based on historical accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from base_model import BaseModel
from pythagorean_model import PythagoreanModel
from elo_model import EloModel
from log5_model import Log5Model

class EnsembleModel(BaseModel):
    name = "ensemble"
    version = "1.0"
    description = "Weighted combination of multiple models"
    
    def __init__(self, weights=None):
        """
        Initialize with model weights
        
        weights: dict of {model_name: weight}
        Default: Equal weights
        """
        self.models = {
            "pythagorean": PythagoreanModel(),
            "elo": EloModel(),
            "log5": Log5Model()
        }
        
        if weights is None:
            # Equal weights
            n = len(self.models)
            self.weights = {name: 1/n for name in self.models}
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
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
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0)
            home_prob += pred['home_win_probability'] * weight
            home_runs += pred['projected_home_runs'] * weight
            away_runs += pred['projected_away_runs'] * weight
        
        home_prob = max(0.1, min(0.9, home_prob))
        run_line = self.calculate_run_line(home_runs, away_runs)
        
        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(home_runs, 1),
            "projected_away_runs": round(away_runs, 1),
            "projected_total": round(home_runs + away_runs, 1),
            "run_line": run_line,
            "component_predictions": {
                name: {
                    "home_prob": pred['home_win_probability'],
                    "weight": round(self.weights.get(name, 0), 3)
                }
                for name, pred in predictions.items()
            },
            "weights": self.weights
        }
    
    def set_weights(self, weights):
        """Update model weights"""
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
