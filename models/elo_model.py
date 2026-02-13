#!/usr/bin/env python3
"""
Elo Rating Model

Uses Elo ratings (like chess/FiveThirtyEight) to predict games.
Ratings update after each game based on expected vs actual outcome.
"""

import sys
import json
import math
from pathlib import Path

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
sys.path.insert(0, str(_models_dir))
sys.path.insert(0, str(_scripts_dir))

from base_model import BaseModel
from database import get_connection

class EloModel(BaseModel):
    name = "elo"
    version = "1.0"
    description = "Elo rating system (FiveThirtyEight style)"
    
    BASE_RATING = 1500
    K_FACTOR = 32  # How much ratings change per game
    HOME_ADVANTAGE = 50  # Elo points for home team
    
    def __init__(self):
        self.ratings = {}
        self._load_ratings()
    
    def _load_ratings(self):
        """Load existing ratings from database or initialize"""
        conn = get_connection()
        c = conn.cursor()
        
        # Check if elo_ratings table exists
        c.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='elo_ratings'
        """)
        
        if not c.fetchone():
            # Create table
            c.execute('''
                CREATE TABLE elo_ratings (
                    team_id TEXT PRIMARY KEY,
                    rating REAL DEFAULT 1500,
                    games_played INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        
        # Load all ratings
        c.execute("SELECT team_id, rating FROM elo_ratings")
        for row in c.fetchall():
            self.ratings[row[0]] = row[1]
        
        conn.close()
    
    def _get_rating(self, team_id):
        """Get team's Elo rating, initialize if new"""
        if team_id not in self.ratings:
            self.ratings[team_id] = self.BASE_RATING
            self._save_rating(team_id, self.BASE_RATING)
        return self.ratings[team_id]
    
    def _save_rating(self, team_id, rating):
        """Save rating to database"""
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO elo_ratings (team_id, rating)
            VALUES (?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                rating = excluded.rating,
                games_played = games_played + 1,
                updated_at = CURRENT_TIMESTAMP
        ''', (team_id, rating))
        conn.commit()
        conn.close()
    
    def _expected_score(self, rating_a, rating_b):
        """Calculate expected score (win probability) for team A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team_id, away_team_id, home_won, margin=None):
        """
        Update ratings after a game
        
        margin: Optional run differential for margin-of-victory adjustment
        """
        home_rating = self._get_rating(home_team_id)
        away_rating = self._get_rating(away_team_id)
        
        # Adjust for home advantage
        home_adj_rating = home_rating + self.HOME_ADVANTAGE
        
        # Expected scores
        home_expected = self._expected_score(home_adj_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Actual scores
        home_actual = 1.0 if home_won else 0.0
        away_actual = 1.0 - home_actual
        
        # K-factor adjustment for margin of victory (optional)
        k = self.K_FACTOR
        if margin is not None:
            # Increase K for blowouts, decrease for close games
            mov_mult = math.log(abs(margin) + 1) * 0.5 + 0.5
            k = self.K_FACTOR * min(mov_mult, 2.0)
        
        # Update ratings
        new_home = home_rating + k * (home_actual - home_expected)
        new_away = away_rating + k * (away_actual - away_expected)
        
        self.ratings[home_team_id] = new_home
        self.ratings[away_team_id] = new_away
        
        self._save_rating(home_team_id, new_home)
        self._save_rating(away_team_id, new_away)
        
        return {
            "home_old": round(home_rating, 1),
            "home_new": round(new_home, 1),
            "home_change": round(new_home - home_rating, 1),
            "away_old": round(away_rating, 1),
            "away_new": round(new_away, 1),
            "away_change": round(new_away - away_rating, 1)
        }
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        home_rating = self._get_rating(home_team_id)
        away_rating = self._get_rating(away_team_id)
        
        # Adjust for home/neutral
        if neutral_site:
            adj_home_rating = home_rating
        else:
            adj_home_rating = home_rating + self.HOME_ADVANTAGE
        
        home_prob = self._expected_score(adj_home_rating, away_rating)
        home_prob = max(0.1, min(0.9, home_prob))
        
        # Project runs based on rating differential
        # Higher rated teams score more, allow less
        rating_diff = (home_rating - away_rating) / 100
        base_runs = 5.5  # Average runs per game
        
        home_runs = base_runs + rating_diff * 0.5
        away_runs = base_runs - rating_diff * 0.5
        
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
                "home_elo": round(home_rating, 1),
                "away_elo": round(away_rating, 1),
                "rating_diff": round(home_rating - away_rating, 1)
            }
        }
    
    def get_top_ratings(self, n=25):
        """Get top N teams by Elo rating"""
        sorted_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_teams[:n]
    
    def initialize_from_rankings(self, rankings):
        """
        Initialize Elo ratings from preseason rankings
        
        rankings: list of team_ids in rank order
        """
        # Spread ratings from 1700 (#1) to 1400 (#25)
        for i, team_id in enumerate(rankings):
            rating = 1700 - (i * 12)  # ~12 points per rank
            self.ratings[team_id] = rating
            self._save_rating(team_id, rating)
        
        return self.ratings
