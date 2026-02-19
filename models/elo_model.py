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
# sys.path.insert(0, str(_models_dir))  # Removed by cleanup
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from models.base_model import BaseModel
from scripts.database import get_connection
from config.model_config import (
    ELO_BASE_RATING, ELO_K_FACTOR, HOME_ADVANTAGE_ELO,
    ELO_CONFERENCE_TIERS, ELO_MOV_MULTIPLIER_CAP,
    ELO_TEAM_START_OVERRIDES,
    ELO_FULLY_TRACKED_CONFERENCES,
    ELO_UNTRACKED_DECAY_FACTOR,
    ELO_UNTRACKED_DECAY_TARGET,
    ELO_LOW_SAMPLE_MAX_GAMES,
    ELO_LOW_SAMPLE_START_RATING,
    ELO_TOP25_SEED_MAX,
    ELO_TOP25_SEED_STEP,
)

class EloModel(BaseModel):
    name = "elo"
    version = "1.0"
    description = "Elo rating system (FiveThirtyEight style)"
    
    BASE_RATING = ELO_BASE_RATING
    K_FACTOR = ELO_K_FACTOR
    HOME_ADVANTAGE = HOME_ADVANTAGE_ELO
    
    def __init__(self):
        self.ratings = {}
        self._conference_cache = {}
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
    
    # Conference-tiered starting Elo for new teams (from shared config)
    CONF_ELO = ELO_CONFERENCE_TIERS

    def _get_rating(self, team_id):
        """Get team's Elo rating, initialize with conference-tiered default if new"""
        if team_id not in self.ratings:
            # Look up conference for tiered starting Elo
            try:
                conn = get_connection()
                row = conn.execute("SELECT conference, current_rank FROM teams WHERE id = ?", (team_id,)).fetchone()
                conf = row['conference'] if row and row['conference'] else ''
                current_rank = row['current_rank'] if row and row['current_rank'] else None

                gp_row = conn.execute('''
                    SELECT COUNT(*) AS gp
                    FROM games
                    WHERE status = 'final'
                      AND home_score IS NOT NULL
                      AND away_score IS NOT NULL
                      AND (home_team_id = ? OR away_team_id = ?)
                ''', (team_id, team_id)).fetchone()
                conn.close()

                games_played = gp_row['gp'] if gp_row and gp_row['gp'] is not None else 0

                if team_id in ELO_TEAM_START_OVERRIDES:
                    start_elo = ELO_TEAM_START_OVERRIDES[team_id]
                elif current_rank is not None and 1 <= int(current_rank) <= 25:
                    start_elo = ELO_TOP25_SEED_MAX - ((int(current_rank) - 1) * ELO_TOP25_SEED_STEP)
                elif games_played <= ELO_LOW_SAMPLE_MAX_GAMES:
                    start_elo = ELO_LOW_SAMPLE_START_RATING
                else:
                    start_elo = self.CONF_ELO.get(conf, self.BASE_RATING)
            except Exception:
                start_elo = ELO_TEAM_START_OVERRIDES.get(team_id, self.BASE_RATING)
            self.ratings[team_id] = start_elo
            self._save_rating(team_id, start_elo)
        return self.ratings[team_id]
    
    def _save_rating(self, team_id, rating, increment_games=False):
        """Save rating to database. Only increment games_played when a game result is processed."""
        conn = get_connection()
        c = conn.cursor()
        if increment_games:
            c.execute('''
                INSERT INTO elo_ratings (team_id, rating, games_played)
                VALUES (?, ?, 1)
                ON CONFLICT(team_id) DO UPDATE SET
                    rating = excluded.rating,
                    games_played = games_played + 1,
                    updated_at = CURRENT_TIMESTAMP
            ''', (team_id, rating))
        else:
            c.execute('''
                INSERT INTO elo_ratings (team_id, rating)
                VALUES (?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    rating = excluded.rating,
                    updated_at = CURRENT_TIMESTAMP
            ''', (team_id, rating))
        conn.commit()
        conn.close()
    
    def _log_history(self, team_id, rating, opponent_id, rating_change, game_id=None, game_date=None):
        """Log Elo rating change for historical charts."""
        try:
            conn = get_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO elo_history (team_id, rating, game_id, game_date, opponent_id, rating_change)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, game_id) DO UPDATE SET
                    rating = excluded.rating,
                    rating_change = excluded.rating_change,
                    game_date = excluded.game_date,
                    opponent_id = excluded.opponent_id,
                    created_at = CURRENT_TIMESTAMP
            ''', (team_id, round(rating, 1), game_id, game_date, opponent_id, round(rating_change, 1)))
            conn.commit()
            conn.close()
        except Exception:
            pass  # Don't break rating updates if history logging fails

    def _get_team_conference(self, team_id):
        """Get/cached conference for team_id."""
        if team_id in self._conference_cache:
            return self._conference_cache[team_id]
        conf = ''
        try:
            conn = get_connection()
            row = conn.execute("SELECT conference FROM teams WHERE id = ?", (team_id,)).fetchone()
            conn.close()
            conf = row['conference'] if row and row['conference'] else ''
        except Exception:
            conf = ''
        self._conference_cache[team_id] = conf
        return conf

    def _is_fully_tracked_team(self, team_id):
        conf = self._get_team_conference(team_id)
        return conf in ELO_FULLY_TRACKED_CONFERENCES

    def _apply_untracked_decay(self, team_id, rating):
        """
        Apply extra regression for teams outside fully tracked conferences.
        Pulls rating toward a conservative target each processed game.
        """
        if self._is_fully_tracked_team(team_id):
            return rating
        return ELO_UNTRACKED_DECAY_TARGET + (rating - ELO_UNTRACKED_DECAY_TARGET) * ELO_UNTRACKED_DECAY_FACTOR

    def _expected_score(self, rating_a, rating_b):
        """Calculate expected score (win probability) for team A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team_id, away_team_id, home_won, margin=None, game_id=None, game_date=None):
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
            k = self.K_FACTOR * min(mov_mult, ELO_MOV_MULTIPLIER_CAP)
        
        # Update ratings
        new_home = home_rating + k * (home_actual - home_expected)
        new_away = away_rating + k * (away_actual - away_expected)

        # Additional decay for teams we don't fully track schedules for
        new_home = self._apply_untracked_decay(home_team_id, new_home)
        new_away = self._apply_untracked_decay(away_team_id, new_away)
        
        self.ratings[home_team_id] = new_home
        self.ratings[away_team_id] = new_away
        
        self._save_rating(home_team_id, new_home, increment_games=True)
        self._save_rating(away_team_id, new_away, increment_games=True)
        
        # Log history for charts
        self._log_history(home_team_id, new_home, away_team_id, new_home - home_rating, game_id, game_date)
        self._log_history(away_team_id, new_away, home_team_id, new_away - away_rating, game_id, game_date)
        
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
        home_prob = max(0.02, min(0.98, home_prob))
        
        # Project runs based on rating differential
        # Higher rated teams score more, allow less
        rating_diff = (home_rating - away_rating) / 100
        # Use actual league average instead of hardcoded 5.5
        try:
            from scripts.database import get_connection
            _conn = get_connection()
            _r = _conn.cursor().execute(
                'SELECT AVG(home_score + away_score) / 2.0 FROM games WHERE home_score IS NOT NULL'
            ).fetchone()
            _conn.close()
            base_runs = _r[0] if _r and _r[0] else 6.5
        except Exception:
            base_runs = 6.5
        
        home_runs = base_runs + rating_diff * 0.5
        away_runs = base_runs - rating_diff * 0.5
        
        if not neutral_site:
            home_runs *= 1.04
            away_runs *= 0.96
        
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
        Initialize Elo ratings from preseason rankings.
        
        Uses max(ranking-based Elo, conference tier) so a ranked team
        never starts below their conference default.
        
        rankings: list of team_ids in rank order
        """
        # Spread ratings from 1700 (#1) down by 12 per rank
        for i, team_id in enumerate(rankings):
            rank_rating = 1700 - (i * 12)
            
            # Get conference tier floor
            try:
                conn = get_connection()
                row = conn.execute("SELECT conference FROM teams WHERE id = ?", (team_id,)).fetchone()
                conn.close()
                conf = row['conference'] if row and row['conference'] else ''
                conf_floor = self.CONF_ELO.get(conf, self.BASE_RATING)
            except Exception:
                conf_floor = self.BASE_RATING
            
            rating = max(rank_rating, conf_floor)
            self.ratings[team_id] = rating
            self._save_rating(team_id, rating)
        
        return self.ratings
