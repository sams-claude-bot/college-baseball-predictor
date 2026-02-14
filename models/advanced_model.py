#!/usr/bin/env python3
"""
Advanced Prediction Model

Features:
- Opponent-adjusted stats (strength of schedule)
- Recency weighting (recent games matter more)
- Margin of victory consideration
- Home/away splits
- Rest days factor
"""

import sys
import math
from datetime import datetime, timedelta
from pathlib import Path

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
# sys.path.insert(0, str(_models_dir))  # Removed by cleanup
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from models.base_model import BaseModel
from scripts.database import get_connection

class AdvancedModel(BaseModel):
    name = "advanced"
    version = "1.0"
    description = "Opponent-adjusted, recency-weighted model"
    
    # Weights for recency (game 1 = most recent)
    RECENCY_WEIGHTS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
    
    HOME_ADVANTAGE = 0.035  # ~3.5% boost
    NEUTRAL_ADVANTAGE = 0.0
    
    # Rest day adjustments
    REST_FACTORS = {
        0: -0.02,   # Back-to-back: slight penalty
        1: 0.0,     # Normal rest
        2: 0.01,    # Extra rest: slight boost
        3: 0.015,   # 3+ days rest
    }
    
    def __init__(self):
        self.team_cache = {}
        self.sos_cache = {}  # Strength of schedule cache
    
    def _get_team_games(self, team_id, limit=20):
        """Get team's recent games with full details"""
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT g.*, 
                   CASE WHEN g.home_team_id = ? THEN g.away_team_id ELSE g.home_team_id END as opponent_id,
                   CASE WHEN g.home_team_id = ? THEN 1 ELSE 0 END as was_home,
                   CASE WHEN g.winner_id = ? THEN 1 ELSE 0 END as won,
                   CASE WHEN g.home_team_id = ? THEN g.home_score ELSE g.away_score END as team_score,
                   CASE WHEN g.home_team_id = ? THEN g.away_score ELSE g.home_score END as opp_score
            FROM games g
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
            AND g.status = 'final'
            ORDER BY g.date DESC
            LIMIT ?
        ''', (team_id, team_id, team_id, team_id, team_id, team_id, team_id, limit))
        
        rows = c.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def _get_opponent_win_pct(self, opponent_id):
        """Get opponent's winning percentage (for SOS calculation)"""
        if opponent_id in self.sos_cache:
            return self.sos_cache[opponent_id]
        
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COUNT(CASE WHEN winner_id = ? THEN 1 END) as wins,
                COUNT(*) as games
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final'
        ''', (opponent_id, opponent_id, opponent_id))
        
        row = c.fetchone()
        conn.close()
        
        if row and row[1] > 0:
            win_pct = row[0] / row[1]
        else:
            win_pct = 0.5  # Unknown opponent, assume average
        
        self.sos_cache[opponent_id] = win_pct
        return win_pct
    
    def _calculate_adjusted_stats(self, team_id):
        """Calculate opponent-adjusted, recency-weighted stats"""
        games = self._get_team_games(team_id, limit=15)
        
        if not games:
            return {
                'adj_win_pct': 0.5,
                'adj_run_diff': 0,
                'adj_runs_scored': 5.0,
                'adj_runs_allowed': 5.0,
                'strength_of_schedule': 0.5,
                'games_played': 0,
                'recent_form': 0.5
            }
        
        total_weight = 0
        weighted_wins = 0
        weighted_run_diff = 0
        weighted_runs_scored = 0
        weighted_runs_allowed = 0
        sos_sum = 0
        
        for i, game in enumerate(games):
            # Recency weight
            weight = self.RECENCY_WEIGHTS[i] if i < len(self.RECENCY_WEIGHTS) else 0.5
            
            # Opponent strength adjustment
            opp_win_pct = self._get_opponent_win_pct(game['opponent_id'])
            opp_factor = 0.5 + (opp_win_pct - 0.5) * 0.5  # Dampen extreme values
            
            # Adjusted weight = recency * opponent strength
            adj_weight = weight * (0.5 + opp_factor)
            
            total_weight += adj_weight
            
            # Accumulate weighted stats
            if game['won']:
                weighted_wins += adj_weight
            
            if game['team_score'] is not None and game['opp_score'] is not None:
                run_diff = game['team_score'] - game['opp_score']
                weighted_run_diff += run_diff * adj_weight
                weighted_runs_scored += game['team_score'] * adj_weight
                weighted_runs_allowed += game['opp_score'] * adj_weight
            
            sos_sum += opp_win_pct
        
        if total_weight == 0:
            total_weight = 1
        
        # Recent form (last 5 games, simple)
        recent_5 = games[:5]
        recent_wins = sum(1 for g in recent_5 if g['won'])
        recent_form = recent_wins / len(recent_5) if recent_5 else 0.5
        
        return {
            'adj_win_pct': weighted_wins / total_weight,
            'adj_run_diff': weighted_run_diff / total_weight,
            'adj_runs_scored': weighted_runs_scored / total_weight if weighted_runs_scored > 0 else 5.0,
            'adj_runs_allowed': weighted_runs_allowed / total_weight if weighted_runs_allowed > 0 else 5.0,
            'strength_of_schedule': sos_sum / len(games) if games else 0.5,
            'games_played': len(games),
            'recent_form': recent_form
        }
    
    def _get_rest_days(self, team_id):
        """Get days since last game"""
        games = self._get_team_games(team_id, limit=1)
        if not games:
            return 3  # No games = well rested
        
        last_game_date = datetime.strptime(games[0]['date'], '%Y-%m-%d')
        today = datetime.now()
        delta = (today - last_game_date).days
        return min(delta, 3)  # Cap at 3
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        home_stats = self._calculate_adjusted_stats(home_team_id)
        away_stats = self._calculate_adjusted_stats(away_team_id)
        
        # Base probability from adjusted win percentages
        home_adj = home_stats['adj_win_pct']
        away_adj = away_stats['adj_win_pct']
        
        # Log5-style combination of adjusted win pcts
        if home_adj + away_adj > 0 and home_adj * away_adj != home_adj + away_adj:
            base_prob = (home_adj - home_adj * away_adj) / (home_adj + away_adj - 2 * home_adj * away_adj)
        else:
            base_prob = 0.5
        
        # Apply home/neutral advantage
        if neutral_site:
            home_prob = base_prob + self.NEUTRAL_ADVANTAGE
        else:
            home_prob = base_prob + self.HOME_ADVANTAGE
        
        # Rest day adjustment
        home_rest = self._get_rest_days(home_team_id)
        away_rest = self._get_rest_days(away_team_id)
        
        rest_diff = self.REST_FACTORS.get(home_rest, 0.015) - self.REST_FACTORS.get(away_rest, 0.015)
        home_prob += rest_diff
        
        # Recent form boost (small weight)
        form_diff = (home_stats['recent_form'] - away_stats['recent_form']) * 0.05
        home_prob += form_diff
        
        # Clamp probability
        home_prob = max(0.1, min(0.9, home_prob))
        
        # Project runs using adjusted stats
        home_runs = (home_stats['adj_runs_scored'] + away_stats['adj_runs_allowed']) / 2
        away_runs = (away_stats['adj_runs_scored'] + home_stats['adj_runs_allowed']) / 2
        
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
                "home_adj_win_pct": round(home_stats['adj_win_pct'], 3),
                "away_adj_win_pct": round(away_stats['adj_win_pct'], 3),
                "home_sos": round(home_stats['strength_of_schedule'], 3),
                "away_sos": round(away_stats['strength_of_schedule'], 3),
                "home_recent_form": round(home_stats['recent_form'], 3),
                "away_recent_form": round(away_stats['recent_form'], 3),
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "home_games_played": home_stats['games_played'],
                "away_games_played": away_stats['games_played']
            }
        }


# For testing
if __name__ == "__main__":
    model = AdvancedModel()
    
    if len(sys.argv) > 2:
        home = sys.argv[1].lower().replace(" ", "-")
        away = sys.argv[2].lower().replace(" ", "-")
        neutral = "--neutral" in sys.argv
        
        pred = model.predict_game(home, away, neutral)
        
        print(f"\n{'='*55}")
        print(f"  ADVANCED MODEL: {away} @ {home}")
        print('='*55)
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nInputs:")
        for k, v in pred['inputs'].items():
            print(f"  {k}: {v}")
    else:
        print("Usage: python advanced_model.py <home_team> <away_team> [--neutral]")
