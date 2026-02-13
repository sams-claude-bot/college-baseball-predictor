#!/usr/bin/env python3
"""
Preseason Prior Model

Solves the cold start problem by providing meaningful predictions
before game data exists.

Inputs:
- Preseason rankings (D1Baseball, Baseball America, etc.)
- Elo seeds / returning talent estimates
- Conference strength expectations
- Previous season performance

Bayesian blending shifts weight from priors to actual performance:
- Games 0-5:   80% prior, 20% actual
- Games 5-15:  Linear blend from 80/20 to 30/70
- Games 15+:   20% prior, 80% actual (priors never fully disappear)
"""

import sys
import json
from pathlib import Path

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
sys.path.insert(0, str(_models_dir))
sys.path.insert(0, str(_scripts_dir))

from base_model import BaseModel
from database import get_connection
from conference_model import get_conference_rating, normalize_conference


# File to store preseason data
PRESEASON_FILE = Path(__file__).parent.parent / "data" / "preseason_priors.json"


def load_preseason_data():
    """Load preseason expectations from JSON file"""
    if PRESEASON_FILE.exists():
        with open(PRESEASON_FILE) as f:
            return json.load(f)
    return {"season": None, "teams": {}, "updated": None}


def save_preseason_data(data):
    """Save preseason expectations to JSON file"""
    PRESEASON_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRESEASON_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def set_preseason_expectations(team_id, ranking=None, projected_win_pct=None,
                                returning_war=None, preseason_elo=None,
                                conference=None, notes=None):
    """
    Set preseason expectations for a team.
    
    Args:
        team_id: Team identifier
        ranking: Preseason poll ranking (1-25, or None for unranked)
        projected_win_pct: Expected win percentage (0-1)
        returning_war: Estimated WAR of returning players
        preseason_elo: Starting Elo rating
        conference: Team's conference
        notes: Any notes about the team
    
    Usage:
        set_preseason_expectations("mississippi-state", 
                                   ranking=5, 
                                   projected_win_pct=0.72,
                                   preseason_elo=1650)
    """
    data = load_preseason_data()
    
    from datetime import datetime
    
    if data["season"] is None:
        data["season"] = datetime.now().year
    
    team_data = data["teams"].get(team_id, {})
    
    if ranking is not None:
        team_data["ranking"] = ranking
    if projected_win_pct is not None:
        team_data["projected_win_pct"] = projected_win_pct
    if returning_war is not None:
        team_data["returning_war"] = returning_war
    if preseason_elo is not None:
        team_data["preseason_elo"] = preseason_elo
    if conference is not None:
        team_data["conference"] = conference
    if notes is not None:
        team_data["notes"] = notes
    
    data["teams"][team_id] = team_data
    data["updated"] = datetime.now().isoformat()
    
    save_preseason_data(data)
    return team_data


def get_preseason_expectations(team_id):
    """
    Get preseason expectations for a team.
    Returns None if no preseason data exists.
    """
    data = load_preseason_data()
    return data["teams"].get(team_id)


def initialize_rankings(rankings_list):
    """
    Initialize preseason data from a rankings list.
    
    Args:
        rankings_list: List of team_ids in rank order (index 0 = #1)
    
    Example:
        initialize_rankings([
            "tennessee", "lsu", "texas-a&m", "florida", "mississippi-state",
            "arkansas", "auburn", "georgia", "south-carolina", "vanderbilt",
            ...
        ])
    """
    data = load_preseason_data()
    from datetime import datetime
    
    data["season"] = datetime.now().year
    
    for i, team_id in enumerate(rankings_list):
        rank = i + 1
        # Generate expected win pct from ranking
        # #1 ~ 0.80, #25 ~ 0.60, unranked ~ 0.50
        if rank <= 25:
            projected_pct = 0.80 - (rank - 1) * 0.008
            preseason_elo = 1700 - (rank - 1) * 8
        else:
            projected_pct = 0.55
            preseason_elo = 1500
        
        data["teams"][team_id] = {
            "ranking": rank,
            "projected_win_pct": round(projected_pct, 3),
            "preseason_elo": preseason_elo
        }
    
    data["updated"] = datetime.now().isoformat()
    save_preseason_data(data)
    return data


class PriorModel(BaseModel):
    """
    Preseason prior-based prediction model.
    
    Uses Bayesian blending to combine preseason expectations
    with actual game performance as the season progresses.
    """
    
    name = "prior"
    version = "1.0"
    description = "Preseason priors with Bayesian blending"
    
    # Blend schedule
    # (games_played, prior_weight, actual_weight)
    EARLY_SEASON_CUTOFF = 5     # Games 0-5: 80% prior
    MID_SEASON_CUTOFF = 15      # Games 5-15: linear blend
    LATE_SEASON_PRIOR = 0.20    # Games 15+: 20% prior minimum
    
    HOME_ADVANTAGE = 0.035
    DEFAULT_STRENGTH = 0.50     # For unknown teams
    
    def __init__(self):
        self.preseason_data = load_preseason_data()
        self.games_cache = {}
    
    def refresh_data(self):
        """Reload preseason data from file"""
        self.preseason_data = load_preseason_data()
    
    def _get_games_played(self, team_id):
        """Get number of games played by team"""
        if team_id in self.games_cache:
            return self.games_cache[team_id]
        
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT COUNT(*) as games
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final'
        ''', (team_id, team_id))
        
        row = c.fetchone()
        conn.close()
        
        games = row['games'] if row else 0
        self.games_cache[team_id] = games
        return games
    
    def _get_actual_performance(self, team_id):
        """Get actual season performance"""
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COUNT(CASE WHEN winner_id = ? THEN 1 END) as wins,
                COUNT(*) as games,
                SUM(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as runs_for,
                SUM(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as runs_against
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final'
        ''', (team_id, team_id, team_id, team_id, team_id))
        
        row = c.fetchone()
        conn.close()
        
        if row and row['games'] and row['games'] > 0:
            return {
                'win_pct': row['wins'] / row['games'],
                'games': row['games'],
                'runs_per_game': row['runs_for'] / row['games'],
                'runs_allowed_per_game': row['runs_against'] / row['games']
            }
        return None
    
    def _get_prior_strength(self, team_id):
        """Get preseason expected strength for a team"""
        team_data = self.preseason_data.get("teams", {}).get(team_id)
        
        if team_data:
            # Use projected win pct if available
            if "projected_win_pct" in team_data:
                return team_data["projected_win_pct"]
            
            # Otherwise estimate from ranking
            ranking = team_data.get("ranking")
            if ranking and ranking <= 25:
                return 0.80 - (ranking - 1) * 0.008
        
        # Check database for preseason rank
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT preseason_rank, conference 
            FROM teams 
            WHERE id = ?
        ''', (team_id,))
        
        row = c.fetchone()
        conn.close()
        
        if row:
            if row['preseason_rank'] and row['preseason_rank'] <= 25:
                return 0.80 - (row['preseason_rank'] - 1) * 0.008
            
            # Use conference strength as a weak prior
            if row['conference']:
                conf_rating = get_conference_rating(row['conference'])
                # Convert conference rating to expected win pct
                # 1.18 (SEC) -> ~0.60, 0.75 (low tier) -> ~0.45
                return 0.40 + (conf_rating - 0.75) * 0.35
        
        return self.DEFAULT_STRENGTH
    
    def _calculate_blend_weights(self, games_played):
        """
        Calculate prior vs actual weights based on games played.
        
        Returns (prior_weight, actual_weight)
        """
        if games_played <= self.EARLY_SEASON_CUTOFF:
            # Early season: 80% prior, 20% actual
            return 0.80, 0.20
        
        elif games_played <= self.MID_SEASON_CUTOFF:
            # Mid season: linear blend from 80/20 to 30/70
            progress = (games_played - self.EARLY_SEASON_CUTOFF) / \
                      (self.MID_SEASON_CUTOFF - self.EARLY_SEASON_CUTOFF)
            prior_weight = 0.80 - progress * 0.50  # 0.80 -> 0.30
            actual_weight = 0.20 + progress * 0.50  # 0.20 -> 0.70
            return prior_weight, actual_weight
        
        else:
            # Late season: 20% prior, 80% actual
            return self.LATE_SEASON_PRIOR, 1 - self.LATE_SEASON_PRIOR
    
    def _get_blended_strength(self, team_id):
        """
        Get team strength using Bayesian blending of
        preseason priors and actual performance.
        """
        games = self._get_games_played(team_id)
        prior_weight, actual_weight = self._calculate_blend_weights(games)
        
        prior_strength = self._get_prior_strength(team_id)
        actual_perf = self._get_actual_performance(team_id)
        
        if actual_perf and actual_perf['games'] > 0:
            actual_strength = actual_perf['win_pct']
            blended = prior_strength * prior_weight + actual_strength * actual_weight
        else:
            # No actual data yet, use pure prior
            blended = prior_strength
        
        return {
            'strength': blended,
            'prior_strength': prior_strength,
            'actual_strength': actual_perf['win_pct'] if actual_perf else None,
            'games': games,
            'prior_weight': prior_weight,
            'actual_weight': actual_weight
        }
    
    def _project_runs(self, team_id, games_played):
        """Project runs scored/allowed using blended expectations"""
        actual = self._get_actual_performance(team_id)
        team_data = self.preseason_data.get("teams", {}).get(team_id, {})
        
        prior_weight, actual_weight = self._calculate_blend_weights(games_played)
        
        # Prior expectation: use ranking or default
        ranking = team_data.get("ranking", 50)
        if ranking <= 25:
            # Top 25 teams: expect ~7 runs scored, ~4 allowed
            prior_rpg = 7.0 - (ranking - 1) * 0.04
            prior_rapg = 4.0 + (ranking - 1) * 0.06
        else:
            prior_rpg = 5.5
            prior_rapg = 5.5
        
        if actual and actual['games'] > 0:
            actual_rpg = actual['runs_per_game']
            actual_rapg = actual['runs_allowed_per_game']
            
            rpg = prior_rpg * prior_weight + actual_rpg * actual_weight
            rapg = prior_rapg * prior_weight + actual_rapg * actual_weight
        else:
            rpg = prior_rpg
            rapg = prior_rapg
        
        return rpg, rapg
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        """
        Predict game using blended preseason/actual performance.
        """
        # Get blended strengths
        home_blend = self._get_blended_strength(home_team_id)
        away_blend = self._get_blended_strength(away_team_id)
        
        home_strength = home_blend['strength']
        away_strength = away_blend['strength']
        
        # Log5 calculation
        if home_strength + away_strength > 0 and \
           (home_strength * away_strength != home_strength + away_strength):
            base_prob = (home_strength - home_strength * away_strength) / \
                       (home_strength + away_strength - 2 * home_strength * away_strength)
        else:
            base_prob = 0.5
        
        # Home advantage
        if not neutral_site:
            home_prob = base_prob + self.HOME_ADVANTAGE
        else:
            home_prob = base_prob
        
        home_prob = max(0.1, min(0.9, home_prob))
        
        # Project runs
        home_rpg, home_rapg = self._project_runs(home_team_id, home_blend['games'])
        away_rpg, away_rapg = self._project_runs(away_team_id, away_blend['games'])
        
        home_runs = (home_rpg + away_rapg) / 2
        away_runs = (away_rpg + home_rapg) / 2
        
        if not neutral_site:
            home_runs *= 1.02
            away_runs *= 0.98
        
        run_line = self.calculate_run_line(home_runs, away_runs)
        
        # Get preseason data for display
        home_preseason = self.preseason_data.get("teams", {}).get(home_team_id, {})
        away_preseason = self.preseason_data.get("teams", {}).get(away_team_id, {})
        
        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(home_runs, 1),
            "projected_away_runs": round(away_runs, 1),
            "projected_total": round(home_runs + away_runs, 1),
            "run_line": run_line,
            "inputs": {
                "home_preseason_rank": home_preseason.get("ranking", "NR"),
                "away_preseason_rank": away_preseason.get("ranking", "NR"),
                "home_blended_strength": round(home_strength, 3),
                "away_blended_strength": round(away_strength, 3),
                "home_prior_strength": round(home_blend['prior_strength'], 3),
                "away_prior_strength": round(away_blend['prior_strength'], 3),
                "home_actual_strength": round(home_blend['actual_strength'], 3) if home_blend['actual_strength'] else "N/A",
                "away_actual_strength": round(away_blend['actual_strength'], 3) if away_blend['actual_strength'] else "N/A",
                "home_games_played": home_blend['games'],
                "away_games_played": away_blend['games'],
                "home_prior_weight": round(home_blend['prior_weight'], 2),
                "away_prior_weight": round(away_blend['prior_weight'], 2)
            }
        }


# For testing
if __name__ == "__main__":
    model = PriorModel()
    
    if len(sys.argv) > 2:
        home = sys.argv[1].lower().replace(" ", "-")
        away = sys.argv[2].lower().replace(" ", "-")
        neutral = "--neutral" in sys.argv
        
        pred = model.predict_game(home, away, neutral)
        
        print(f"\n{'='*55}")
        print(f"  PRIOR MODEL: {away} @ {home}")
        print('='*55)
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nInputs:")
        for k, v in pred['inputs'].items():
            print(f"  {k}: {v}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "init":
        # Initialize with sample rankings
        print("Initializing sample preseason rankings...")
        sample_rankings = [
            "tennessee", "lsu", "texas-am", "florida", "mississippi-state",
            "arkansas", "auburn", "georgia", "south-carolina", "vanderbilt",
            "ole-miss", "texas", "clemson", "miami", "nc-state",
            "louisville", "wake-forest", "virginia", "florida-state", "virginia-tech",
            "oregon-state", "stanford", "arizona", "usc", "ucla"
        ]
        initialize_rankings(sample_rankings)
        print(f"Initialized {len(sample_rankings)} teams")
        
    else:
        print("Usage:")
        print("  python prior_model.py <home_team> <away_team> [--neutral]")
        print("  python prior_model.py init   # Initialize sample rankings")
        print("\nPreseason Data:")
        data = load_preseason_data()
        print(f"  Season: {data.get('season', 'Not set')}")
        print(f"  Teams: {len(data.get('teams', {}))}")
        if data.get('teams'):
            print("\n  Top 5:")
            sorted_teams = sorted(
                data['teams'].items(),
                key=lambda x: x[1].get('ranking', 999)
            )[:5]
            for team_id, info in sorted_teams:
                rank = info.get('ranking', 'NR')
                pct = info.get('projected_win_pct', 'N/A')
                print(f"    #{rank}: {team_id} ({pct})")
