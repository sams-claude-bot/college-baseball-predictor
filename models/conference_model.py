#!/usr/bin/env python3
"""
Conference Strength Adjustment Model

Adjusts team ratings based on conference quality:
- Conference strength ratings (SEC, ACC, Big 12, etc.)
- Strength of schedule adjustment
- Non-conference opponent quality analysis
- Can be used standalone or as a modifier by other models

The key insight: A .600 team in the SEC is probably better than
a .700 team in a weaker conference.
"""

import sys
from pathlib import Path

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
sys.path.insert(0, str(_models_dir))
sys.path.insert(0, str(_scripts_dir))

from base_model import BaseModel
from database import get_connection


# Conference strength ratings (updated annually)
# Based on RPI, tournament performance, head-to-head records
# Scale: 1.0 = average D1, >1.0 = above average, <1.0 = below
CONFERENCE_RATINGS = {
    # Power conferences
    "SEC": 1.18,
    "ACC": 1.12,
    "Big 12": 1.10,
    "Big Ten": 1.08,
    "Pac-12": 1.06,
    
    # Strong mid-majors
    "AAC": 1.02,
    "Big East": 1.00,
    "Sun Belt": 0.98,
    "Conference USA": 0.97,
    "Mountain West": 0.96,
    "Missouri Valley": 0.95,
    "Colonial": 0.94,
    "West Coast": 0.93,
    
    # Mid-tier
    "Atlantic 10": 0.92,
    "MAC": 0.91,
    "Southern": 0.90,
    "Big West": 0.90,
    "ASUN": 0.89,
    "Horizon": 0.88,
    "WAC": 0.87,
    
    # Lower tier
    "America East": 0.86,
    "Big South": 0.85,
    "Patriot": 0.84,
    "Ivy League": 0.83,
    "MAAC": 0.82,
    "OVC": 0.81,
    "Southland": 0.80,
    "Summit": 0.79,
    "NEC": 0.78,
    "MEAC": 0.75,
    "SWAC": 0.74,
}

# Conference name normalization
CONFERENCE_ALIASES = {
    "southeastern": "SEC",
    "southeastern conference": "SEC",
    "atlantic coast": "ACC",
    "atlantic coast conference": "ACC",
    "big twelve": "Big 12",
    "big 10": "Big Ten",
    "pacific-12": "Pac-12",
    "pacific 12": "Pac-12",
    "american athletic": "AAC",
    "american": "AAC",
    "mid-american": "MAC",
    "mid american": "MAC",
}


def normalize_conference(conf):
    """Normalize conference name to standard form"""
    if conf is None:
        return None
    
    conf_lower = conf.lower().strip()
    
    # Check aliases
    if conf_lower in CONFERENCE_ALIASES:
        return CONFERENCE_ALIASES[conf_lower]
    
    # Check exact matches (case insensitive)
    for standard in CONFERENCE_RATINGS:
        if standard.lower() == conf_lower:
            return standard
    
    # Return as-is if not found
    return conf


def get_conference_rating(conference):
    """Get the strength rating for a conference"""
    if conference is None:
        return 1.0  # Assume average
    
    normalized = normalize_conference(conference)
    return CONFERENCE_RATINGS.get(normalized, 0.90)  # Default to mid-tier


def get_conference_adjustment(team_id):
    """
    Get conference strength adjustment factor for a team.
    
    This is the primary function other models should call.
    
    Returns a multiplier:
    - > 1.0 = inflate stats (weaker conference, stats are inflated)
    - < 1.0 = deflate stats (stronger conference, stats are deflated)
    - = 1.0 = no adjustment
    
    Usage in other models:
        from conference_model import get_conference_adjustment
        adjustment = get_conference_adjustment(team_id)
        adjusted_win_pct = raw_win_pct * adjustment
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT conference FROM teams WHERE id = ?", (team_id,))
    row = c.fetchone()
    conn.close()
    
    if row and row['conference']:
        conf_rating = get_conference_rating(row['conference'])
        # Invert: strong conference (>1.0) means team is underrated, adjust up
        # This returns factor to apply to opponent's expected performance
        return conf_rating
    
    return 1.0


def get_opponent_adjustment(team_id, opponent_id):
    """
    Calculate adjustment factor for a specific matchup.
    
    Returns adjustment to apply to the team's win probability
    based on relative conference strengths.
    """
    team_conf = get_conference_adjustment(team_id)
    opp_conf = get_conference_adjustment(opponent_id)
    
    # If team is from stronger conference than opponent
    # they should get a bump (their record vs weaker opponents understates them)
    diff = (team_conf - opp_conf) / 2
    
    return diff


class ConferenceModel(BaseModel):
    """
    Conference-adjusted prediction model.
    
    Adjusts baseline 50/50 probability based on:
    - Each team's conference strength
    - Non-conference results
    - Strength of schedule
    """
    
    name = "conference"
    version = "1.0"
    description = "Conference strength and SOS adjustments"
    
    HOME_ADVANTAGE = 0.035
    
    def __init__(self):
        self.team_cache = {}
        self.sos_cache = {}
    
    def _get_team_conference(self, team_id):
        """Get team's conference"""
        if team_id in self.team_cache:
            return self.team_cache[team_id]
        
        conn = get_connection()
        c = conn.cursor()
        
        c.execute("SELECT conference FROM teams WHERE id = ?", (team_id,))
        row = c.fetchone()
        conn.close()
        
        conf = row['conference'] if row else None
        self.team_cache[team_id] = conf
        return conf
    
    def _calculate_ooc_strength(self, team_id):
        """
        Calculate strength of non-conference schedule.
        Returns average conference rating of OOC opponents.
        """
        if team_id in self.sos_cache:
            return self.sos_cache[team_id]
        
        conn = get_connection()
        c = conn.cursor()
        
        team_conf = self._get_team_conference(team_id)
        
        # Get all opponents
        c.execute('''
            SELECT 
                CASE WHEN home_team_id = ? THEN away_team_id ELSE home_team_id END as opponent_id
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND is_conference_game = 0
            AND status = 'final'
        ''', (team_id, team_id, team_id))
        
        opponents = [row['opponent_id'] for row in c.fetchall()]
        
        if not opponents:
            self.sos_cache[team_id] = 1.0
            conn.close()
            return 1.0
        
        # Get conference ratings for opponents
        total_rating = 0
        counted = 0
        
        for opp_id in opponents:
            c.execute("SELECT conference FROM teams WHERE id = ?", (opp_id,))
            row = c.fetchone()
            if row and row['conference']:
                opp_conf = normalize_conference(row['conference'])
                total_rating += get_conference_rating(opp_conf)
                counted += 1
        
        conn.close()
        
        if counted > 0:
            sos = total_rating / counted
        else:
            sos = 1.0
        
        self.sos_cache[team_id] = sos
        return sos
    
    def _get_team_record(self, team_id):
        """Get team win-loss record"""
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COUNT(CASE WHEN winner_id = ? THEN 1 END) as wins,
                COUNT(CASE WHEN winner_id != ? AND winner_id IS NOT NULL THEN 1 END) as losses
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final'
        ''', (team_id, team_id, team_id, team_id))
        
        row = c.fetchone()
        conn.close()
        
        wins = row['wins'] if row else 0
        losses = row['losses'] if row else 0
        return wins, losses
    
    def _get_conf_record(self, team_id):
        """Get conference win-loss record"""
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                COUNT(CASE WHEN winner_id = ? THEN 1 END) as wins,
                COUNT(CASE WHEN winner_id != ? AND winner_id IS NOT NULL THEN 1 END) as losses
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND is_conference_game = 1
            AND status = 'final'
        ''', (team_id, team_id, team_id, team_id))
        
        row = c.fetchone()
        conn.close()
        
        wins = row['wins'] if row else 0
        losses = row['losses'] if row else 0
        return wins, losses
    
    def _calculate_adjusted_strength(self, team_id):
        """
        Calculate conference-adjusted team strength.
        Returns a 0-1 score.
        """
        wins, losses = self._get_team_record(team_id)
        games = wins + losses
        
        if games == 0:
            return 0.5  # No data
        
        raw_win_pct = wins / games
        
        # Get conference strength
        conf = self._get_team_conference(team_id)
        conf_rating = get_conference_rating(conf)
        
        # Get OOC strength
        ooc_strength = self._calculate_ooc_strength(team_id)
        
        # Conference record matters more in strong conferences
        conf_wins, conf_losses = self._get_conf_record(team_id)
        conf_games = conf_wins + conf_losses
        
        if conf_games > 0:
            conf_win_pct = conf_wins / conf_games
            # Weight conference record by conference strength
            conf_weight = min(0.4, conf_games / 20 * 0.4)  # Up to 40% weight
            adjusted_pct = (raw_win_pct * (1 - conf_weight) + 
                          conf_win_pct * conf_weight)
        else:
            adjusted_pct = raw_win_pct
        
        # Apply conference adjustment
        # Strong conference teams get a boost
        conf_adjustment = (conf_rating - 1.0) * 0.15
        
        # Strong OOC schedule gets a boost
        ooc_adjustment = (ooc_strength - 1.0) * 0.10
        
        final_strength = adjusted_pct + conf_adjustment + ooc_adjustment
        return max(0.1, min(0.9, final_strength))
    
    def _get_team_runs(self, team_id):
        """Get team run statistics"""
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                SUM(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as runs_scored,
                SUM(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as runs_allowed,
                COUNT(*) as games
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final'
        ''', (team_id, team_id, team_id, team_id))
        
        row = c.fetchone()
        conn.close()
        
        if row and row['games'] and row['games'] > 0:
            return {
                'rpg': row['runs_scored'] / row['games'],
                'rapg': row['runs_allowed'] / row['games'],
                'games': row['games']
            }
        return {'rpg': 5.0, 'rapg': 5.0, 'games': 0}
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        """
        Predict game using conference-adjusted strengths.
        """
        # Get adjusted strengths
        home_strength = self._calculate_adjusted_strength(home_team_id)
        away_strength = self._calculate_adjusted_strength(away_team_id)
        
        # Get conference ratings
        home_conf = self._get_team_conference(home_team_id)
        away_conf = self._get_team_conference(away_team_id)
        home_conf_rating = get_conference_rating(home_conf)
        away_conf_rating = get_conference_rating(away_conf)
        
        # Log5-style probability calculation
        if home_strength + away_strength > 0:
            base_prob = (home_strength - home_strength * away_strength) / \
                       (home_strength + away_strength - 2 * home_strength * away_strength)
        else:
            base_prob = 0.5
        
        # Cross-conference adjustment
        # When teams from different conferences meet, adjust based on conference gap
        conf_diff = (home_conf_rating - away_conf_rating) * 0.05
        home_prob = base_prob + conf_diff
        
        # Home advantage
        if not neutral_site:
            home_prob += self.HOME_ADVANTAGE
        
        home_prob = max(0.1, min(0.9, home_prob))
        
        # Project runs
        home_runs_data = self._get_team_runs(home_team_id)
        away_runs_data = self._get_team_runs(away_team_id)
        
        # Adjust runs by conference (strong conf teams allow fewer)
        home_runs = (home_runs_data['rpg'] + away_runs_data['rapg']) / 2
        away_runs = (away_runs_data['rpg'] + home_runs_data['rapg']) / 2
        
        # Small conference adjustment to runs
        home_runs *= (2 - home_conf_rating) / 1.5  # Strong conf pitching = fewer runs
        away_runs *= (2 - away_conf_rating) / 1.5
        
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
                "home_conference": home_conf or "Unknown",
                "away_conference": away_conf or "Unknown",
                "home_conf_rating": round(home_conf_rating, 3),
                "away_conf_rating": round(away_conf_rating, 3),
                "home_adjusted_strength": round(home_strength, 3),
                "away_adjusted_strength": round(away_strength, 3),
                "home_ooc_sos": round(self._calculate_ooc_strength(home_team_id), 3),
                "away_ooc_sos": round(self._calculate_ooc_strength(away_team_id), 3)
            }
        }


# For testing
if __name__ == "__main__":
    model = ConferenceModel()
    
    if len(sys.argv) > 2:
        home = sys.argv[1].lower().replace(" ", "-")
        away = sys.argv[2].lower().replace(" ", "-")
        neutral = "--neutral" in sys.argv
        
        pred = model.predict_game(home, away, neutral)
        
        print(f"\n{'='*55}")
        print(f"  CONFERENCE MODEL: {away} @ {home}")
        print('='*55)
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nInputs:")
        for k, v in pred['inputs'].items():
            print(f"  {k}: {v}")
    else:
        print("Conference Strength Ratings:")
        print("-" * 40)
        for conf, rating in sorted(CONFERENCE_RATINGS.items(), key=lambda x: -x[1]):
            print(f"  {conf}: {rating:.2f}")
        print("\nUsage: python conference_model.py <home_team> <away_team> [--neutral]")
