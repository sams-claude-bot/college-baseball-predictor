#!/usr/bin/env python3
"""
Pitching Matchup Model

Predicts game outcomes based on pitching matchup analysis:
- Starting pitcher quality (ERA, WHIP, K/9, BB/9, IP)
- Opponent hitting adjustments (BA, OBP, K% vs LHP/RHP)
- Fatigue factors (days rest, pitch count trends, workload)
- Bullpen availability (recent usage)
- Weekend series context (game 1/2/3 pitcher quality gaps)

Falls back to team-level pitching stats when individual data unavailable.
"""

import sys
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

_models_dir = Path(__file__).parent
_scripts_dir = _models_dir.parent / "scripts"
# sys.path.insert(0, str(_models_dir))  # Removed by cleanup
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from models.base_model import BaseModel
from scripts.database import get_connection


# Weather defaults (when data is missing)
DEFAULT_WEATHER = {
    'temp_f': 65.0,
    'humidity_pct': 55.0,
    'wind_speed_mph': 6.0,
    'wind_direction_deg': 180,
    'precip_prob_pct': 5.0,
    'is_dome': 0,
}

# Reference temperature for weather adjustments
BASELINE_TEMP = 70.0


class PitchingModel(BaseModel):
    """Pitching matchup-based prediction model"""
    
    name = "pitching"
    version = "1.0"
    description = "Starting pitcher and bullpen matchup analysis"
    
    # Weights for different factors
    STARTER_WEIGHT = 0.50      # Starting pitcher quality
    BULLPEN_WEIGHT = 0.20      # Bullpen availability
    MATCHUP_WEIGHT = 0.15      # Pitcher vs opponent lineup
    FATIGUE_WEIGHT = 0.10      # Rest and workload
    SERIES_WEIGHT = 0.05       # Weekend series position
    
    HOME_ADVANTAGE = 0.035
    
    # League average stats for comparison (D1 baseball)
    LEAGUE_AVG = {
        'era': 5.50,
        'whip': 1.45,
        'k_per_9': 8.0,
        'bb_per_9': 4.0,
        'batting_avg': 0.275,
        'obp': 0.360,
        'k_pct': 0.22
    }
    
    # Rest day impact on pitcher performance
    REST_FACTORS = {
        0: -0.15,   # Same day (unlikely for starter)
        1: -0.10,   # 1 day rest - fatigued
        2: -0.05,   # 2 days rest - short rest
        3: -0.02,   # 3 days rest - slightly short
        4: 0.0,     # Normal college rest
        5: 0.02,    # Extra rest - optimal
        6: 0.03,    # Well rested
        7: 0.02,    # Very rested (might be rusty)
    }
    
    def __init__(self):
        self.starter_cache = {}
        self.team_pitching_cache = {}
        self.bullpen_cache = {}
        self.weather_cache = {}
    
    def _get_weather_for_game(self, game_id: str) -> Optional[Dict]:
        """Fetch weather data for a game from game_weather table."""
        if not game_id:
            return None
        
        if game_id in self.weather_cache:
            return self.weather_cache[game_id]
        
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            SELECT temp_f, humidity_pct, wind_speed_mph, wind_direction_deg,
                   precip_prob_pct, is_dome
            FROM game_weather WHERE game_id = ?
        ''', (game_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            weather = {
                'temp_f': row['temp_f'],
                'humidity_pct': row['humidity_pct'],
                'wind_speed_mph': row['wind_speed_mph'],
                'wind_direction_deg': row['wind_direction_deg'],
                'precip_prob_pct': row['precip_prob_pct'],
                'is_dome': row['is_dome'],
            }
            self.weather_cache[game_id] = weather
            return weather
        
        self.weather_cache[game_id] = None
        return None
    
    def _calculate_weather_pitcher_adjustment(self, weather_data: Optional[Dict] = None) -> tuple:
        """
        Calculate weather adjustment to pitcher effectiveness.
        
        Returns (adjustment, components) where adjustment is added to pitcher score.
        Positive adjustment = pitcher performs better (cold, calm conditions).
        Negative adjustment = pitcher performs worse (hot, windy conditions).
        
        Weather effects on pitching:
        - Cold weather (<55°F): Harder grip, less movement, -2% to -5% effectiveness
        - Hot weather (>85°F): Fatigue, sweat affects grip, -1% to -3% effectiveness
        - Ideal range (60-75°F): No adjustment
        - Wind: Affects breaking ball movement (strong wind = less predictable)
        - Humidity: High humidity can affect grip (slight negative)
        """
        if weather_data is None:
            weather_data = DEFAULT_WEATHER.copy()
        
        # Use defaults for missing values (careful with 0 being falsy)
        temp = weather_data.get('temp_f') if weather_data.get('temp_f') is not None else DEFAULT_WEATHER['temp_f']
        humidity = weather_data.get('humidity_pct') if weather_data.get('humidity_pct') is not None else DEFAULT_WEATHER['humidity_pct']
        wind_speed = weather_data.get('wind_speed_mph') if weather_data.get('wind_speed_mph') is not None else DEFAULT_WEATHER['wind_speed_mph']
        is_dome = weather_data.get('is_dome', 0)
        
        # If it's a dome, return neutral adjustment
        if is_dome:
            return 0.0, {'dome': True, 'temp_adj': 0, 'wind_adj': 0, 'humidity_adj': 0}
        
        components = {'dome': False}
        adjustment = 0.0
        
        # Temperature adjustment for pitching
        # Cold weather hurts pitchers more (grip issues, arm stiffness)
        # Hot weather also hurts (fatigue, sweat)
        # Ideal range: 60-75°F
        if temp < 55:
            # Cold: harder to grip ball, less break on pitches
            # -0.5% per degree below 55
            cold_penalty = (55 - temp) * 0.005
            cold_penalty = min(cold_penalty, 0.05)  # Cap at -5%
            adjustment -= cold_penalty
            components['temp_adj'] = round(-cold_penalty, 4)
            components['temp_condition'] = 'cold'
        elif temp > 85:
            # Hot: fatigue, sweat affects grip
            # -0.3% per degree above 85
            hot_penalty = (temp - 85) * 0.003
            hot_penalty = min(hot_penalty, 0.04)  # Cap at -4%
            adjustment -= hot_penalty
            components['temp_adj'] = round(-hot_penalty, 4)
            components['temp_condition'] = 'hot'
        elif 60 <= temp <= 75:
            # Ideal conditions: slight bonus
            adjustment += 0.01
            components['temp_adj'] = 0.01
            components['temp_condition'] = 'ideal'
        else:
            components['temp_adj'] = 0.0
            components['temp_condition'] = 'normal'
        
        components['temp_f'] = temp
        
        # Wind adjustment for pitching
        # Strong wind makes breaking balls less effective/predictable
        # Slight penalty for wind > 10 mph
        if wind_speed > 10:
            wind_penalty = (wind_speed - 10) * 0.002  # -0.2% per mph above 10
            wind_penalty = min(wind_penalty, 0.03)  # Cap at -3%
            adjustment -= wind_penalty
            components['wind_adj'] = round(-wind_penalty, 4)
        else:
            components['wind_adj'] = 0.0
        components['wind_speed_mph'] = wind_speed
        
        # Humidity adjustment for pitching
        # Very high humidity (>80%) makes ball slick, harder to grip
        # Slight negative effect
        if humidity > 80:
            humidity_penalty = (humidity - 80) * 0.001  # -0.1% per % above 80
            humidity_penalty = min(humidity_penalty, 0.02)  # Cap at -2%
            adjustment -= humidity_penalty
            components['humidity_adj'] = round(-humidity_penalty, 4)
        else:
            components['humidity_adj'] = 0.0
        components['humidity_pct'] = humidity
        
        # Cap total adjustment
        adjustment = max(-0.08, min(0.03, adjustment))
        components['total_adjustment'] = round(adjustment, 4)
        
        return adjustment, components
    
    def _get_starting_pitcher(self, team_id, game_date=None):
        """
        Get projected starting pitcher for a team.
        Returns player stats dict or None if unavailable.
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"{team_id}_{game_date}"
        if cache_key in self.starter_cache:
            return self.starter_cache[cache_key]
        
        conn = get_connection()
        c = conn.cursor()
        
        # First try to get from pitching_matchups table
        c.execute('''
            SELECT ps.*, pm.game_id
            FROM pitching_matchups pm
            JOIN games g ON pm.game_id = g.id
            LEFT JOIN player_stats ps ON (
                (g.home_team_id = ? AND pm.home_starter_id = ps.id) OR
                (g.away_team_id = ? AND pm.away_starter_id = ps.id)
            )
            WHERE g.date = ?
            AND (g.home_team_id = ? OR g.away_team_id = ?)
        ''', (team_id, team_id, game_date, team_id, team_id))
        
        row = c.fetchone()
        if row and row['id']:
            starter = dict(row)
            self.starter_cache[cache_key] = starter
            conn.close()
            return starter
        
        # Fall back to best available starter (by ERA with minimum IP)
        c.execute('''
            SELECT * FROM player_stats
            WHERE team_id = ?
            AND is_starter = 1
            AND innings_pitched >= 10
            ORDER BY era ASC
            LIMIT 1
        ''', (team_id,))
        
        row = c.fetchone()
        if row:
            starter = dict(row)
            self.starter_cache[cache_key] = starter
            conn.close()
            return starter
        
        # No individual data - return None (will use team-level)
        conn.close()
        self.starter_cache[cache_key] = None
        return None
    
    def _get_team_pitching_stats(self, team_id):
        """Get team-level pitching statistics as fallback"""
        if team_id in self.team_pitching_cache:
            return self.team_pitching_cache[team_id]
        
        conn = get_connection()
        c = conn.cursor()
        
        # Calculate from player_stats aggregation
        c.execute('''
            SELECT 
                SUM(innings_pitched) as total_ip,
                SUM(earned_runs) as total_er,
                SUM(hits_allowed) as total_hits,
                SUM(walks_allowed) as total_walks,
                SUM(strikeouts_pitched) as total_k,
                AVG(era) as avg_era,
                AVG(whip) as avg_whip,
                AVG(k_per_9) as avg_k9,
                AVG(bb_per_9) as avg_bb9
            FROM player_stats
            WHERE team_id = ?
            AND innings_pitched > 0
        ''', (team_id,))
        
        row = c.fetchone()
        conn.close()
        
        if row and row['total_ip'] and row['total_ip'] > 0:
            total_ip = row['total_ip']
            stats = {
                'era': (row['total_er'] / total_ip) * 9 if total_ip > 0 else self.LEAGUE_AVG['era'],
                'whip': (row['total_hits'] + row['total_walks']) / total_ip if total_ip > 0 else self.LEAGUE_AVG['whip'],
                'k_per_9': (row['total_k'] / total_ip) * 9 if total_ip > 0 else self.LEAGUE_AVG['k_per_9'],
                'bb_per_9': (row['total_walks'] / total_ip) * 9 if total_ip > 0 else self.LEAGUE_AVG['bb_per_9'],
                'innings_pitched': total_ip
            }
        else:
            # No pitching data - use league average
            stats = {
                'era': self.LEAGUE_AVG['era'],
                'whip': self.LEAGUE_AVG['whip'],
                'k_per_9': self.LEAGUE_AVG['k_per_9'],
                'bb_per_9': self.LEAGUE_AVG['bb_per_9'],
                'innings_pitched': 0
            }
        
        self.team_pitching_cache[team_id] = stats
        return stats
    
    def _get_team_hitting_stats(self, team_id):
        """Get team hitting statistics for matchup analysis"""
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                AVG(batting_avg) as team_ba,
                AVG(obp) as team_obp,
                SUM(strikeouts) as total_k,
                SUM(at_bats) as total_ab
            FROM player_stats
            WHERE team_id = ?
            AND at_bats > 0
        ''', (team_id,))
        
        row = c.fetchone()
        conn.close()
        
        if row and row['total_ab'] and row['total_ab'] > 0:
            return {
                'batting_avg': row['team_ba'] or self.LEAGUE_AVG['batting_avg'],
                'obp': row['team_obp'] or self.LEAGUE_AVG['obp'],
                'k_pct': row['total_k'] / row['total_ab'] if row['total_ab'] > 0 else self.LEAGUE_AVG['k_pct']
            }
        return {
            'batting_avg': self.LEAGUE_AVG['batting_avg'],
            'obp': self.LEAGUE_AVG['obp'],
            'k_pct': self.LEAGUE_AVG['k_pct']
        }
    
    def _get_pitcher_rest_days(self, pitcher_id, current_date=None):
        """Calculate days since pitcher's last start"""
        if pitcher_id is None:
            return 5  # Assume normal rest if unknown
        
        if current_date is None:
            current_date = datetime.now()
        elif isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d')
        
        conn = get_connection()
        c = conn.cursor()
        
        # Find last game where this pitcher started
        c.execute('''
            SELECT g.date
            FROM pitching_matchups pm
            JOIN games g ON pm.game_id = g.id
            WHERE (pm.home_starter_id = ? OR pm.away_starter_id = ?)
            AND g.date < ?
            AND g.status = 'final'
            ORDER BY g.date DESC
            LIMIT 1
        ''', (pitcher_id, pitcher_id, current_date.strftime('%Y-%m-%d')))
        
        row = c.fetchone()
        conn.close()
        
        if row:
            last_start = datetime.strptime(row['date'], '%Y-%m-%d')
            return (current_date - last_start).days
        
        return 5  # No previous start found, assume normal rest
    
    def _get_bullpen_availability(self, team_id, game_date=None):
        """
        Calculate bullpen availability based on recent usage.
        Returns a score 0-1 (1 = fully available, 0 = depleted)
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        cache_key = f"{team_id}_{game_date}"
        if cache_key in self.bullpen_cache:
            return self.bullpen_cache[cache_key]
        
        # Look at last 3 days of games
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT COUNT(*) as games_3_days,
                   SUM(CASE WHEN innings > 9 THEN 1 ELSE 0 END) as extra_innings
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND date >= date(?, '-3 days')
            AND date < ?
            AND status = 'final'
        ''', (team_id, team_id, game_date, game_date))
        
        row = c.fetchone()
        conn.close()
        
        games_3_days = row['games_3_days'] if row else 0
        extra_innings = row['extra_innings'] if row and row['extra_innings'] else 0
        
        # Calculate availability (1 = fresh, 0 = depleted)
        # More games = less available, extra innings games hurt more
        availability = 1.0 - (games_3_days * 0.15) - (extra_innings * 0.10)
        availability = max(0.3, min(1.0, availability))  # Floor at 30%
        
        self.bullpen_cache[cache_key] = availability
        return availability
    
    def _calculate_pitcher_score(self, pitcher, opponent_hitting):
        """
        Calculate a 0-1 score for pitcher quality adjusted for opponent.
        Higher = better pitcher.
        """
        if pitcher is None:
            return 0.5  # League average
        
        # Base score from ERA (inverted - lower ERA = higher score)
        era = pitcher.get('era', self.LEAGUE_AVG['era'])
        era_score = 1 - (era / (self.LEAGUE_AVG['era'] * 2))  # Normalize
        era_score = max(0.2, min(0.9, era_score))
        
        # WHIP component
        whip = pitcher.get('whip', self.LEAGUE_AVG['whip'])
        whip_score = 1 - (whip / (self.LEAGUE_AVG['whip'] * 1.5))
        whip_score = max(0.2, min(0.9, whip_score))
        
        # Strikeout ability
        k9 = pitcher.get('k_per_9', self.LEAGUE_AVG['k_per_9'])
        k_score = k9 / (self.LEAGUE_AVG['k_per_9'] * 1.5)
        k_score = max(0.2, min(0.9, k_score))
        
        # Walk rate (inverted - lower is better)
        bb9 = pitcher.get('bb_per_9', self.LEAGUE_AVG['bb_per_9'])
        bb_score = 1 - (bb9 / (self.LEAGUE_AVG['bb_per_9'] * 1.5))
        bb_score = max(0.2, min(0.9, bb_score))
        
        # Workload/experience bonus for IP
        ip = pitcher.get('innings_pitched', 0)
        ip_bonus = min(0.05, ip / 200 * 0.05)  # Up to 5% bonus for heavy workload
        
        # Combine scores
        base_score = (era_score * 0.35 + whip_score * 0.25 + 
                     k_score * 0.25 + bb_score * 0.15) + ip_bonus
        
        # Adjust for opponent hitting quality
        opp_ba = opponent_hitting.get('batting_avg', self.LEAGUE_AVG['batting_avg'])
        opp_factor = self.LEAGUE_AVG['batting_avg'] / opp_ba if opp_ba > 0 else 1.0
        opp_adjustment = (opp_factor - 1.0) * 0.15  # Subtle adjustment
        
        final_score = base_score + opp_adjustment
        return max(0.1, min(0.9, final_score))
    
    def _get_series_position(self, team_id, game_date):
        """
        Determine position in weekend series (game 1, 2, or 3).
        Returns 1, 2, or 3, or 0 if not a series game.
        """
        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d')
        
        day_of_week = game_date.weekday()
        
        # Typical college schedule: Fri/Sat/Sun series
        if day_of_week == 4:  # Friday
            return 1
        elif day_of_week == 5:  # Saturday
            return 2
        elif day_of_week == 6:  # Sunday
            return 3
        
        return 0  # Midweek game
    
    def _calculate_series_adjustment(self, home_starter, away_starter, series_position):
        """
        Calculate adjustment based on series position and pitcher quality gap.
        Teams often save their ace for game 1.
        """
        if series_position == 0:
            return 0  # No adjustment for midweek
        
        # Get pitcher quality scores
        home_quality = self._calculate_pitcher_score(home_starter, {'batting_avg': self.LEAGUE_AVG['batting_avg']})
        away_quality = self._calculate_pitcher_score(away_starter, {'batting_avg': self.LEAGUE_AVG['batting_avg']})
        
        quality_gap = home_quality - away_quality
        
        # Game 1: Quality gap matters more (aces usually pitch)
        # Game 3: Often bullpen games, quality gap matters less
        position_weights = {1: 1.2, 2: 1.0, 3: 0.8}
        weight = position_weights.get(series_position, 1.0)
        
        return quality_gap * weight * 0.03  # Small adjustment
    
    def predict_game(self, home_team_id, away_team_id, neutral_site=False, game_date=None,
                     game_id: str = None, weather_data: Dict = None):
        """
        Predict game outcome based on pitching matchup.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            neutral_site: If True, no home advantage
            game_date: Date string (YYYY-MM-DD) or None for today
            game_id: Optional game ID for weather lookup
            weather_data: Optional dict with weather (overrides database lookup)
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get weather data and calculate adjustment
        if weather_data is None and game_id:
            weather_data = self._get_weather_for_game(game_id)
        weather_adjustment, weather_components = self._calculate_weather_pitcher_adjustment(weather_data)
        
        # Get starting pitchers
        home_starter = self._get_starting_pitcher(home_team_id, game_date)
        away_starter = self._get_starting_pitcher(away_team_id, game_date)
        
        # Get team pitching stats as fallback
        home_team_pitching = self._get_team_pitching_stats(home_team_id)
        away_team_pitching = self._get_team_pitching_stats(away_team_id)
        
        # Get hitting stats for matchup adjustment
        home_hitting = self._get_team_hitting_stats(home_team_id)
        away_hitting = self._get_team_hitting_stats(away_team_id)
        
        # Calculate pitcher scores (opponent adjusted)
        if home_starter:
            home_pitcher_score = self._calculate_pitcher_score(home_starter, away_hitting)
        else:
            home_pitcher_score = self._calculate_pitcher_score(home_team_pitching, away_hitting)
        
        if away_starter:
            away_pitcher_score = self._calculate_pitcher_score(away_starter, home_hitting)
        else:
            away_pitcher_score = self._calculate_pitcher_score(away_team_pitching, home_hitting)
        
        # Apply weather adjustment to pitcher scores
        # Weather affects both pitchers equally (same conditions)
        home_pitcher_score += weather_adjustment
        away_pitcher_score += weather_adjustment
        
        # Clamp scores after weather adjustment
        home_pitcher_score = max(0.1, min(0.9, home_pitcher_score))
        away_pitcher_score = max(0.1, min(0.9, away_pitcher_score))
        
        # Rest day factors
        home_rest = self._get_pitcher_rest_days(
            home_starter.get('id') if home_starter else None, 
            game_date
        )
        away_rest = self._get_pitcher_rest_days(
            away_starter.get('id') if away_starter else None, 
            game_date
        )
        
        home_rest_factor = self.REST_FACTORS.get(min(home_rest, 7), 0.02)
        away_rest_factor = self.REST_FACTORS.get(min(away_rest, 7), 0.02)
        
        # Bullpen availability
        home_bullpen = self._get_bullpen_availability(home_team_id, game_date)
        away_bullpen = self._get_bullpen_availability(away_team_id, game_date)
        
        # Series position adjustment
        series_pos = self._get_series_position(home_team_id, game_date)
        series_adj = self._calculate_series_adjustment(home_starter, away_starter, series_pos)
        
        # Combine factors
        # Base probability from pitcher matchup
        total_pitcher_score = home_pitcher_score + away_pitcher_score
        if total_pitcher_score > 0:
            base_prob = home_pitcher_score / total_pitcher_score
        else:
            base_prob = 0.5
        
        # Apply adjustments
        prob = base_prob * self.STARTER_WEIGHT
        
        # Fatigue adjustment
        fatigue_diff = home_rest_factor - away_rest_factor
        prob += (0.5 + fatigue_diff) * self.FATIGUE_WEIGHT
        
        # Bullpen adjustment
        bullpen_diff = (home_bullpen - away_bullpen) / 2
        prob += (0.5 + bullpen_diff * 0.3) * self.BULLPEN_WEIGHT
        
        # Matchup adjustment (pitcher vs lineup)
        matchup_diff = (home_pitcher_score - away_pitcher_score) * 0.3
        prob += (0.5 + matchup_diff) * self.MATCHUP_WEIGHT
        
        # Series adjustment
        prob += series_adj * self.SERIES_WEIGHT
        
        # Home advantage
        if not neutral_site:
            prob += self.HOME_ADVANTAGE
        
        # Clamp probability
        home_prob = max(0.1, min(0.9, prob))
        
        # Project runs based on pitcher quality (inverted for ERA)
        home_era = home_starter.get('era', home_team_pitching['era']) if home_starter else home_team_pitching['era']
        away_era = away_starter.get('era', away_team_pitching['era']) if away_starter else away_team_pitching['era']
        
        # Runs = opponent's ERA indicates how many runs they typically allow
        away_runs = (home_era / 9) * 9 * 0.7 + away_hitting.get('batting_avg', 0.275) * 20
        home_runs = (away_era / 9) * 9 * 0.7 + home_hitting.get('batting_avg', 0.275) * 20
        
        # Adjust for bullpen
        home_runs += (1 - away_bullpen) * 0.5
        away_runs += (1 - home_bullpen) * 0.5
        
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
                "home_starter": home_starter.get('name') if home_starter else "Team Average",
                "away_starter": away_starter.get('name') if away_starter else "Team Average",
                "home_starter_era": round(home_era, 2),
                "away_starter_era": round(away_era, 2),
                "home_pitcher_score": round(home_pitcher_score, 3),
                "away_pitcher_score": round(away_pitcher_score, 3),
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "home_bullpen_avail": round(home_bullpen, 2),
                "away_bullpen_avail": round(away_bullpen, 2),
                "series_position": series_pos
            },
            "weather": {
                "adjustment": weather_components.get('total_adjustment', 0.0),
                "components": weather_components,
                "has_data": weather_data is not None
            }
        }


# For testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pitching Matchup Model')
    parser.add_argument('home_team', nargs='?', help='Home team ID')
    parser.add_argument('away_team', nargs='?', help='Away team ID')
    parser.add_argument('--neutral', action='store_true', help='Neutral site game')
    parser.add_argument('--game-id', type=str, help='Game ID for weather lookup')
    parser.add_argument('--temp', type=float, help='Temperature (°F)')
    parser.add_argument('--wind', type=float, help='Wind speed (mph)')
    parser.add_argument('--humidity', type=float, help='Humidity (%)')
    parser.add_argument('--dome', action='store_true', help='Indoor dome')
    args = parser.parse_args()
    
    model = PitchingModel()
    
    if args.home_team and args.away_team:
        home = args.home_team.lower().replace(" ", "-")
        away = args.away_team.lower().replace(" ", "-")
        
        # Build weather data from CLI args if provided
        weather_data = None
        if any([args.temp is not None, args.wind is not None, args.humidity is not None, args.dome]):
            weather_data = {}
            if args.temp is not None:
                weather_data['temp_f'] = args.temp
            if args.wind is not None:
                weather_data['wind_speed_mph'] = args.wind
            if args.humidity is not None:
                weather_data['humidity_pct'] = args.humidity
            if args.dome:
                weather_data['is_dome'] = 1
        
        pred = model.predict_game(home, away, args.neutral, 
                                  game_id=args.game_id, weather_data=weather_data)
        
        print(f"\n{'='*55}")
        print(f"  PITCHING MODEL: {away} @ {home}")
        print('='*55)
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nInputs:")
        for k, v in pred['inputs'].items():
            print(f"  {k}: {v}")
        
        # Weather info
        weather = pred.get('weather', {})
        if weather.get('has_data') or weather_data:
            print(f"\n🌤️ Weather Adjustment:")
            comp = weather.get('components', {})
            if comp.get('dome'):
                print(f"  Dome: Indoor game, no weather adjustment")
            else:
                print(f"  Pitcher adjustment: {weather.get('adjustment', 0):+.3f}")
                if 'temp_condition' in comp:
                    print(f"  Temperature: {comp.get('temp_f', 'N/A')}°F ({comp['temp_condition']}, {comp['temp_adj']:+.3f})")
                if 'wind_speed_mph' in comp:
                    print(f"  Wind: {comp['wind_speed_mph']} mph ({comp['wind_adj']:+.3f})")
                if 'humidity_pct' in comp:
                    print(f"  Humidity: {comp['humidity_pct']}% ({comp['humidity_adj']:+.3f})")
    else:
        print("Usage: python pitching_model.py <home_team> <away_team> [--neutral] [--game-id ID] [--temp F] [--wind MPH] [--humidity %] [--dome]")
