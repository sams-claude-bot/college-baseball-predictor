#!/usr/bin/env python3
"""Lambda/stat calculation helpers for the Poisson model."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.database import get_connection

try:
    from config import model_config as _model_config
except Exception:
    _model_config = None

def _cfg(name: str, default):
    """Config lookup with safe fallback for backward compatibility."""
    return getattr(_model_config, name, default) if _model_config else default

def clamp_lambda(lambda_: float) -> float:
    """Clamp expected runs to a conservative, configurable range."""
    return max(
        _cfg('POISSON_LAMBDA_MIN', 0.5),
        min(lambda_, _cfg('POISSON_LAMBDA_MAX', 12.0))
    )

def get_team_run_stats(team_id: str, last_n_games: int = None) -> Dict:
    """Get team's runs scored/allowed statistics."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Base query
    query = '''
        SELECT 
            CASE WHEN home_team_id = ? THEN home_score ELSE away_score END as runs_scored,
            CASE WHEN home_team_id = ? THEN away_score ELSE home_score END as runs_allowed,
            CASE WHEN home_team_id = ? THEN 1 ELSE 0 END as is_home
        FROM games
        WHERE (home_team_id = ? OR away_team_id = ?)
        AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date DESC
    '''
    
    if last_n_games:
        query += f' LIMIT {last_n_games}'
    
    cur.execute(query, (team_id, team_id, team_id, team_id, team_id))
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        # Return league average if no data
        return {
            'games': 0,
            'avg_scored': 5.5,
            'avg_allowed': 5.5,
            'avg_scored_home': 5.8,
            'avg_scored_away': 5.2,
            'avg_allowed_home': 5.2,
            'avg_allowed_away': 5.8,
            'total_scored': 0,
            'total_allowed': 0
        }
    
    total_scored = sum(r[0] for r in rows)
    total_allowed = sum(r[1] for r in rows)
    games = len(rows)
    
    home_games = [r for r in rows if r[2] == 1]
    away_games = [r for r in rows if r[2] == 0]
    
    return {
        'games': games,
        'avg_scored': total_scored / games,
        'avg_allowed': total_allowed / games,
        'avg_scored_home': sum(r[0] for r in home_games) / len(home_games) if home_games else 5.5,
        'avg_scored_away': sum(r[0] for r in away_games) / len(away_games) if away_games else 5.5,
        'avg_allowed_home': sum(r[1] for r in home_games) / len(home_games) if home_games else 5.5,
        'avg_allowed_away': sum(r[1] for r in away_games) / len(away_games) if away_games else 5.5,
        'total_scored': total_scored,
        'total_allowed': total_allowed
    }

def get_recent_totals_history(team_a: str, team_b: str, last_n_games: int = None) -> List[float]:
    """
    Get recent combined totals for games involving either team.
    Used only for optional totals overdispersion handling.
    """
    sample_limit = last_n_games or 20
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            '''
            SELECT (home_score + away_score) AS total_runs
            FROM games
            WHERE (home_team_id IN (?, ?) OR away_team_id IN (?, ?))
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date DESC
            LIMIT ?
            ''',
            (team_a, team_b, team_a, team_b, sample_limit)
        )
        rows = cur.fetchall()
        totals = [float(r[0]) for r in rows if r and r[0] is not None]
        return totals
    except Exception:
        return []
    finally:
        conn.close()

def get_league_average() -> float:
    """Get league average runs per game."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT AVG(home_score + away_score) / 2.0
        FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
    ''')
    result = cur.fetchone()[0]
    conn.close()
    return result if result else 5.5

def get_weather_for_game(game_id: str) -> Optional[Dict]:
    """Fetch weather data for a game from game_weather table."""
    if not game_id:
        return None
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT temp_f, humidity_pct, wind_speed_mph, wind_direction_deg,
               precip_prob_pct, is_dome
        FROM game_weather WHERE game_id = ?
    ''', (game_id,))
    row = cur.fetchone()
    conn.close()
    
    if row:
        return {
            'temp_f': row[0],
            'humidity_pct': row[1],
            'wind_speed_mph': row[2],
            'wind_direction_deg': row[3],
            'precip_prob_pct': row[4],
            'is_dome': row[5],
        }
    return None

def get_quality_adjustment(team_id: str, opponent_id: str) -> tuple:
    """
    Get batting/pitching quality adjustment factors.
    Returns (offense_mult, defense_mult) where 1.0 = league average.
    Blends game-results-based stats with player-quality-based stats.
    """
    conn = get_connection()
    c = conn.cursor()

    offense_mult = 1.0
    defense_mult = 1.0

    # Batting quality for the offense
    try:
        c.execute("""SELECT lineup_wrc_plus, lineup_ops FROM team_batting_quality
                     WHERE team_id = ?""", (team_id,))
        row = c.fetchone()
        if row and row['lineup_wrc_plus']:
            # wRC+ of 120 means 20% better than league avg offense
            offense_mult = (row['lineup_wrc_plus'] / 100.0)
            # Dampen to avoid over-adjustment early season
            offense_mult = 0.5 + offense_mult * 0.5  # range: ~0.75 - 1.25
    except Exception:
        pass

    # Pitching quality for the opponent's defense
    try:
        c.execute("""SELECT staff_era, staff_fip FROM team_pitching_quality
                     WHERE team_id = ?""", (opponent_id,))
        row = c.fetchone()
        if row and row['staff_era']:
            # Lower ERA = better pitching = fewer runs allowed
            # League avg ERA ~4.50 for D1
            era_ratio = row['staff_era'] / 4.50
            defense_mult = 0.5 + era_ratio * 0.5  # range: ~0.67 - 1.3
    except Exception:
        pass

    conn.close()
    return offense_mult, defense_mult

def _calculate_strength_factors(stats: Dict, league_avg: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Derive conservative offense/defense factors from team run stats.
    offense_strength > 1 increases own lambda
    defense_factor < 1 lowers opponent lambda (strong defense)
    """
    if not stats or league_avg <= 0:
        return None, None
    if 'avg_scored' not in stats or 'avg_allowed' not in stats:
        return None, None

    try:
        games = max(int(stats.get('games', 0) or 0), 0)
        k = 12.0  # small-sample regression toward neutral
        w = games / (games + k)

        scored = float(stats['avg_scored'])
        allowed = float(stats['avg_allowed'])
        neutral = float(league_avg)

        reg_scored = w * scored + (1.0 - w) * neutral
        reg_allowed = w * allowed + (1.0 - w) * neutral

        offense_strength = reg_scored / neutral
        defense_factor = reg_allowed / neutral

        offense_strength = max(
            _cfg('POISSON_OFFENSE_STRENGTH_MIN', 0.85),
            min(offense_strength, _cfg('POISSON_OFFENSE_STRENGTH_MAX', 1.15))
        )
        defense_factor = max(
            _cfg('POISSON_DEFENSE_FACTOR_MIN', 0.85),
            min(defense_factor, _cfg('POISSON_DEFENSE_FACTOR_MAX', 1.15))
        )
        return offense_strength, defense_factor
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None

def apply_opponent_adjusted_lambdas(
    base_home: float,
    base_away: float,
    stats_home: Dict,
    stats_away: Dict,
    league_avg: float
) -> Tuple[float, float, Dict]:
    """
    Apply conservative opponent-adjusted lambda scaling.

    Formula:
      lambda_home = base_home * offense_strength_home * defense_factor_away
      lambda_away = base_away * offense_strength_away * defense_factor_home

    Falls back to original lambdas if required stats are unavailable.
    """
    if not _cfg('POISSON_ENABLE_OPPONENT_ADJUSTMENT', True):
        return clamp_lambda(base_home), clamp_lambda(base_away), {
            'enabled': False,
            'applied': False,
            'fallback': True
        }

    home_offense, home_defense = _calculate_strength_factors(stats_home, league_avg)
    away_offense, away_defense = _calculate_strength_factors(stats_away, league_avg)

    if None in (home_offense, home_defense, away_offense, away_defense):
        return clamp_lambda(base_home), clamp_lambda(base_away), {
            'enabled': True,
            'applied': False,
            'fallback': True
        }

    try:
        strength = float(_cfg('POISSON_OPPONENT_ADJUSTMENT_STRENGTH', 1.0))
    except (TypeError, ValueError):
        strength = 1.0
    strength = max(0.0, min(strength, 1.0))

    def blend_to_neutral(factor: float) -> float:
        return 1.0 + strength * (factor - 1.0)

    home_offense_blended = blend_to_neutral(home_offense)
    away_offense_blended = blend_to_neutral(away_offense)
    home_defense_blended = blend_to_neutral(home_defense)
    away_defense_blended = blend_to_neutral(away_defense)

    lambda_home = clamp_lambda(base_home * home_offense_blended * away_defense_blended)
    lambda_away = clamp_lambda(base_away * away_offense_blended * home_defense_blended)
    return lambda_home, lambda_away, {
        'enabled': True,
        'applied': True,
        'fallback': False,
        'strength': round(strength, 4),
        'offense_strength_home': round(home_offense, 4),
        'offense_strength_away': round(away_offense, 4),
        'defense_factor_home': round(home_defense, 4),
        'defense_factor_away': round(away_defense, 4),
        'offense_strength_home_blended': round(home_offense_blended, 4),
        'offense_strength_away_blended': round(away_offense_blended, 4),
        'defense_factor_home_blended': round(home_defense_blended, 4),
        'defense_factor_away_blended': round(away_defense_blended, 4),
    }

def calculate_expected_runs(team_offense: float, opponent_defense: float, 
                            league_avg: float, home_advantage: float = 0.3,
                            quality_offense: float = 1.0,
                            quality_defense: float = 1.0,
                            team_games: int = 0,
                            opponent_games: int = 0) -> float:
    """
    Calculate expected runs for a team.
    
    Uses log5-style adjustment with quality modifiers:
    Expected = (Team_Offense * Opponent_AllowedRate / League_Average) * quality_blend
    
    With small samples, regresses inputs toward league average to prevent
    extreme splits (e.g. 7 runs/home game × 8 allowed/away game) from
    producing absurd projections.
    
    quality_offense: multiplier from batting quality (1.0 = avg)
    quality_defense: multiplier from opponent pitching quality (1.0 = avg)
    team_games: number of games for the offensive team (for regression)
    opponent_games: number of games for the defensive team (for regression)
    """
    if league_avg == 0:
        league_avg = 5.5
    
    # Regress toward league average with small samples
    # At 0 games: 100% league avg. At 20+ games: ~85% real data.
    # This prevents 3-game home splits from dominating.
    def regress(value, n_games, prior=league_avg):
        # Bayesian-style: weight = n / (n + k), where k = 10 (prior strength)
        k = 10
        weight = n_games / (n_games + k)
        return weight * value + (1 - weight) * prior
    
    team_offense = regress(team_offense, team_games)
    opponent_defense = regress(opponent_defense, opponent_games)
    
    expected = (team_offense * opponent_defense) / league_avg
    
    # Blend in quality adjustments (10% weight — conservative early season,
    # will become more meaningful as sample sizes grow)
    quality_blend = (quality_offense * quality_defense)
    expected = expected * (0.9 + 0.1 * quality_blend)
    
    expected += home_advantage  # Home teams score ~0.3 more runs on average
    
    # Hard cap: no team realistically averages more than 12 runs/game
    return max(0.5, min(expected, 12.0))
