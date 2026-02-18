#!/usr/bin/env python3
"""
Poisson Run Distribution Model - Models exact run probabilities.

Theory: Runs scored in baseball follow approximately a Poisson distribution.
Instead of just predicting "who wins", we model:
- P(Team A scores X runs) for any X
- P(Team B scores Y runs) for any Y
- Full joint probability matrix
- Better estimates for totals, run lines, and game variance

Applications:
- More accurate win probabilities (sum of all P(A > B) scenarios)
- Over/under predictions with confidence intervals
- Run line probabilities (-1.5 spread)
- Variance estimation (close game vs blowout likelihood)
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection


# Maximum runs to model (beyond this is negligible probability)
MAX_RUNS = 25

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


@lru_cache(maxsize=1000)
def poisson_pmf(k: int, lambda_: float) -> float:
    """Poisson probability mass function: P(X = k) given mean lambda."""
    if lambda_ <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    # Use log to avoid overflow for large values
    log_prob = k * math.log(lambda_) - lambda_ - math.lgamma(k + 1)
    return math.exp(log_prob)


def poisson_cdf(k: int, lambda_: float) -> float:
    """Cumulative distribution: P(X <= k)."""
    return sum(poisson_pmf(i, lambda_) for i in range(k + 1))


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


def calculate_weather_multiplier(weather_data: Optional[Dict] = None) -> Tuple[float, Dict]:
    """
    Calculate a weather multiplier for expected runs.
    
    Returns a multiplier (e.g., 0.95 to 1.08) and breakdown of components.
    
    Research-based adjustments:
    - Temperature: ~0.5% more runs per 10°F above 70°F (ball carries better, pitchers tire)
    - Wind out (0-90° or 270-360°): +0.03 runs per mph (helps fly balls)
    - Wind in (135-225°): -0.03 runs per mph (hurts fly balls)
    - Humidity: ~0.3% more runs per 10% above 50% (slightly affects ball flight)
    - Dome: Controlled environment, use baseline (multiplier = 1.0)
    """
    if weather_data is None:
        weather_data = DEFAULT_WEATHER.copy()
    
    # Use defaults for missing values (careful with 0 being falsy)
    temp = weather_data.get('temp_f') if weather_data.get('temp_f') is not None else DEFAULT_WEATHER['temp_f']
    humidity = weather_data.get('humidity_pct') if weather_data.get('humidity_pct') is not None else DEFAULT_WEATHER['humidity_pct']
    wind_speed = weather_data.get('wind_speed_mph') if weather_data.get('wind_speed_mph') is not None else DEFAULT_WEATHER['wind_speed_mph']
    wind_dir = weather_data.get('wind_direction_deg') if weather_data.get('wind_direction_deg') is not None else DEFAULT_WEATHER['wind_direction_deg']
    is_dome = weather_data.get('is_dome', 0)
    
    # If it's a dome, return neutral multiplier
    if is_dome:
        return 1.0, {'dome': True, 'temp_adj': 0, 'wind_adj': 0, 'humidity_adj': 0}
    
    components = {'dome': False}
    multiplier = 1.0
    
    # Temperature adjustment: ~0.5% per 10°F deviation from 70°F
    # Hot weather = more runs, cold = fewer
    temp_diff = temp - BASELINE_TEMP
    temp_adjustment = (temp_diff / 10.0) * 0.005  # 0.5% per 10°F
    multiplier += temp_adjustment
    components['temp_adj'] = round(temp_adjustment, 4)
    components['temp_f'] = temp
    
    # Wind adjustment based on direction
    # 0° = North (blowing from N to S, toward home plate at most parks)
    # 90° = East, 180° = South (blowing out), 270° = West
    # Standard park orientation: home plate to the south, CF to the north
    # Wind blowing "out" (helping fly balls): 135-225° (from S/SE/SW toward CF)
    # Wind blowing "in" (hurting fly balls): 315-360° or 0-45° (from N toward HP)
    # Crosswinds have mixed effect
    
    # Convert to radians for trig
    wind_rad = math.radians(wind_dir)
    
    # Calculate wind effect: positive = blowing out (helps offense)
    # cos(180°) = -1 (blowing out), cos(0°) = 1 (blowing in)
    wind_effect = -math.cos(wind_rad)  # -1 to 1, positive = out
    
    # Scale by wind speed: ~0.03 runs per 10 mph at full effect
    wind_adjustment = (wind_speed / 10.0) * wind_effect * 0.03
    
    # Cap wind adjustment at ±5%
    wind_adjustment = max(-0.05, min(0.05, wind_adjustment))
    multiplier += wind_adjustment
    components['wind_adj'] = round(wind_adjustment, 4)
    components['wind_speed_mph'] = wind_speed
    components['wind_dir_deg'] = wind_dir
    components['wind_effect'] = 'out' if wind_effect > 0.3 else 'in' if wind_effect < -0.3 else 'cross'
    
    # Humidity adjustment: ~0.3% per 10% above 50%
    # Higher humidity = slightly more runs (ball travels slightly less but grip harder)
    humidity_diff = humidity - 50.0
    humidity_adjustment = (humidity_diff / 10.0) * 0.003  # 0.3% per 10%
    humidity_adjustment = max(-0.02, min(0.02, humidity_adjustment))  # Cap at ±2%
    multiplier += humidity_adjustment
    components['humidity_adj'] = round(humidity_adjustment, 4)
    components['humidity_pct'] = humidity
    
    # Cap total multiplier to reasonable range
    multiplier = max(0.90, min(1.12, multiplier))
    components['total_multiplier'] = round(multiplier, 4)
    
    return multiplier, components


def calculate_expected_runs(team_offense: float, opponent_defense: float, 
                            league_avg: float, home_advantage: float = 0.3) -> float:
    """
    Calculate expected runs for a team.
    
    Uses log5-style adjustment:
    Expected = Team_Offense * Opponent_AllowedRate / League_Average
    """
    if league_avg == 0:
        league_avg = 5.5
    
    expected = (team_offense * opponent_defense) / league_avg
    expected += home_advantage  # Home teams score ~0.3 more runs on average
    
    return max(0.5, expected)  # Floor at 0.5 runs


def build_probability_matrix(lambda_a: float, lambda_b: float) -> List[List[float]]:
    """
    Build joint probability matrix P(A=i, B=j) for all i,j in [0, MAX_RUNS].
    
    Returns 2D list where matrix[i][j] = P(Team A scores i, Team B scores j)
    """
    matrix = []
    for i in range(MAX_RUNS + 1):
        row = []
        for j in range(MAX_RUNS + 1):
            # Assuming independence (simplification - could add correlation)
            prob = poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
            row.append(prob)
        matrix.append(row)
    return matrix


def calculate_win_probability(matrix: List[List[float]]) -> Tuple[float, float, float]:
    """
    Calculate win/loss/tie probabilities from joint probability matrix.
    
    Returns (P(A wins), P(B wins), P(tie))
    """
    p_a_wins = 0.0
    p_b_wins = 0.0
    p_tie = 0.0
    
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            prob = matrix[i][j]
            if i > j:
                p_a_wins += prob
            elif j > i:
                p_b_wins += prob
            else:
                p_tie += prob
    
    # In baseball, ties go to extra innings - redistribute tie probability
    # Assume 50/50 split in extras (slight simplification)
    p_a_wins += p_tie * 0.5
    p_b_wins += p_tie * 0.5
    
    return p_a_wins, p_b_wins, p_tie


def calculate_total_probability(matrix: List[List[float]], total: float) -> Dict:
    """
    Calculate over/under probabilities for a total line.
    
    Returns probabilities for over, under, and push.
    """
    p_over = 0.0
    p_under = 0.0
    p_push = 0.0
    
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            prob = matrix[i][j]
            combined = i + j
            if combined > total:
                p_over += prob
            elif combined < total:
                p_under += prob
            else:
                p_push += prob
    
    return {
        'over': round(p_over, 4),
        'under': round(p_under, 4),
        'push': round(p_push, 4),
        'total_line': total
    }


def calculate_spread_probability(matrix: List[List[float]], spread: float) -> Dict:
    """
    Calculate probability Team A covers the spread.
    
    spread > 0 means A is underdog (e.g., +1.5)
    spread < 0 means A is favorite (e.g., -1.5)
    """
    p_cover = 0.0
    p_not_cover = 0.0
    p_push = 0.0
    
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            prob = matrix[i][j]
            margin = i - j + spread  # A's margin adjusted by spread
            if margin > 0:
                p_cover += prob
            elif margin < 0:
                p_not_cover += prob
            else:
                p_push += prob
    
    return {
        'cover': round(p_cover, 4),
        'not_cover': round(p_not_cover, 4),
        'push': round(p_push, 4),
        'spread': spread
    }


def get_most_likely_scores(matrix: List[List[float]], top_n: int = 10) -> List[Dict]:
    """Get the most likely final scores."""
    scores = []
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            scores.append({
                'team_a_runs': i,
                'team_b_runs': j,
                'probability': matrix[i][j]
            })
    
    scores.sort(key=lambda x: x['probability'], reverse=True)
    return scores[:top_n]


def predict(team_a: str, team_b: str, 
            neutral_site: bool = False,
            team_a_home: bool = True,
            last_n_games: int = None,
            game_id: str = None,
            weather_data: Dict = None) -> Dict:
    """
    Full Poisson prediction for a matchup.
    
    Args:
        team_a: First team ID
        team_b: Second team ID
        neutral_site: If True, no home advantage
        team_a_home: If True (default), team_a is home team
        last_n_games: Limit stats to last N games (for recency weighting)
        game_id: Optional game ID to fetch weather from database
        weather_data: Optional dict with weather (overrides database lookup)
    
    Returns:
        Comprehensive prediction with win prob, totals, spreads, likely scores
    """
    # Get team stats
    stats_a = get_team_run_stats(team_a, last_n_games)
    stats_b = get_team_run_stats(team_b, last_n_games)
    league_avg = get_league_average()
    
    # Get weather data and calculate multiplier
    if weather_data is None and game_id:
        weather_data = get_weather_for_game(game_id)
    weather_multiplier, weather_components = calculate_weather_multiplier(weather_data)
    
    # Calculate expected runs for each team
    home_adv = 0.0 if neutral_site else 0.3
    
    if team_a_home:
        lambda_a = calculate_expected_runs(
            stats_a['avg_scored_home'] if not neutral_site else stats_a['avg_scored'],
            stats_b['avg_allowed_away'] if not neutral_site else stats_b['avg_allowed'],
            league_avg,
            home_adv
        )
        lambda_b = calculate_expected_runs(
            stats_b['avg_scored_away'] if not neutral_site else stats_b['avg_scored'],
            stats_a['avg_allowed_home'] if not neutral_site else stats_a['avg_allowed'],
            league_avg,
            0.0
        )
    else:
        lambda_a = calculate_expected_runs(
            stats_a['avg_scored_away'] if not neutral_site else stats_a['avg_scored'],
            stats_b['avg_allowed_home'] if not neutral_site else stats_b['avg_allowed'],
            league_avg,
            0.0
        )
        lambda_b = calculate_expected_runs(
            stats_b['avg_scored_home'] if not neutral_site else stats_b['avg_scored'],
            stats_a['avg_allowed_away'] if not neutral_site else stats_a['avg_allowed'],
            league_avg,
            home_adv
        )
    
    # Apply weather multiplier to expected runs
    lambda_a *= weather_multiplier
    lambda_b *= weather_multiplier
    
    # Build probability matrix
    matrix = build_probability_matrix(lambda_a, lambda_b)
    
    # Calculate outcomes
    p_a_wins, p_b_wins, p_tie = calculate_win_probability(matrix)
    
    # Expected total
    expected_total = lambda_a + lambda_b
    
    # Common totals analysis
    totals_analysis = {}
    for total in [expected_total - 1, expected_total - 0.5, expected_total, 
                  expected_total + 0.5, expected_total + 1]:
        totals_analysis[total] = calculate_total_probability(matrix, total)
    
    # Spread analysis (run line)
    spread_analysis = {
        '-1.5': calculate_spread_probability(matrix, -1.5),
        '+1.5': calculate_spread_probability(matrix, 1.5),
        '-2.5': calculate_spread_probability(matrix, -2.5),
        '+2.5': calculate_spread_probability(matrix, 2.5)
    }
    
    # Most likely scores
    likely_scores = get_most_likely_scores(matrix, 10)
    
    return {
        'team_a': team_a,
        'team_b': team_b,
        'team_a_home': team_a_home,
        'neutral_site': neutral_site,
        
        # Win probabilities
        'win_prob_a': round(p_a_wins, 4),
        'win_prob_b': round(p_b_wins, 4),
        
        # Expected runs
        'expected_runs_a': round(lambda_a, 2),
        'expected_runs_b': round(lambda_b, 2),
        'expected_total': round(expected_total, 2),
        
        # Team stats used
        'team_a_stats': {
            'games': stats_a['games'],
            'avg_scored': round(stats_a['avg_scored'], 2),
            'avg_allowed': round(stats_a['avg_allowed'], 2)
        },
        'team_b_stats': {
            'games': stats_b['games'],
            'avg_scored': round(stats_b['avg_scored'], 2),
            'avg_allowed': round(stats_b['avg_allowed'], 2)
        },
        
        # Betting analysis
        'totals': totals_analysis,
        'spreads': spread_analysis,
        
        # Score probabilities
        'most_likely_scores': likely_scores,
        
        # Variance indicators
        'blowout_prob': sum(matrix[i][j] for i in range(MAX_RUNS+1) 
                           for j in range(MAX_RUNS+1) if abs(i-j) >= 5),
        'one_run_game_prob': sum(matrix[i][j] for i in range(MAX_RUNS+1) 
                                 for j in range(MAX_RUNS+1) if abs(i-j) == 1),
        
        # Weather adjustment info
        'weather': {
            'multiplier': weather_components.get('total_multiplier', 1.0),
            'components': weather_components,
            'has_data': weather_data is not None
        }
    }


def compare_to_line(prediction: Dict, dk_total: float = None, 
                    dk_spread_a: float = None) -> Dict:
    """Compare Poisson prediction to DraftKings lines for edge detection."""
    edges = {}
    
    if dk_total:
        model_total = prediction['expected_total']
        total_diff = model_total - dk_total
        
        # Find closest line in our analysis
        closest_line = min(prediction['totals'].keys(), 
                          key=lambda x: abs(x - dk_total))
        total_probs = prediction['totals'][closest_line]
        
        edges['total'] = {
            'dk_line': dk_total,
            'model_expected': model_total,
            'difference': round(total_diff, 2),
            'recommendation': 'OVER' if total_diff > 0.5 else 'UNDER' if total_diff < -0.5 else 'NO EDGE',
            'over_prob': total_probs['over'],
            'under_prob': total_probs['under']
        }
    
    if dk_spread_a is not None:
        spread_key = f"{dk_spread_a:+.1f}".replace('.0', '')
        if spread_key in prediction['spreads']:
            spread_probs = prediction['spreads'][spread_key]
        else:
            spread_probs = calculate_spread_probability(
                build_probability_matrix(
                    prediction['expected_runs_a'], 
                    prediction['expected_runs_b']
                ), 
                dk_spread_a
            )
        
        edges['spread'] = {
            'dk_spread': dk_spread_a,
            'cover_prob': spread_probs['cover'],
            'recommendation': 'COVER' if spread_probs['cover'] > 0.55 else 
                             'FADE' if spread_probs['cover'] < 0.45 else 'NO EDGE'
        }
    
    return edges


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Poisson Run Distribution Model')
    parser.add_argument('team_a', help='First team ID')
    parser.add_argument('team_b', help='Second team ID')
    parser.add_argument('--neutral', action='store_true', help='Neutral site game')
    parser.add_argument('--away', action='store_true', help='Team A is away team')
    parser.add_argument('--last-n', type=int, help='Use only last N games')
    parser.add_argument('--dk-total', type=float, help='DraftKings total line')
    parser.add_argument('--dk-spread', type=float, help='DraftKings spread for team A')
    parser.add_argument('--game-id', type=str, help='Game ID for weather lookup')
    parser.add_argument('--temp', type=float, help='Temperature (°F)')
    parser.add_argument('--wind', type=float, help='Wind speed (mph)')
    parser.add_argument('--wind-dir', type=int, help='Wind direction (degrees)')
    parser.add_argument('--humidity', type=float, help='Humidity (%)')
    parser.add_argument('--dome', action='store_true', help='Indoor dome')
    args = parser.parse_args()
    
    # Build weather data from CLI args if provided
    weather_data = None
    if any([args.temp is not None, args.wind is not None, args.wind_dir is not None, 
            args.humidity is not None, args.dome]):
        weather_data = {}
        if args.temp is not None:
            weather_data['temp_f'] = args.temp
        if args.wind is not None:
            weather_data['wind_speed_mph'] = args.wind
        if args.wind_dir is not None:
            weather_data['wind_direction_deg'] = args.wind_dir
        if args.humidity is not None:
            weather_data['humidity_pct'] = args.humidity
        if args.dome:
            weather_data['is_dome'] = 1
    
    result = predict(args.team_a, args.team_b, 
                     neutral_site=args.neutral,
                     team_a_home=not args.away,
                     last_n_games=args.last_n,
                     game_id=args.game_id,
                     weather_data=weather_data)
    
    print(f"\n{'='*60}")
    print(f"POISSON MODEL: {args.team_a} vs {args.team_b}")
    print(f"{'='*60}")
    
    venue = "Neutral" if args.neutral else f"@ {args.team_a if not args.away else args.team_b}"
    print(f"Venue: {venue}")
    
    print(f"\n📊 WIN PROBABILITY:")
    print(f"  {args.team_a}: {result['win_prob_a']:.1%}")
    print(f"  {args.team_b}: {result['win_prob_b']:.1%}")
    
    print(f"\n🎯 EXPECTED RUNS:")
    print(f"  {args.team_a}: {result['expected_runs_a']:.1f}")
    print(f"  {args.team_b}: {result['expected_runs_b']:.1f}")
    print(f"  Total: {result['expected_total']:.1f}")
    
    print(f"\n📈 RUN LINE (-1.5):")
    print(f"  {args.team_a} -1.5: {result['spreads']['-1.5']['cover']:.1%}")
    print(f"  {args.team_b} +1.5: {result['spreads']['+1.5']['cover']:.1%}")
    
    print(f"\n🎲 GAME CHARACTER:")
    print(f"  One-run game: {result['one_run_game_prob']:.1%}")
    print(f"  Blowout (5+): {result['blowout_prob']:.1%}")
    
    print(f"\n🏆 MOST LIKELY SCORES:")
    for score in result['most_likely_scores'][:5]:
        print(f"  {score['team_a_runs']}-{score['team_b_runs']}: {score['probability']:.1%}")
    
    # Weather info
    weather = result.get('weather', {})
    if weather.get('has_data') or weather_data:
        print(f"\n🌤️ WEATHER ADJUSTMENT:")
        comp = weather.get('components', {})
        if comp.get('dome'):
            print(f"  Dome: Indoor game, no weather adjustment")
        else:
            print(f"  Multiplier: {weather.get('multiplier', 1.0):.3f}x")
            if 'temp_f' in comp:
                print(f"  Temperature: {comp['temp_f']:.0f}°F ({comp['temp_adj']:+.3f})")
            if 'wind_speed_mph' in comp:
                print(f"  Wind: {comp['wind_speed_mph']:.0f} mph {comp.get('wind_effect', '')} ({comp['wind_adj']:+.3f})")
            if 'humidity_pct' in comp:
                print(f"  Humidity: {comp['humidity_pct']:.0f}% ({comp['humidity_adj']:+.3f})")
    
    # DK comparison if provided
    if args.dk_total or args.dk_spread:
        edges = compare_to_line(result, args.dk_total, args.dk_spread)
        print(f"\n💰 VS DRAFTKINGS:")
        if 'total' in edges:
            t = edges['total']
            print(f"  Total: DK {t['dk_line']} vs Model {t['model_expected']:.1f} "
                  f"→ {t['recommendation']} ({t['over_prob']:.1%} over)")
        if 'spread' in edges:
            s = edges['spread']
            print(f"  Spread: {args.team_a} {s['dk_spread']:+.1f} "
                  f"→ {s['recommendation']} ({s['cover_prob']:.1%} cover)")
