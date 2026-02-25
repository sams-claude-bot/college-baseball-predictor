#!/usr/bin/env python3
"""Prediction orchestration for the Poisson model."""

from typing import Dict

from models.weather_model import calculate_weather_adjustment
from .lambda_calc import (
    apply_opponent_adjusted_lambdas,
    calculate_expected_runs,
    clamp_lambda,
    get_league_average,
    get_quality_adjustment,
    get_recent_totals_history,
    get_team_run_stats,
    get_weather_for_game,
)
from .distribution import (
    MAX_RUNS,
    build_probability_matrix,
    build_total_pmf_from_matrix,
    calculate_spread_probability,
    calculate_total_probability_from_pmf,
    calculate_win_probability,
    get_most_likely_scores,
    maybe_apply_overdispersion_to_total_pmf,
)

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
    
    # Get weather data and calculate adjustment (learned from historical data)
    if weather_data is None and game_id:
        weather_data = get_weather_for_game(game_id)
    
    # Weather adjustment (learned from historical data)
    # Note: Backtest shows adjustment doesn't significantly improve predictions,
    # so we show weather info but apply adjustment only if explicitly requested
    weather_adjustment = 0.0
    weather_components = {'has_data': False, 'adjustment_applied': False}
    if weather_data:
        # Set apply_adjustment=False by default (backtest showed it doesn't help)
        # The raw_adjustment is still computed and shown for informational purposes
        weather_adjustment, weather_components = calculate_weather_adjustment(
            weather_data, apply_adjustment=True
        )
        weather_components['has_data'] = True
    
    # Calculate expected runs for each team
    home_adv = 0.0 if neutral_site else 0.5
    
    # Get quality adjustments from batting/pitching tables
    qa_off, qa_def = get_quality_adjustment(team_a, team_b)  # A's offense, B's pitching
    qb_off, qb_def = get_quality_adjustment(team_b, team_a)  # B's offense, A's pitching
    
    if team_a_home:
        lambda_a = calculate_expected_runs(
            stats_a['avg_scored_home'] if not neutral_site else stats_a['avg_scored'],
            stats_b['avg_allowed_away'] if not neutral_site else stats_b['avg_allowed'],
            league_avg,
            home_adv,
            quality_offense=qa_off,
            quality_defense=qb_def,  # B's pitching quality affects A's runs
            team_games=stats_a['games'],
            opponent_games=stats_b['games']
        )
        lambda_b = calculate_expected_runs(
            stats_b['avg_scored_away'] if not neutral_site else stats_b['avg_scored'],
            stats_a['avg_allowed_home'] if not neutral_site else stats_a['avg_allowed'],
            league_avg,
            0.0,
            quality_offense=qb_off,
            quality_defense=qa_def,  # A's pitching quality affects B's runs
            team_games=stats_b['games'],
            opponent_games=stats_a['games']
        )
        base_lambda_home, base_lambda_away = lambda_a, lambda_b
        lambda_a, lambda_b, lambda_adjustment_meta = apply_opponent_adjusted_lambdas(
            base_lambda_home, base_lambda_away, stats_a, stats_b, league_avg
        )
    else:
        lambda_a = calculate_expected_runs(
            stats_a['avg_scored_away'] if not neutral_site else stats_a['avg_scored'],
            stats_b['avg_allowed_home'] if not neutral_site else stats_b['avg_allowed'],
            league_avg,
            0.0,
            quality_offense=qa_off,
            quality_defense=qb_def,
            team_games=stats_a['games'],
            opponent_games=stats_b['games']
        )
        lambda_b = calculate_expected_runs(
            stats_b['avg_scored_home'] if not neutral_site else stats_b['avg_scored'],
            stats_a['avg_allowed_away'] if not neutral_site else stats_a['avg_allowed'],
            league_avg,
            home_adv,
            quality_offense=qb_off,
            quality_defense=qa_def,
            team_games=stats_b['games'],
            opponent_games=stats_a['games']
        )
        # team_b is home in this branch; apply adjustment in home/away space then map back
        base_lambda_away, base_lambda_home = lambda_a, lambda_b
        adj_home, adj_away, lambda_adjustment_meta = apply_opponent_adjusted_lambdas(
            base_lambda_home, base_lambda_away, stats_b, stats_a, league_avg
        )
        lambda_b, lambda_a = adj_home, adj_away
    
    # Apply weather adjustment to expected runs (learned from historical data)
    # Split adjustment equally between teams (weather affects both)
    if weather_adjustment != 0:
        per_team_adj = weather_adjustment / 2.0
        lambda_a += per_team_adj
        lambda_b += per_team_adj
    lambda_a = clamp_lambda(lambda_a)
    lambda_b = clamp_lambda(lambda_b)
    
    # Build probability matrix
    matrix = build_probability_matrix(lambda_a, lambda_b)
    
    # Calculate outcomes
    p_a_wins, p_b_wins, p_tie = calculate_win_probability(matrix)
    
    # Expected total
    expected_total = lambda_a + lambda_b

    # Totals-only dispersion adjustment (keeps win/spread outputs stable)
    total_pmf = build_total_pmf_from_matrix(matrix)
    recent_totals = get_recent_totals_history(team_a, team_b, last_n_games)
    total_pmf_for_totals, overdispersion_meta = maybe_apply_overdispersion_to_total_pmf(
        total_pmf, expected_total, recent_totals
    )
    
    # Common totals analysis
    totals_analysis = {}
    for total in [expected_total - 1, expected_total - 0.5, expected_total, 
                  expected_total + 0.5, expected_total + 1]:
        totals_analysis[total] = calculate_total_probability_from_pmf(total_pmf_for_totals, total)
    
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
        
        # Weather adjustment info (learned from historical data)
        'weather': {
            'adjustment': weather_components.get('total_adjustment', 0.0),
            'components': weather_components,
            'has_data': weather_components.get('has_data', False),
            'model_r_squared': weather_components.get('model_r_squared', 0.0)
        },

        # Model assumption diagnostics (additive, backward-compatible)
        'poisson_adjustments': {
            'lambda_adjustment': lambda_adjustment_meta,
            'totals_overdispersion': overdispersion_meta
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
