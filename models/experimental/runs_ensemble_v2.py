"""
Experimental Runs Ensemble V2 — Champion/Challenger model.

Changes from production (models/runs_ensemble.py):
1. Removed pitching component (worst performer: +1.50 bias, 34% OVER accuracy)
2. New weights: poisson 0.60, advanced 0.40
3. Negative Binomial O/U alongside Poisson
4. Game-context adjustment (conference/non-conf, ranked/unranked)
5. OVER confidence gate (require >8% edge for OVER predictions)
"""

import math
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.database import get_connection
from models.negbin_model import negbin_over_under, fit_dispersion_from_db

# V2 weights: no pitching
DEFAULT_RUN_WEIGHTS_V2 = {
    'poisson': 0.60,
    'advanced': 0.40,
}

# OVER confidence gate threshold
OVER_EDGE_THRESHOLD = 8.0  # Require >8% edge for OVER predictions

# P4 conferences (Power 4 + independents that play like P4)
P4_CONFERENCES = {
    'SEC', 'ACC', 'Big 12', 'Big Ten', 'Pac-12',
    # Common DB variants
    'Southeastern', 'Atlantic Coast', 'Big XII', 'Big East',
}


def get_team_info(team_id: str) -> Optional[Dict]:
    """Get team info including conference and ranking."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT id, name, conference, current_rank
        FROM teams WHERE id = ?
    ''', (team_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row['id'],
        'name': row['name'],
        'conference': row['conference'],
        'rank': row['current_rank'],
    }


def get_game_context(home_team_id: str, away_team_id: str,
                     game_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Determine game context for variance adjustment.
    
    Returns context dict with is_conference, is_p4_vs_mid, rank info.
    """
    home = get_team_info(home_team_id)
    away = get_team_info(away_team_id)
    
    context = {
        'is_conference': False,
        'is_p4_vs_mid': False,
        'is_ranked_vs_unranked': False,
        'variance_boost': 1.0,
    }
    
    if not home or not away:
        return context
    
    # Check if conference game (from game record or team conferences)
    if game_id:
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT is_conference_game FROM games WHERE id = ?', (game_id,))
        row = c.fetchone()
        conn.close()
        if row:
            context['is_conference'] = bool(row['is_conference_game'])
    
    if not context['is_conference'] and home.get('conference') and away.get('conference'):
        context['is_conference'] = (home['conference'] == away['conference'])
    
    # P4 vs mid-major detection
    home_p4 = home.get('conference', '') in P4_CONFERENCES
    away_p4 = away.get('conference', '') in P4_CONFERENCES
    if (home_p4 and not away_p4) or (away_p4 and not home_p4):
        context['is_p4_vs_mid'] = True
    
    # Ranked vs unranked
    home_ranked = home.get('rank') is not None and home['rank'] > 0
    away_ranked = away.get('rank') is not None and away['rank'] > 0
    if (home_ranked and not away_ranked) or (away_ranked and not home_ranked):
        context['is_ranked_vs_unranked'] = True
    
    # Calculate variance boost
    # Non-conference P4 vs mid-major = higher blowout risk
    boost = 1.0
    if not context['is_conference']:
        boost += 0.1  # Non-conf games slightly more variable
    if context['is_p4_vs_mid']:
        boost += 0.2  # Talent gap → blowout risk
    if context['is_ranked_vs_unranked']:
        boost += 0.1
    
    context['variance_boost'] = boost
    return context


def get_model_projections(home_id: str, away_id: str) -> Dict[str, Dict[str, float]]:
    """Get run projections from poisson and advanced models only (no pitching)."""
    projections = {}
    
    # Advanced model
    try:
        from models.advanced_model import AdvancedModel
        model = AdvancedModel()
        pred = model.predict_game(home_id, away_id)
        if pred.get('projected_home_runs'):
            projections['advanced'] = {
                'home': pred['projected_home_runs'],
                'away': pred['projected_away_runs'],
                'total': pred['projected_total']
            }
    except Exception:
        pass
    
    # Poisson model
    try:
        import models.poisson_model as pm
        pred = pm.predict(home_id, away_id)
        if pred.get('expected_runs_a'):
            projections['poisson'] = {
                'home': pred['expected_runs_a'],
                'away': pred['expected_runs_b'],
                'total': pred['expected_total']
            }
    except Exception:
        pass
    
    return projections


def weighted_average(projections: Dict[str, Dict[str, float]],
                     weights: Dict[str, float]):
    """Calculate weighted average of run projections."""
    total_weight = 0
    home_sum = 0
    away_sum = 0
    
    for model, proj in projections.items():
        weight = weights.get(model, 0)
        if weight > 0:
            home_sum += proj['home'] * weight
            away_sum += proj['away'] * weight
            total_weight += weight
    
    if total_weight == 0:
        return 5.5, 5.5, 11.0
    
    home = home_sum / total_weight
    away = away_sum / total_weight
    return home, away, home + away


def _get_bias_correction() -> float:
    """Calculate bias correction from production predictions."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            SELECT AVG(projected_total - actual_total) as bias, COUNT(*) as n
            FROM totals_predictions
            WHERE model_name = 'runs_ensemble'
            AND actual_total IS NOT NULL
        ''')
        row = c.fetchone()
        conn.close()
        if row and row['n'] and row['n'] >= 50:
            return row['bias'] or 0.0
        return 0.0
    except Exception:
        return 0.0


def apply_over_confidence_gate(prediction: str, edge_pct: float,
                                confidence: float) -> tuple:
    """
    Gate OVER predictions: require >8% edge or downgrade.
    
    UNDER at 76% hit rate is solid. OVER at 44% needs higher bar.
    """
    if prediction == 'OVER' and edge_pct <= OVER_EDGE_THRESHOLD:
        # Downgrade to low confidence / no-play
        return prediction, max(confidence * 0.5, 0.1), True
    return prediction, confidence, False


def predict(home_team_id: str, away_team_id: str,
            total_line: Optional[float] = None,
            neutral_site: bool = False,
            game_id: str = None,
            weather_data: dict = None) -> Dict[str, Any]:
    """
    Generate V2 ensemble run predictions.
    
    Same interface as production runs_ensemble.predict().
    """
    from models.runs_ensemble import get_team_stats, get_league_average, poisson_over_under
    
    projections = get_model_projections(home_team_id, away_team_id)
    
    if not projections:
        home_stats = get_team_stats(home_team_id)
        away_stats = get_team_stats(away_team_id)
        league_avg = get_league_average() / 2
        home_runs = home_stats['avg_scored'] if home_stats else league_avg
        away_runs = away_stats['avg_scored'] if away_stats else league_avg
        projections = {'fallback': {'home': home_runs, 'away': away_runs, 'total': home_runs + away_runs}}
    
    home_runs, away_runs, total = weighted_average(projections, DEFAULT_RUN_WEIGHTS_V2)
    
    # Home field adjustment
    if not neutral_site:
        home_runs *= 1.04
        away_runs *= 0.96
        total = home_runs + away_runs
    
    # Bias correction
    bias = _get_bias_correction()
    if bias != 0.0:
        total -= bias
        if home_runs + away_runs > 0:
            ratio = home_runs / (home_runs + away_runs)
            home_runs = total * ratio
            away_runs = total * (1 - ratio)
    
    # Weather adjustment
    weather_adj = 0.0
    try:
        from models.weather_model import calculate_weather_adjustment, get_weather_for_game
        if weather_data is None and game_id:
            weather_data = get_weather_for_game(game_id)
        if weather_data:
            weather_adj, _ = calculate_weather_adjustment(weather_data, apply_adjustment=True)
            if weather_adj != 0.0:
                total += weather_adj
                if home_runs + away_runs > 0:
                    ratio = home_runs / (home_runs + away_runs)
                    home_runs = total * ratio
                    away_runs = total * (1 - ratio)
    except Exception:
        pass
    
    # Game context
    context = get_game_context(home_team_id, away_team_id, game_id)
    
    # Confidence metrics
    std_dev = 2.0
    if len(projections) >= 2:
        totals = [p['total'] for p in projections.values()]
        mean_t = sum(totals) / len(totals)
        variance = sum((t - mean_t) ** 2 for t in totals) / len(totals)
        std_dev = math.sqrt(variance)
    
    ci_lower = total - 1.645 * std_dev
    ci_upper = total + 1.645 * std_dev
    
    max_spread = max(p['total'] for p in projections.values()) - min(p['total'] for p in projections.values())
    agreement = max(0, 1 - max_spread / 10)
    
    result = {
        'home_team': home_team_id,
        'away_team': away_team_id,
        'projected_home_runs': round(home_runs, 1),
        'projected_away_runs': round(away_runs, 1),
        'projected_total': round(total, 1),
        'confidence_interval': {
            'lower': round(ci_lower, 1),
            'upper': round(ci_upper, 1),
            'level': '90%'
        },
        'std_dev': round(std_dev, 2),
        'model_agreement': round(agreement, 2),
        'weather_adjustment': round(weather_adj, 2),
        'bias_correction': round(bias, 2),
        'game_context': context,
        'model_breakdown': {
            name: {
                'home': round(p['home'], 1),
                'away': round(p['away'], 1),
                'total': round(p['total'], 1),
                'weight': DEFAULT_RUN_WEIGHTS_V2.get(name, 0)
            }
            for name, p in projections.items()
        },
        'model_version': 'v2_experimental',
    }
    
    # Over/under analysis
    if total_line:
        # Poisson O/U (same as production)
        poisson_ou = poisson_over_under(home_runs, away_runs, total_line)
        
        # NegBin O/U (new)
        negbin_ou = negbin_over_under(
            home_runs, away_runs, total_line,
            variance_boost=context['variance_boost']
        )
        
        # Blend: 50/50 Poisson and NegBin for O/U probabilities
        blended_over = (poisson_ou['over'] + negbin_ou['over']) / 2
        blended_under = (poisson_ou['under'] + negbin_ou['under']) / 2
        
        edge = round(abs(blended_over - 0.5) * 100, 1)
        lean = 'OVER' if blended_over > 0.5 else 'UNDER'
        
        # Confidence based on edge and model agreement
        confidence = min(edge / 100.0 * 2, 1.0) * agreement
        
        # Apply OVER confidence gate
        lean_gated, confidence_gated, was_gated = apply_over_confidence_gate(
            lean, edge, confidence
        )
        
        result['over_under'] = {
            'line': total_line,
            'over_prob': round(blended_over, 4),
            'under_prob': round(blended_under, 4),
            'push_prob': round((poisson_ou['push'] + negbin_ou['push']) / 2, 4),
            'edge': edge,
            'lean': lean_gated,
            'confidence': round(confidence_gated, 4),
            'over_gated': was_gated,
            'poisson_over': poisson_ou['over'],
            'poisson_under': poisson_ou['under'],
            'negbin_over': negbin_ou['over'],
            'negbin_under': negbin_ou['under'],
            'dispersion': negbin_ou['dispersion'],
        }
        result['diff_from_line'] = round(total - total_line, 1)
    
    return result
