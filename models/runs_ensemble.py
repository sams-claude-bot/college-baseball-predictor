"""
Ensemble model for run total predictions.

Combines projections from multiple models with confidence-weighted averaging.
Uses Poisson distribution for over/under probability calculations.
"""

import sqlite3
import math
from scipy import stats
from typing import Dict, Any, Optional, Tuple

DB_PATH = '/home/sam/college-baseball-predictor/data/baseball.db'

# Model weights for run projections (tuned for accuracy)
RUN_MODEL_WEIGHTS = {
    'poisson': 0.30,      # Best for run distributions
    'advanced': 0.25,     # Good overall, opponent-adjusted
    'elo': 0.20,          # Solid baseline
    'pythagorean': 0.15,  # Classic, uses historical RS/RA
    'pitching': 0.10,     # Tends to inflate, down-weighted
}

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_team_stats(team_id: str) -> Optional[Dict]:
    """Get team's run scoring and allowing stats."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            t.name,
            COUNT(CASE WHEN g.status = 'final' THEN 1 END) as games,
            SUM(CASE 
                WHEN g.home_team_id = t.id AND g.status = 'final' THEN g.home_score
                WHEN g.away_team_id = t.id AND g.status = 'final' THEN g.away_score
            END) as runs_scored,
            SUM(CASE 
                WHEN g.home_team_id = t.id AND g.status = 'final' THEN g.away_score
                WHEN g.away_team_id = t.id AND g.status = 'final' THEN g.home_score
            END) as runs_allowed
        FROM teams t
        LEFT JOIN games g ON t.id = g.home_team_id OR t.id = g.away_team_id
        WHERE t.id = ?
        GROUP BY t.id
    ''', (team_id,))
    
    row = c.fetchone()
    conn.close()
    
    if not row or not row['games'] or row['games'] == 0:
        return None
    
    return {
        'name': row['name'],
        'games': row['games'],
        'runs_scored': row['runs_scored'] or 0,
        'runs_allowed': row['runs_allowed'] or 0,
        'avg_scored': (row['runs_scored'] or 0) / row['games'],
        'avg_allowed': (row['runs_allowed'] or 0) / row['games']
    }

def get_league_average() -> float:
    """Get average runs per game across all teams."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT AVG(home_score + away_score) as avg_total
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL
    ''')
    
    row = c.fetchone()
    conn.close()
    
    return row['avg_total'] if row and row['avg_total'] else 11.0  # Default to ~11 total

# Cache model instances at module level for performance
_MODEL_CACHE = {}

def _get_cached_model(name: str):
    """Get or create a cached model instance."""
    if name not in _MODEL_CACHE:
        if name == 'advanced':
            from models.advanced_model import AdvancedModel
            _MODEL_CACHE[name] = AdvancedModel()
        elif name == 'pythagorean':
            from models.pythagorean_model import PythagoreanModel
            _MODEL_CACHE[name] = PythagoreanModel()
        elif name == 'elo':
            from models.elo_model import EloModel
            _MODEL_CACHE[name] = EloModel()
        elif name == 'pitching':
            from models.pitching_model import PitchingModel
            _MODEL_CACHE[name] = PitchingModel()
    return _MODEL_CACHE.get(name)

def get_model_projections(home_id: str, away_id: str) -> Dict[str, Dict[str, float]]:
    """Get run projections from each individual model."""
    projections = {}
    
    # Use cached models
    for name in ['advanced', 'pythagorean', 'elo', 'pitching']:
        try:
            model = _get_cached_model(name)
            if model:
                pred = model.predict_game(home_id, away_id)
                if pred.get('projected_home_runs'):
                    projections[name] = {
                        'home': pred['projected_home_runs'],
                        'away': pred['projected_away_runs'],
                        'total': pred['projected_total']
                    }
        except Exception:
            pass
    
    # Poisson model (module-level function)
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
                     weights: Dict[str, float]) -> Tuple[float, float, float]:
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
        return 5.5, 5.5, 11.0  # Fallback
    
    home = home_sum / total_weight
    away = away_sum / total_weight
    return home, away, home + away

def calculate_std_dev(projections: Dict[str, Dict[str, float]], 
                      mean_total: float) -> float:
    """Calculate standard deviation of model projections for confidence."""
    if len(projections) < 2:
        return 2.0  # Default uncertainty
    
    totals = [p['total'] for p in projections.values()]
    variance = sum((t - mean_total) ** 2 for t in totals) / len(totals)
    return math.sqrt(variance)

def poisson_over_under(home_runs: float, away_runs: float, 
                       total_line: float) -> Dict[str, float]:
    """Calculate over/under probabilities using Poisson distribution."""
    # Simulate distribution
    over_prob = 0
    under_prob = 0
    push_prob = 0
    
    # Calculate probabilities for each score combination
    for h in range(25):
        for a in range(25):
            h_prob = stats.poisson.pmf(h, home_runs)
            a_prob = stats.poisson.pmf(a, away_runs)
            combined = h_prob * a_prob
            
            total = h + a
            if total > total_line:
                over_prob += combined
            elif total < total_line:
                under_prob += combined
            else:
                push_prob += combined
    
    return {
        'over': round(over_prob, 4),
        'under': round(under_prob, 4),
        'push': round(push_prob, 4)
    }

def predict(home_team_id: str, away_team_id: str, 
            total_line: Optional[float] = None,
            neutral_site: bool = False) -> Dict[str, Any]:
    """
    Generate ensemble run predictions.
    
    Returns projected runs for each team, total, over/under probs,
    confidence interval, and individual model breakdown.
    """
    # Get projections from all models
    projections = get_model_projections(home_team_id, away_team_id)
    
    if not projections:
        # Fallback to simple average
        home_stats = get_team_stats(home_team_id)
        away_stats = get_team_stats(away_team_id)
        league_avg = get_league_average() / 2
        
        home_runs = home_stats['avg_scored'] if home_stats else league_avg
        away_runs = away_stats['avg_scored'] if away_stats else league_avg
        
        projections = {'fallback': {'home': home_runs, 'away': away_runs, 'total': home_runs + away_runs}}
    
    # Calculate weighted ensemble
    home_runs, away_runs, total = weighted_average(projections, RUN_MODEL_WEIGHTS)
    
    # Apply home field adjustment if not neutral
    if not neutral_site:
        home_runs *= 1.02  # Slight home boost
        away_runs *= 0.98
        total = home_runs + away_runs
    
    # Calculate confidence (std dev of projections)
    std_dev = calculate_std_dev(projections, total)
    
    # 90% confidence interval
    ci_lower = total - 1.645 * std_dev
    ci_upper = total + 1.645 * std_dev
    
    # Model agreement score (1 = perfect agreement, 0 = wide disagreement)
    max_spread = max(p['total'] for p in projections.values()) - min(p['total'] for p in projections.values())
    agreement = max(0, 1 - max_spread / 10)  # Normalize to 0-1
    
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
        'model_breakdown': {
            name: {
                'home': round(p['home'], 1),
                'away': round(p['away'], 1),
                'total': round(p['total'], 1),
                'weight': RUN_MODEL_WEIGHTS.get(name, 0)
            }
            for name, p in projections.items()
        }
    }
    
    # Over/under analysis
    if total_line:
        ou = poisson_over_under(home_runs, away_runs, total_line)
        result['over_under'] = {
            'line': total_line,
            'over_prob': ou['over'],
            'under_prob': ou['under'],
            'push_prob': ou['push'],
            'edge': round(abs(ou['over'] - 0.5) * 100, 1),
            'lean': 'OVER' if ou['over'] > 0.5 else 'UNDER'
        }
        result['diff_from_line'] = round(total - total_line, 1)
    
    # Common lines analysis
    common_lines = [8.5, 9.5, 10.5, 11.5, 12.5]
    result['line_analysis'] = {}
    for line in common_lines:
        ou = poisson_over_under(home_runs, away_runs, line)
        result['line_analysis'][line] = {
            'over': round(ou['over'] * 100, 1),
            'under': round(ou['under'] * 100, 1)
        }
    
    return result

def get_weights() -> Dict[str, float]:
    """Return current model weights for runs ensemble."""
    return RUN_MODEL_WEIGHTS.copy()

def set_weights(new_weights: Dict[str, float]):
    """Update model weights (for tuning)."""
    global RUN_MODEL_WEIGHTS
    RUN_MODEL_WEIGHTS.update(new_weights)

if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python runs_ensemble.py <home_team> <away_team> [total_line]")
        print("\nExample: python runs_ensemble.py mississippi-state hofstra 10.5")
        sys.exit(1)
    
    home = sys.argv[1]
    away = sys.argv[2]
    line = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = predict(home, away, total_line=line)
    
    print(f"\n🏟️  {away} @ {home} - Run Projection Ensemble")
    print("=" * 55)
    print(f"\n📊 Projected Score: {result['away_team']} {result['projected_away_runs']} - {result['projected_home_runs']} {result['home_team']}")
    print(f"📈 Projected Total: {result['projected_total']} runs")
    print(f"📉 90% CI: [{result['confidence_interval']['lower']} - {result['confidence_interval']['upper']}]")
    print(f"🤝 Model Agreement: {result['model_agreement']:.0%}")
    
    print("\n📋 Model Breakdown:")
    for model, data in result['model_breakdown'].items():
        print(f"   {model:12} → {data['total']:5.1f} total ({data['weight']:.0%} weight)")
    
    if 'over_under' in result:
        ou = result['over_under']
        print(f"\n🎯 vs Line {ou['line']}:")
        print(f"   {ou['lean']}: {ou['over_prob' if ou['lean'] == 'OVER' else 'under_prob']:.1%} ({ou['edge']:.1f}% edge)")
        print(f"   Projection vs Line: {result['diff_from_line']:+.1f}")
    
    print("\n📊 Common Lines:")
    for line, probs in result['line_analysis'].items():
        print(f"   {line}: Over {probs['over']:.0f}% / Under {probs['under']:.0f}%")
