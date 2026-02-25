"""
Ensemble model for run total predictions (v2.1).

Combines Poisson and Advanced models with confidence-weighted averaging.
Uses both Poisson and Negative Binomial distributions for over/under probability calculations.
Includes game-context variance adjustment, OVER confidence gate,
day-of-week adjustment, and team scoring volatility.
"""

import math
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, Any, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.negbin_model import negbin_over_under, fit_dispersion_from_db

# Default weights for run projections — poisson + advanced only (v2)
DEFAULT_RUN_WEIGHTS = {
    'poisson': 0.60,
    'advanced': 0.40,
}

# Active weights (updated by auto-adjustment)
RUN_MODEL_WEIGHTS = DEFAULT_RUN_WEIGHTS.copy()

# Minimum games before adjusting weights
MIN_GAMES_FOR_ADJUSTMENT = 20
# How fast weights move toward target (0-1, higher = faster)
ADJUSTMENT_RATE = 0.3
# Minimum weight for any model
MIN_WEIGHT = 0.05

# OVER confidence gate threshold
OVER_EDGE_THRESHOLD = 8.0  # Require >8% edge for OVER predictions

# ── TASK A: Day-of-Week Adjustments ──────────────────────────────────
# Based on verified data: Sun=13.6, Mon=14.2, Tue=15.1, Wed=13.5,
# Thu=15.9, Fri=12.0, Sat=12.8, overall avg ~13.0
DOW_ADJUSTMENTS = {
    6: +0.6,   # Sunday
    0: +1.2,   # Monday
    1: +2.1,   # Tuesday
    2: +0.5,   # Wednesday
    3: +2.9,   # Thursday
    4: -1.0,   # Friday
    5: -0.2,   # Saturday
}
# Dampening: ramp from 0.5x at <100 games to 1.0x at 500+ games
DOW_DAMPENING_MIN_GAMES = 100
DOW_DAMPENING_FULL_GAMES = 500

# ── TASK C: League average volatility (MAD) baseline ─────────────────
LEAGUE_AVG_VOLATILITY = 3.0  # fallback MAD if can't compute

# P4 conferences
P4_CONFERENCES = {
    'SEC', 'ACC', 'Big 12', 'Big Ten', 'Pac-12',
    'Southeastern', 'Atlantic Coast', 'Big XII', 'Big East',
}


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
    
    return row['avg_total'] if row and row['avg_total'] else 11.0


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
    Returns context dict with is_conference, is_p4_vs_mid, rank info, variance_boost.
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
    
    # Check if conference game
    if game_id:
        try:
            conn = get_connection()
            c = conn.cursor()
            c.execute('SELECT is_conference_game FROM games WHERE id = ?', (game_id,))
            row = c.fetchone()
            conn.close()
            if row:
                context['is_conference'] = bool(row['is_conference_game'])
        except Exception:
            pass
    
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
    boost = 1.0
    if not context['is_conference']:
        boost += 0.1
    if context['is_p4_vs_mid']:
        boost += 0.2
    if context['is_ranked_vs_unranked']:
        boost += 0.1
    
    context['variance_boost'] = boost
    return context


def apply_over_confidence_gate(prediction: str, edge_pct: float,
                                confidence: float) -> tuple:
    """
    Gate OVER predictions: require >8% edge or downgrade.
    UNDER at 76% hit rate is solid. OVER at 44% needs higher bar.
    """
    if prediction == 'OVER' and edge_pct <= OVER_EDGE_THRESHOLD:
        return prediction, max(confidence * 0.5, 0.1), True
    return prediction, confidence, False


def _update_weights_from_accuracy():
    """Auto-adjust run model weights based on O/U prediction accuracy.
    
    Reads totals_predictions table for per-component accuracy (runs_poisson, 
    runs_advanced). Uses recency-weighted accuracy^2 to shift weights.
    """
    global RUN_MODEL_WEIGHTS
    
    try:
        conn = get_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT model_name, was_correct,
                   ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY predicted_at DESC) as recency_rank
            FROM totals_predictions
            WHERE was_correct IS NOT NULL
            AND model_name IN ('runs_poisson', 'runs_advanced')
        ''')
        
        model_scores = {}
        model_counts = {}
        for row in c.fetchall():
            name = row['model_name'].replace('runs_', '')
            if name not in model_scores:
                model_scores[name] = {'weighted_correct': 0.0, 'weighted_total': 0.0}
                model_counts[name] = 0
            
            rank = row['recency_rank']
            weight = 0.977 ** (rank - 1)
            model_scores[name]['weighted_correct'] += row['was_correct'] * weight
            model_scores[name]['weighted_total'] += weight
            model_counts[name] += 1
        
        conn.close()
        
        total_evaluated = sum(model_counts.values())
        if total_evaluated < MIN_GAMES_FOR_ADJUSTMENT:
            return
        
        accuracy = {}
        for name in DEFAULT_RUN_WEIGHTS:
            if name in model_scores and model_scores[name]['weighted_total'] > 0:
                accuracy[name] = model_scores[name]['weighted_correct'] / model_scores[name]['weighted_total']
            else:
                accuracy[name] = 0.5
        
        # O/U accuracy-based weights
        scores = {n: max(acc, 0.3) ** 2 for n, acc in accuracy.items()}
        total_score = sum(scores.values())
        
        if total_score > 0:
            ou_weights = {n: s / total_score for n, s in scores.items()}
            
            # ── TASK D: MAE-based weights ────────────────────────────
            mae_weights = None
            try:
                conn2 = get_connection()
                c2 = conn2.cursor()
                c2.execute('''
                    SELECT model_name, AVG(ABS(projected_total - actual_total)) as mae
                    FROM totals_predictions
                    WHERE actual_total IS NOT NULL
                    AND model_name IN ('runs_poisson', 'runs_advanced')
                    GROUP BY model_name
                ''')
                mae_rows = {r['model_name'].replace('runs_', ''): r['mae'] for r in c2.fetchall()}
                conn2.close()
                
                if len(mae_rows) >= 2 and all(v > 0 for v in mae_rows.values()):
                    inv_mae = {n: 1.0 / v for n, v in mae_rows.items()}
                    inv_total = sum(inv_mae.values())
                    mae_weights = {n: v / inv_total for n, v in inv_mae.items()}
            except Exception:
                pass
            
            # Blend O/U and MAE weights (50/50 if both available)
            if mae_weights:
                target_weights = {}
                for name in DEFAULT_RUN_WEIGHTS:
                    ou_w = ou_weights.get(name, MIN_WEIGHT)
                    mae_w = mae_weights.get(name, MIN_WEIGHT)
                    target_weights[name] = 0.5 * ou_w + 0.5 * mae_w
                print(f"[runs_ensemble] Weight update: O/U={ou_weights} MAE={mae_weights} → blended={target_weights}")
            else:
                target_weights = ou_weights
            
            for name in DEFAULT_RUN_WEIGHTS:
                current = RUN_MODEL_WEIGHTS.get(name, DEFAULT_RUN_WEIGHTS[name])
                target = target_weights.get(name, MIN_WEIGHT)
                new_weight = current + (target - current) * ADJUSTMENT_RATE
                RUN_MODEL_WEIGHTS[name] = max(new_weight, MIN_WEIGHT)
            
            total = sum(RUN_MODEL_WEIGHTS.values())
            RUN_MODEL_WEIGHTS = {n: w / total for n, w in RUN_MODEL_WEIGHTS.items()}
    
    except Exception:
        pass


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
    return _MODEL_CACHE.get(name)

def get_model_projections(home_id: str, away_id: str) -> Dict[str, Dict[str, float]]:
    """Get run projections from poisson and advanced models (no pitching)."""
    projections = {}
    
    # Advanced model
    try:
        model = _get_cached_model('advanced')
        if model:
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
        return 5.5, 5.5, 11.0
    
    home = home_sum / total_weight
    away = away_sum / total_weight
    return home, away, home + away

def calculate_std_dev(projections: Dict[str, Dict[str, float]], 
                      mean_total: float) -> float:
    """Calculate standard deviation of model projections for confidence."""
    if len(projections) < 2:
        return 2.0
    
    totals = [p['total'] for p in projections.values()]
    variance = sum((t - mean_total) ** 2 for t in totals) / len(totals)
    return math.sqrt(variance)

def poisson_over_under(home_runs: float, away_runs: float, 
                       total_line: float) -> Dict[str, float]:
    """Calculate over/under probabilities using Poisson distribution."""
    over_prob = 0
    under_prob = 0
    push_prob = 0
    
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

def _get_dow_dampening() -> float:
    """Get dampening factor for DOW adjustment based on season game count."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as n FROM games WHERE status = 'final'")
        n = c.fetchone()['n'] or 0
        conn.close()
        if n <= DOW_DAMPENING_MIN_GAMES:
            return 0.5
        if n >= DOW_DAMPENING_FULL_GAMES:
            return 1.0
        return 0.5 + 0.5 * (n - DOW_DAMPENING_MIN_GAMES) / (DOW_DAMPENING_FULL_GAMES - DOW_DAMPENING_MIN_GAMES)
    except Exception:
        return 0.5


def get_team_scoring_variance(team_id: str) -> Dict[str, Any]:
    """
    Get team scoring volatility (mean absolute deviation) from last 10 games.
    Returns {'avg_scored': float, 'variance': float, 'games': int}
    """
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            SELECT CASE WHEN home_team_id = ? THEN home_score ELSE away_score END as runs
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?) AND status = 'final'
            ORDER BY date DESC LIMIT 10
        ''', (team_id, team_id, team_id))
        rows = c.fetchall()
        conn.close()

        if len(rows) < 3:
            return {'avg_scored': 6.0, 'variance': LEAGUE_AVG_VOLATILITY, 'games': len(rows)}

        scores = [r['runs'] for r in rows]
        avg = sum(scores) / len(scores)
        mad = sum(abs(s - avg) for s in scores) / len(scores)
        return {'avg_scored': avg, 'variance': mad, 'games': len(scores)}
    except Exception:
        return {'avg_scored': 6.0, 'variance': LEAGUE_AVG_VOLATILITY, 'games': 0}


def _get_bias_correction() -> float:
    """Calculate bias correction from recent evaluated predictions."""
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


def predict(home_team_id: str, away_team_id: str, 
            total_line: Optional[float] = None,
            neutral_site: bool = False,
            game_id: str = None,
            weather_data: dict = None) -> Dict[str, Any]:
    """
    Generate ensemble run predictions (v2).
    
    Returns projected runs for each team, total, over/under probs,
    confidence interval, and individual model breakdown.
    """
    # Auto-adjust weights based on tracked accuracy
    _update_weights_from_accuracy()
    
    # Get projections from all models
    projections = get_model_projections(home_team_id, away_team_id)
    
    if not projections:
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
        home_runs *= 1.04
        away_runs *= 0.96
        total = home_runs + away_runs
    
    # Apply bias correction
    bias = _get_bias_correction()
    if bias != 0.0:
        total -= bias
        if home_runs + away_runs > 0:
            ratio = home_runs / (home_runs + away_runs)
            home_runs = total * ratio
            away_runs = total * (1 - ratio)
    
    # Apply weather adjustment
    weather_adj = 0.0
    weather_components = {}
    try:
        from models.weather_model import calculate_weather_adjustment, get_weather_for_game
        if weather_data is None and game_id:
            weather_data = get_weather_for_game(game_id)
        if weather_data:
            weather_adj, weather_components = calculate_weather_adjustment(
                weather_data, apply_adjustment=True
            )
            if weather_adj != 0.0:
                total += weather_adj
                if home_runs + away_runs > 0:
                    ratio = home_runs / (home_runs + away_runs)
                    home_runs = total * ratio
                    away_runs = total * (1 - ratio)
    except Exception:
        pass
    
    # ── TASK A: Day-of-Week Adjustment ──────────────────────────────
    dow_adj = 0.0
    try:
        game_date = None
        if game_id and len(game_id) >= 10:
            game_date = datetime.strptime(game_id[:10], '%Y-%m-%d')
        if game_date:
            raw_adj = DOW_ADJUSTMENTS.get(game_date.weekday(), 0.0)
            dampening = _get_dow_dampening()
            dow_adj = raw_adj * dampening
            if dow_adj != 0.0:
                total += dow_adj
                if home_runs + away_runs > 0:
                    ratio = home_runs / (home_runs + away_runs)
                    home_runs = total * ratio
                    away_runs = total * (1 - ratio)
    except Exception:
        pass

    # ── TASK C: Team Scoring Volatility ──────────────────────────────
    volatility_adj = 0.0
    combined_volatility = 0.0
    try:
        home_var_data = get_team_scoring_variance(home_team_id)
        away_var_data = get_team_scoring_variance(away_team_id)
        combined_volatility = (home_var_data['variance'] + away_var_data['variance']) / 2

        if combined_volatility > LEAGUE_AVG_VOLATILITY * 1.5:
            volatility_adj = min(0.5 * (combined_volatility / LEAGUE_AVG_VOLATILITY - 1.0), 2.0)
            total += volatility_adj
            if home_runs + away_runs > 0:
                ratio = home_runs / (home_runs + away_runs)
                home_runs = total * ratio
                away_runs = total * (1 - ratio)
    except Exception:
        pass

    # Game context for variance adjustment
    context = get_game_context(home_team_id, away_team_id, game_id)
    
    # Calculate confidence (std dev of projections)
    std_dev = calculate_std_dev(projections, total)
    
    # 90% confidence interval (widen for high volatility matchups)
    ci_multiplier = 1.3 if combined_volatility > LEAGUE_AVG_VOLATILITY * 1.5 else 1.0
    ci_lower = total - 1.645 * std_dev * ci_multiplier
    ci_upper = total + 1.645 * std_dev * ci_multiplier
    
    # Model agreement score
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
                'weight': RUN_MODEL_WEIGHTS.get(name, 0)
            }
            for name, p in projections.items()
        },
        'dow_adjustment': round(dow_adj, 2),
        'volatility_adjustment': round(volatility_adj, 2),
        'combined_volatility': round(combined_volatility, 2),
        'model_version': 'v2.1',
    }
    
    # Over/under analysis
    if total_line:
        # Poisson O/U
        poisson_ou = poisson_over_under(home_runs, away_runs, total_line)
        
        # NegBin O/U
        nb_ou = negbin_over_under(
            home_runs, away_runs, total_line,
            variance_boost=context['variance_boost']
        )
        
        # Blend: 50/50 Poisson and NegBin
        blended_over = (poisson_ou['over'] + nb_ou['over']) / 2
        blended_under = (poisson_ou['under'] + nb_ou['under']) / 2
        
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
            'push_prob': round((poisson_ou['push'] + nb_ou['push']) / 2, 4),
            'edge': edge,
            'lean': lean_gated,
            'confidence': round(confidence_gated, 4),
            'over_gated': was_gated,
            'poisson_over': poisson_ou['over'],
            'poisson_under': poisson_ou['under'],
            'negbin_over': nb_ou['over'],
            'negbin_under': nb_ou['under'],
            'dispersion': nb_ou['dispersion'],
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
        sys.exit(1)
    
    home = sys.argv[1]
    away = sys.argv[2]
    line = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = predict(home, away, total_line=line)
    
    print(f"\n🏟️  {away} @ {home} - Run Projection Ensemble v2")
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
        if ou.get('over_gated'):
            print(f"   ⚠️  OVER confidence gated (edge < {OVER_EDGE_THRESHOLD}%)")
        print(f"   Poisson: O {ou['poisson_over']:.1%} / U {ou['poisson_under']:.1%}")
        print(f"   NegBin:  O {ou['negbin_over']:.1%} / U {ou['negbin_under']:.1%}")
    
    print("\n📊 Common Lines:")
    for line, probs in result['line_analysis'].items():
        print(f"   {line}: Over {probs['over']:.0f}% / Under {probs['under']:.0f}%")
