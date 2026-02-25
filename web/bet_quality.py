"""Bet quality gates and scoring for the college baseball predictor.

Based on analysis of 36 settled bets showing -$742 P&L:
- Heavy favorites (<-200): 6W-2L, +$56, +7% ROI  ← ONLY profitable bucket
- Light favorites (-200 to -100): 5W-5L, -$175
- Underdogs (+ML): 2W-6L, -$325
- Higher model "margin" over breakeven = WORSE results (calibration broken)
- Model disagrees with Vegas by 30-40pts on some games = model errors, not value

Created 2026-02-24 to stop bleeding money on bad bet patterns.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'


def vegas_implied_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml is None:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def get_recent_team_losses(team_name, days=3):
    """Check if a team lost a tracked bet within the last N days."""
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        # Check both consensus and EV tables
        for table in ('tracked_confident_bets', 'tracked_bets'):
            row = conn.execute(
                f'SELECT won FROM {table} WHERE pick_team_name=? AND date>=? AND won=0 ORDER BY date DESC LIMIT 1',
                (team_name, cutoff)
            ).fetchone()
            if row:
                conn.close()
                return True
        conn.close()
    except Exception:
        pass
    return False


def get_meta_ensemble_prob(game_id, pick_team_id):
    """Get meta_ensemble probability for a team in a game, if available."""
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        row = conn.execute('''
            SELECT home_win_prob FROM model_predictions
            WHERE game_id=? AND model_name='meta_ensemble'
            ORDER BY created_at DESC LIMIT 1
        ''', (game_id,)).fetchone()
        conn.close()
        if row:
            # We need to know if pick is home or away to return correct prob
            # For now return the raw home_win_prob - caller must handle direction
            return row['home_win_prob']
    except Exception:
        pass
    return None


def passes_quality_gate(bet, category='consensus'):
    """Smart filter to avoid money-losing bet patterns.
    
    Args:
        bet: dict with keys like moneyline, avg_prob/model_prob, models_agree, etc.
        category: 'consensus', 'ev', or 'totals'
    
    Returns:
        (passes: bool, reason: str or None)
    """
    if category == 'totals':
        return _gate_totals(bet)

    ml = bet.get('moneyline') or bet.get('ml', 0)
    # Use meta_ensemble prob if available, fall back to avg_prob/model_prob
    model_prob = (bet.get('meta_prob')
                  or bet.get('avg_prob')
                  or bet.get('model_prob', 0.5))

    # RULE 1: Never bet underdogs — 25% win rate, -$325 historically
    if ml is not None and ml > 0:
        return False, 'underdog (2W-6L historically, -$325)'

    # RULE 2: Require minimum model probability
    if model_prob < 0.65:
        return False, f'model prob {model_prob:.0%} < 65% minimum'

    # RULE 3: Check Vegas disagreement — model errors, not value
    vi = vegas_implied_prob(ml)
    if vi is not None and abs(model_prob - vi) > 0.25:
        return False, f'model ({model_prob:.0%}) disagrees with Vegas ({vi:.0%}) by >{25}pp'

    # RULE 4: Require meaningful margin over breakeven
    if vi is not None and (model_prob - vi) < 0.05:
        return False, f'margin {(model_prob - vi)*100:.1f}pp < 5pp minimum'

    if category == 'consensus':
        # RULE 5: Require strong consensus (8+ of 12 models)
        models_agree = bet.get('models_agree', 0)
        if models_agree < 8:
            return False, f'only {models_agree} models agree (need 8+)'

    if category == 'ev':
        # RULE 6: Require 10%+ edge for EV bets
        edge = bet.get('edge', 0)
        if edge < 10.0:
            return False, f'edge {edge:.1f}% < 10% minimum'

    # RULE 7: Don't bet team that lost within 3 days
    team_name = bet.get('pick_team_name', '')
    if team_name and get_recent_team_losses(team_name, days=3):
        return False, f'{team_name} lost a tracked bet in last 3 days'

    return True, None


def _gate_totals(bet):
    """Quality gate for totals bets."""
    pick = bet.get('pick', '')
    total_diff = abs(bet.get('edge', 0) if 'edge' in bet else bet.get('total_diff', 0))

    # Require 2+ runs edge
    if total_diff < 2.0:
        return False, f'total diff {total_diff:.1f} < 2.0 runs minimum'

    # Check actual probability if available
    over_prob = bet.get('over_prob')
    under_prob = bet.get('under_prob')
    prob = over_prob if pick == 'OVER' else under_prob

    if prob is not None:
        if prob < 0.60:
            return False, f'total prob {prob:.0%} < 60% minimum'
        # OVER predictions are weaker — require higher threshold
        if pick == 'OVER' and prob < 0.65:
            return False, f'OVER prob {prob:.0%} < 65% (OVER predictions weaker)'

    return True, None


def bet_quality_score(bet, category='consensus'):
    """Score a bet for ranking (higher = better).
    
    Combines:
    - Model probability (40%)
    - Agreement with Vegas (30%) — closer = safer
    - Model agreement count (20%)
    - Meta-ensemble probability (10%)
    """
    ml = bet.get('moneyline') or bet.get('ml', 0)
    model_prob = bet.get('avg_prob') or bet.get('model_prob', 0.5)
    meta_prob = bet.get('meta_prob', model_prob)
    models_agree = bet.get('models_agree', 6)
    vi = vegas_implied_prob(ml) or 0.5

    score = (
        model_prob * 0.4
        + (1.0 - abs(model_prob - vi)) * 0.3
        + (models_agree / 12.0) * 0.2
        + (meta_prob or model_prob) * 0.1
    )
    return round(score, 4)


def has_vegas_disagreement(model_prob, ml, threshold=0.20):
    """Check if model disagrees with Vegas by more than threshold.
    
    Returns (disagrees: bool, model_prob, vegas_implied, diff).
    """
    vi = vegas_implied_prob(ml)
    if vi is None:
        return False, model_prob, None, 0
    diff = abs(model_prob - vi)
    return diff > threshold, model_prob, vi, diff


# Maximum bets per category per day
MAX_PER_TYPE = 4
