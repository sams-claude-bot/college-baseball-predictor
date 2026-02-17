#!/usr/bin/env python3
"""
Improved Bet Selection Logic (v2)

Key changes from v1:
1. Higher edge thresholds (8% ML, 3 runs totals)
2. Require model consensus (7+/10 models agree) for ML bets
3. DISABLE spreads entirely (model not calibrated - 0/5 historical)
4. Kelly criterion for bet sizing
5. Avoid heavy favorites (ML worse than -200)
6. Track rejection reasons for analysis

Usage:
    python3 scripts/bet_selection_v2.py analyze   # Show what would be bet
    python3 scripts/bet_selection_v2.py record    # Record to DB
"""

import sys
import json
import sqlite3
import requests
from datetime import datetime
from typing import Optional

DB_PATH = 'data/baseball.db'
API_URL = 'http://localhost:5000/api/best-bets'

# ============ THRESHOLDS ============
# These are intentionally conservative after 0/5 spreads, 1/3 ML
ML_EDGE_THRESHOLD = 8.0       # Was 5%, raised to 8%
ML_EDGE_UNDERDOG = 15.0       # Higher threshold for underdogs (+100 or more)
ML_CONSENSUS_MIN = 7          # Require 7/10 models to agree
ML_MAX_FAVORITE = -200        # Don't bet heavy favorites for EV bets
ML_MAX_FAVORITE_CONSENSUS = -300  # Allow heavier favorites when consensus is high
ML_MIN_UNDERDOG = 250         # Don't bet extreme underdogs (unlikely to hit)

# Model probability sanity checks
ML_MAX_MODEL_PROB = 0.88      # Cap model probability (avoid overconfidence)
ML_MIN_MODEL_PROB = 0.55      # Don't bet near coin-flips

TOTALS_EDGE_THRESHOLD = 3.0   # Runs diff (was 15%, now 3 runs)
TOTALS_MIN_CONFIDENCE = 0.6   # Model confidence in the pick

SPREADS_ENABLED = False       # DISABLED - 0/5 record, model not calibrated

MAX_BETS_PER_DAY = 3          # Reduced from 6 to be more selective
BASE_BET = 100                # Base unit

# ============ KELLY CRITERION ============
def kelly_fraction(win_prob: float, odds: int, fraction: float = 0.25) -> float:
    """
    Calculate Kelly bet size as fraction of bankroll.
    Uses fractional Kelly (default 25%) to reduce variance.
    
    Returns multiplier for base bet (0 to ~2x).
    """
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))
    
    # Kelly formula: (bp - q) / b
    # where b = decimal odds - 1, p = win prob, q = 1 - p
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b if b > 0 else 0
    
    # Apply fractional Kelly and cap
    kelly_adj = kelly * fraction
    return max(0, min(2.0, kelly_adj))  # Cap at 2x base bet


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return abs(odds) / (abs(odds) + 100)


def analyze_games(date_str: Optional[str] = None) -> dict:
    """
    Analyze all games for a date and return bet recommendations.
    Returns dict with 'bets' and 'rejections' for transparency.
    """
    try:
        url = API_URL
        if date_str:
            url += f'?date={date_str}'
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ Failed to fetch from API: {e}")
        return {'bets': [], 'rejections': [], 'error': str(e)}
    
    date = data['date']
    results = {
        'date': date,
        'bets': [],
        'rejections': [],
        'spreads_disabled': True,
    }
    
    # ===== CONFIDENT BETS (Model Consensus) - HIGHEST PRIORITY =====
    # These are games where 7+ models agree - much more reliable
    confident_lookup = {g['game_id']: g for g in data.get('confident_bets', [])}
    
    for game in data.get('confident_bets', []):
        ml = game.get('moneyline')
        if not ml:
            continue
            
        models_agree = game['models_agree']
        avg_prob = game['avg_prob']
        
        rejection_reasons = []
        
        # Consensus filter
        if models_agree < ML_CONSENSUS_MIN:
            rejection_reasons.append(f"only {models_agree}/10 models agree (need {ML_CONSENSUS_MIN}+)")
        
        # Heavy favorite filter - more relaxed for high consensus
        max_fav = ML_MAX_FAVORITE_CONSENSUS if models_agree >= 9 else ML_MAX_FAVORITE
        if ml < max_fav:
            rejection_reasons.append(f"ML {ml} too heavy (max {max_fav} for {models_agree}/10 consensus)")
        
        # Probability sanity
        if avg_prob > ML_MAX_MODEL_PROB:
            rejection_reasons.append(f"avg prob {avg_prob*100:.0f}% overconfident")
        
        if rejection_reasons:
            results['rejections'].append({
                'type': 'CONSENSUS',
                'game': f"{game['pick_team_name']} vs {game['opponent_name']}",
                'models_agree': models_agree,
                'ml': ml,
                'reasons': rejection_reasons
            })
        else:
            # Consensus bets get kelly sizing too
            kelly_mult = kelly_fraction(avg_prob, ml)
            bet_size = round(BASE_BET * max(1.0, kelly_mult), 0)
            
            results['bets'].append({
                'type': 'CONSENSUS',
                'game_id': game['game_id'],
                'date': game['date'],
                'pick_team_id': game['pick_team_id'],
                'pick_team_name': game['pick_team_name'],
                'opponent_name': game['opponent_name'],
                'is_home': game['is_home'],
                'moneyline': ml,
                'model_prob': avg_prob,
                'models_agree': models_agree,
                'edge': (avg_prob - american_to_prob(ml)) * 100,  # Calculate edge
                'kelly_mult': kelly_mult,
                'bet_amount': bet_size,
            })
    
    # ===== EV MONEYLINE ANALYSIS (secondary to consensus) =====
    for game in data.get('moneylines', []):
        # Skip if already added as confident bet
        if game['game_id'] in [b['game_id'] for b in results['bets'] if b['type'] == 'CONSENSUS']:
            continue
        ml = game['moneyline']
        edge = game['edge']
        model_prob = game['model_prob']
        is_underdog = ml > 0
        
        rejection_reasons = []
        
        # Different edge threshold for underdogs (markets are usually right)
        edge_threshold = ML_EDGE_UNDERDOG if is_underdog else ML_EDGE_THRESHOLD
        if edge < edge_threshold:
            rejection_reasons.append(f"edge {edge:.1f}% < {edge_threshold}% {'underdog ' if is_underdog else ''}threshold")
        
        # Model probability sanity checks
        if model_prob > ML_MAX_MODEL_PROB:
            rejection_reasons.append(f"model {model_prob*100:.0f}% overconfident (max {ML_MAX_MODEL_PROB*100:.0f}%)")
        if model_prob < ML_MIN_MODEL_PROB:
            rejection_reasons.append(f"model {model_prob*100:.0f}% too low (min {ML_MIN_MODEL_PROB*100:.0f}%)")
        
        # Heavy favorite filter
        if ml < ML_MAX_FAVORITE:
            rejection_reasons.append(f"ML {ml} too heavy (max {ML_MAX_FAVORITE})")
        
        # Extreme underdog filter  
        if ml > ML_MIN_UNDERDOG:
            rejection_reasons.append(f"ML +{ml} too long (max +{ML_MIN_UNDERDOG})")
        
        if rejection_reasons:
            results['rejections'].append({
                'type': 'ML',
                'game': f"{game['pick_team_name']} vs {game['opponent_name']}",
                'edge': edge,
                'ml': ml,
                'reasons': rejection_reasons
            })
        else:
            # Calculate Kelly-adjusted bet size
            implied_prob = american_to_prob(ml)
            model_prob = game['model_prob']
            kelly_mult = kelly_fraction(model_prob, ml)
            bet_size = round(BASE_BET * max(1.0, kelly_mult), 0)
            
            results['bets'].append({
                'type': 'ML',
                'game_id': game['game_id'],
                'date': game['date'],
                'pick_team_id': game['pick_team_id'],
                'pick_team_name': game['pick_team_name'],
                'opponent_name': game['opponent_name'],
                'is_home': game['is_home'],
                'moneyline': ml,
                'model_prob': model_prob,
                'dk_implied': game['dk_implied'],
                'edge': edge,
                'kelly_mult': kelly_mult,
                'bet_amount': bet_size,
            })
    
    # ===== TOTALS ANALYSIS =====
    for game in data.get('totals', []):
        edge = game['edge']  # In runs
        
        rejection_reasons = []
        
        if edge < TOTALS_EDGE_THRESHOLD:
            rejection_reasons.append(f"edge {edge:.1f} runs < {TOTALS_EDGE_THRESHOLD} threshold")
        
        if rejection_reasons:
            results['rejections'].append({
                'type': 'TOTAL',
                'game': f"{game['pick']} {game['line']}",
                'edge': edge,
                'reasons': rejection_reasons
            })
        else:
            results['bets'].append({
                'type': 'TOTAL',
                'game_id': game['game_id'],
                'date': game['date'],
                'pick': game['pick'],
                'line': game['line'],
                'odds': game['odds'],
                'model_projection': game['model_projection'],
                'edge': edge,
                'bet_amount': BASE_BET,
            })
    
    # ===== SPREADS - DISABLED =====
    if SPREADS_ENABLED:
        # Future: re-enable when model is calibrated
        pass
    else:
        for game in data.get('spreads', []):
            results['rejections'].append({
                'type': 'SPREAD',
                'game': f"{game['pick']} {game['line']:+.1f}",
                'edge': game['edge'],
                'reasons': ['SPREADS DISABLED - model not calibrated (0/5 historical)']
            })
    
    # Sort bets: CONSENSUS first (most reliable), then by edge descending
    def bet_priority(b):
        # Consensus bets get priority (sorted by model count, then edge)
        if b['type'] == 'CONSENSUS':
            return (0, -b.get('models_agree', 0), -b.get('edge', 0))
        # Totals second (our only profitable category historically)
        elif b['type'] == 'TOTAL':
            return (1, 0, -b.get('edge', 0))
        # EV ML bets last (underperforming)
        else:
            return (2, 0, -b.get('edge', 0))
    
    results['bets'].sort(key=bet_priority)
    if len(results['bets']) > MAX_BETS_PER_DAY:
        overflow = results['bets'][MAX_BETS_PER_DAY:]
        results['bets'] = results['bets'][:MAX_BETS_PER_DAY]
        for bet in overflow:
            results['rejections'].append({
                'type': bet['type'],
                'game': bet.get('pick_team_name') or bet.get('pick'),
                'edge': bet['edge'],
                'reasons': [f'exceeded {MAX_BETS_PER_DAY} bets/day limit']
            })
    
    return results


def print_analysis(results: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"BET SELECTION ANALYSIS - {results['date']}")
    print(f"{'='*60}")
    
    print(f"\n🎯 RECOMMENDED BETS ({len(results['bets'])})")
    print("-" * 40)
    
    if not results['bets']:
        print("  No bets meet the criteria today")
    else:
        for bet in results['bets']:
            if bet['type'] == 'CONSENSUS':
                sign = '+' if bet['moneyline'] > 0 else ''
                print(f"  🎯 CONSENSUS: {bet['pick_team_name']} ({sign}{bet['moneyline']})")
                print(f"       Models: {bet['models_agree']}/10 agree | Avg: {bet['model_prob']*100:.0f}%")
                print(f"       Edge: {bet['edge']:.1f}% | Bet: ${bet['bet_amount']:.0f}")
            elif bet['type'] == 'ML':
                sign = '+' if bet['moneyline'] > 0 else ''
                print(f"  💰 ML: {bet['pick_team_name']} ({sign}{bet['moneyline']})")
                print(f"       Edge: {bet['edge']:.1f}% | Model: {bet['model_prob']*100:.0f}%")
                print(f"       Bet: ${bet['bet_amount']:.0f} (Kelly: {bet['kelly_mult']:.2f}x)")
            elif bet['type'] == 'TOTAL':
                print(f"  📊 TOTAL: {bet['pick']} {bet['line']} ({bet['odds']:+d})")
                print(f"       Edge: {bet['edge']:.1f} runs | Proj: {bet['model_projection']:.1f}")
                print(f"       Bet: ${bet['bet_amount']:.0f}")
            print()
    
    print(f"\n❌ REJECTIONS ({len(results['rejections'])})")
    print("-" * 40)
    
    # Group by reason
    by_reason = {}
    for rej in results['rejections']:
        for reason in rej['reasons']:
            key = reason.split(' ')[0]  # First word as category
            by_reason.setdefault(key, []).append(rej)
    
    for reason_cat, rejs in by_reason.items():
        print(f"  {reason_cat}: {len(rejs)} games")
    
    if results.get('spreads_disabled'):
        print("\n  ⚠️  SPREADS DISABLED (0/5 historical, needs recalibration)")


def record_bets(results: dict):
    """Record recommended bets to database."""
    if not results['bets']:
        print("No bets to record")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    recorded = 0
    for bet in results['bets']:
        if bet['type'] == 'CONSENSUS':
            # Record consensus bets to tracked_confident_bets table
            c.execute('''
                INSERT OR IGNORE INTO tracked_confident_bets
                (game_id, date, pick_team_id, pick_team_name, opponent_name,
                 is_home, moneyline, models_agree, models_total, avg_prob, 
                 confidence, bet_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 10, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick_team_id'],
                  bet['pick_team_name'], bet['opponent_name'], bet['is_home'],
                  bet['moneyline'], bet['models_agree'], bet['model_prob'],
                  bet['model_prob'], bet['bet_amount']))
            if c.rowcount > 0:
                recorded += 1
                
        elif bet['type'] == 'ML':
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets
                (game_id, date, pick_team_id, pick_team_name, opponent_name,
                 is_home, moneyline, model_prob, dk_implied, edge, bet_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick_team_id'],
                  bet['pick_team_name'], bet['opponent_name'], bet['is_home'],
                  bet['moneyline'], bet['model_prob'], 
                  bet.get('dk_implied', american_to_prob(bet['moneyline'])),
                  bet['edge'], bet['bet_amount']))
            if c.rowcount > 0:
                recorded += 1
                
        elif bet['type'] == 'TOTAL':
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets_spreads
                (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                VALUES (?, ?, 'total', ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick'], bet['line'],
                  bet['odds'], bet['model_projection'], bet['edge'], bet['bet_amount']))
            if c.rowcount > 0:
                recorded += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Recorded {recorded} bets")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bet_selection_v2.py [analyze|record]")
        print("\nThresholds:")
        print(f"  ML: {ML_EDGE_THRESHOLD}%+ edge, ML between {ML_MAX_FAVORITE} and +{ML_MIN_UNDERDOG}")
        print(f"  Totals: {TOTALS_EDGE_THRESHOLD}+ runs edge")
        print(f"  Spreads: {'ENABLED' if SPREADS_ENABLED else 'DISABLED'}")
        print(f"  Max bets/day: {MAX_BETS_PER_DAY}")
        sys.exit(1)
    
    cmd = sys.argv[1]
    date_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = analyze_games(date_arg)
    
    if 'error' in results:
        sys.exit(1)
    
    print_analysis(results)
    
    if cmd == 'record':
        record_bets(results)
    elif cmd != 'analyze':
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == '__main__':
    main()
