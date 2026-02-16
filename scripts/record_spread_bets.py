#!/usr/bin/env python3
"""Record and evaluate spread and totals bets.

Uses the SAME models as the betting page:
- NN Totals model for over/under picks
- NN Spread model for spread picks  
- Falls back to blended ensemble/neural run projections if NN models unavailable

Usage:
    python3 scripts/record_spread_bets.py record          # Record bets for upcoming games
    python3 scripts/record_spread_bets.py evaluate         # Score completed bets
    python3 scripts/record_spread_bets.py summary          # Print P&L summary
"""
import sys, os, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'baseball.db')
SPREAD_EDGE_THRESHOLD = 1.0  # Min run margin beyond spread to bet
TOTAL_EDGE_THRESHOLD = 1.0   # Min run diff from total line to bet
TOTAL_PROB_THRESHOLD = 0.55  # Min NN probability to bet (if NN available)
BET_AMOUNT = 100.0
AGREEMENT_BONUS = 1.5        # Multiply edge by this when NN and ensemble agree

# Lazy-loaded models
_nn_totals = None
_nn_spread = None

def get_nn_totals():
    global _nn_totals
    if _nn_totals is None:
        try:
            from models.nn_totals_model import NNTotalsModel
            _nn_totals = NNTotalsModel(use_model_predictions=False)
            if not _nn_totals.is_trained():
                _nn_totals = False
        except:
            _nn_totals = False
    return _nn_totals if _nn_totals else None

def get_nn_spread():
    global _nn_spread
    if _nn_spread is None:
        try:
            from models.nn_spread_model import NNSpreadModel
            _nn_spread = NNSpreadModel(use_model_predictions=False)
            if not _nn_spread.is_trained():
                _nn_spread = False
        except:
            _nn_spread = False
    return _nn_spread if _nn_spread else None

def get_connection():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def american_to_payout(ml, stake=100):
    """Calculate profit from American odds on a winning bet."""
    if ml > 0:
        return ml
    else:
        return stake / abs(ml) * 100

def get_blended_runs(conn, game_id):
    """Fallback: get blended run projections from stored predictions."""
    c = conn.cursor()
    ens = c.execute(
        'SELECT predicted_home_runs, predicted_away_runs FROM model_predictions WHERE game_id=? AND model_name="ensemble"',
        (game_id,)
    ).fetchone()
    nn = c.execute(
        'SELECT predicted_home_runs, predicted_away_runs FROM model_predictions WHERE game_id=? AND model_name="neural"',
        (game_id,)
    ).fetchone()
    
    ens_hr = ens['predicted_home_runs'] if ens and ens['predicted_home_runs'] else None
    ens_ar = ens['predicted_away_runs'] if ens and ens['predicted_away_runs'] else None
    nn_hr = nn['predicted_home_runs'] if nn and nn['predicted_home_runs'] else None
    nn_ar = nn['predicted_away_runs'] if nn and nn['predicted_away_runs'] else None
    
    if ens_hr is not None and nn_hr is not None:
        return nn_hr * 0.60 + ens_hr * 0.40, nn_ar * 0.60 + ens_ar * 0.40
    elif nn_hr is not None:
        return nn_hr, nn_ar
    elif ens_hr is not None:
        return ens_hr, ens_ar
    return None, None

def record_bets(max_spread=6, max_total=6):
    """Record top spread and total bets by edge using NN models (matching betting page)."""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    three_days = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT b.game_id, b.date, b.home_team_id, b.away_team_id,
               b.home_spread, b.home_spread_odds, b.away_spread, b.away_spread_odds,
               b.over_under, b.over_odds, b.under_odds,
               ht.name as home_name, at.name as away_name
        FROM betting_lines b
        JOIN teams ht ON b.home_team_id = ht.id
        JOIN teams at ON b.away_team_id = at.id
        WHERE b.date >= ? AND b.date <= ?
        ORDER BY b.date
    ''', (today, three_days))
    
    lines = [dict(r) for r in c.fetchall()]
    nn_totals = get_nn_totals()
    nn_spread = get_nn_spread()
    
    print(f"Models: nn_totals={'✅' if nn_totals else '❌'} nn_spread={'✅' if nn_spread else '❌'}")
    
    # Collect all candidates first, then pick top N by edge
    spread_candidates = []
    total_candidates = []
    
    for line in lines:
        home_id = line['home_team_id']
        away_id = line['away_team_id']
        
        # --- SPREAD CANDIDATE ---
        if line['home_spread'] is not None and line['home_spread'] != 0:
            existing = c.execute(
                'SELECT 1 FROM tracked_bets_spreads WHERE game_id=? AND bet_type="spread"',
                (line['game_id'],)
            ).fetchone()
            
            if not existing:
                nn_margin = None
                ens_margin = None
                
                # Get NN spread projection
                if nn_spread:
                    try:
                        sp = nn_spread.predict_game(home_id, away_id)
                        nn_margin = sp.get('projected_margin', 0)
                    except:
                        pass
                
                # Get ensemble/blended projection
                hr, ar = get_blended_runs(conn, line['game_id'])
                if hr is not None:
                    ens_margin = hr - ar
                
                # Need at least one projection
                if nn_margin is not None or ens_margin is not None:
                    spread = line['home_spread']
                    
                    # Determine agreement and pick best projection
                    if nn_margin is not None and ens_margin is not None:
                        nn_covers_home = (nn_margin - spread) > 0
                        ens_covers_home = (ens_margin - spread) > 0
                        agree = nn_covers_home == ens_covers_home
                        # Use NN as primary, ensemble as confirmation
                        proj_margin = nn_margin
                        source = 'AGREE' if agree else 'nn_spread'
                    elif nn_margin is not None:
                        proj_margin = nn_margin
                        agree = False
                        source = 'nn_spread'
                    else:
                        proj_margin = ens_margin
                        agree = False
                        source = 'blended'
                    
                    home_cover_margin = proj_margin - spread
                    raw_edge = abs(home_cover_margin)
                    
                    # Boost edge when models agree
                    scored_edge = raw_edge * AGREEMENT_BONUS if agree else raw_edge
                    
                    if raw_edge >= SPREAD_EDGE_THRESHOLD:
                        if home_cover_margin > 0:
                            spread_candidates.append({
                                'game_id': line['game_id'], 'date': line['date'],
                                'pick': line['home_name'], 'line': spread,
                                'odds': line['home_spread_odds'] or -110,
                                'proj': round(proj_margin, 2), 'edge': round(raw_edge, 2),
                                'scored_edge': round(scored_edge, 2),
                                'agree': agree, 'source': source,
                                'nn_margin': nn_margin, 'ens_margin': ens_margin
                            })
                        else:
                            spread_candidates.append({
                                'game_id': line['game_id'], 'date': line['date'],
                                'pick': line['away_name'], 'line': line['away_spread'],
                                'odds': line['away_spread_odds'] or -110,
                                'proj': round(proj_margin, 2), 'edge': round(raw_edge, 2),
                                'scored_edge': round(scored_edge, 2),
                                'agree': agree, 'source': source,
                                'nn_margin': nn_margin, 'ens_margin': ens_margin
                            })
        
        # --- TOTAL CANDIDATE ---
        if line['over_under'] is not None and line['over_under'] > 0:
            existing = c.execute(
                'SELECT 1 FROM tracked_bets_spreads WHERE game_id=? AND bet_type="total"',
                (line['game_id'],)
            ).fetchone()
            
            if not existing:
                nn_total = None
                nn_pick = None
                ens_total = None
                ens_pick = None
                ou_line = line['over_under']
                
                # Get NN totals projection
                if nn_totals:
                    try:
                        tp = nn_totals.predict_game(home_id, away_id, over_under_line=ou_line)
                        nn_total = tp.get('projected_total')
                        over_prob = tp.get('over_prob', 0.5)
                        under_prob = tp.get('under_prob', 0.5)
                        if over_prob >= TOTAL_PROB_THRESHOLD:
                            nn_pick = 'OVER'
                        elif under_prob >= TOTAL_PROB_THRESHOLD:
                            nn_pick = 'UNDER'
                    except:
                        pass
                
                # Get ensemble/blended projection
                hr, ar = get_blended_runs(conn, line['game_id'])
                if hr is not None:
                    ens_total = hr + ar
                    diff = ens_total - ou_line
                    if abs(diff) >= TOTAL_EDGE_THRESHOLD:
                        ens_pick = 'OVER' if diff > 0 else 'UNDER'
                
                # Need at least one model to have a pick
                if nn_pick or ens_pick:
                    agree = nn_pick is not None and ens_pick is not None and nn_pick == ens_pick
                    
                    # Use NN as primary pick, fall back to ensemble
                    pick = nn_pick or ens_pick
                    proj = nn_total if nn_total else ens_total
                    raw_edge = abs(proj - ou_line) if proj else 1.0
                    scored_edge = raw_edge * AGREEMENT_BONUS if agree else raw_edge
                    source = 'AGREE' if agree else ('nn_totals' if nn_pick else 'blended')
                    
                    odds = line['over_odds'] if pick == 'OVER' else line['under_odds']
                    total_candidates.append({
                        'game_id': line['game_id'], 'date': line['date'],
                        'pick': pick, 'line': ou_line,
                        'odds': odds or -110, 'proj': round(proj, 2) if proj else 0,
                        'edge': round(raw_edge, 2), 'scored_edge': round(scored_edge, 2),
                        'agree': agree, 'source': source,
                        'nn_total': nn_total, 'ens_total': ens_total
                    })
    
    # Sort by scored_edge (agreement-boosted) and take top N
    spread_candidates.sort(key=lambda x: x['scored_edge'], reverse=True)
    total_candidates.sort(key=lambda x: x['scored_edge'], reverse=True)
    
    top_spreads = spread_candidates[:max_spread]
    top_totals = total_candidates[:max_total]
    
    spread_recorded = 0
    for s in top_spreads:
        # Check per-day cap: only 6 spread bets per date
        existing_count = c.execute(
            'SELECT COUNT(*) FROM tracked_bets_spreads WHERE date=? AND bet_type="spread"',
            (s['date'],)
        ).fetchone()[0]
        if existing_count >= max_spread:
            break
        c.execute('''
            INSERT OR IGNORE INTO tracked_bets_spreads
            (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
            VALUES (?, ?, 'spread', ?, ?, ?, ?, ?, ?)
        ''', (s['game_id'], s['date'], s['pick'], s['line'], s['odds'],
              s['proj'], s['edge'], BET_AMOUNT))
        spread_recorded += 1
        agree_tag = " ✅ AGREE" if s['agree'] else ""
        nn_str = f"NN={s['nn_margin']:+.1f}" if s['nn_margin'] is not None else "NN=—"
        ens_str = f"Ens={s['ens_margin']:+.1f}" if s['ens_margin'] is not None else "Ens=—"
        print(f"  📊 SPREAD: {s['pick']} {s['line']:+.1f} ({s['odds']:+g}) | "
              f"{nn_str} {ens_str} | edge {s['edge']:.1f} (score {s['scored_edge']:.1f}){agree_tag}")
    
    total_recorded = 0
    for t in top_totals:
        # Check per-day cap: only 6 total bets per date
        existing_count = c.execute(
            'SELECT COUNT(*) FROM tracked_bets_spreads WHERE date=? AND bet_type="total"',
            (t['date'],)
        ).fetchone()[0]
        if existing_count >= max_total:
            break
        c.execute('''
            INSERT OR IGNORE INTO tracked_bets_spreads
            (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
            VALUES (?, ?, 'total', ?, ?, ?, ?, ?, ?)
        ''', (t['game_id'], t['date'], t['pick'], t['line'], t['odds'],
              t['proj'], t['edge'], BET_AMOUNT))
        total_recorded += 1
        agree_tag = " ✅ AGREE" if t['agree'] else ""
        nn_str = f"NN={t['nn_total']:.1f}" if t.get('nn_total') else "NN=—"
        ens_str = f"Ens={t['ens_total']:.1f}" if t.get('ens_total') else "Ens=—"
        print(f"  🎯 TOTAL: {t['pick']} {t['line']} ({t['odds']:+g}) | "
              f"{nn_str} {ens_str} | edge {t['edge']:.1f} (score {t['scored_edge']:.1f}){agree_tag}")
    
    agree_spreads = sum(1 for s in top_spreads if s['agree'])
    agree_totals = sum(1 for t in top_totals if t['agree'])
    if spread_candidates:
        print(f"\n  Spreads: {len(spread_candidates)} candidates → top {max_spread} ({agree_spreads} with agreement)")
    if total_candidates:
        print(f"  Totals: {len(total_candidates)} candidates → top {max_total} ({agree_totals} with agreement)")
    
    conn.commit()
    conn.close()
    print(f"\n✅ Recorded {spread_recorded} spread bets + {total_recorded} total bets")

def evaluate_bets():
    """Score completed spread and total bets."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT tb.*, g.home_score, g.away_score, g.home_team_id, g.away_team_id
        FROM tracked_bets_spreads tb
        JOIN games g ON tb.game_id = g.id
        WHERE tb.won IS NULL
          AND g.status = 'final'
          AND g.home_score IS NOT NULL
    ''')
    
    pending = [dict(r) for r in c.fetchall()]
    evaluated = 0
    
    for bet in pending:
        home_score = bet['home_score']
        away_score = bet['away_score']
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        
        if bet['bet_type'] == 'spread':
            spread_line = bet['line']
            
            # Get team names to figure out which side we picked
            conn2 = get_connection()
            c2 = conn2.cursor()
            home_name = c2.execute('SELECT name FROM teams WHERE id=?', (bet['home_team_id'],)).fetchone()['name']
            conn2.close()
            
            if bet['pick'] == home_name:
                result = actual_margin + spread_line
            else:
                result = -actual_margin + spread_line
            
            if result == 0:
                won = None; profit = 0; status = "🔄 PUSH"
            elif result > 0:
                won = 1; profit = american_to_payout(bet['odds'] or -110); status = "✅ WIN"
            else:
                won = 0; profit = -BET_AMOUNT; status = "❌ LOSS"
        
        elif bet['bet_type'] == 'total':
            total_line = bet['line']
            
            if actual_total == total_line:
                won = None; profit = 0; status = "🔄 PUSH"
            elif bet['pick'] == 'OVER':
                won = 1 if actual_total > total_line else 0
                profit = american_to_payout(bet['odds'] or -110) if won else -BET_AMOUNT
                status = "✅ WIN" if won else "❌ LOSS"
            else:
                won = 1 if actual_total < total_line else 0
                profit = american_to_payout(bet['odds'] or -110) if won else -BET_AMOUNT
                status = "✅ WIN" if won else "❌ LOSS"
        
        c.execute('UPDATE tracked_bets_spreads SET won=?, profit=? WHERE id=?',
                  (won, round(profit, 2), bet['id']))
        
        print(f"  {status} {bet['bet_type'].upper()} {bet['pick']} {bet['line']} | "
              f"{'+' if profit >= 0 else ''}{profit:.0f}")
        evaluated += 1
    
    conn.commit()
    conn.close()
    print(f"\n✅ Evaluated {evaluated} bets")

def summary():
    """Print spread/totals P&L summary."""
    conn = get_connection()
    c = conn.cursor()
    
    for bet_type in ('spread', 'total'):
        c.execute('SELECT * FROM tracked_bets_spreads WHERE bet_type=? AND won IS NOT NULL ORDER BY date', (bet_type,))
        bets = [dict(r) for r in c.fetchall()]
        
        if not bets:
            print(f"\n📊 {bet_type.upper()} P&L: No completed bets yet.")
            continue
        
        wins = sum(1 for b in bets if b['won'])
        total_pl = sum(b['profit'] for b in bets)
        total_wagered = len(bets) * BET_AMOUNT
        
        print(f"\n📊 {bet_type.upper()} P&L")
        print(f"  Bets: {len(bets)} | Wins: {wins} | Win Rate: {wins/len(bets):.1%}")
        print(f"  Total P&L: ${total_pl:+.2f} | ROI: {total_pl/total_wagered:.1%}")
    
    conn.close()

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'record'
    if action == 'record':
        record_bets()
    elif action == 'evaluate':
        evaluate_bets()
    elif action == 'summary':
        summary()
    else:
        print(f"Unknown action: {action}")
