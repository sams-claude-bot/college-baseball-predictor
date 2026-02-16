#!/usr/bin/env python3
"""Record and evaluate spread and totals bets.

Uses blended model run projections to find edges on DraftKings spreads and totals.

Usage:
    python3 scripts/record_spread_bets.py record     # Record spread + total bets
    python3 scripts/record_spread_bets.py evaluate    # Score completed bets
    python3 scripts/record_spread_bets.py summary     # Print P&L summary
"""
import sys, os, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'baseball.db')
SPREAD_EDGE_THRESHOLD = 1.0  # Minimum run margin beyond the spread to bet
TOTAL_EDGE_THRESHOLD = 1.0   # Minimum run difference from total line to bet
BET_AMOUNT = 100.0

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
    """Get blended run projections using stored model predictions (60/40 NN/ensemble)."""
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
        home_runs = nn_hr * 0.60 + ens_hr * 0.40
        away_runs = nn_ar * 0.60 + ens_ar * 0.40
    elif nn_hr is not None:
        home_runs, away_runs = nn_hr, nn_ar
    elif ens_hr is not None:
        home_runs, away_runs = ens_hr, ens_ar
    else:
        return None, None
    
    return home_runs, away_runs

def record_bets():
    """Record spread and total bets for upcoming games with DK lines."""
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
    spread_recorded = 0
    total_recorded = 0
    
    for line in lines:
        home_runs, away_runs = get_blended_runs(conn, line['game_id'])
        if home_runs is None:
            continue
        
        projected_margin = home_runs - away_runs  # positive = home favored
        projected_total = home_runs + away_runs
        
        # --- SPREAD BET ---
        if line['home_spread'] is not None and line['home_spread'] != 0:
            # Check if already tracked
            existing = c.execute(
                'SELECT 1 FROM tracked_bets_spreads WHERE game_id=? AND bet_type="spread"',
                (line['game_id'],)
            ).fetchone()
            
            if not existing:
                spread = line['home_spread']  # e.g., -1.5 means home favored by 1.5
                # Our projected margin vs the spread
                # If spread is -1.5 and we project home wins by 3, edge = 3 - 1.5 = 1.5 runs
                home_cover_margin = projected_margin - spread  # positive = home covers
                
                if abs(home_cover_margin) >= SPREAD_EDGE_THRESHOLD:
                    if home_cover_margin > 0:
                        # Bet home to cover
                        pick = line['home_name']
                        bet_line = spread
                        odds = line['home_spread_odds'] or -110
                        edge = home_cover_margin
                    else:
                        # Bet away to cover
                        pick = line['away_name']
                        bet_line = line['away_spread']
                        odds = line['away_spread_odds'] or -110
                        edge = abs(home_cover_margin)
                    
                    c.execute('''
                        INSERT OR IGNORE INTO tracked_bets_spreads
                        (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                        VALUES (?, ?, 'spread', ?, ?, ?, ?, ?, ?)
                    ''', (line['game_id'], line['date'], pick, bet_line, odds,
                          round(projected_margin, 2), round(edge, 2), BET_AMOUNT))
                    
                    spread_recorded += 1
                    print(f"  📊 SPREAD: {pick} {bet_line:+.1f} ({odds:+g}) | "
                          f"proj margin {projected_margin:+.1f} | edge {edge:.1f} runs")
        
        # --- TOTAL BET ---
        if line['over_under'] is not None and line['over_under'] > 0:
            existing = c.execute(
                'SELECT 1 FROM tracked_bets_spreads WHERE game_id=? AND bet_type="total"',
                (line['game_id'],)
            ).fetchone()
            
            if not existing:
                total_line = line['over_under']
                total_diff = projected_total - total_line
                
                if abs(total_diff) >= TOTAL_EDGE_THRESHOLD:
                    if total_diff > 0:
                        pick = 'OVER'
                        odds = line['over_odds'] or -110
                    else:
                        pick = 'UNDER'
                        odds = line['under_odds'] or -110
                    
                    c.execute('''
                        INSERT OR IGNORE INTO tracked_bets_spreads
                        (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                        VALUES (?, ?, 'total', ?, ?, ?, ?, ?, ?)
                    ''', (line['game_id'], line['date'], pick, total_line, odds,
                          round(projected_total, 2), round(abs(total_diff), 2), BET_AMOUNT))
                    
                    total_recorded += 1
                    print(f"  🎯 TOTAL: {pick} {total_line} ({odds:+g}) | "
                          f"proj {projected_total:.1f} | edge {abs(total_diff):.1f} runs")
    
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
            # Did our pick cover?
            spread_line = bet['line']
            # If pick is home team: home_score + spread > away_score (cover)
            # If pick is away team: away_score + spread > home_score (cover)
            # Since line is from the picked team's perspective:
            # Check: did actual result beat the line?
            
            # Figure out if pick was home or away
            conn2 = get_connection()
            c2 = conn2.cursor()
            home_name = c2.execute('SELECT name FROM teams WHERE id=?', (bet['home_team_id'],)).fetchone()['name']
            conn2.close()
            
            if bet['pick'] == home_name:
                # Home pick: home wins by more than |spread| (spread is negative for favorite)
                covered = actual_margin + spread_line > 0  # e.g., margin=3, spread=-1.5 → 3+(-1.5)=1.5 > 0 ✓
            else:
                # Away pick: away_spread = -home_spread
                covered = -actual_margin + spread_line > 0
            
            # Push
            if bet['pick'] == home_name:
                push = (actual_margin + spread_line) == 0
            else:
                push = (-actual_margin + spread_line) == 0
            
            if push:
                won = None
                profit = 0
                status = "🔄 PUSH"
            elif covered:
                won = 1
                profit = american_to_payout(bet['odds'] or -110)
                status = "✅ WIN"
            else:
                won = 0
                profit = -BET_AMOUNT
                status = "❌ LOSS"
        
        elif bet['bet_type'] == 'total':
            total_line = bet['line']
            if bet['pick'] == 'OVER':
                won = 1 if actual_total > total_line else 0
                push = actual_total == total_line
            else:
                won = 1 if actual_total < total_line else 0
                push = actual_total == total_line
            
            if push:
                won = None
                profit = 0
                status = "🔄 PUSH"
            elif won:
                profit = american_to_payout(bet['odds'] or -110)
                status = "✅ WIN"
            else:
                won = 0
                profit = -BET_AMOUNT
                status = "❌ LOSS"
        
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
