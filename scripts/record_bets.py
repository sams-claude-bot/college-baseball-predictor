#!/usr/bin/env python3
"""Record and evaluate tracked bets.

Uses the SAME blended prediction logic as the betting page's "best bets"
so P&L always matches what was recommended.

Usage:
    python3 scripts/record_bets.py record          # Record bets for upcoming games with DK lines
    python3 scripts/record_bets.py evaluate         # Score completed bets
    python3 scripts/record_bets.py summary          # Print P&L summary
"""
import sys, os, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'baseball.db')
EDGE_THRESHOLD = 5.0  # Minimum edge % to place a bet
BET_AMOUNT = 100.0

def get_connection():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def american_to_implied_prob(ml):
    """Convert American odds to implied probability."""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)

def get_blended_prob(conn, game_id, home_team_id, away_team_id):
    """Get blended prediction using stored model predictions.
    
    Uses the same 60/40 NN/ensemble blend as the betting page.
    Falls back to stored predictions so bets are locked in at record time.
    """
    c = conn.cursor()
    
    # Get ensemble prediction
    ens_row = c.execute(
        'SELECT predicted_home_prob FROM model_predictions WHERE game_id=? AND model_name="ensemble"',
        (game_id,)
    ).fetchone()
    
    # Get neural prediction
    nn_row = c.execute(
        'SELECT predicted_home_prob FROM model_predictions WHERE game_id=? AND model_name="neural"',
        (game_id,)
    ).fetchone()
    
    ens_prob = ens_row['predicted_home_prob'] if ens_row else None
    nn_prob = nn_row['predicted_home_prob'] if nn_row else None
    
    if ens_prob is not None and nn_prob is not None:
        return nn_prob * 0.60 + ens_prob * 0.40
    elif nn_prob is not None:
        return nn_prob
    elif ens_prob is not None:
        return ens_prob
    return None

def record_bets():
    """Record bets for games with DK lines, using same logic as best bets page."""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    three_days = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    
    # Get games with betting lines that don't already have tracked bets
    c.execute('''
        SELECT b.game_id, b.date, b.home_team_id, b.away_team_id,
               b.home_ml, b.away_ml,
               ht.name as home_name, at.name as away_name
        FROM betting_lines b
        JOIN teams ht ON b.home_team_id = ht.id
        JOIN teams at ON b.away_team_id = at.id
        WHERE b.date >= ? AND b.date <= ?
          AND b.home_ml IS NOT NULL AND b.away_ml IS NOT NULL
          AND b.game_id NOT IN (SELECT game_id FROM tracked_bets)
        ORDER BY b.date
    ''', (today, three_days))
    
    lines = [dict(r) for r in c.fetchall()]
    recorded = 0
    
    for line in lines:
        model_home_prob = get_blended_prob(conn, line['game_id'], 
                                           line['home_team_id'], line['away_team_id'])
        if model_home_prob is None:
            continue
        
        # DK implied probability (vig-removed)
        try:
            dk_home = american_to_implied_prob(line['home_ml'])
            dk_away = american_to_implied_prob(line['away_ml'])
            dk_home_fair = dk_home / (dk_home + dk_away)
        except:
            continue
        
        # Calculate edges
        home_edge = (model_home_prob - dk_home_fair) * 100
        away_edge = ((1 - model_home_prob) - (1 - dk_home_fair)) * 100
        
        # Pick the side with bigger edge (same as best bets)
        if home_edge >= away_edge:
            pick_home = True
            edge = home_edge
            ml = line['home_ml']
            pick_id = line['home_team_id']
            pick_name = line['home_name']
            opp_name = line['away_name']
            model_prob = model_home_prob
            dk_imp = dk_home_fair
        else:
            pick_home = False
            edge = away_edge
            ml = line['away_ml']
            pick_id = line['away_team_id']
            pick_name = line['away_name']
            opp_name = line['home_name']
            model_prob = 1 - model_home_prob
            dk_imp = 1 - dk_home_fair
        
        # Only bet if edge >= threshold
        if edge < EDGE_THRESHOLD:
            continue
        
        # Per-day cap: only 6 moneyline bets per date
        existing_count = c.execute(
            'SELECT COUNT(*) FROM tracked_bets WHERE date=?',
            (line['date'],)
        ).fetchone()[0]
        if existing_count >= 6:
            continue
        
        c.execute('''
            INSERT OR IGNORE INTO tracked_bets 
            (game_id, date, pick_team_id, pick_team_name, opponent_name,
             is_home, moneyline, model_prob, dk_implied, edge, bet_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (line['game_id'], line['date'], pick_id, pick_name, opp_name,
              1 if pick_home else 0, ml, round(model_prob, 4), round(dk_imp, 4),
              round(edge, 2), BET_AMOUNT))
        
        recorded += 1
        print(f"  📌 {pick_name} ({'+' if ml > 0 else ''}{ml}) | edge {edge:.1f}% | "
              f"model {model_prob:.1%} vs DK {dk_imp:.1%}")
    
    conn.commit()
    conn.close()
    print(f"\n✅ Recorded {recorded} new bets")

def evaluate_bets():
    """Score completed bets."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT tb.rowid, tb.*, g.winner_id, g.home_team_id, g.away_team_id
        FROM tracked_bets tb
        JOIN games g ON tb.game_id = g.id
        WHERE tb.won IS NULL
          AND g.status = 'final'
          AND g.winner_id IS NOT NULL
    ''')
    
    pending = [dict(r) for r in c.fetchall()]
    evaluated = 0
    
    for bet in pending:
        won = bet['winner_id'] == bet['pick_team_id']
        ml = bet['moneyline']
        
        if won:
            if ml > 0:
                profit = ml
            else:
                profit = BET_AMOUNT / abs(ml) * 100
        else:
            profit = -BET_AMOUNT
        
        c.execute('UPDATE tracked_bets SET won=?, profit=? WHERE game_id=?',
                  (1 if won else 0, round(profit, 2), bet['game_id']))
        
        status = "✅ WIN" if won else "❌ LOSS"
        print(f"  {status} {bet['pick_team_name']} | {'+' if profit > 0 else ''}{profit:.0f}")
        evaluated += 1
    
    conn.commit()
    conn.close()
    print(f"\n✅ Evaluated {evaluated} bets")

def summary():
    """Print P&L summary."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('SELECT * FROM tracked_bets WHERE won IS NOT NULL ORDER BY date')
    bets = [dict(r) for r in c.fetchall()]
    
    if not bets:
        print("No completed bets yet.")
        return
    
    wins = sum(1 for b in bets if b['won'])
    total_pl = sum(b['profit'] for b in bets)
    total_wagered = len(bets) * BET_AMOUNT
    
    print(f"📊 P&L Summary")
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
