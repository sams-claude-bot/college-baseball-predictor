#!/usr/bin/env bash
# Record daily bets from the betting page selections into tracked_bets tables.
# Runs after odds are loaded so bets reflect current lines.
# System cron: 15 8 * * *
set -euo pipefail
cd /home/sam/college-baseball-predictor

echo "$(date '+%Y-%m-%d %H:%M:%S') — Recording daily bets"

python3 << 'PYEOF'
import sqlite3
from datetime import date
from web.services.betting_page import build_betting_page_context

data = build_betting_page_context()
today = date.today().isoformat()

conn = sqlite3.connect('data/baseball.db')
conn.row_factory = sqlite3.Row

# Record EV bets
ev_bets = data.get('ev_bets', [])
ev_recorded = 0
for g in ev_bets:
    game_id = g['game_id']
    pick = g.get('best_pick', 'home')
    pick_team_id = g['home_team_id'] if pick == 'home' else g['away_team_id']
    pick_team_name = g.get('home_team_name') if pick == 'home' else g.get('away_team_name')
    opponent_name = g.get('away_team_name') if pick == 'home' else g.get('home_team_name')
    is_home = 1 if pick == 'home' else 0
    ml = g.get('home_ml') if pick == 'home' else g.get('away_ml')
    model_prob = g.get('model_home_prob', 0.5) if pick == 'home' else g.get('model_away_prob', 0.5)
    
    if ml and ml < 0:
        dk_implied = abs(ml) / (abs(ml) + 100)
    elif ml and ml > 0:
        dk_implied = 100 / (ml + 100)
    else:
        dk_implied = 0.5
    edge = (model_prob - dk_implied) * 100
    
    existing = conn.execute("SELECT id FROM tracked_bets WHERE game_id=? AND date=?",
                           (game_id, today)).fetchone()
    if existing:
        continue
    conn.execute("""
        INSERT INTO tracked_bets (game_id, date, pick_team_id, pick_team_name, opponent_name,
                                  is_home, moneyline, model_prob, dk_implied, edge, bet_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 100)
    """, (game_id, today, pick_team_id, pick_team_name, opponent_name,
          is_home, ml, model_prob, dk_implied, edge))
    ev_recorded += 1
    print(f"  EV: {pick_team_name} ML {ml} (edge: {edge:.1f}%)")

# Record totals bets
totals = data.get('best_totals', [])
tot_recorded = 0
for g in totals:
    game_id = g['game_id']
    ou = g.get('over_under')
    nn = g.get('nn_total')
    if not ou or nn is None:
        continue
    diff = nn - ou
    pick = 'OVER' if diff > 0 else 'UNDER'
    odds = g.get('over_odds', -110) if pick == 'OVER' else g.get('under_odds', -110)
    
    existing = conn.execute(
        "SELECT id FROM tracked_bets_spreads WHERE game_id=? AND date=? AND bet_type='total'",
        (game_id, today)).fetchone()
    if existing:
        continue
    conn.execute("""
        INSERT INTO tracked_bets_spreads (game_id, date, bet_type, pick, line, odds,
                                          model_projection, edge, bet_amount)
        VALUES (?, ?, 'total', ?, ?, ?, ?, ?, 100)
    """, (game_id, today, pick, ou, odds, nn, abs(diff)))
    tot_recorded += 1
    print(f"  TOT: {game_id} {pick} {ou} (NN: {nn:.1f}, edge: {abs(diff):.1f})")

conn.commit()
conn.close()

print(f"\nRecorded: {ev_recorded} EV bets, {tot_recorded} totals bets")
PYEOF

echo "$(date '+%Y-%m-%d %H:%M:%S') — Bet recording complete"
