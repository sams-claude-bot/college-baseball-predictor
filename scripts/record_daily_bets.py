#!/usr/bin/env python3
"""Record daily best bets from the web dashboard's betting page logic.

This calls /api/best-bets which uses the exact same model predictions
and selection criteria as the Betting page's "Best Bets" and "Best Totals" sections.

Usage:
    python3 scripts/record_daily_bets.py record   # Record today's best bets
    python3 scripts/record_daily_bets.py evaluate  # Grade completed bets
"""
import sys
import json
import sqlite3
import requests
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from run_utils import ScriptRunner

DB_PATH = 'data/baseball.db'
API_URL = 'http://localhost:5000/api/best-bets'
BET_AMOUNT = 100
MAX_PER_TYPE = 6


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def record(date_override=None, runner=None):
    """Fetch best bets from the API and record them."""
    if runner:
        runner.info("Recording daily best bets...")
    else:
        print(f"\n📊 Recording daily best bets...")
    
    try:
        url = API_URL
        if date_override:
            url += f'?date={date_override}'
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        msg = f"Failed to fetch best bets from API: {e}"
        if runner:
            runner.error(msg)
            runner.finish()
        else:
            print(f"❌ {msg}")
            print("   Make sure the web dashboard is running (python3 web/app.py)")
            sys.exit(1)
    
    date = data['date']
    if runner:
        runner.info(f"Date: {date}")
    else:
        print(f"   Date: {date}")
    
    conn = get_conn()
    c = conn.cursor()
    
    # --- CONFIDENT BETS (Model Consensus) ---
    # Check if table exists; create if not
    c.execute('''
        CREATE TABLE IF NOT EXISTS tracked_confident_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            pick_team_id TEXT,
            pick_team_name TEXT,
            opponent_name TEXT,
            is_home INTEGER,
            moneyline INTEGER,
            models_agree INTEGER,
            models_total INTEGER,
            avg_prob REAL,
            confidence REAL,
            bet_amount REAL DEFAULT 100,
            won INTEGER,
            profit REAL,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, pick_team_id)
        )
    ''')
    
    existing_conf = c.execute(
        'SELECT COUNT(*) FROM tracked_confident_bets WHERE date=?', (date,)
    ).fetchone()[0]
    
    conf_recorded = 0
    confident_bets = data.get('confident_bets', [])
    if existing_conf >= MAX_PER_TYPE:
        if runner:
            runner.info(f"Confident Bets: Already have {existing_conf} bets for {date}, skipping")
        else:
            print(f"\n🎯 Confident Bets: Already have {existing_conf} bets for {date}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_conf
        if runner:
            runner.info(f"Confident Bets ({len(confident_bets)} candidates, {slots} slots):")
        else:
            print(f"\n🎯 Confident Bets ({len(confident_bets)} candidates, {slots} slots):")
        for bet in confident_bets[:slots]:
            c.execute('''
                INSERT OR IGNORE INTO tracked_confident_bets
                (game_id, date, pick_team_id, pick_team_name, opponent_name,
                 is_home, moneyline, models_agree, models_total, avg_prob, confidence, bet_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick_team_id'],
                  bet['pick_team_name'], bet['opponent_name'], bet['is_home'],
                  bet['moneyline'], bet['models_agree'], bet['models_total'],
                  bet['avg_prob'], bet['confidence'], BET_AMOUNT))
            if c.rowcount > 0:
                conf_recorded += 1
                sign = '+' if bet.get('moneyline', 0) and bet['moneyline'] > 0 else ''
                ml_str = f"({sign}{bet['moneyline']})" if bet.get('moneyline') else ""
                msg = f"   {bet['pick_team_name']} {ml_str} | {bet['models_agree']}/{bet['models_total']} models | {bet['avg_prob']*100:.0f}% avg | vs {bet['opponent_name']}"
                if runner:
                    runner.info(msg)
                else:
                    print(f"   🎯 {bet['pick_team_name']} {ml_str} | "
                          f"{bet['models_agree']}/{bet['models_total']} models | "
                          f"{bet['avg_prob']*100:.0f}% avg | vs {bet['opponent_name']}")
    
    # --- MONEYLINES (EV-based) ---
    existing_ml = c.execute(
        'SELECT COUNT(*) FROM tracked_bets WHERE date=?', (date,)
    ).fetchone()[0]
    
    ml_recorded = 0
    if existing_ml >= MAX_PER_TYPE:
        if runner:
            runner.info(f"EV Moneylines: Already have {existing_ml} bets for {date}, skipping")
        else:
            print(f"\n💰 EV Moneylines: Already have {existing_ml} bets for {date}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_ml
        if runner:
            runner.info(f"EV Moneylines ({len(data['moneylines'])} candidates, {slots} slots):")
        else:
            print(f"\n💰 EV Moneylines ({len(data['moneylines'])} candidates, {slots} slots):")
        for bet in data['moneylines'][:slots]:
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets
                (game_id, date, pick_team_id, pick_team_name, opponent_name,
                 is_home, moneyline, model_prob, dk_implied, edge, bet_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick_team_id'],
                  bet['pick_team_name'], bet['opponent_name'], bet['is_home'],
                  bet['moneyline'], bet['model_prob'], bet['dk_implied'],
                  bet['edge'], BET_AMOUNT))
            if c.rowcount > 0:
                ml_recorded += 1
                sign = '+' if bet['moneyline'] > 0 else ''
                msg = f"   {bet['pick_team_name']} ({sign}{bet['moneyline']}) | edge {bet['edge']:.1f}% | vs {bet['opponent_name']}"
                if runner:
                    runner.info(msg)
                else:
                    print(f"   📌 {bet['pick_team_name']} ({sign}{bet['moneyline']}) | "
                          f"edge {bet['edge']:.1f}% | vs {bet['opponent_name']}")
    
    # --- SPREADS ---
    existing_sp = c.execute(
        'SELECT COUNT(*) FROM tracked_bets_spreads WHERE date=? AND bet_type="spread"',
        (date,)
    ).fetchone()[0]
    
    sp_recorded = 0
    if existing_sp >= MAX_PER_TYPE:
        if runner:
            runner.info(f"Spreads: Already have {existing_sp} bets for {date}, skipping")
        else:
            print(f"\n📊 Spreads: Already have {existing_sp} bets for {date}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_sp
        if runner:
            runner.info(f"Spreads ({len(data['spreads'])} candidates, {slots} slots):")
        else:
            print(f"\n📊 Spreads ({len(data['spreads'])} candidates, {slots} slots):")
        for bet in data['spreads'][:slots]:
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets_spreads
                (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                VALUES (?, ?, 'spread', ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick'], bet['line'],
                  bet['odds'], bet['model_projection'], bet['edge'], BET_AMOUNT))
            if c.rowcount > 0:
                sp_recorded += 1
                msg = f"   {bet['pick']} {bet['line']:+.1f} ({bet['odds']:+g}) | edge {bet['edge']:.1f} runs"
                if runner:
                    runner.info(msg)
                else:
                    print(f"   📊 {bet['pick']} {bet['line']:+.1f} ({bet['odds']:+g}) | "
                          f"edge {bet['edge']:.1f} runs")
    
    # --- TOTALS ---
    existing_tot = c.execute(
        'SELECT COUNT(*) FROM tracked_bets_spreads WHERE date=? AND bet_type="total"',
        (date,)
    ).fetchone()[0]
    
    tot_recorded = 0
    if existing_tot >= MAX_PER_TYPE:
        if runner:
            runner.info(f"Totals: Already have {existing_tot} bets for {date}, skipping")
        else:
            print(f"\n🎯 Totals: Already have {existing_tot} bets for {date}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_tot
        if runner:
            runner.info(f"Totals ({len(data['totals'])} candidates, {slots} slots):")
        else:
            print(f"\n🎯 Totals ({len(data['totals'])} candidates, {slots} slots):")
        for bet in data['totals'][:slots]:
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets_spreads
                (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                VALUES (?, ?, 'total', ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick'], bet['line'],
                  bet['odds'], bet['model_projection'], bet['edge'], BET_AMOUNT))
            if c.rowcount > 0:
                tot_recorded += 1
                msg = f"   {bet['pick']} {bet['line']} ({bet['odds']:+g}) | proj {bet['model_projection']:.1f} | edge {bet['edge']:.1f} runs"
                if runner:
                    runner.info(msg)
                else:
                    print(f"   🎯 {bet['pick']} {bet['line']} ({bet['odds']:+g}) | "
                          f"proj {bet['model_projection']:.1f} | edge {bet['edge']:.1f} runs")
    
    conn.commit()
    conn.close()
    
    total = conf_recorded + ml_recorded + sp_recorded + tot_recorded
    if runner:
        runner.info(f"Recorded {total} new bets (CONF:{conf_recorded} EV-ML:{ml_recorded} SPR:{sp_recorded} TOT:{tot_recorded})")
        runner.add_stat("bets_recorded", total)
    else:
        print(f"\n✅ Recorded {total} new bets (CONF:{conf_recorded} EV-ML:{ml_recorded} SPR:{sp_recorded} TOT:{tot_recorded})")
    
    if not confident_bets and not data['moneylines'] and not data['spreads'] and not data['totals']:
        if runner:
            runner.info("No games with betting lines today (off day or no DK odds scraped)")
        else:
            print("ℹ️  No games with betting lines today (off day or no DK odds scraped)")


def evaluate(runner=None):
    """Grade completed bets based on final scores."""
    conn = get_conn()
    c = conn.cursor()
    
    total_profit = 0.0
    
    # --- Evaluate confident bets (model consensus) ---
    try:
        c.execute('''
            SELECT tb.id, tb.game_id, tb.date, tb.pick_team_id, tb.moneyline,
                   g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.status
            FROM tracked_confident_bets tb
            LEFT JOIN betting_lines bl ON tb.game_id = bl.game_id
            LEFT JOIN games g ON g.date = tb.date AND g.status = 'final'
                AND ((g.home_team_id = bl.home_team_id AND g.away_team_id = bl.away_team_id)
                  OR (g.home_team_id = bl.away_team_id AND g.away_team_id = bl.home_team_id)
                  OR g.id = tb.game_id)
            WHERE tb.won IS NULL AND g.status = 'final'
        ''')
        conf_evaluated = 0
        for row in c.fetchall():
            row = dict(row)
            home_won = row['home_score'] > row['away_score']
            picked_home = row['pick_team_id'] == row['home_team_id']
            won = 1 if (picked_home == home_won) else 0
            
            ml = row['moneyline'] or -110  # Default to -110 if no line
            if won:
                profit = (ml / 100) * BET_AMOUNT if ml > 0 else (100 / abs(ml)) * BET_AMOUNT
            else:
                profit = -BET_AMOUNT
            
            total_profit += profit
            
            c.execute('UPDATE tracked_confident_bets SET won=?, profit=? WHERE id=?',
                      (won, round(profit, 2), row['id']))
            conf_evaluated += 1
            icon = '✅' if won else '❌'
            msg = f"CONF: {'W' if won else 'L'} | ${profit:+.2f}"
            if runner:
                runner.info(f"  {msg}")
            else:
                print(f"  {icon} {msg}")
    except Exception as e:
        conf_evaluated = 0
        if runner:
            runner.warn(f"Confident bets table not found or error: {e}")
        else:
            print(f"  ⚠️  Confident bets table not found or error: {e}")
    
    # --- Evaluate moneylines (EV-based) ---
    # Join via betting_lines to resolve team IDs, then find game by teams+date
    # Note: DraftKings sometimes lists home/away reversed from our games table
    c.execute('''
        SELECT DISTINCT tb.id, tb.game_id, tb.date, tb.pick_team_id, tb.moneyline,
               g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.status, g.winner_id
        FROM tracked_bets tb
        LEFT JOIN betting_lines bl ON tb.game_id = bl.game_id
        LEFT JOIN games g ON g.date = tb.date AND g.status = 'final'
            AND ((g.home_team_id = bl.home_team_id AND g.away_team_id = bl.away_team_id)
              OR (g.home_team_id = bl.away_team_id AND g.away_team_id = bl.home_team_id)
              OR g.id = tb.game_id)
        WHERE tb.won IS NULL AND g.status = 'final'
    ''')
    ml_evaluated = 0
    seen_ids = set()
    for row in c.fetchall():
        row = dict(row)
        if row['id'] in seen_ids:
            continue  # skip duplicate rows from multi-book join
        seen_ids.add(row['id'])
        # Grade by comparing pick directly to winner — no is_home dependency
        winner_id = row.get('winner_id')
        if not winner_id:
            # Derive winner from scores
            winner_id = row['home_team_id'] if row['home_score'] > row['away_score'] else row['away_team_id']
        won = 1 if row['pick_team_id'] == winner_id else 0
        
        ml = row['moneyline']
        if won:
            profit = (ml / 100) * BET_AMOUNT if ml > 0 else (100 / abs(ml)) * BET_AMOUNT
        else:
            profit = -BET_AMOUNT
        
        total_profit += profit
        
        c.execute('UPDATE tracked_bets SET won=?, profit=? WHERE id=?',
                  (won, round(profit, 2), row['id']))
        ml_evaluated += 1
        icon = '✅' if won else '❌'
        msg = f"ML: {'W' if won else 'L'} | ${profit:+.2f}"
        if runner:
            runner.info(f"  {msg}")
        else:
            print(f"  {icon} {msg}")
    
    # --- Evaluate spreads & totals ---
    # Note: DraftKings sometimes lists home/away reversed from our games table
    c.execute('''
        SELECT tb.id, tb.game_id, tb.date, tb.pick, tb.line, tb.odds, tb.bet_type,
               g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.status,
               ht.name as home_name, at.name as away_name
        FROM tracked_bets_spreads tb
        LEFT JOIN betting_lines bl ON tb.game_id = bl.game_id
        LEFT JOIN games g ON g.date = tb.date AND g.status = 'final'
            AND ((g.home_team_id = bl.home_team_id AND g.away_team_id = bl.away_team_id)
              OR (g.home_team_id = bl.away_team_id AND g.away_team_id = bl.home_team_id)
              OR g.id = tb.game_id)
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE tb.won IS NULL AND g.status = 'final'
    ''')
    sp_evaluated = 0
    tot_evaluated = 0
    for row in c.fetchall():
        row = dict(row)
        hs, aws = row['home_score'], row['away_score']
        actual_total = hs + aws
        actual_margin = hs - aws  # positive = home won by X
        
        if row['bet_type'] == 'spread':
            # Spread: pick + line, e.g. "Kansas -1.5" means Kansas must win by >1.5
            pick_is_home = (row['pick'] == row['home_name'])
            if pick_is_home:
                covered = actual_margin + row['line'] > 0  # line is negative for favorites
            else:
                covered = -actual_margin + row['line'] > 0
            
            won = 1 if covered else 0
            odds = row['odds'] or -110
            if won:
                profit = (odds / 100) * BET_AMOUNT if odds > 0 else (100 / abs(odds)) * BET_AMOUNT
            else:
                profit = -BET_AMOUNT
            
            total_profit += profit
            
            c.execute('UPDATE tracked_bets_spreads SET won=?, profit=? WHERE id=?',
                      (won, round(profit, 2), row['id']))
            sp_evaluated += 1
            icon = '✅' if won else '❌'
            msg = f"SPR: {row['pick']} {row['line']:+.1f} | actual margin {actual_margin:+d} | ${profit:+.2f}"
            if runner:
                runner.info(f"  {msg}")
            else:
                print(f"  {icon} {msg}")
        
        else:  # total
            if row['pick'] == 'OVER':
                won = 1 if actual_total > row['line'] else 0
            else:
                won = 1 if actual_total < row['line'] else 0
            
            # Push
            if actual_total == row['line']:
                won = None
                profit = 0
            else:
                odds = row['odds'] or -110
                if won:
                    profit = (odds / 100) * BET_AMOUNT if odds > 0 else (100 / abs(odds)) * BET_AMOUNT
                else:
                    profit = -BET_AMOUNT
            
            total_profit += profit
            
            c.execute('UPDATE tracked_bets_spreads SET won=?, profit=? WHERE id=?',
                      (won, round(profit, 2), row['id']))
            tot_evaluated += 1
            icon = '✅' if won else '❌'
            msg = f"TOT: {row['pick']} {row['line']} | actual {actual_total} | ${profit:+.2f}"
            if runner:
                runner.info(f"  {msg}")
            else:
                print(f"  {icon} {msg}")
    
    # === PARLAY EVALUATION ===
    parlay_evaluated = 0
    c = conn.cursor()
    c.execute('''
        SELECT tp.id, tp.date, tp.legs_json, tp.num_legs, tp.decimal_odds, tp.bet_amount
        FROM tracked_parlays tp
        WHERE tp.won IS NULL
    ''')
    pending_parlays = [dict(r) for r in c.fetchall()]
    
    for row in pending_parlays:
        legs = json.loads(row['legs_json'])
        legs_won = 0
        legs_resolved = 0
        all_resolved = True
        
        for leg in legs:
            game_id = leg['game_id']
            g = c.execute('SELECT home_team_id, away_team_id, home_score, away_score, status FROM games WHERE id=?',
                         (game_id,)).fetchone()
            if not g or g['status'] != 'final':
                all_resolved = False
                break
            
            legs_resolved += 1
            leg_type = leg.get('type', '')
            
            if leg_type in ('CONSENSUS', 'ML'):
                pick_team = leg.get('pick', '')
                # Find which team was picked by matching name
                # Check if the pick won
                pick_id = leg.get('pick_team_id', '')
                if not pick_id:
                    # Match by name from consensus/ev data
                    # The pick field has team name — check winner
                    home_won = g['home_score'] > g['away_score']
                    # We need to figure out if pick was home or away
                    # Look it up from tracked tables
                    tb = c.execute('SELECT is_home FROM tracked_confident_bets WHERE game_id=? AND pick_team_name=?',
                                  (game_id, pick_team)).fetchone()
                    if not tb:
                        tb = c.execute('SELECT is_home FROM tracked_bets WHERE game_id=? AND pick_team_name=?',
                                      (game_id, pick_team)).fetchone()
                    if tb:
                        won = (home_won and tb['is_home']) or (not home_won and not tb['is_home'])
                    else:
                        # Fallback: check if pick matches home or away team name
                        home_name = c.execute('SELECT name FROM teams WHERE id=?', (g['home_team_id'],)).fetchone()
                        won = home_won if (home_name and pick_team == home_name['name']) else not home_won
                    if won:
                        legs_won += 1
                else:
                    won = (g['home_score'] > g['away_score'] and pick_id == g['home_team_id']) or \
                          (g['away_score'] > g['home_score'] and pick_id == g['away_team_id'])
                    if won:
                        legs_won += 1
                        
            elif leg_type == 'Total':
                pick_str = leg.get('pick', '')  # e.g. "OVER 13.0"
                parts = pick_str.split()
                if len(parts) == 2:
                    direction = parts[0]  # OVER or UNDER
                    line = float(parts[1])
                    actual = g['home_score'] + g['away_score']
                    if direction == 'OVER' and actual > line:
                        legs_won += 1
                    elif direction == 'UNDER' and actual < line:
                        legs_won += 1
                    elif actual == line:
                        legs_won += 1  # Push = win for parlay purposes
        
        if not all_resolved:
            continue
        
        bet_amount = row['bet_amount']
        won = 1 if legs_won == row['num_legs'] else 0
        if won:
            profit = round(bet_amount * (row['decimal_odds'] - 1), 2)
        else:
            profit = -bet_amount
        
        c.execute('UPDATE tracked_parlays SET won=?, profit=?, legs_won=? WHERE id=?',
                  (won, profit, legs_won, row['id']))
        parlay_evaluated += 1
        total_profit += profit
        icon = '✅' if won else '❌'
        msg = f"PARLAY: {legs_won}/{row['num_legs']} legs | ${profit:+.2f}"
        if runner:
            runner.info(f"  {msg}")
        else:
            print(f"  {icon} {msg}")

    conn.commit()
    conn.close()
    
    total = conf_evaluated + ml_evaluated + sp_evaluated + tot_evaluated + parlay_evaluated
    if runner:
        runner.info(f"Evaluated {total} bets (CONF:{conf_evaluated} EV-ML:{ml_evaluated} SPR:{sp_evaluated} TOT:{tot_evaluated} PARLAY:{parlay_evaluated})")
        runner.add_stat("bets_evaluated", total)
        runner.add_stat("profit", f"${total_profit:+.2f}")
    else:
        print(f"\n✅ Evaluated {total} bets (CONF:{conf_evaluated} EV-ML:{ml_evaluated} SPR:{sp_evaluated} TOT:{tot_evaluated} PARLAY:{parlay_evaluated})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 record_daily_bets.py [record|evaluate]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    date_arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == 'record':
        runner = ScriptRunner("record_daily_bets_record")
        record(date_override=date_arg, runner=runner)
        runner.finish()
    elif cmd == 'evaluate':
        runner = ScriptRunner("record_daily_bets_evaluate")
        evaluate(runner=runner)
        runner.finish()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
