#!/usr/bin/env python3
"""Backfill tracked bets for past dates that had DK lines.

Usage:
    python3 scripts/backfill_bets.py 2026-02-15 2026-02-16
"""
import sys
import json
import sqlite3
import requests
from datetime import datetime

DB_PATH = 'data/baseball.db'
API_URL = 'http://localhost:5000/api/best-bets'
BET_AMOUNT = 100
MAX_PER_TYPE = 6


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def backfill_date(date_str):
    """Fetch best bets for a specific date and record them."""
    print(f"\n📊 Backfilling bets for {date_str}...")
    
    try:
        resp = requests.get(f"{API_URL}?date={date_str}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ Failed to fetch best bets from API: {e}")
        return 0
    
    if data['date'] != date_str:
        print(f"⚠️  API returned date {data['date']} instead of {date_str}")
    
    conn = get_conn()
    c = conn.cursor()
    
    # --- MONEYLINES ---
    existing_ml = c.execute(
        'SELECT COUNT(*) FROM tracked_bets WHERE date=?', (date_str,)
    ).fetchone()[0]
    
    ml_recorded = 0
    if existing_ml >= MAX_PER_TYPE:
        print(f"💰 Moneylines: Already have {existing_ml} bets for {date_str}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_ml
        print(f"💰 Moneylines ({len(data['moneylines'])} candidates, {slots} slots):")
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
                print(f"   📌 {bet['pick_team_name']} ({sign}{bet['moneyline']}) | "
                      f"edge {bet['edge']:.1f}% | vs {bet['opponent_name']}")
    
    # --- SPREADS ---
    existing_sp = c.execute(
        'SELECT COUNT(*) FROM tracked_bets_spreads WHERE date=? AND bet_type="spread"',
        (date_str,)
    ).fetchone()[0]
    
    sp_recorded = 0
    if existing_sp >= MAX_PER_TYPE:
        print(f"📊 Spreads: Already have {existing_sp} bets for {date_str}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_sp
        print(f"📊 Spreads ({len(data['spreads'])} candidates, {slots} slots):")
        for bet in data['spreads'][:slots]:
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets_spreads
                (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                VALUES (?, ?, 'spread', ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick'], bet['line'],
                  bet['odds'], bet['model_projection'], bet['edge'], BET_AMOUNT))
            if c.rowcount > 0:
                sp_recorded += 1
                print(f"   📊 {bet['pick']} {bet['line']:+.1f} ({bet['odds']:+g}) | "
                      f"edge {bet['edge']:.1f} runs")
    
    # --- TOTALS ---
    existing_tot = c.execute(
        'SELECT COUNT(*) FROM tracked_bets_spreads WHERE date=? AND bet_type="total"',
        (date_str,)
    ).fetchone()[0]
    
    tot_recorded = 0
    if existing_tot >= MAX_PER_TYPE:
        print(f"🎯 Totals: Already have {existing_tot} bets for {date_str}, skipping")
    else:
        slots = MAX_PER_TYPE - existing_tot
        print(f"🎯 Totals ({len(data['totals'])} candidates, {slots} slots):")
        for bet in data['totals'][:slots]:
            c.execute('''
                INSERT OR IGNORE INTO tracked_bets_spreads
                (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount)
                VALUES (?, ?, 'total', ?, ?, ?, ?, ?, ?)
            ''', (bet['game_id'], bet['date'], bet['pick'], bet['line'],
                  bet['odds'], bet['model_projection'], bet['edge'], BET_AMOUNT))
            if c.rowcount > 0:
                tot_recorded += 1
                print(f"   🎯 {bet['pick']} {bet['line']} ({bet['odds']:+g}) | "
                      f"proj {bet['model_projection']:.1f} | edge {bet['edge']:.1f} runs")
    
    conn.commit()
    conn.close()
    
    total = ml_recorded + sp_recorded + tot_recorded
    print(f"✅ Recorded {total} bets for {date_str} (ML:{ml_recorded} SPR:{sp_recorded} TOT:{tot_recorded})")
    return total


def evaluate_all():
    """Grade all ungraded bets."""
    conn = get_conn()
    c = conn.cursor()
    
    # --- Evaluate moneylines ---
    c.execute('''
        SELECT tb.id, tb.game_id, tb.date, tb.pick_team_id, tb.moneyline,
               g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.status
        FROM tracked_bets tb
        JOIN games g ON g.id = tb.game_id
        WHERE tb.won IS NULL AND g.status = 'final'
    ''')
    ml_evaluated = 0
    for row in c.fetchall():
        row = dict(row)
        home_won = row['home_score'] > row['away_score']
        picked_home = row['pick_team_id'] == row['home_team_id']
        won = 1 if (picked_home == home_won) else 0
        
        ml = row['moneyline']
        if won:
            profit = (ml / 100) * BET_AMOUNT if ml > 0 else (100 / abs(ml)) * BET_AMOUNT
        else:
            profit = -BET_AMOUNT
        
        c.execute('UPDATE tracked_bets SET won=?, profit=? WHERE id=?',
                  (won, round(profit, 2), row['id']))
        ml_evaluated += 1
        icon = '✅' if won else '❌'
        print(f"  {icon} ML: {'W' if won else 'L'} | ${profit:+.2f}")
    
    # --- Evaluate spreads & totals ---
    c.execute('''
        SELECT tb.id, tb.game_id, tb.date, tb.pick, tb.line, tb.odds, tb.bet_type,
               g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.status,
               ht.name as home_name, at.name as away_name
        FROM tracked_bets_spreads tb
        JOIN games g ON g.id = tb.game_id
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
            
            c.execute('UPDATE tracked_bets_spreads SET won=?, profit=? WHERE id=?',
                      (won, round(profit, 2), row['id']))
            sp_evaluated += 1
            icon = '✅' if won else '❌'
            print(f"  {icon} SPR: {row['pick']} {row['line']:+.1f} | actual margin {actual_margin:+d} | ${profit:+.2f}")
        
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
            
            c.execute('UPDATE tracked_bets_spreads SET won=?, profit=? WHERE id=?',
                      (won, round(profit, 2), row['id']))
            tot_evaluated += 1
            icon = '✅' if won else ('❌' if won == 0 else '➖')
            print(f"  {icon} TOT: {row['pick']} {row['line']} | actual {actual_total} | ${profit:+.2f}")
    
    conn.commit()
    conn.close()
    
    total = ml_evaluated + sp_evaluated + tot_evaluated
    print(f"\n✅ Evaluated {total} bets (ML:{ml_evaluated} SPR:{sp_evaluated} TOT:{tot_evaluated})")


def show_summary():
    """Show P&L summary."""
    conn = get_conn()
    c = conn.cursor()
    
    print("\n" + "="*60)
    print("📊 P&L SUMMARY")
    print("="*60)
    
    # Moneylines
    c.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) as losses,
               SUM(profit) as total_profit
        FROM tracked_bets WHERE won IS NOT NULL
    ''')
    row = dict(c.fetchone())
    if row['total']:
        pct = row['wins'] / row['total'] * 100 if row['total'] else 0
        print(f"\n💰 Moneylines: {row['wins']}-{row['losses']} ({pct:.1f}%) | "
              f"Profit: ${row['total_profit']:+.2f}")
    
    # Spreads
    c.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) as losses,
               SUM(profit) as total_profit
        FROM tracked_bets_spreads WHERE bet_type='spread' AND won IS NOT NULL
    ''')
    row = dict(c.fetchone())
    if row['total']:
        pct = row['wins'] / row['total'] * 100 if row['total'] else 0
        print(f"📊 Spreads: {row['wins']}-{row['losses']} ({pct:.1f}%) | "
              f"Profit: ${row['total_profit']:+.2f}")
    
    # Totals
    c.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) as losses,
               SUM(profit) as total_profit
        FROM tracked_bets_spreads WHERE bet_type='total' AND won IS NOT NULL
    ''')
    row = dict(c.fetchone())
    if row['total']:
        pct = row['wins'] / row['total'] * 100 if row['total'] else 0
        print(f"🎯 Totals: {row['wins']}-{row['losses']} ({pct:.1f}%) | "
              f"Profit: ${row['total_profit']:+.2f}")
    
    # Overall
    c.execute('''
        SELECT SUM(profit) FROM (
            SELECT profit FROM tracked_bets WHERE won IS NOT NULL
            UNION ALL
            SELECT profit FROM tracked_bets_spreads WHERE won IS NOT NULL
        )
    ''')
    total_profit = c.fetchone()[0] or 0
    print(f"\n💵 TOTAL PROFIT: ${total_profit:+.2f}")
    
    conn.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 backfill_bets.py DATE1 [DATE2 ...]")
        print("       python3 backfill_bets.py evaluate")
        print("       python3 backfill_bets.py summary")
        sys.exit(1)
    
    if sys.argv[1] == 'evaluate':
        evaluate_all()
        show_summary()
    elif sys.argv[1] == 'summary':
        show_summary()
    else:
        dates = sys.argv[1:]
        for date in dates:
            backfill_date(date)
        
        print("\n" + "="*60)
        print("Evaluating bets against final scores...")
        print("="*60)
        evaluate_all()
        show_summary()
