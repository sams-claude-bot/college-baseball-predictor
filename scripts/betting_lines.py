#!/usr/bin/env python3
"""
Track betting lines and compare to model predictions

Find value: games where our models disagree most with DraftKings
"""

import sys
import json
import math
from datetime import datetime
from pathlib import Path

_scripts_dir = Path(__file__).parent
_models_dir = _scripts_dir.parent / "models"
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup
# sys.path.insert(0, str(_models_dir))  # Removed by cleanup

from scripts.database import get_connection
from compare_models import MODELS, normalize_team_id

def init_betting_table():
    """Create betting lines table"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            book TEXT DEFAULT 'draftkings',
            -- Moneyline (American odds)
            home_ml INTEGER,
            away_ml INTEGER,
            -- Run line
            home_spread REAL,
            home_spread_odds INTEGER,
            away_spread REAL,
            away_spread_odds INTEGER,
            -- Totals
            over_under REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            -- Timestamps
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (home_team_id) REFERENCES teams(id),
            FOREIGN KEY (away_team_id) REFERENCES teams(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Betting lines table ready")

def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def implied_prob_to_american(prob):
    """Convert probability to American odds"""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

def add_line(date, home_team, away_team, home_ml, away_ml, 
             home_spread=None, away_spread=None, over_under=None,
             home_spread_odds=-110, away_spread_odds=-110,
             over_odds=-110, under_odds=-110, book="draftkings"):
    """Add betting lines for a game"""
    init_betting_table()
    
    conn = get_connection()
    c = conn.cursor()
    
    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)
    game_id = f"{date}_{away_id}_{home_id}"
    
    # Default spread is -1.5/+1.5
    if home_spread is None:
        home_spread = -1.5
    if away_spread is None:
        away_spread = 1.5
    
    c.execute('''
        INSERT INTO betting_lines 
        (game_id, date, home_team_id, away_team_id, book,
         home_ml, away_ml, home_spread, home_spread_odds,
         away_spread, away_spread_odds, over_under, over_odds, under_odds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (game_id, date, home_id, away_id, book,
          home_ml, away_ml, home_spread, home_spread_odds,
          away_spread, away_spread_odds, over_under, over_odds, under_odds))
    
    conn.commit()
    conn.close()
    
    # Show comparison immediately
    print(f"\n✓ Added {book} lines for {away_team} @ {home_team}")
    compare_to_models(home_team, away_team, home_ml, away_ml, over_under)

def compare_to_models(home_team, away_team, home_ml, away_ml, over_under=None):
    """Compare betting lines to model predictions"""
    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)
    
    # Get implied probabilities from odds
    dk_home_prob = american_to_implied_prob(home_ml)
    dk_away_prob = american_to_implied_prob(away_ml)
    
    # Normalize (remove vig)
    total_prob = dk_home_prob + dk_away_prob
    dk_home_prob_fair = dk_home_prob / total_prob
    dk_away_prob_fair = dk_away_prob / total_prob
    
    print(f"\n{'='*70}")
    print(f"  VALUE FINDER: {away_team} @ {home_team}")
    print('='*70)
    
    print(f"\n📊 DRAFTKINGS LINES:")
    print(f"   {home_team}: {home_ml:+d} ({dk_home_prob_fair*100:.1f}% implied)")
    print(f"   {away_team}: {away_ml:+d} ({dk_away_prob_fair*100:.1f}% implied)")
    if over_under:
        print(f"   Total: {over_under}")
    
    print(f"\n🤖 MODEL PREDICTIONS:")
    print(f"{'Model':<15} {'Home Prob':<12} {'Edge':<10} {'Value?'}")
    print("-" * 55)
    
    best_edge = 0
    best_edge_pick = None
    
    for name, model in MODELS.items():
        try:
            pred = model.predict_game(home_id, away_id)
            model_home_prob = pred['home_win_probability']
            
            # Edge = our probability - their implied probability
            home_edge = (model_home_prob - dk_home_prob_fair) * 100
            away_edge = ((1 - model_home_prob) - dk_away_prob_fair) * 100
            
            if abs(home_edge) > abs(best_edge):
                best_edge = home_edge
                best_edge_pick = home_team if home_edge > 0 else away_team
            
            # Determine value
            if home_edge > 5:
                value = f"✅ {home_team} +{home_edge:.1f}%"
            elif home_edge < -5:
                value = f"✅ {away_team} +{abs(home_edge):.1f}%"
            else:
                value = "—"
            
            edge_str = f"{home_edge:+.1f}%" if home_edge != 0 else "0%"
            print(f"{name:<15} {model_home_prob*100:.1f}%        {edge_str:<10} {value}")
            
        except Exception as e:
            print(f"{name:<15} Error: {e}")
    
    # Totals comparison
    if over_under:
        print(f"\n📈 TOTALS:")
        print(f"   DraftKings O/U: {over_under}")
        
        for name, model in MODELS.items():
            try:
                pred = model.predict_game(home_id, away_id)
                proj_total = pred['projected_total']
                diff = proj_total - over_under
                
                if abs(diff) > 1:
                    lean = "OVER" if diff > 0 else "UNDER"
                    print(f"   {name}: {proj_total:.1f} ({lean} by {abs(diff):.1f})")
                else:
                    print(f"   {name}: {proj_total:.1f} (close)")
            except:
                pass
    
    # Summary
    print(f"\n{'='*70}")
    if abs(best_edge) >= 5:
        print(f"💰 BEST VALUE: {best_edge_pick} ({abs(best_edge):.1f}% edge)")
    else:
        print(f"📊 No significant edge found (largest: {abs(best_edge):.1f}%)")
    print()

def get_todays_value():
    """Show value picks for today's games with lines"""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT home_team_id, away_team_id, home_ml, away_ml, over_under
        FROM betting_lines
        WHERE date = ?
        ORDER BY captured_at DESC
    ''', (today,))
    
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        print(f"\n⚠️  No betting lines for {today}")
        print("Add lines with: python betting_lines.py add <date> <home> <away> <home_ml> <away_ml> [over_under]")
        return
    
    for row in rows:
        home_id, away_id, home_ml, away_ml, over_under = row
        compare_to_models(home_id, away_id, home_ml, away_ml, over_under)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python betting_lines.py today                    - Show today's value picks")
        print("  python betting_lines.py add <date> <home> <away> <home_ml> <away_ml> [over_under]")
        print("  python betting_lines.py compare <home> <away> <home_ml> <away_ml> [over_under]")
        print()
        print("Examples:")
        print("  python betting_lines.py add 2026-02-13 'Mississippi State' 'Hofstra' -450 +350 9.5")
        print("  python betting_lines.py compare 'Mississippi State' 'Hofstra' -450 +350")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "today":
        get_todays_value()
    
    elif cmd == "add":
        if len(sys.argv) < 7:
            print("Usage: python betting_lines.py add <date> <home> <away> <home_ml> <away_ml> [over_under]")
            return
        
        date = sys.argv[2]
        home = sys.argv[3]
        away = sys.argv[4]
        home_ml = int(sys.argv[5])
        away_ml = int(sys.argv[6])
        over_under = float(sys.argv[7]) if len(sys.argv) > 7 else None
        
        add_line(date, home, away, home_ml, away_ml, over_under=over_under)
    
    elif cmd == "compare":
        if len(sys.argv) < 6:
            print("Usage: python betting_lines.py compare <home> <away> <home_ml> <away_ml> [over_under]")
            return
        
        home = sys.argv[2]
        away = sys.argv[3]
        home_ml = int(sys.argv[4])
        away_ml = int(sys.argv[5])
        over_under = float(sys.argv[6]) if len(sys.argv) > 6 else None
        
        compare_to_models(home, away, home_ml, away_ml, over_under)
    
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
