#!/usr/bin/env python3
"""
Scrape DraftKings NCAA Baseball odds from a browser snapshot.

This script is designed to be called from an OpenClaw cron job that:
1. Opens the DK page in the openclaw browser
2. Takes a snapshot
3. Calls this script with the snapshot data piped in

Standalone usage (reads from saved snapshot file):
    python3 scripts/scrape_dk_odds.py --file snapshot.txt

Or pipe snapshot text directly:
    cat snapshot.txt | python3 scripts/scrape_dk_odds.py
"""

import argparse
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "baseball.db"

# Import database-backed team resolver
import sys
sys.path.insert(0, str(Path(__file__).parent))
from team_resolver import resolve_team as db_resolve_team, add_alias

# DraftKings team name -> our team_id

# Team aliases are now stored in the database (team_aliases table).
# Use: python3 scripts/team_resolver.py --add <alias> <team_id> draftkings

def resolve_team(name):
    """Resolve a DraftKings team name to our team_id using database."""
    result = db_resolve_team(name)
    if result:
        return result
    # Fallback: slugify for unknown teams
    slug = name.lower().strip().replace(' ', '-').replace("'", '').replace('&', 'and')
    return slug


def parse_odds(text):
    """Parse odds string like '+105' or '−135' to integer."""
    text = text.replace('−', '-').replace('–', '-').replace('\u2212', '-')
    match = re.search(r'([+-]?\d+)', text)
    return int(match.group(1)) if match else None


def parse_snapshot(text):
    """Parse a browser snapshot of the DK NCAA baseball page into game data."""
    games = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for pattern: team link, "at", team link, then 6 buttons (spread, total, ML x2)
        # DK format: away_team "at" home_team
        if ' at ' in line or (i + 1 < len(lines) and lines[i + 1].strip() == 'at'):
            # Try to find the away and home team names
            # They appear as link text before/after "at"
            away_name = None
            home_name = None
            
            # Look back for away team link
            for j in range(max(0, i - 3), i + 1):
                l = lines[j].strip()
                if l.startswith('link "') and '/event/' in lines[min(j+1, len(lines)-1)]:
                    match = re.match(r'link "([^"]+)"', l)
                    if match:
                        away_name = match.group(1)
            
            # Look forward for home team
            for j in range(i, min(i + 5, len(lines))):
                l = lines[j].strip()
                if l.startswith('link "') and away_name and l != f'link "{away_name}"':
                    match = re.match(r'link "([^"]+)"', l)
                    if match and 'More Bets' not in match.group(1):
                        home_name = match.group(1)
                        break
            
            if not away_name or not home_name:
                i += 1
                continue
            
            # Now find the 6 buttons (away_spread, over, away_ml, home_spread, under, home_ml)
            buttons = []
            for j in range(i, min(i + 20, len(lines))):
                l = lines[j].strip()
                if l.startswith('button "') and ('−' in l or '+' in l or '-' in l):
                    match = re.match(r'button "([^"]+)"', l)
                    if match:
                        buttons.append(match.group(1))
                if len(buttons) >= 6:
                    break
            
            if len(buttons) >= 6:
                # Parse: away_spread, over, away_ml, home_spread, under, home_ml
                # e.g. "+1.5 −145", "O 13.5 −115", "+105", "-1.5 +114", "U 13.5 −115", "−135"
                game = {
                    'away': away_name,
                    'home': home_name,
                    'away_id': resolve_team(away_name),
                    'home_id': resolve_team(home_name),
                }
                
                # Away spread
                sp = re.match(r'([+-]?\d+\.?\d*)\s+([+-−–]\d+)', buttons[0].replace('−', '-').replace('–', '-'))
                if sp:
                    game['away_spread'] = float(sp.group(1))
                    game['away_spread_odds'] = int(sp.group(2))
                
                # Over
                ov = re.match(r'O\s+(\d+\.?\d*)\s+([+-−–]\d+)', buttons[1].replace('−', '-').replace('–', '-'))
                if ov:
                    game['over_under'] = float(ov.group(1))
                    game['over_odds'] = int(ov.group(2))
                
                # Away ML
                game['away_ml'] = parse_odds(buttons[2])
                
                # Home spread
                sp2 = re.match(r'([+-]?\d+\.?\d*)\s+([+-−–]\d+)', buttons[3].replace('−', '-').replace('–', '-'))
                if sp2:
                    game['home_spread'] = float(sp2.group(1))
                    game['home_spread_odds'] = int(sp2.group(2))
                
                # Under
                un = re.match(r'U\s+(\d+\.?\d*)\s+([+-−–]\d+)', buttons[4].replace('−', '-').replace('–', '-'))
                if un:
                    game['under_odds'] = int(un.group(2))
                
                # Home ML
                game['home_ml'] = parse_odds(buttons[5])
                
                games.append(game)
        
        i += 1
    
    return games


def load_to_db(games, date_str):
    """Load parsed games into the betting_lines table."""
    conn = sqlite3.connect(DB_PATH)
    added = 0
    updated = 0
    
    for g in games:
        game_id = f"{date_str}_{g['away_id']}_{g['home_id']}"
        
        existing = conn.execute("SELECT id FROM betting_lines WHERE game_id=?", (game_id,)).fetchone()
        
        if existing:
            conn.execute("""
                UPDATE betting_lines SET 
                    home_ml=?, away_ml=?, home_spread=?, home_spread_odds=?,
                    away_spread=?, away_spread_odds=?, over_under=?, over_odds=?, under_odds=?,
                    captured_at=CURRENT_TIMESTAMP
                WHERE game_id=?
            """, (g.get('home_ml'), g.get('away_ml'), g.get('home_spread'), g.get('home_spread_odds'),
                  g.get('away_spread'), g.get('away_spread_odds'), g.get('over_under'),
                  g.get('over_odds'), g.get('under_odds'), game_id))
            updated += 1
        else:
            conn.execute("""
                INSERT INTO betting_lines (game_id, date, home_team_id, away_team_id, book,
                    home_ml, away_ml, home_spread, home_spread_odds, away_spread, away_spread_odds,
                    over_under, over_odds, under_odds)
                VALUES (?, ?, ?, ?, 'draftkings', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, date_str, g['home_id'], g['away_id'],
                  g.get('home_ml'), g.get('away_ml'), g.get('home_spread'), g.get('home_spread_odds'),
                  g.get('away_spread'), g.get('away_spread_odds'), g.get('over_under'),
                  g.get('over_odds'), g.get('under_odds')))
            added += 1
        
        print(f"  {'ADD' if not existing else 'UPD'}: {g['away']} @ {g['home']} | "
              f"ML: {g.get('away_ml')}/{g.get('home_ml')} | O/U: {g.get('over_under')}")
    
    conn.commit()
    conn.close()
    
    print(f"\nDraftKings: {added} added, {updated} updated for {date_str}")
    return added + updated


def main():
    parser = argparse.ArgumentParser(description='Parse DraftKings NCAA baseball odds')
    parser.add_argument('--file', help='Read snapshot from file')
    parser.add_argument('--date', help='Date (YYYY-MM-DD), defaults to today')
    args = parser.parse_args()
    
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')
    
    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    
    games = parse_snapshot(text)
    
    if not games:
        print("No games found in snapshot!")
        return
    
    print(f"Parsed {len(games)} games from DraftKings snapshot")
    load_to_db(games, date_str)


if __name__ == '__main__':
    main()
