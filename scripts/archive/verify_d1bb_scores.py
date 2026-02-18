#!/usr/bin/env python3
"""
D1Baseball Score Verification Script
Compares D1Baseball scraped data against our database.
"""

import sqlite3
import re
import json
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

# Import database-backed team resolver
sys.path.insert(0, str(Path(__file__).parent))
from team_resolver import resolve_team as db_resolve_team

def get_db_games(date):
    """Get all games from our DB for a given date."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get games with team names
    cursor.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id, 
               g.home_score, g.away_score,
               ht.name as home_name, at.name as away_name
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.date = ? AND g.home_score IS NOT NULL
    """, (date,))
    
    games = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return games

def get_all_teams():
    """Get all team mappings from DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM teams")
    teams = {row[1].lower(): row[0] for row in cursor.fetchall()}
    # Also map by id
    cursor.execute("SELECT id, name FROM teams")
    teams_by_id = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return teams, teams_by_id

def parse_d1bb_text(text):
    """
    Parse D1Baseball text output to extract games.
    Returns list of dicts: {away_team, away_score, home_team, home_score}
    """
    games = []
    
    # Pattern to match game blocks
    # Format after FINAL:
    # [Rank]Away Team\n(Record)\nR [runs] H [hits] E [errors]\n[Rank]Home Team\n(Record)\n[runs] [hits] [errors]
    
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for FINAL or FINAL (X) pattern
        if line.startswith('FINAL'):
            # Skip Win Prob, Box Score, Recap lines
            j = i + 1
            while j < len(lines) and lines[j].strip() in ['Win Prob.', 'Box Score', 'Recap', '']:
                j += 1
            
            if j + 5 < len(lines):
                # Away team (may have rank prefix like "21Wake Forest")
                away_team_line = lines[j].strip()
                # Remove rank number prefix
                away_team = re.sub(r'^\d+', '', away_team_line).strip()
                
                # Record line (skip)
                j += 1
                
                # Runs/Hits/Errors line for away team: "R 2 H 7 E 1" or just numbers
                j += 1
                rhe_line = lines[j].strip()
                
                # Parse away team score
                rhe_match = re.search(r'R\s+(\d+)', rhe_line)
                if rhe_match:
                    away_score = int(rhe_match.group(1))
                else:
                    # Try just numbers
                    nums = re.findall(r'\d+', rhe_line)
                    if nums:
                        away_score = int(nums[0])
                    else:
                        i += 1
                        continue
                
                # Home team
                j += 1
                home_team_line = lines[j].strip()
                home_team = re.sub(r'^\d+', '', home_team_line).strip()
                
                # Record line (skip)
                j += 1
                
                # Home team scores (just numbers)
                j += 1
                home_rhe = lines[j].strip()
                nums = re.findall(r'\d+', home_rhe)
                if nums:
                    home_score = int(nums[0])
                else:
                    i += 1
                    continue
                
                games.append({
                    'away_team': away_team,
                    'away_score': away_score,
                    'home_team': home_team,
                    'home_score': home_score
                })
            
            i = j + 1
        else:
            i += 1
    
    return games

# Team name mapping from D1Baseball display names to our DB IDs
# Team mappings are now in the database (team_aliases table)
# The normalize_team_name() function uses db_resolve_team() from team_resolver.py


def normalize_team_name(name):
    """Normalize team name for matching using database resolver."""
    name = name.lower().strip()
    
    # Use database resolver
    result = db_resolve_team(name)
    if result:
        return result
    
    # Try without special characters
    name_clean = re.sub(r'[^\w\s]', '', name)
    result = db_resolve_team(name_clean)
    if result:
        return result
    
    return name

def compare_games(date, d1bb_games, db_games):
    """
    Compare D1Baseball games with our database games.
    Returns: (matches, mismatches, missing_from_db, extra_in_db)
    """
    matches = []
    mismatches = []
    missing_from_db = []
    
    # Create lookup for DB games
    db_lookup = {}
    for g in db_games:
        key = (g['home_team_id'], g['away_team_id'])
        db_lookup[key] = g
    
    matched_db_ids = set()
    
    for d1g in d1bb_games:
        home_id = normalize_team_name(d1g['home_team'])
        away_id = normalize_team_name(d1g['away_team'])
        
        key = (home_id, away_id)
        
        if key in db_lookup:
            dbg = db_lookup[key]
            matched_db_ids.add(dbg['id'])
            
            if dbg['home_score'] == d1g['home_score'] and dbg['away_score'] == d1g['away_score']:
                matches.append({
                    'd1bb': d1g,
                    'db': dbg,
                    'status': 'match'
                })
            else:
                mismatches.append({
                    'd1bb': d1g,
                    'db': dbg,
                    'status': 'mismatch'
                })
        else:
            missing_from_db.append(d1g)
    
    # Find games in DB but not in D1BB
    extra_in_db = [g for g in db_games if g['id'] not in matched_db_ids]
    
    return matches, mismatches, missing_from_db, extra_in_db


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_d1bb_scores.py <d1bb_text_file> <date>")
        print("Example: python verify_d1bb_scores.py feb13.txt 2026-02-13")
        sys.exit(1)
    
    text_file = sys.argv[1]
    date = sys.argv[2]
    
    with open(text_file, 'r') as f:
        text = f.read()
    
    d1bb_games = parse_d1bb_text(text)
    db_games = get_db_games(date)
    
    print(f"\n=== {date} ===")
    print(f"D1Baseball games found: {len(d1bb_games)}")
    print(f"Database games found: {len(db_games)}")
    
    matches, mismatches, missing, extra = compare_games(date, d1bb_games, db_games)
    
    print(f"\nMatches: {len(matches)}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Missing from DB: {len(missing)}")
    print(f"Extra in DB: {len(extra)}")
    
    if mismatches:
        print("\n--- MISMATCHES ---")
        for m in mismatches:
            d1 = m['d1bb']
            db = m['db']
            print(f"  {d1['away_team']} @ {d1['home_team']}")
            print(f"    D1BB: {d1['away_score']}-{d1['home_score']}")
            print(f"    DB:   {db['away_score']}-{db['home_score']}")
    
    if missing:
        print("\n--- MISSING FROM DB ---")
        for g in missing[:20]:  # Show first 20
            print(f"  {g['away_team']} {g['away_score']} @ {g['home_team']} {g['home_score']}")
