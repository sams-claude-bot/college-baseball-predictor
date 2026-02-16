#!/usr/bin/env python3
"""
D1Baseball Score Verification Script
Compares D1Baseball scraped data against our database.
"""

import sqlite3
import re
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

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
TEAM_MAPPING = {
    # SEC
    'alabama': 'alabama',
    'arkansas': 'arkansas',
    'auburn': 'auburn',
    'florida': 'florida',
    'georgia': 'georgia',
    'kentucky': 'kentucky',
    'lsu': 'lsu',
    'mississippi state': 'mississippi-state',
    'missouri': 'missouri',
    'oklahoma': 'oklahoma',
    'ole miss': 'ole-miss',
    'south carolina': 'south-carolina',
    'tennessee': 'tennessee',
    'texas': 'texas',
    'texas a&m': 'texas-am',
    'vanderbilt': 'vanderbilt',
    
    # ACC
    'boston college': 'boston-college',
    'clemson': 'clemson',
    'duke': 'duke',
    'florida state': 'florida-state',
    'georgia tech': 'georgia-tech',
    'louisville': 'louisville',
    'miami': 'miami-fl',
    'nc state': 'nc-state',
    'north carolina': 'north-carolina',
    'notre dame': 'notre-dame',
    'pittsburgh': 'pittsburgh',
    'virginia': 'virginia',
    'virginia tech': 'virginia-tech',
    'wake forest': 'wake-forest',
    'california': 'california',
    'stanford': 'stanford',
    
    # Big 12
    'arizona': 'arizona',
    'arizona state': 'arizona-state',
    'baylor': 'baylor',
    'byu': 'byu',
    'cincinnati': 'cincinnati',
    'colorado': 'colorado',
    'houston': 'houston',
    'iowa state': 'iowa-state',
    'kansas': 'kansas',
    'kansas state': 'kansas-state',
    'oklahoma state': 'oklahoma-state',
    'tcu': 'tcu',
    'texas tech': 'texas-tech',
    'ucf': 'ucf',
    'utah': 'utah',
    'west virginia': 'west-virginia',
    
    # Big Ten
    'illinois': 'illinois',
    'indiana': 'indiana',
    'iowa': 'iowa',
    'maryland': 'maryland',
    'michigan': 'michigan',
    'michigan state': 'michigan-state',
    'minnesota': 'minnesota',
    'nebraska': 'nebraska',
    'northwestern': 'northwestern',
    'ohio state': 'ohio-state',
    'oregon': 'oregon',
    'oregon state': 'oregon-state',
    'penn state': 'penn-state',
    'purdue': 'purdue',
    'rutgers': 'rutgers',
    'ucla': 'ucla',
    'usc': 'usc',
    'southern california': 'usc',
    'washington': 'washington',
    'wisconsin': 'wisconsin',
    
    # Sun Belt
    'appalachian state': 'appalachian-state',
    'coastal carolina': 'coastal-carolina',
    'georgia southern': 'georgia-southern',
    'georgia state': 'georgia-state',
    'james madison': 'james-madison',
    'louisiana': 'louisiana',
    'marshall': 'marshall',
    'old dominion': 'old-dominion',
    'south alabama': 'south-alabama',
    'southern miss': 'southern-miss',
    'texas state': 'texas-state',
    'troy': 'troy',
    'ul monroe': 'ul-monroe',
    
    # American
    'charlotte': 'charlotte',
    'east carolina': 'east-carolina',
    'florida atlantic': 'florida-atlantic',
    'memphis': 'memphis',
    'rice': 'rice',
    'south florida': 'south-florida',
    'tulane': 'tulane',
    'uab': 'uab',
    'utsa': 'utsa',
    'wichita state': 'wichita-state',
    
    # Others - will add as needed
    'air force': 'air-force',
    'army': 'army',
    'ball state': 'ball-state',
    'bowling green': 'bowling-green',
    'bryant': 'bryant',
    'butler': 'butler',
    'cal poly': 'cal-poly',
    'cal state fullerton': 'cal-state-fullerton',
    'cal state northridge': 'cal-state-northridge',
    'campbell': 'campbell',
    'central michigan': 'central-michigan',
    'college of charleston': 'college-of-charleston',
    'connecticut': 'connecticut',
    'creighton': 'creighton',
    'dallas baptist': 'dallas-baptist',
    'davidson': 'davidson',
    'dayton': 'dayton',
    'delaware': 'delaware',
    'elon': 'elon',
    'evansville': 'evansville',
    'fairfield': 'fairfield',
    'florida gulf coast': 'fgcu',
    'fgcu': 'fgcu',
    'fordham': 'fordham',
    'fresno state': 'fresno-state',
    'gardner-webb': 'gardner-webb',
    'george mason': 'george-mason',
    'george washington': 'george-washington',
    'georgetown': 'georgetown',
    'gonzaga': 'gonzaga',
    'grambling': 'grambling-state',
    'grand canyon': 'grand-canyon',
    'hawaii': 'hawaii',
    'high point': 'high-point',
    'hofstra': 'hofstra',
    'holy cross': 'holy-cross',
    'houston christian': 'houston-christian',
    'incarnate word': 'incarnate-word',
    'iona': 'iona',
    'jacksonville': 'jacksonville',
    'jacksonville state': 'jacksonville-state',
    'kennesaw state': 'kennesaw-state',
    'kent state': 'kent-state',
    'la salle': 'la-salle',
    'lamar': 'lamar',
    'lehigh': 'lehigh',
    'liberty': 'liberty',
    'lipscomb': 'lipscomb',
    'little rock': 'little-rock',
    'long beach state': 'long-beach-state',
    'longwood': 'longwood',
    'louisiana tech': 'louisiana-tech',
    'loyola marymount': 'loyola-marymount',
    'maine': 'maine',
    'marist': 'marist',
    'mcneese': 'mcneese',
    'mercer': 'mercer',
    'miami (oh)': 'miami-oh',
    'middle tennessee': 'middle-tennessee',
    'milwaukee': 'milwaukee',
    'missouri state': 'missouri-state',
    'monmouth': 'monmouth',
    'morehead state': 'morehead-state',
    'murray state': 'murray-state',
    'navy': 'navy',
    'nevada': 'nevada',
    'new jersey tech': 'njit',
    'njit': 'njit',
    'new mexico': 'new-mexico',
    'new mexico state': 'new-mexico-state',
    'new orleans': 'new-orleans',
    'nicholls': 'nicholls',
    'north alabama': 'north-alabama',
    'north dakota state': 'north-dakota-state',
    'north florida': 'north-florida',
    'northern colorado': 'northern-colorado',
    'northern illinois': 'northern-illinois',
    'omaha': 'omaha',
    'oral roberts': 'oral-roberts',
    'pacific': 'pacific',
    'pepperdine': 'pepperdine',
    'portland': 'portland',
    'prairie view': 'prairie-view',
    'presbyterian': 'presbyterian',
    'queens (nc)': 'queens',
    'radford': 'radford',
    'rhode island': 'rhode-island',
    'richmond': 'richmond',
    'rider': 'rider',
    'sacramento state': 'sacramento-state',
    'sacred heart': 'sacred-heart',
    'saint joseph\'s': 'saint-josephs',
    'saint louis': 'saint-louis',
    'saint mary\'s': 'saint-marys',
    'saint mary\'s (ca)': 'saint-marys',
    'sam houston state': 'sam-houston',
    'samford': 'samford',
    'san diego': 'san-diego',
    'san diego state': 'san-diego-state',
    'san francisco': 'san-francisco',
    'san jose state': 'san-jose-state',
    'santa clara': 'santa-clara',
    'seattle': 'seattle',
    'seton hall': 'seton-hall',
    'siena': 'siena',
    'siu edwardsville': 'siue',
    'siue': 'siue',
    'south dakota state': 'south-dakota-state',
    'southeast missouri': 'southeast-missouri',
    'southeastern louisiana': 'southeastern-louisiana',
    'southern': 'southern',
    'southern illinois': 'southern-illinois',
    'southern indiana': 'southern-indiana',
    'st. bonaventure': 'st-bonaventure',
    'st. john\'s': 'st-johns',
    'stetson': 'stetson',
    'stony brook': 'stony-brook',
    'tarleton state': 'tarleton-state',
    'tennessee tech': 'tennessee-tech',
    'tennessee-martin': 'ut-martin',
    'the citadel': 'citadel',
    'toledo': 'toledo',
    'towson': 'towson',
    'uc davis': 'uc-davis',
    'uc irvine': 'uc-irvine',
    'uc riverside': 'uc-riverside',
    'uc san diego': 'uc-san-diego',
    'uc santa barbara': 'uc-santa-barbara',
    'umass': 'umass',
    'umass lowell': 'umass-lowell',
    'umes': 'maryland-eastern-shore',
    'unc asheville': 'unc-asheville',
    'unc greensboro': 'unc-greensboro',
    'unc wilmington': 'unc-wilmington',
    'unlv': 'unlv',
    'usc upstate': 'usc-upstate',
    'ut rio grande valley': 'utrgv',
    'utah tech': 'utah-tech',
    'utah valley': 'utah-valley',
    'valparaiso': 'valparaiso',
    'vcu': 'vcu',
    'villanova': 'villanova',
    'wagner': 'wagner',
    'western carolina': 'western-carolina',
    'western illinois': 'western-illinois',
    'western kentucky': 'western-kentucky',
    'western michigan': 'western-michigan',
    'william & mary': 'william-mary',
    'winthrop': 'winthrop',
    'wofford': 'wofford',
    'wright state': 'wright-state',
    'xavier': 'xavier',
    'youngstown state': 'youngstown-state',
    'akron': 'akron',
    'alabama state': 'alabama-state',
    'alcorn state': 'alcorn-state',
    'austin peay': 'austin-peay',
    'bellarmine': 'bellarmine',
    'bucknell': 'bucknell',
    'california baptist': 'california-baptist',
    'central arkansas': 'central-arkansas',
    'charleston southern': 'charleston-southern',
    'coppin state': 'coppin-state',
    'csu bakersfield': 'csu-bakersfield',
    'eastern kentucky': 'eastern-kentucky',
    'florida a&m': 'florida-am',
    'florida international': 'fiu',
    'jackson state': 'jackson-state',
    'texas a&m-corpus christi': 'texas-am-corpus-christi',
    'texas southern': 'texas-southern',
}

def normalize_team_name(name):
    """Normalize team name for matching."""
    name = name.lower().strip()
    # Try direct mapping first
    if name in TEAM_MAPPING:
        return TEAM_MAPPING[name]
    # Try without special characters
    name_clean = re.sub(r'[^\w\s]', '', name)
    if name_clean in TEAM_MAPPING:
        return TEAM_MAPPING[name_clean]
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
