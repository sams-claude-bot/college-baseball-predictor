#!/usr/bin/env python3
"""
Comprehensive Box Score Collection

Strategy:
1. Get all games from ESPN scoreboard API
2. Try ESPN summary API for detailed stats (~30% have data)
3. Fall back to team site scraping for Power 4 teams
4. Track which games have stats vs which don't

Usage:
    python collect_all_boxscores.py 2026-02-13
    python collect_all_boxscores.py --today
    python collect_all_boxscores.py --recent 3
"""

import sys
import sqlite3
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Import modules
try:
    from scripts.scrape_team_boxscore import scrape_game, TEAM_SITES
    from scripts.espn_boxscore import collect_espn_boxscore, parse_boxscore, get_espn_boxscore
except ImportError:
    from scrape_team_boxscore import scrape_game, TEAM_SITES
    from espn_boxscore import collect_espn_boxscore, parse_boxscore, get_espn_boxscore


def get_espn_games(date_str):
    """Get all games for a date from ESPN API"""
    date_compact = date_str.replace('-', '')
    url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={date_compact}&limit=200'
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        data = resp.json()
        events = data.get('events', [])
        
        games = []
        for event in events:
            status = event.get('status', {}).get('type', {}).get('name', '')
            if status != 'STATUS_FINAL':
                continue
            
            game_id = event.get('id')
            name = event.get('shortName', '')
            
            # Get team IDs
            competitors = event.get('competitions', [{}])[0].get('competitors', [])
            home_team = away_team = None
            home_score = away_score = 0
            
            for comp in competitors:
                team_name = comp.get('team', {}).get('displayName', '')
                score = int(comp.get('score', 0))
                if comp.get('homeAway') == 'home':
                    home_team = team_name
                    home_score = score
                else:
                    away_team = team_name
                    away_score = score
            
            games.append({
                'espn_id': game_id,
                'name': name,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'date': date_str
            })
        
        return games
    except Exception as e:
        print(f"  Error fetching ESPN games: {e}")
        return []


def normalize_team_id(team_name):
    """Convert team display name to our team_id format"""
    if not team_name:
        return None
    
    # Common mappings
    mappings = {
        'Mississippi State Bulldogs': 'mississippi-state',
        'Ole Miss Rebels': 'ole-miss',
        'Texas A&M Aggies': 'texas-am',
        'South Carolina Gamecocks': 'south-carolina',
        'Georgia Tech Yellow Jackets': 'georgia-tech',
        'Florida State Seminoles': 'florida-state',
        'NC State Wolfpack': 'nc-state',
        'Boston College Eagles': 'boston-college',
        'Miami Hurricanes': 'miami-fl',
        'Virginia Tech Hokies': 'virginia-tech',
        'Wake Forest Demon Deacons': 'wake-forest',
        'Oklahoma State Cowboys': 'oklahoma-state',
        'Kansas State Wildcats': 'kansas-state',
        'West Virginia Mountaineers': 'west-virginia',
        'Texas Tech Red Raiders': 'texas-tech',
        'Arizona State Sun Devils': 'arizona-state',
        'Oregon State Beavers': 'oregon-state',
        'Penn State Nittany Lions': 'penn-state',
        'Ohio State Buckeyes': 'ohio-state',
        'Michigan State Spartans': 'michigan-state',
        'North Carolina Tar Heels': 'north-carolina',
    }
    
    for key, val in mappings.items():
        if key in team_name or team_name in key:
            return val
    
    # Simplified name extraction
    name = team_name.split(' ')[0].lower() if team_name else None
    return name


def collect_for_date(date_str, dry_run=False):
    """Collect all box scores for a date"""
    print(f"\n{'='*60}")
    print(f"Collecting box scores for {date_str}")
    print(f"{'='*60}")
    
    # Step 1: Get all completed games from ESPN
    print("\n📡 Fetching games from ESPN...")
    games = get_espn_games(date_str)
    print(f"   Found {len(games)} completed games")
    
    if not games:
        return {'total': 0, 'espn': 0, 'team_site': 0, 'missing': 0}
    
    stats = {'total': len(games), 'espn': 0, 'team_site': 0, 'missing': 0}
    
    for game in games:
        print(f"\n📋 {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
        
        # Step 2: Try ESPN detailed stats first (fastest, works for ~30%)
        try:
            if collect_espn_boxscore(game['espn_id'], dry_run=dry_run):
                stats['espn'] += 1
                continue
        except Exception as e:
            pass  # Fall through to team site
        
        # Step 3: Try team site scraping for Power 4 teams
        home_id = normalize_team_id(game['home_team'])
        away_id = normalize_team_id(game['away_team'])
        
        scraped = False
        for team_id in [home_id, away_id]:
            if team_id and team_id in TEAM_SITES:
                print(f"   → Trying {team_id} team site...")
                try:
                    if scrape_game(team_id, date_str, dry_run=dry_run):
                        scraped = True
                        break
                except Exception as e:
                    print(f"   ⚠ Error: {e}")
        
        if scraped:
            stats['team_site'] += 1
        else:
            print(f"   ⊘ No box score source found")
            stats['missing'] += 1
        
        time.sleep(0.3)  # Rate limiting
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Collection complete:")
    print(f"  Total games:     {stats['total']}")
    print(f"  ESPN box scores: {stats['espn']}")
    print(f"  Team site:       {stats['team_site']}")
    print(f"  No data:         {stats['missing']}")
    print(f"  Coverage:        {((stats['espn'] + stats['team_site']) / stats['total'] * 100):.1f}%")
    print(f"{'='*60}")
    
    return stats


def collect_recent(days=3, dry_run=False):
    """Collect box scores for recent days"""
    today = datetime.now()
    all_stats = {'total': 0, 'espn': 0, 'team_site': 0, 'missing': 0}
    
    for i in range(days):
        date = today - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        stats = collect_for_date(date_str, dry_run)
        
        for key in all_stats:
            all_stats[key] += stats.get(key, 0)
    
    return all_stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python collect_all_boxscores.py DATE")
        print("  python collect_all_boxscores.py --today")
        print("  python collect_all_boxscores.py --recent N")
        sys.exit(1)
    
    dry_run = '--dry' in sys.argv
    
    if sys.argv[1] == '--today':
        collect_for_date(datetime.now().strftime('%Y-%m-%d'), dry_run)
    elif sys.argv[1] == '--recent':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        collect_recent(days, dry_run)
    else:
        collect_for_date(sys.argv[1], dry_run)
