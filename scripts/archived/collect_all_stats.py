#!/usr/bin/env python3
"""
Multi-source stats collection pipeline for college baseball

Sources (in priority order):
1. ESPN Scoreboard - Daily scores, rankings, live games
2. NCAA.com Stats - Top 10 team/individual stats (BA, ERA)
3. D1Baseball - Top 25 rankings

This script tries multiple sources and aggregates data into the SQLite database.
"""

import re
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup path
BASE_DIR = Path(__file__).parent.parent
# sys.path.insert(0, str(BASE_DIR / "scripts"))  # Removed by cleanup

from scripts.database import (
    get_connection, add_game, add_team, add_ranking, 
    init_rankings_table, init_database
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - web fetching disabled")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# ESPN team ID mappings for SEC teams
ESPN_TEAM_IDS = {
    'alabama': 333,
    'arkansas': 8,
    'auburn': 2,
    'florida': 57,
    'georgia': 61,
    'kentucky': 96,
    'lsu': 99,
    'mississippi-state': 344,
    'missouri': 142,
    'ole-miss': 145,
    'south-carolina': 2579,
    'tennessee': 2633,
    'texas': 251,
    'texas-a&m': 245,
    'vanderbilt': 238,
    'oklahoma': 201,
}

# Team name normalizations
TEAM_ALIASES = {
    'miss state': 'mississippi-state',
    'mississippi state': 'mississippi-state',
    'mississippi st.': 'mississippi-state',
    'ole miss': 'ole-miss',
    'texas a&m': 'texas-a&m',
    'south carolina': 'south-carolina',
    'nc state': 'nc-state',
    'georgia tech': 'georgia-tech',
    'florida st.': 'florida-state',
    'florida state': 'florida-state',
    'oregon st.': 'oregon-state',
    'oregon state': 'oregon-state',
    'wake forest': 'wake-forest',
    'virginia tech': 'virginia-tech',
    'east carolina': 'east-carolina',
    'coastal caro.': 'coastal-carolina',
    'coastal carolina': 'coastal-carolina',
    'southern miss.': 'southern-miss',
    'southern miss': 'southern-miss',
    'arizona st.': 'arizona-state',
    'arizona state': 'arizona-state',
    'miami (fl)': 'miami',
    'central conn. st.': 'central-connecticut',
    'usc upstate': 'usc-upstate',
    'unc wilmington': 'unc-wilmington',
    'uncw': 'unc-wilmington',
    'unc greensboro': 'unc-greensboro',
    'unc asheville': 'unc-asheville',
    'fgcu': 'florida-gulf-coast',
    'louisiana': 'louisiana-lafayette',
    'ul monroe': 'louisiana-monroe',
}


def normalize_team_id(name):
    """Convert team name to standardized team_id format"""
    if not name:
        return None
    lower = name.lower().strip()
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]
    # Remove parentheses, periods, replace spaces with dashes
    result = lower.replace('(', '').replace(')', '').replace('.', '')
    result = re.sub(r'\s+', '-', result)
    return result


def fetch_url(url, timeout=15):
    """Fetch URL with error handling"""
    if not HAS_REQUESTS:
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        logger.warning(f"HTTP {resp.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return None


# ============================================================
# ESPN SCOREBOARD COLLECTION
# ============================================================

def parse_espn_scoreboard(html):
    """
    Parse ESPN college baseball scoreboard for game data.
    Returns list of game dicts with teams, scores, rankings.
    
    ESPN pages are JavaScript-rendered, so raw HTML parsing is limited.
    This works best with web_fetch extracted text content.
    """
    games = []
    if not html:
        return games
    
    # Skip BeautifulSoup for raw HTML - it's too slow on 800KB+ pages
    # and ESPN data is in JavaScript anyway
    
    # For web_fetch extracted text content, use line-based parsing
    lines = html.split('\n')
    current_game = {}
    
    # Pattern: ranked team like "21Wake Forest(0-0)" or "- 21Wake Forest(0-0)"
    team_pattern = re.compile(r'^-?\s*(\d+)?([A-Za-z][A-Za-z &\'\.\-]+?)(?:\((\d+-\d+)\))?(\d*)$')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for game blocks starting with "R" or "- " team entries  
        if line == 'R' and i+2 < len(lines):
            # Start of a game block - look for team lines after R H E
            i += 3
            current_game = {'teams': [], 'tournament': None}
            continue
        
        # Check for tournament name
        if any(x in line for x in ['Challenge', 'Classic', 'Invitational', 'Showdown', 'Series']):
            if current_game:
                current_game['tournament'] = line
            continue
        
        # Parse team lines - handle "- 21Wake Forest(0-0)" format from web_fetch
        original_line = line
        if line.startswith('- '):
            line = line[2:].strip()
        
        match = team_pattern.match(line)
        if match:
            rank = int(match.group(1)) if match.group(1) else None
            team_name = match.group(2).strip()
            record = match.group(3)
            score_str = match.group(4)
            
            team_data = {
                'name': team_name,
                'rank': rank,
                'record': record,
                'score': int(score_str[0]) if score_str and len(score_str) >= 1 else None
            }
            
            if 'teams' not in current_game:
                current_game = {'teams': [], 'tournament': None}
            
            current_game['teams'].append(team_data)
            
            # If we have 2 teams, game is complete
            if len(current_game['teams']) == 2:
                games.append(current_game.copy())
                current_game = {'teams': [], 'tournament': None}
        
        i += 1
    
    return games


def collect_espn_scoreboard():
    """
    Collect today's scores from ESPN college baseball scoreboard.
    
    Note: ESPN pages are JavaScript-rendered and don't parse well with requests.
    This collector works best when HTML is pre-extracted via web_fetch or browser.
    Returns minimal data from raw HTML parsing.
    """
    url = "https://www.espn.com/college-baseball/scoreboard"
    logger.info(f"Fetching ESPN scoreboard: {url}")
    
    # Use short timeout since ESPN parsing often fails
    html = fetch_url(url, timeout=8)
    if not html:
        logger.warning("ESPN scoreboard fetch failed or timed out")
        return {'success': False, 'error': 'Failed to fetch ESPN scoreboard', 'games': []}
    
    # ESPN HTML is mostly JavaScript-rendered, but try to extract any available data
    games = parse_espn_scoreboard(html)
    
    results = {
        'success': True,
        'source': 'espn',
        'games_found': len(games),
        'games': [],
        'rankings_found': [],
        'note': 'ESPN data limited - page is JavaScript-rendered'
    }
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    for game in games:
        if len(game.get('teams', [])) != 2:
            continue
        
        away_team = game['teams'][0]
        home_team = game['teams'][1]
        
        game_data = {
            'date': today,
            'away_team': away_team['name'],
            'home_team': home_team['name'],
            'away_team_id': normalize_team_id(away_team['name']),
            'home_team_id': normalize_team_id(home_team['name']),
            'away_score': away_team.get('score'),
            'home_score': home_team.get('score'),
            'away_rank': away_team.get('rank'),
            'home_rank': home_team.get('rank'),
            'tournament': game.get('tournament'),
            'status': 'final' if away_team.get('score') is not None else 'scheduled'
        }
        results['games'].append(game_data)
        
        # Track rankings found
        for team_data in [away_team, home_team]:
            if team_data.get('rank'):
                results['rankings_found'].append({
                    'team': team_data['name'],
                    'rank': team_data['rank']
                })
    
    logger.info(f"ESPN: Found {len(results['games'])} games, {len(results['rankings_found'])} ranked teams")
    if not games:
        logger.info("ESPN parsing returned no games - this is expected (JS-rendered page)")
    
    return results


def save_espn_games(games_data):
    """Save ESPN games to database"""
    if not games_data.get('success'):
        return 0
    
    saved = 0
    for game in games_data.get('games', []):
        if game.get('status') == 'final' and game.get('home_score') is not None:
            try:
                # Ensure teams exist
                add_team(game['home_team_id'], game['home_team'])
                add_team(game['away_team_id'], game['away_team'])
                
                # Add the game
                add_game(
                    date=game['date'],
                    home_team_id=game['home_team_id'],
                    away_team_id=game['away_team_id'],
                    home_score=game['home_score'],
                    away_score=game['away_score'],
                    tournament_id=None,
                    is_neutral_site=bool(game.get('tournament')),
                    notes=game.get('tournament')
                )
                saved += 1
            except Exception as e:
                logger.warning(f"Failed to save game: {e}")
    
    logger.info(f"Saved {saved} completed games from ESPN")
    return saved


# ============================================================
# NCAA.COM STATS COLLECTION
# ============================================================

def parse_ncaa_stats(html):
    """
    Parse NCAA.com main stats page for top 10 leaders.
    Returns dict with individual and team stats.
    Uses BeautifulSoup for robust HTML parsing.
    """
    results = {
        'individual_batting': [],
        'individual_pitching': [],
        'team_batting': [],
        'team_pitching': []
    }
    
    if not html:
        return results
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Parse HTML tables - NCAA uses tables for stats
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            # Determine table type from headers
            header_row = table.find('thead')
            is_individual = False
            is_team = False
            stat_type = None  # 'ba' or 'era'
            
            if header_row:
                headers = header_row.get_text().lower()
                is_individual = 'name' in headers or 'player' in headers
                is_team = 'team' in headers and 'name' not in headers
                if 'ba' in headers or 'avg' in headers:
                    stat_type = 'ba'
                elif 'era' in headers:
                    stat_type = 'era'
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                
                # Try to extract rank, name/team, and stat
                # Typical structure: Rank | Name | Team | Stat or Rank | Team | Stat
                cell_texts = [c.get_text(strip=True) for c in cells]
                
                # Look for rank in first cell
                rank = None
                if cell_texts and re.match(r'^\d+$', cell_texts[0]):
                    rank = int(cell_texts[0])
                
                if not rank or rank > 10:
                    continue
                
                # Extract team from link
                team_link = row.find('a', href=re.compile(r'/schools/'))
                team_name = team_link.get_text(strip=True) if team_link else None
                
                # Extract stat value (last decimal number)
                stat_value = None
                for cell_text in reversed(cell_texts):
                    if re.match(r'^\d*\.\d+$', cell_text):
                        stat_value = float(cell_text)
                        break
                
                if not stat_value or not team_name:
                    continue
                
                # Determine stat type from value if not from headers
                if not stat_type:
                    if stat_value > 0.2 and stat_value < 0.6:
                        stat_type = 'ba'
                    elif stat_value < 15:
                        stat_type = 'era'
                
                # Individual stats have player name
                player_links = row.find_all('a')
                player_name = None
                for link in player_links:
                    href = link.get('href', '')
                    if '/schools/' not in href and link.get_text(strip=True):
                        text = link.get_text(strip=True)
                        if len(text) > 2 and ' ' in text:  # Likely a player name
                            player_name = text
                            break
                
                if player_name:
                    # Individual stat
                    if stat_type == 'ba':
                        results['individual_batting'].append({
                            'rank': rank,
                            'name': player_name,
                            'team': team_name,
                            'team_id': normalize_team_id(team_name),
                            'batting_avg': stat_value
                        })
                    elif stat_type == 'era':
                        results['individual_pitching'].append({
                            'rank': rank,
                            'name': player_name,
                            'team': team_name,
                            'team_id': normalize_team_id(team_name),
                            'era': stat_value
                        })
                else:
                    # Team stat
                    if stat_type == 'ba':
                        results['team_batting'].append({
                            'rank': rank,
                            'team': team_name,
                            'team_id': normalize_team_id(team_name),
                            'batting_avg': stat_value
                        })
                    elif stat_type == 'era':
                        results['team_pitching'].append({
                            'rank': rank,
                            'team': team_name,
                            'team_id': normalize_team_id(team_name),
                            'era': stat_value
                        })
    
    except ImportError:
        logger.warning("BeautifulSoup not available, using regex fallback")
        # Fallback: Use regex to extract team/ERA pairs
        team_era = re.findall(
            r'<a[^>]*href="/schools/([^"]+)"[^>]*>([^<]+)</a></td>[^<]*<td[^>]*>(\d+\.\d+)</td>',
            html
        )
        for i, (slug, name, stat) in enumerate(team_era[:20], 1):
            stat_value = float(stat)
            if stat_value < 1:  # BA
                if i <= 10:
                    results['team_batting'].append({
                        'rank': i,
                        'team': name,
                        'team_id': normalize_team_id(name),
                        'batting_avg': stat_value
                    })
            else:  # ERA
                if i <= 10:
                    results['team_pitching'].append({
                        'rank': i,
                        'team': name,
                        'team_id': normalize_team_id(name),
                        'era': stat_value
                    })
    
    return results


def collect_ncaa_stats():
    """Collect top 10 stats from NCAA.com main stats page"""
    url = "https://www.ncaa.com/stats/baseball/d1"
    logger.info(f"Fetching NCAA stats: {url}")
    
    html = fetch_url(url)
    if not html:
        return {'success': False, 'error': 'Failed to fetch NCAA stats page'}
    
    stats = parse_ncaa_stats(html)
    
    results = {
        'success': True,
        'source': 'ncaa',
        'individual_batting': stats['individual_batting'],
        'individual_pitching': stats['individual_pitching'],
        'team_batting': stats['team_batting'],
        'team_pitching': stats['team_pitching'],
        'total_stats': (
            len(stats['individual_batting']) + 
            len(stats['individual_pitching']) +
            len(stats['team_batting']) + 
            len(stats['team_pitching'])
        )
    }
    
    logger.info(f"NCAA: Found {results['total_stats']} stat entries")
    return results


def save_ncaa_stats(stats_data):
    """Save NCAA stats to database"""
    if not stats_data.get('success'):
        return 0
    
    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    c = conn.cursor()
    
    # Ensure tables exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS ncaa_team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_id TEXT,
            stat_category TEXT NOT NULL,
            stat_value REAL,
            rank INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS ncaa_individual_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            team_name TEXT NOT NULL,
            team_id TEXT,
            stat_category TEXT NOT NULL,
            stat_value REAL,
            rank INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    saved = 0
    
    # Save team batting
    for entry in stats_data.get('team_batting', []):
        c.execute('''
            INSERT INTO ncaa_team_stats (team_name, team_id, stat_category, stat_value, rank, date)
            VALUES (?, ?, 'batting_avg', ?, ?, ?)
        ''', (entry['team'], entry['team_id'], entry['batting_avg'], entry['rank'], today))
        saved += 1
    
    # Save team pitching
    for entry in stats_data.get('team_pitching', []):
        c.execute('''
            INSERT INTO ncaa_team_stats (team_name, team_id, stat_category, stat_value, rank, date)
            VALUES (?, ?, 'era', ?, ?, ?)
        ''', (entry['team'], entry['team_id'], entry['era'], entry['rank'], today))
        saved += 1
    
    # Save individual batting
    for entry in stats_data.get('individual_batting', []):
        c.execute('''
            INSERT INTO ncaa_individual_stats (player_name, team_name, team_id, stat_category, stat_value, rank, date)
            VALUES (?, ?, ?, 'batting_avg', ?, ?, ?)
        ''', (entry['name'], entry['team'], entry['team_id'], entry['batting_avg'], entry['rank'], today))
        saved += 1
    
    # Save individual pitching
    for entry in stats_data.get('individual_pitching', []):
        c.execute('''
            INSERT INTO ncaa_individual_stats (player_name, team_name, team_id, stat_category, stat_value, rank, date)
            VALUES (?, ?, ?, 'era', ?, ?, ?)
        ''', (entry['name'], entry['team'], entry['team_id'], entry['era'], entry['rank'], today))
        saved += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved {saved} NCAA stat entries")
    return saved


# ============================================================
# D1BASEBALL RANKINGS COLLECTION
# ============================================================

def parse_d1baseball_rankings(html):
    """Parse D1Baseball rankings page for Top 25"""
    rankings = []
    
    if not html:
        return rankings
    
    # Try BeautifulSoup for raw HTML first
    try:
        from bs4 import BeautifulSoup
        
        if '<html' in html.lower() or '<table' in html.lower():
            soup = BeautifulSoup(html, 'html.parser')
            
            # D1Baseball has team links in order /team/slug/
            team_links = soup.find_all('a', href=re.compile(r'^/team/[a-z]+/?$'))
            
            # Filter to only ranking table links (first 25 unique teams)
            seen = set()
            for link in team_links:
                team_name = link.get_text(strip=True)
                href = link.get('href', '')
                slug_match = re.search(r'/team/([^/]+)', href)
                
                if team_name and slug_match and team_name not in seen and len(rankings) < 25:
                    seen.add(team_name)
                    rankings.append({
                        'rank': len(rankings) + 1,
                        'team': team_name,
                        'team_id': normalize_team_id(team_name),
                        'd1baseball_slug': slug_match.group(1)
                    })
            
            if rankings:
                return rankings
    except ImportError:
        pass
    
    # Fallback to text/markdown parsing (for web_fetch extracted content)
    lines = html.split('\n')
    current_rank = None
    
    for line in lines:
        line = line.strip()
        
        # Look for rank numbers 1-25
        if re.match(r'^([1-9]|1[0-9]|2[0-5])$', line):
            current_rank = int(line)
            continue
        
        # Look for team links like [UCLA](/team/ucla/)
        team_match = re.match(r'\[([^\]]+)\]\(/team/([^/]+)/?\)', line)
        if team_match and current_rank:
            team_name = team_match.group(1)
            team_slug = team_match.group(2)
            
            rankings.append({
                'rank': current_rank,
                'team': team_name,
                'team_id': normalize_team_id(team_name),
                'd1baseball_slug': team_slug
            })
            current_rank = None
    
    return rankings


def collect_d1baseball_rankings():
    """Collect Top 25 rankings from D1Baseball"""
    url = "https://d1baseball.com/rankings/"
    logger.info(f"Fetching D1Baseball rankings: {url}")
    
    html = fetch_url(url)
    if not html:
        return {'success': False, 'error': 'Failed to fetch D1Baseball rankings'}
    
    rankings = parse_d1baseball_rankings(html)
    
    results = {
        'success': True,
        'source': 'd1baseball',
        'rankings': rankings,
        'count': len(rankings)
    }
    
    logger.info(f"D1Baseball: Found {len(rankings)} ranked teams")
    return results


def save_d1baseball_rankings(rankings_data):
    """Save D1Baseball rankings to database"""
    if not rankings_data.get('success'):
        return 0
    
    init_rankings_table()
    today = datetime.now().strftime('%Y-%m-%d')
    saved = 0
    
    for entry in rankings_data.get('rankings', []):
        try:
            # Add team if not exists
            add_team(
                team_id=entry['team_id'],
                name=entry['team'],
                preseason_rank=entry['rank']
            )
            
            # Add ranking
            add_ranking(
                team_id=entry['team_id'],
                rank=entry['rank'],
                poll='d1baseball',
                date=today
            )
            saved += 1
        except Exception as e:
            logger.warning(f"Failed to save ranking for {entry['team']}: {e}")
    
    logger.info(f"Saved {saved} rankings from D1Baseball")
    return saved


# ============================================================
# MAIN COLLECTION FUNCTIONS
# ============================================================

def collect_all(save_to_db=True):
    """
    Run all collection sources and optionally save to database.
    Returns comprehensive results dict.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'sources': {},
        'summary': {
            'games_found': 0,
            'games_saved': 0,
            'stats_found': 0,
            'stats_saved': 0,
            'rankings_found': 0,
            'rankings_saved': 0
        },
        'errors': []
    }
    
    # 1. ESPN Scoreboard
    try:
        espn_data = collect_espn_scoreboard()
        results['sources']['espn'] = espn_data
        results['summary']['games_found'] = espn_data.get('games_found', 0)
        
        if save_to_db and espn_data.get('success'):
            saved = save_espn_games(espn_data)
            results['summary']['games_saved'] = saved
    except Exception as e:
        logger.error(f"ESPN collection failed: {e}")
        results['errors'].append(f"ESPN: {str(e)}")
    
    # 2. NCAA Stats
    try:
        ncaa_data = collect_ncaa_stats()
        results['sources']['ncaa'] = ncaa_data
        results['summary']['stats_found'] = ncaa_data.get('total_stats', 0)
        
        if save_to_db and ncaa_data.get('success'):
            saved = save_ncaa_stats(ncaa_data)
            results['summary']['stats_saved'] = saved
    except Exception as e:
        logger.error(f"NCAA collection failed: {e}")
        results['errors'].append(f"NCAA: {str(e)}")
    
    # 3. D1Baseball Rankings
    try:
        d1b_data = collect_d1baseball_rankings()
        results['sources']['d1baseball'] = d1b_data
        results['summary']['rankings_found'] = d1b_data.get('count', 0)
        
        if save_to_db and d1b_data.get('success'):
            saved = save_d1baseball_rankings(d1b_data)
            results['summary']['rankings_saved'] = saved
    except Exception as e:
        logger.error(f"D1Baseball collection failed: {e}")
        results['errors'].append(f"D1Baseball: {str(e)}")
    
    return results


def collect_for_daily():
    """Entry point for daily_collection.py integration"""
    print("\n[Multi-Source Stats Collection]")
    print("=" * 50)
    
    results = collect_all(save_to_db=True)
    
    print(f"\n📊 Collection Summary ({results['date']})")
    print(f"  ESPN Games: {results['summary']['games_found']} found, {results['summary']['games_saved']} saved")
    print(f"  NCAA Stats: {results['summary']['stats_found']} found, {results['summary']['stats_saved']} saved")
    print(f"  Rankings:   {results['summary']['rankings_found']} found, {results['summary']['rankings_saved']} saved")
    
    if results['errors']:
        print(f"\n⚠️  Errors: {len(results['errors'])}")
        for err in results['errors']:
            print(f"    - {err}")
    
    return results


def show_collection_status():
    """Show current database status"""
    conn = get_connection()
    c = conn.cursor()
    
    print("\n📊 Database Status")
    print("=" * 50)
    
    # Teams
    c.execute("SELECT COUNT(*) FROM teams")
    print(f"Teams: {c.fetchone()[0]}")
    
    # Games
    c.execute("SELECT COUNT(*) FROM games")
    total_games = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM games WHERE status='final'")
    completed = c.fetchone()[0]
    print(f"Games: {total_games} ({completed} completed)")
    
    # Rankings
    try:
        c.execute("SELECT COUNT(DISTINCT team_id) FROM rankings_history")
        ranked_teams = c.fetchone()[0]
        print(f"Ranked teams: {ranked_teams}")
    except:
        print("Ranked teams: N/A")
    
    # NCAA stats
    try:
        c.execute("SELECT COUNT(*) FROM ncaa_team_stats")
        team_stats = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM ncaa_individual_stats")
        ind_stats = c.fetchone()[0]
        print(f"NCAA stats: {team_stats} team, {ind_stats} individual")
    except:
        print("NCAA stats: N/A")
    
    conn.close()


def main():
    """CLI interface"""
    import argparse
    parser = argparse.ArgumentParser(description='Multi-source college baseball stats collection')
    parser.add_argument('command', nargs='?', default='collect',
                       choices=['collect', 'status', 'espn', 'ncaa', 'd1baseball', 'test'],
                       help='Command to run')
    parser.add_argument('--no-save', action='store_true', help='Do not save to database')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        results = collect_all(save_to_db=not args.no_save)
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            collect_for_daily()
    
    elif args.command == 'status':
        show_collection_status()
    
    elif args.command == 'espn':
        results = collect_espn_scoreboard()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\n📺 ESPN Scoreboard")
            print(f"Games found: {results.get('games_found', 0)}")
            for game in results.get('games', [])[:10]:
                away = game['away_team']
                home = game['home_team']
                if game.get('away_rank'):
                    away = f"#{game['away_rank']} {away}"
                if game.get('home_rank'):
                    home = f"#{game['home_rank']} {home}"
                if game.get('status') == 'final':
                    print(f"  {away} {game['away_score']} @ {home} {game['home_score']} (F)")
                else:
                    print(f"  {away} @ {home}")
    
    elif args.command == 'ncaa':
        results = collect_ncaa_stats()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\n📊 NCAA Stats (Top 10)")
            print("\nTeam Batting Average:")
            for entry in results.get('team_batting', []):
                print(f"  #{entry['rank']} {entry['team']}: {entry['batting_avg']:.3f}")
            print("\nTeam ERA:")
            for entry in results.get('team_pitching', []):
                print(f"  #{entry['rank']} {entry['team']}: {entry['era']:.2f}")
    
    elif args.command == 'd1baseball':
        results = collect_d1baseball_rankings()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\n🏆 D1Baseball Top 25")
            for entry in results.get('rankings', []):
                print(f"  #{entry['rank']:>2} {entry['team']}")
    
    elif args.command == 'test':
        print("Testing all sources (no save)...")
        results = collect_all(save_to_db=False)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
