#!/usr/bin/env python3
"""
Collect player stats for scheduled games.

Workflow:
1. Get games from our schedule for target date
2. For each game, try ESPN box score first
3. Fall back to StatBroadcast if needed
4. Parse player batting/pitching stats
5. Update database with results and player stats

Usage:
    python3 collect_game_stats.py                    # Yesterday's games
    python3 collect_game_stats.py --date 2026-02-13  # Specific date
    python3 collect_game_stats.py --delay 15         # Custom delay
"""

import re
import sys
import json
import time
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def get_connection():
    return sqlite3.connect(str(DATA_DIR / 'baseball.db'))


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def init_player_stats_tables():
    """Create tables for detailed player stats"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS game_batting_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            at_bats INTEGER DEFAULT 0,
            runs INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0,
            rbi INTEGER DEFAULT 0,
            home_runs INTEGER DEFAULT 0,
            walks INTEGER DEFAULT 0,
            strikeouts INTEGER DEFAULT 0,
            stolen_bases INTEGER DEFAULT 0,
            batting_avg REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, team_id, player_name)
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS game_pitching_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            innings_pitched REAL DEFAULT 0,
            hits_allowed INTEGER DEFAULT 0,
            runs_allowed INTEGER DEFAULT 0,
            earned_runs INTEGER DEFAULT 0,
            walks INTEGER DEFAULT 0,
            strikeouts INTEGER DEFAULT 0,
            home_runs_allowed INTEGER DEFAULT 0,
            pitches INTEGER DEFAULT 0,
            strikes INTEGER DEFAULT 0,
            era REAL,
            win INTEGER DEFAULT 0,
            loss INTEGER DEFAULT 0,
            save INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, team_id, player_name)
        )
    ''')
    
    conn.commit()
    conn.close()


def get_scheduled_games(date_str):
    """Get games from our schedule for a date"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        SELECT id, home_team_id, away_team_id, home_score, away_score
        FROM games 
        WHERE date = ?
        ORDER BY home_team_id
    ''', (date_str,))
    
    games = []
    for row in cur.fetchall():
        games.append({
            'game_id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'home_score': row[3],
            'away_score': row[4],
            'has_result': row[3] is not None
        })
    
    conn.close()
    return games


def extract_text_from_html(html):
    """Extract readable text from HTML (similar to web_fetch)"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(line for line in lines if line)
    except ImportError:
        # Fallback: basic regex extraction
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '\n', text)
        return text


def search_espn_game(home_team, away_team, date_str, delay=10):
    """Search ESPN for a specific game's box score"""
    if not HAS_REQUESTS:
        return None
    
    # First get the scoreboard to find game IDs
    date_param = date_str.replace('-', '')
    url = f"https://www.espn.com/college-baseball/scoreboard?date={date_param}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        
        # Find game IDs and match to our teams
        game_ids = re.findall(r'/gameId/(\d+)', resp.text)
        
        # For each game ID, check if it matches our teams
        for gid in set(game_ids):
            box_url = f"https://www.espn.com/college-baseball/boxscore/_/gameId/{gid}"
            box_resp = requests.get(box_url, headers=HEADERS, timeout=30)
            
            if box_resp.status_code == 200:
                content = box_resp.text.lower()
                
                # Check if both teams appear in the page
                home_variants = [home_team, home_team.replace('-', ' '), home_team.replace('-', '')]
                away_variants = [away_team, away_team.replace('-', ' '), away_team.replace('-', '')]
                
                home_found = any(v in content for v in home_variants)
                away_found = any(v in content for v in away_variants)
                
                if home_found and away_found:
                    # Extract text from HTML for parsing
                    text = extract_text_from_html(box_resp.text)
                    time.sleep(delay)
                    return {'espn_id': gid, 'html': text}
            
            time.sleep(2)  # Small delay between checks
        
        time.sleep(delay)
        
    except Exception as e:
        log(f"ESPN search error: {e}")
    
    return None


def search_statbroadcast_game(home_team, away_team, date_str, delay=10):
    """Search StatBroadcast for a specific game"""
    if not HAS_PLAYWRIGHT:
        return None
    
    # Map team IDs to StatBroadcast school codes
    school_codes = {
        'mississippi-state': 'msst',
        'lsu': 'lsu',
        'texas': 'texas',
        'ucla': 'ucla',
        'florida': 'florida',
        'georgia': 'uga',
        'tennessee': 'utenn',
        'alabama': 'bama',
        'arkansas': 'ark',
        'auburn': 'auburn',
        'ole-miss': 'olemiss',
        'texas-am': 'tamu',
        'vanderbilt': 'vandy',
        'kentucky': 'uky',
        'south-carolina': 'scar',
        'missouri': 'mizzou',
    }
    
    # Try home team's portal first
    school_code = school_codes.get(home_team)
    if not school_code:
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            url = f"https://statbroadcast.com/events/statmonitr.php?gid={school_code}"
            page.goto(url, timeout=30000)
            time.sleep(3)
            
            html = page.content()
            
            # Look for baseball game IDs (BASE sport marker)
            # Find IDs near "BASE" text
            baseball_section = re.search(r'BASE.*?(?=MBB|WBB|SOFT|$)', html, re.DOTALL)
            if baseball_section:
                game_ids = re.findall(r'id=(\d{6,})', baseball_section.group())
                
                # Check each game
                for gid in game_ids[:5]:  # Limit to first 5
                    game_url = f"https://stats.statbroadcast.com/statmonitr/?id={gid}"
                    page.goto(game_url, timeout=30000)
                    time.sleep(2)
                    
                    title = page.title().lower()
                    
                    # Check if both teams in title
                    home_short = home_team.split('-')[0][:4]
                    away_short = away_team.split('-')[0][:4]
                    
                    if home_short in title and away_short in title:
                        game_html = page.content()
                        browser.close()
                        time.sleep(delay)
                        return {'statbroadcast_id': gid, 'html': game_html, 'title': page.title()}
            
            browser.close()
            
    except Exception as e:
        log(f"StatBroadcast search error: {e}")
    
    time.sleep(delay)
    return None


def parse_espn_boxscore(html, game_id, home_team, away_team):
    """Parse ESPN box score HTML for player stats"""
    stats = {
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
        'home_score': None,
        'away_score': None
    }
    
    lines = html.strip().split('\n')
    
    current_team = None
    current_section = None
    player_names = []
    player_positions = []
    stat_values = []  # Individual stat values
    in_stats = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Team hitting header
        if ' Hitting' in line or line == 'Hitting':
            # Save previous section
            if current_team and player_names and current_section == 'batting':
                team_players = build_batting_players(player_names, player_positions, stat_values)
                team_lower = current_team.lower().replace(' ', '-')
                if home_team.lower() in team_lower or team_lower in home_team.lower():
                    stats['home_batting'] = team_players
                else:
                    stats['away_batting'] = team_players
            
            current_team = line.replace(' Hitting', '').replace('Hitting', '').strip()
            current_section = 'batting'
            player_names = []
            player_positions = []
            stat_values = []
            in_stats = False
            continue
        
        # Team pitching header
        if ' Pitching' in line or line == 'Pitching':
            # Save batting first
            if current_team and player_names and current_section == 'batting':
                team_players = build_batting_players(player_names, player_positions, stat_values)
                team_lower = current_team.lower().replace(' ', '-')
                if home_team.lower() in team_lower or team_lower in home_team.lower():
                    stats['home_batting'] = team_players
                else:
                    stats['away_batting'] = team_players
            
            current_team = line.replace(' Pitching', '').replace('Pitching', '').strip()
            current_section = 'pitching'
            player_names = []
            player_positions = []
            stat_values = []
            in_stats = False
            continue
        
        # Skip non-batting sections
        if current_section != 'batting':
            continue
        
        # Skip headers
        if line in ('hitters', 'pitchers', 'team', 'AB', 'R', 'H', 'RBI', 'HR', 'BB', 'K'):
            if line == 'AB':
                in_stats = True
            continue
        
        # Player name: "X. Lastname"
        if re.match(r'^[A-Z]\. [A-Za-z\'\-]+$', line):
            player_names.append(line)
            continue
        
        # Position
        if re.match(r'^[A-Z]{1,3}(-[A-Z]{1,3})?$', line) and len(line) <= 6:
            player_positions.append(line)
            continue
        
        # Single digit stat values OR concatenated stats
        if re.match(r'^\d+$', line):
            if len(line) >= 7:
                # Concatenated format (original web_fetch style)
                stat_values.append(line)
            else:
                # Single value format (BeautifulSoup style)
                stat_values.append(int(line))
    
    # Save last section
    if current_team and player_names and current_section == 'batting':
        team_players = build_batting_players(player_names, player_positions, stat_values)
        team_lower = current_team.lower().replace(' ', '-')
        if home_team.lower() in team_lower or team_lower in home_team.lower():
            stats['home_batting'] = team_players
        else:
            stats['away_batting'] = team_players
    
    return stats


def build_batting_players(names, positions, stat_values):
    """Build player stat dictionaries from parsed values"""
    players = []
    
    # Check if stats are concatenated strings or individual values
    if stat_values and isinstance(stat_values[0], str):
        # Concatenated format: "3000011"
        for i, name in enumerate(names):
            if i < len(stat_values) and i < len(positions):
                s = stat_values[i]
                if len(s) >= 7:
                    players.append({
                        'name': name, 'pos': positions[i],
                        'ab': int(s[0]), 'r': int(s[1]), 'h': int(s[2]),
                        'rbi': int(s[3]), 'hr': int(s[4]), 'bb': int(s[5]), 'k': int(s[6])
                    })
    else:
        # Individual values format: 7 values per player
        stats_per_player = 7
        num_players = len(stat_values) // stats_per_player
        
        for i in range(min(num_players, len(names))):
            if i < len(positions):
                offset = i * stats_per_player
                if offset + 6 < len(stat_values):
                    players.append({
                        'name': names[i], 'pos': positions[i],
                        'ab': stat_values[offset],
                        'r': stat_values[offset + 1],
                        'h': stat_values[offset + 2],
                        'rbi': stat_values[offset + 3],
                        'hr': stat_values[offset + 4],
                        'bb': stat_values[offset + 5],
                        'k': stat_values[offset + 6]
                    })
    
    return players


def parse_statbroadcast_boxscore(html, title, game_id, home_team, away_team):
    """Parse StatBroadcast box score for player stats"""
    stats = {
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
        'home_score': None,
        'away_score': None
    }
    
    # Parse score from title: "AWAY #, HOME # - Final"
    match = re.match(r'([A-Z\s]+)\s*(\d+),\s*([A-Z\s]+)\s*(\d+)\s*-\s*Final', title, re.IGNORECASE)
    if match:
        stats['away_score'] = int(match.group(2))
        stats['home_score'] = int(match.group(4))
    
    return stats


def update_game_result(game_id, home_score, away_score):
    """Update game with final score"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        UPDATE games 
        SET home_score = ?, away_score = ?, status = 'final'
        WHERE id = ?
    ''', (home_score, away_score, game_id))
    
    updated = cur.rowcount
    conn.commit()
    conn.close()
    
    return updated > 0


def save_batting_stats(game_id, team_id, players):
    """Save batting stats to database"""
    if not players:
        return 0
    
    conn = get_connection()
    cur = conn.cursor()
    
    count = 0
    for p in players:
        try:
            cur.execute('''
                INSERT OR REPLACE INTO game_batting_stats
                (game_id, team_id, player_name, position, at_bats, runs, hits, rbi, 
                 home_runs, walks, strikeouts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id, team_id, p.get('name', ''), p.get('pos', ''),
                p.get('ab', 0), p.get('r', 0), p.get('h', 0), p.get('rbi', 0),
                p.get('hr', 0), p.get('bb', 0), p.get('k', 0)
            ))
            count += 1
        except Exception as e:
            log(f"Error saving batting: {e}")
    
    conn.commit()
    conn.close()
    return count


def process_game(game, date_str, delay=15):
    """Process a single game - find box score and extract stats"""
    game_id = game['game_id']
    home = game['home_team']
    away = game['away_team']
    
    log(f"Processing: {away} @ {home}")
    
    # Try ESPN first
    result = search_espn_game(home, away, date_str, delay)
    source = 'espn'
    
    if not result:
        # Fall back to StatBroadcast
        result = search_statbroadcast_game(home, away, date_str, delay)
        source = 'statbroadcast'
    
    if not result:
        log(f"  No box score found")
        return False
    
    log(f"  Found on {source}")
    
    # Parse stats
    if source == 'espn':
        stats = parse_espn_boxscore(result['html'], game_id, home, away)
    else:
        stats = parse_statbroadcast_boxscore(
            result['html'], result.get('title', ''), game_id, home, away
        )
    
    # Update game result if we got scores
    if stats['home_score'] is not None and stats['away_score'] is not None:
        if update_game_result(game_id, stats['home_score'], stats['away_score']):
            log(f"  Updated score: {away} {stats['away_score']} @ {home} {stats['home_score']}")
    
    # Save player stats
    home_batters = save_batting_stats(game_id, home, stats['home_batting'])
    away_batters = save_batting_stats(game_id, away, stats['away_batting'])
    
    if home_batters or away_batters:
        log(f"  Saved {home_batters + away_batters} player batting records")
    
    return True


def cleanup_browsers():
    """Clean up any stray browser processes"""
    import subprocess
    try:
        subprocess.run(['pkill', '-f', 'chromium.*headless'], 
                      capture_output=True, timeout=5)
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Collect game stats from schedule')
    parser.add_argument('--date', help='Date (YYYY-MM-DD), default: yesterday')
    parser.add_argument('--delay', type=int, default=15, help='Delay between requests')
    parser.add_argument('--limit', type=int, help='Max games to process')
    parser.add_argument('--missing-only', action='store_true', help='Only games without results')
    args = parser.parse_args()
    
    # Default to yesterday
    if args.date:
        date_str = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')
    
    log(f"Collecting stats for {date_str}")
    
    # Initialize tables
    init_player_stats_tables()
    
    # Get scheduled games
    games = get_scheduled_games(date_str)
    log(f"Found {len(games)} scheduled games")
    
    # Filter if needed
    if args.missing_only:
        games = [g for g in games if not g['has_result']]
        log(f"  {len(games)} missing results")
    
    if args.limit:
        games = games[:args.limit]
    
    # Process each game
    found = 0
    for i, game in enumerate(games):
        log(f"\n[{i+1}/{len(games)}]")
        try:
            if process_game(game, date_str, args.delay):
                found += 1
        except Exception as e:
            log(f"  Error: {e}")
    
    log(f"\n=== Complete ===")
    log(f"Processed {len(games)} games, found {found} box scores")
    
    cleanup_browsers()


if __name__ == '__main__':
    main()
