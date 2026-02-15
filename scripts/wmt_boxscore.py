#!/usr/bin/env python3
"""
WMT Games API - Universal Box Score Collector

This is the holy grail - covers ALL D1 teams including small schools.
Uses api.wmt.games which powers most college athletic sites.

Usage:
    python wmt_boxscore.py 6492544                # Single game by WMT ID
    python wmt_boxscore.py 6492544 --dry          # Dry run (no DB writes)
"""

import sys
import sqlite3
import requests
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'


def fetch_wmt_boxscore(game_id):
    """Fetch full box score from WMT API"""
    url = f'https://api.wmt.games/api/statistics/games/{game_id}?with[0]=actions&with[1]=players'
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    })
    
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"   Error fetching: {e}")
    return None


def parse_wmt_boxscore(data):
    """Parse WMT box score into standardized format"""
    result = {
        'game_date': None,
        'home_team': None,
        'away_team': None,
        'home_score': 0,
        'away_score': 0,
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
    }
    
    game_data = data.get('data', {})
    result['game_date'] = game_data.get('game_date', '')[:10]
    
    # Get competitors info (has team names and home/away)
    competitors = game_data.get('competitors', [])
    team_info = {}  # team_id -> {name, is_home, score}
    
    for comp in competitors:
        team_id = comp.get('teamId')
        team_info[team_id] = {
            'name': comp.get('nameTabular', ''),
            'is_home': comp.get('homeTeam', False),
            'score': comp.get('score', 0),
        }
        
        if comp.get('homeTeam'):
            result['home_team'] = comp.get('nameTabular', '')
            result['home_score'] = comp.get('score', 0)
        else:
            result['away_team'] = comp.get('nameTabular', '')
            result['away_score'] = comp.get('score', 0)
    
    # Parse player stats
    players_data = game_data.get('players', {}).get('data', [])
    
    for player in players_data:
        team_id = player.get('team_id')
        name = player.get('xml_name', '').title()
        position = player.get('xml_position', '').upper()
        
        # Determine if home or away based on team_id
        if team_id not in team_info:
            continue
        is_home = team_info[team_id]['is_home']
        
        # Get stats from statistic array (period 0 = cumulative)
        stats_list = player.get('statistic', [])
        if not stats_list:
            continue
        
        stats = stats_list[0].get('statistic', {})
        if not stats:
            continue
        
        # Batting stats
        at_bats = int(stats.get('sAtBats') or 0)
        walks = int(stats.get('sWalks') or 0)
        hbp = int(stats.get('sHitByPitch') or 0)
        
        if at_bats > 0 or walks > 0 or hbp > 0:
            batting = {
                'name': name,
                'position': position,
                'ab': at_bats,
                'runs': int(stats.get('sRuns') or 0),
                'hits': int(stats.get('sHits') or 0),
                'rbi': int(stats.get('sRunsBattedIn') or 0),
                'bb': walks,
                'so': int(stats.get('sStrikeoutsHitting') or 0),
                'doubles': int(stats.get('sDoubles') or 0),
                'triples': int(stats.get('sTriples') or 0),
                'hr': int(stats.get('sHomeRuns') or 0),
                'sb': int(stats.get('sStolenBases') or 0),
                'hbp': hbp,
                'avg': round(float(stats.get('sBattingAverage') or 0), 3),
            }
            
            if is_home:
                result['home_batting'].append(batting)
            else:
                result['away_batting'].append(batting)
        
        # Pitching stats
        ip = float(stats.get('sInningsPitched') or 0)
        
        if ip > 0:
            pitching = {
                'name': name,
                'ip': ip,
                'hits': int(stats.get('sHitsAllowed') or 0),
                'runs': int(stats.get('sRunsAllowed') or 0),
                'er': int(stats.get('sEarnedRuns') or 0),
                'bb': int(stats.get('sBasesOnBallsAllowed') or 0),
                'so': int(stats.get('sStrikeouts') or 0),
                'hr': int(stats.get('sHomeRunsAllowed') or 0),
                'pitches': int(stats.get('sNumberOfPitches') or 0),
                'era': round(float(stats.get('sERA') or 0), 2),
                'whip': round(float(stats.get('sWHIP') or 0), 2),
            }
            
            if is_home:
                result['home_pitching'].append(pitching)
            else:
                result['away_pitching'].append(pitching)
    
    return result


def save_to_db(boxscore, game_id):
    """Save box score stats to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    inserted = 0
    
    for is_home, batting_list, pitching_list in [
        (True, boxscore['home_batting'], boxscore['home_pitching']),
        (False, boxscore['away_batting'], boxscore['away_pitching']),
    ]:
        team_name = boxscore['home_team'] if is_home else boxscore['away_team']
        
        # Try to find team in DB
        c.execute("SELECT id FROM teams WHERE LOWER(name) LIKE LOWER(?) OR LOWER(short_name) LIKE LOWER(?)", 
                  (f'%{team_name}%', f'%{team_name}%'))
        team_row = c.fetchone()
        team_id = team_row[0] if team_row else None
        
        # Insert batting stats
        for b in batting_list:
            c.execute("""
                INSERT OR REPLACE INTO player_stats
                (game_id, player_name, team_id, stat_type, 
                 ab, runs, hits, rbi, doubles, triples, hr, bb, so, sb, hbp, avg)
                VALUES (?, ?, ?, 'batting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, b['name'], team_id, 
                  b['ab'], b['runs'], b['hits'], b['rbi'],
                  b['doubles'], b['triples'], b['hr'], b['bb'], b['so'], b['sb'], b['hbp'], b['avg']))
            inserted += 1
        
        # Insert pitching stats
        for p in pitching_list:
            c.execute("""
                INSERT OR REPLACE INTO player_stats
                (game_id, player_name, team_id, stat_type,
                 ip, hits_allowed, runs_allowed, er, bb, so, hr_allowed, pitches, era)
                VALUES (?, ?, ?, 'pitching', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, p['name'], team_id,
                  p['ip'], p['hits'], p['runs'], p['er'], p['bb'], p['so'], p['hr'], p['pitches'], p['era']))
            inserted += 1
    
    conn.commit()
    conn.close()
    return inserted


def collect_wmt_boxscore(wmt_game_id, dry_run=False, db_game_id=None):
    """Collect and optionally store box score"""
    print(f"\n📊 Fetching WMT game {wmt_game_id}")
    
    data = fetch_wmt_boxscore(wmt_game_id)
    if not data:
        print("   ⊘ Failed to fetch data")
        return None
    
    boxscore = parse_wmt_boxscore(data)
    
    print(f"   {boxscore['away_team']} @ {boxscore['home_team']}")
    print(f"   Score: {boxscore['away_score']}-{boxscore['home_score']}")
    print(f"   ✓ {len(boxscore['home_batting'])} home batters, {len(boxscore['away_batting'])} away batters")
    print(f"     {len(boxscore['home_pitching'])} home pitchers, {len(boxscore['away_pitching'])} away pitchers")
    
    if dry_run:
        print("   [DRY RUN - showing sample data]")
        if boxscore['home_batting']:
            sample = boxscore['home_batting'][0]
            print(f"   Sample: {sample['name']} - {sample['ab']} AB, {sample['hits']} H, {sample['rbi']} RBI")
        if boxscore['home_pitching']:
            sample = boxscore['home_pitching'][0]
            print(f"   Sample: {sample['name']} - {sample['ip']} IP, {sample['so']} K, {sample['era']} ERA")
    elif db_game_id:
        inserted = save_to_db(boxscore, db_game_id)
        print(f"   ✓ Saved {inserted} stat entries to DB")
    
    return boxscore


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("WMT Games API Box Score Collector")
        print("\nUsage:")
        print("  python wmt_boxscore.py GAME_ID [--dry]")
        print("\nExample: python wmt_boxscore.py 6492544 --dry")
        sys.exit(1)
    
    dry_run = '--dry' in sys.argv
    game_id = sys.argv[1]
    
    if game_id.isdigit():
        result = collect_wmt_boxscore(game_id, dry_run)
        if result:
            print("\n   ✅ Box score collected successfully")
    else:
        print(f"Invalid game ID: {game_id}")
