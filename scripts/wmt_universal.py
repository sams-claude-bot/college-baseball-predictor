#!/usr/bin/env python3
"""
WMT Universal Box Score Collector

The holy grail - universal coverage for ALL D1 teams via api.wmt.games.
Discovers school IDs automatically and collects stats for any game.

Usage:
    python wmt_universal.py --discover              # Build school mapping
    python wmt_universal.py --school lsu            # All games for a school
    python wmt_universal.py --date 2026-02-13       # All games on a date
    python wmt_universal.py --collect 2026-02-13    # Collect stats for date
"""

import sys
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'
SCHOOLS_CACHE = Path(__file__).parent.parent / 'data' / 'wmt_schools.json'

def get_session():
    """Create requests session with proper headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    })
    return session


def discover_schools(force=False):
    """Discover all D1 baseball schools from WMT"""
    if SCHOOLS_CACHE.exists() and not force:
        with open(SCHOOLS_CACHE) as f:
            return json.load(f)
    
    print("🔍 Discovering schools from WMT...")
    session = get_session()
    
    # Fetch baseball teams for 2026 season
    url = 'https://api.wmt.games/api/statistics/teams'
    params = {
        'sport_code': 'MBA',
        'season_academic_year': 2026,
        'per_page': 1000,
    }
    
    try:
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            print(f"   Error: {resp.status_code}")
            return {}
        
        data = resp.json()
        teams = data.get('data', [])
        print(f"   Found {len(teams)} teams")
        
        # Build school mapping (dedupe by org_id)
        schools = {}
        for t in teams:
            org_id = t.get('org_id')
            if org_id and org_id not in schools:
                schools[org_id] = {
                    'id': org_id,
                    'name': t.get('name_tabular', ''),
                    'conference': t.get('conference_name_tabular', ''),
                    'nickname': t.get('team_nickname', ''),
                }
        
        # Save cache
        SCHOOLS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCHOOLS_CACHE, 'w') as f:
            json.dump(schools, f, indent=2)
        
        print(f"   Cached {len(schools)} unique schools")
        return schools
    
    except Exception as e:
        print(f"   Error: {e}")
        return {}


def find_school_id(name, schools):
    """Find school ID by name (fuzzy match)"""
    name_lower = name.lower().strip()
    
    # Direct match
    for sid, info in schools.items():
        if name_lower == info['name'].lower():
            return int(sid)
    
    # Partial match
    for sid, info in schools.items():
        school_name = info['name'].lower()
        if name_lower in school_name or school_name in name_lower:
            return int(sid)
    
    # Common aliases
    aliases = {
        'mississippi state': 430,
        'miss state': 430,
        'ms state': 430,
        'ole miss': 433,
        'texas a&m': 697,
        'texas am': 697,
        'unc': 546,
        'north carolina': 546,
    }
    if name_lower in aliases:
        return aliases[name_lower]
    
    return None


def get_school_games(school_id, season=2026):
    """Get all baseball games for a school"""
    session = get_session()
    url = 'https://api.wmt.games/api/statistics/games'
    params = {
        'school_id': school_id,
        'sport_code': 'MBA',  # Men's Baseball only
        'season_academic_year': season,
        'per_page': 200,
    }
    
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json().get('data', [])
    except:
        pass
    return []


def get_games_by_date(date_str, schools):
    """Get all games on a specific date"""
    session = get_session()
    all_games = []
    seen_ids = set()
    
    # Query each school's games and filter by date
    # This is slow but comprehensive
    print(f"🔍 Finding games on {date_str}...")
    
    # Use a subset of major conferences for speed
    major_confs = ['SEC', 'ACC', 'Big Ten', 'Big 12', 'Pac-12']
    major_schools = [s for sid, s in schools.items() 
                     if s['conference'] in major_confs]
    
    for i, school in enumerate(major_schools):
        if i % 20 == 0:
            print(f"   Checking {i}/{len(major_schools)} schools...")
        
        games = get_school_games(school['id'])
        for g in games:
            game_date = g.get('game_date', '')[:10]
            game_id = g['id']
            
            if game_date == date_str and game_id not in seen_ids:
                seen_ids.add(game_id)
                all_games.append(g)
    
    print(f"   Found {len(all_games)} games")
    return all_games


def parse_wmt_boxscore(game_data, players_data):
    """Parse WMT box score into standardized format"""
    result = {
        'game_date': game_data.get('game_date', '')[:10],
        'wmt_game_id': game_data.get('id'),
        'home_team': None,
        'away_team': None,
        'home_score': 0,
        'away_score': 0,
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
    }
    
    # Get competitors info
    competitors = game_data.get('competitors', [])
    team_info = {}
    
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
    for player in players_data:
        team_id = player.get('team_id')
        name = player.get('xml_name', '').title()
        position = player.get('xml_position', '').upper()
        
        if team_id not in team_info:
            continue
        is_home = team_info[team_id]['is_home']
        
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
            }
            
            if is_home:
                result['home_pitching'].append(pitching)
            else:
                result['away_pitching'].append(pitching)
    
    return result


def fetch_full_boxscore(wmt_game_id):
    """Fetch full box score with player stats"""
    session = get_session()
    url = f'https://api.wmt.games/api/statistics/games/{wmt_game_id}'
    params = {'with[0]': 'actions', 'with[1]': 'players'}
    
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            game_data = data.get('data', {})
            players_data = game_data.get('players', {}).get('data', [])
            return game_data, players_data
    except Exception as e:
        print(f"   Error fetching {wmt_game_id}: {e}")
    return None, []


def collect_date(date_str, dry_run=False):
    """Collect all box scores for a date"""
    schools = discover_schools()
    
    # Find games with scores (completed)
    session = get_session()
    
    print(f"\n📊 Collecting box scores for {date_str}")
    
    # Check SEC schools
    sec_schools = [s for sid, s in schools.items() if s['conference'] == 'SEC']
    print(f"   Checking {len(sec_schools)} SEC schools...")
    
    collected = 0
    errors = 0
    
    for school in sec_schools:
        games = get_school_games(school['id'])
        
        for g in games:
            if g.get('game_date', '')[:10] != date_str:
                continue
            
            # Check if game has scores (completed)
            comps = g.get('competitors', [])
            has_score = any(c.get('score') is not None for c in comps)
            if not has_score:
                continue
            
            wmt_id = g['id']
            home = next((c['nameTabular'] for c in comps if c.get('homeTeam')), '?')
            away = next((c['nameTabular'] for c in comps if not c.get('homeTeam')), '?')
            
            print(f"\n   {away} @ {home} (WMT: {wmt_id})")
            
            game_data, players_data = fetch_full_boxscore(wmt_id)
            if not game_data or not players_data:
                print("      ⊘ No player data")
                errors += 1
                continue
            
            boxscore = parse_wmt_boxscore(game_data, players_data)
            
            total_batters = len(boxscore['home_batting']) + len(boxscore['away_batting'])
            total_pitchers = len(boxscore['home_pitching']) + len(boxscore['away_pitching'])
            
            print(f"      ✓ {total_batters} batters, {total_pitchers} pitchers")
            
            if not dry_run:
                # TODO: Save to database
                pass
            
            collected += 1
    
    print(f"\n✅ Collected {collected} box scores ({errors} errors)")
    return collected


def main():
    if len(sys.argv) < 2:
        print("WMT Universal Box Score Collector")
        print("\nUsage:")
        print("  python wmt_universal.py --discover          # Build school mapping")
        print("  python wmt_universal.py --school <name>     # Show school games")
        print("  python wmt_universal.py --collect <date>    # Collect stats (YYYY-MM-DD)")
        print("  python wmt_universal.py --dry <date>        # Dry run for date")
        return
    
    cmd = sys.argv[1]
    
    if cmd == '--discover':
        schools = discover_schools(force='--force' in sys.argv)
        
        # Print SEC schools
        print("\n=== SEC Schools ===")
        sec = [(sid, s) for sid, s in schools.items() if s['conference'] == 'SEC']
        for sid, s in sorted(sec, key=lambda x: x[1]['name']):
            print(f"  {sid:5} {s['name']}")
    
    elif cmd == '--school':
        if len(sys.argv) < 3:
            print("Usage: python wmt_universal.py --school <name>")
            return
        
        schools = discover_schools()
        name = ' '.join(sys.argv[2:])
        school_id = find_school_id(name, schools)
        
        if not school_id:
            print(f"School '{name}' not found")
            return
        
        school = schools.get(str(school_id), {})
        print(f"\n📊 {school.get('name', name)} (ID: {school_id})")
        
        games = get_school_games(school_id)
        print(f"   Found {len(games)} games\n")
        
        for g in games[:15]:
            date = g.get('game_date', '')[:10]
            comps = g.get('competitors', [])
            home = next((c['nameTabular'] for c in comps if c.get('homeTeam')), '?')
            away = next((c['nameTabular'] for c in comps if not c.get('homeTeam')), '?')
            h_score = next((c.get('score', '-') for c in comps if c.get('homeTeam')), '-')
            a_score = next((c.get('score', '-') for c in comps if not c.get('homeTeam')), '-')
            score = f"{a_score}-{h_score}" if h_score != '-' else "TBD"
            
            print(f"  {date}: {away} @ {home} ({score}) → WMT: {g['id']}")
    
    elif cmd in ('--collect', '--dry'):
        if len(sys.argv) < 3:
            print("Usage: python wmt_universal.py --collect <date>")
            return
        
        date_str = sys.argv[2]
        dry_run = cmd == '--dry'
        
        collect_date(date_str, dry_run=dry_run)


if __name__ == "__main__":
    main()
