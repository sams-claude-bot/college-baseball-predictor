#!/usr/bin/env python3
"""
Box Score Collection

Collects box scores for completed games and updates player stats.
Supports multiple sources with fallback.

Usage:
    python collect_box_scores.py                    # Collect today's games
    python collect_box_scores.py --date 2026-02-14  # Specific date
    python collect_box_scores.py --game <game_id>   # Specific game
    python collect_box_scores.py --recent 7         # Last N days
"""

import sys
import re
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

_scripts_dir = Path(__file__).parent
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from scripts.database import get_connection
from scripts.player_stats import add_player, init_player_tables

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Team ID mappings for different sources
TEAM_ALIASES = {
    "mississippi-state": ["Mississippi State", "Miss State", "Miss. State", "MSU", "MS State"],
    "ole-miss": ["Ole Miss", "Mississippi", "Rebels"],
    "alabama": ["Alabama", "Bama", "Crimson Tide"],
    "auburn": ["Auburn", "Tigers"],
    "arkansas": ["Arkansas", "Razorbacks"],
    "florida": ["Florida", "Gators", "UF"],
    "georgia": ["Georgia", "Bulldogs", "UGA"],
    "kentucky": ["Kentucky", "Wildcats", "UK"],
    "lsu": ["LSU", "Louisiana State", "Tigers"],
    "missouri": ["Missouri", "Mizzou", "Tigers"],
    "oklahoma": ["Oklahoma", "Sooners", "OU"],
    "south-carolina": ["South Carolina", "Gamecocks", "USC"],
    "tennessee": ["Tennessee", "Volunteers", "Vols"],
    "texas": ["Texas", "Longhorns", "UT"],
    "texas-am": ["Texas A&M", "Texas A and M", "Aggies", "TAMU"],
    "vanderbilt": ["Vanderbilt", "Commodores", "Vandy"],
}


def team_name_to_id(name):
    """Convert various team names to standard team_id"""
    name_lower = name.lower().strip()
    
    for team_id, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            if alias.lower() == name_lower or alias.lower() in name_lower:
                return team_id
    
    # Fallback: normalize the name
    return name_lower.replace(' ', '-').replace('.', '')


def get_completed_games(date_str=None, team_id=None, limit=50):
    """
    Get completed games that need box score collection
    """
    conn = get_connection()
    c = conn.cursor()
    
    query = """
        SELECT g.*, 
               ht.name as home_team_name,
               at.name as away_team_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE g.status = 'final'
    """
    params = []
    
    if date_str:
        query += " AND g.date = ?"
        params.append(date_str)
    
    if team_id:
        query += " AND (g.home_team_id = ? OR g.away_team_id = ?)"
        params.extend([team_id, team_id])
    
    query += " ORDER BY g.date DESC LIMIT ?"
    params.append(limit)
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_player_by_name(team_id, name):
    """
    Find a player by name (fuzzy match)
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Exact match first
    c.execute("""
        SELECT * FROM player_stats 
        WHERE team_id = ? AND LOWER(name) = LOWER(?)
    """, (team_id, name))
    
    row = c.fetchone()
    if row:
        conn.close()
        return dict(row)
    
    # Try last name match
    parts = name.split()
    if parts:
        last_name = parts[-1]
        c.execute("""
            SELECT * FROM player_stats 
            WHERE team_id = ? AND LOWER(name) LIKE ?
        """, (team_id, f'%{last_name.lower()}%'))
        
        rows = c.fetchall()
        if len(rows) == 1:
            conn.close()
            return dict(rows[0])
    
    conn.close()
    return None


def update_player_batting_stats(player_id, game_stats):
    """
    Update a player's cumulative batting stats with a game's stats.
    game_stats: dict with ab, runs, hits, doubles, triples, hr, rbi, bb, so, sb
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Get current stats
    c.execute("SELECT * FROM player_stats WHERE id = ?", (player_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    
    current = dict(row)
    
    # Add game stats to cumulative
    new_games = current.get('games', 0) + 1
    new_ab = current.get('at_bats', 0) + game_stats.get('ab', 0)
    new_runs = current.get('runs', 0) + game_stats.get('runs', 0)
    new_hits = current.get('hits', 0) + game_stats.get('hits', 0)
    new_doubles = current.get('doubles', 0) + game_stats.get('doubles', 0)
    new_triples = current.get('triples', 0) + game_stats.get('triples', 0)
    new_hr = current.get('home_runs', 0) + game_stats.get('hr', 0)
    new_rbi = current.get('rbi', 0) + game_stats.get('rbi', 0)
    new_bb = current.get('walks', 0) + game_stats.get('bb', 0)
    new_so = current.get('strikeouts', 0) + game_stats.get('so', 0)
    new_sb = current.get('stolen_bases', 0) + game_stats.get('sb', 0)
    
    # Calculate averages
    new_ba = new_hits / new_ab if new_ab > 0 else 0.0
    new_obp = (new_hits + new_bb) / (new_ab + new_bb) if (new_ab + new_bb) > 0 else 0.0
    
    # SLG = (1B + 2*2B + 3*3B + 4*HR) / AB
    singles = new_hits - new_doubles - new_triples - new_hr
    total_bases = singles + (2 * new_doubles) + (3 * new_triples) + (4 * new_hr)
    new_slg = total_bases / new_ab if new_ab > 0 else 0.0
    new_ops = new_obp + new_slg
    
    c.execute("""
        UPDATE player_stats SET
            games = ?,
            at_bats = ?,
            runs = ?,
            hits = ?,
            doubles = ?,
            triples = ?,
            home_runs = ?,
            rbi = ?,
            walks = ?,
            strikeouts = ?,
            stolen_bases = ?,
            batting_avg = ?,
            obp = ?,
            slg = ?,
            ops = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (new_games, new_ab, new_runs, new_hits, new_doubles, new_triples, 
          new_hr, new_rbi, new_bb, new_so, new_sb, new_ba, new_obp, new_slg, 
          new_ops, player_id))
    
    conn.commit()
    conn.close()
    return True


def update_player_pitching_stats(player_id, game_stats, is_starter=False):
    """
    Update a player's cumulative pitching stats with a game's stats.
    game_stats: dict with ip, hits, runs, er, bb, so, win, loss, save, pitches
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Get current stats
    c.execute("SELECT * FROM player_stats WHERE id = ?", (player_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    
    current = dict(row)
    
    # Add game stats
    new_gp = current.get('games_pitched', 0) + 1
    new_gs = current.get('games_started', 0) + (1 if is_starter else 0)
    new_ip = current.get('innings_pitched', 0) + game_stats.get('ip', 0)
    new_h = current.get('hits_allowed', 0) + game_stats.get('hits', 0)
    new_r = current.get('runs_allowed', 0) + game_stats.get('runs', 0)
    new_er = current.get('earned_runs', 0) + game_stats.get('er', 0)
    new_bb = current.get('walks_allowed', 0) + game_stats.get('bb', 0)
    new_so = current.get('strikeouts_pitched', 0) + game_stats.get('so', 0)
    new_wins = current.get('wins', 0) + game_stats.get('win', 0)
    new_losses = current.get('losses', 0) + game_stats.get('loss', 0)
    new_saves = current.get('saves', 0) + game_stats.get('save', 0)
    
    # Calculate rate stats
    new_era = (new_er / new_ip) * 9 if new_ip > 0 else 0.0
    new_whip = (new_h + new_bb) / new_ip if new_ip > 0 else 0.0
    new_k9 = (new_so / new_ip) * 9 if new_ip > 0 else 0.0
    new_bb9 = (new_bb / new_ip) * 9 if new_ip > 0 else 0.0
    
    # Track pitch counts
    pitches = game_stats.get('pitches', 0)
    new_season_pitches = current.get('season_pitches', 0) + pitches
    new_avg_pitch = new_season_pitches / new_gp if new_gp > 0 else 0.0
    
    c.execute("""
        UPDATE player_stats SET
            games_pitched = ?,
            games_started = ?,
            innings_pitched = ?,
            hits_allowed = ?,
            runs_allowed = ?,
            earned_runs = ?,
            walks_allowed = ?,
            strikeouts_pitched = ?,
            wins = ?,
            losses = ?,
            saves = ?,
            era = ?,
            whip = ?,
            k_per_9 = ?,
            bb_per_9 = ?,
            is_starter = CASE WHEN ? > 0 THEN 1 ELSE is_starter END,
            season_pitches = ?,
            avg_pitch_count = ?,
            last_start_date = CASE WHEN ? THEN ? ELSE last_start_date END,
            last_start_pitches = CASE WHEN ? THEN ? ELSE last_start_pitches END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (new_gp, new_gs, new_ip, new_h, new_r, new_er, new_bb, new_so,
          new_wins, new_losses, new_saves, new_era, new_whip, new_k9, new_bb9,
          new_gs, new_season_pitches, new_avg_pitch,
          is_starter, game_stats.get('game_date'), is_starter, pitches,
          player_id))
    
    conn.commit()
    conn.close()
    return True


def record_pitching_matchup(game_id, home_starter_id, away_starter_id, 
                            home_starter_name=None, away_starter_name=None):
    """
    Record the pitching matchup for a game
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO pitching_matchups 
            (game_id, home_starter_id, away_starter_id, home_starter_name, away_starter_name)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            home_starter_id = excluded.home_starter_id,
            away_starter_id = excluded.away_starter_id,
            home_starter_name = excluded.home_starter_name,
            away_starter_name = excluded.away_starter_name
    """, (game_id, home_starter_id, away_starter_id, home_starter_name, away_starter_name))
    
    conn.commit()
    conn.close()


def parse_innings_pitched(ip_str):
    """Parse innings pitched (e.g., '6.2' means 6 2/3 innings)"""
    try:
        if not ip_str:
            return 0.0
        ip_str = str(ip_str).strip()
        if '.' in ip_str:
            whole, frac = ip_str.split('.')
            # Baseball: .1 = 1/3, .2 = 2/3
            thirds = int(frac) if frac else 0
            return int(whole) + (thirds / 3.0)
        return float(ip_str)
    except:
        return 0.0


def fetch_espn_boxscore(game_id):
    """
    Try to fetch box score from ESPN
    Returns dict with batting and pitching stats by team
    """
    # ESPN game ID format varies - this is a stub for the pattern
    # In practice, you'd need to search for the game or have ESPN IDs mapped
    
    # Example URL pattern (would need actual game ID):
    # https://www.espn.com/college-baseball/boxscore/_/gameId/401XXXXX
    
    return None  # TODO: Implement when ESPN mapping is available


def fetch_team_site_boxscore(team_id, opponent_id, date_str):
    """
    Try to fetch box score from team athletics site
    """
    from scripts.track_sec_teams import SEC_TEAMS
    
    if team_id not in SEC_TEAMS:
        return None
    
    team_config = SEC_TEAMS[team_id]
    base_url = team_config['schedule_url'].rsplit('/', 1)[0]
    
    # Try stats page pattern
    stats_url = f"{base_url}/stats"
    
    try:
        resp = requests.get(stats_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            # Parse for the specific game
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Implementation depends on site structure
            pass
    except:
        pass
    
    return None


def parse_box_score_text(text, home_team_id, away_team_id):
    """
    Parse box score from raw text (fallback parser)
    """
    result = {
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
        'home_starter': None,
        'away_starter': None
    }
    
    # Look for common box score patterns
    # This is a simplified parser - real implementation would be more robust
    
    # Pattern: Name AB R H RBI BB SO
    batting_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
    
    # Pattern: Name IP H R ER BB SO
    pitching_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+\.?\d*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
    
    return result


def collect_box_score(game):
    """
    Collect box score for a single game from available sources
    """
    game_id = game['id']
    home_team_id = game['home_team_id']
    away_team_id = game['away_team_id']
    date_str = game['date']
    
    print(f"  📋 {away_team_id} @ {home_team_id} ({date_str})")
    
    # Try sources in order
    box_score = None
    source = None
    
    # Source 1: ESPN
    # box_score = fetch_espn_boxscore(game_id)
    # if box_score:
    #     source = 'espn'
    
    # Source 2: Team sites
    if not box_score:
        box_score = fetch_team_site_boxscore(home_team_id, away_team_id, date_str)
        if box_score:
            source = 'team_site'
    
    if not box_score:
        print(f"    ⊘ No box score found")
        return False
    
    print(f"    ✓ Found via {source}")
    
    # Process batting stats
    for team_key, team_id in [('home_batting', home_team_id), ('away_batting', away_team_id)]:
        for batting in box_score.get(team_key, []):
            player = get_player_by_name(team_id, batting['name'])
            if player:
                update_player_batting_stats(player['id'], batting)
    
    # Process pitching stats and identify starters
    home_starter_id = None
    away_starter_id = None
    
    for team_key, team_id in [('home_pitching', home_team_id), ('away_pitching', away_team_id)]:
        pitching_list = box_score.get(team_key, [])
        for i, pitching in enumerate(pitching_list):
            player = get_player_by_name(team_id, pitching['name'])
            if player:
                is_starter = (i == 0)  # First pitcher listed is typically the starter
                update_player_pitching_stats(player['id'], pitching, is_starter)
                
                if is_starter:
                    if team_key == 'home_pitching':
                        home_starter_id = player['id']
                    else:
                        away_starter_id = player['id']
    
    # Record pitching matchup
    if home_starter_id or away_starter_id:
        record_pitching_matchup(
            game_id, 
            home_starter_id, 
            away_starter_id,
            box_score.get('home_starter'),
            box_score.get('away_starter')
        )
    
    return True


def collect_for_date(date_str, team_id=None):
    """
    Collect all box scores for a specific date
    """
    print(f"\n📅 Collecting box scores for {date_str}")
    
    games = get_completed_games(date_str=date_str, team_id=team_id)
    
    if not games:
        print("  No completed games found")
        return {'collected': 0, 'missing': 0}
    
    print(f"  Found {len(games)} completed games")
    
    collected = 0
    missing = 0
    
    for game in games:
        if collect_box_score(game):
            collected += 1
        else:
            missing += 1
        time.sleep(1)  # Rate limiting
    
    return {'collected': collected, 'missing': missing}


def collect_recent(days=7, team_id=None):
    """
    Collect box scores for recent days
    """
    print(f"\n📆 Collecting box scores for last {days} days")
    
    total_collected = 0
    total_missing = 0
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        
        result = collect_for_date(date_str, team_id)
        total_collected += result['collected']
        total_missing += result['missing']
    
    print(f"\n✓ Total: {total_collected} collected, {total_missing} missing")
    return {'collected': total_collected, 'missing': total_missing}


def manual_entry_game(game_id):
    """
    Interactive manual entry for a game's box score
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT g.*, ht.name as home_name, at.name as away_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE g.id = ?
    """, (game_id,))
    
    game = c.fetchone()
    conn.close()
    
    if not game:
        print(f"Game not found: {game_id}")
        return
    
    game = dict(game)
    print(f"\n📝 Manual Box Score Entry")
    print(f"   {game['away_name']} @ {game['home_name']}")
    print(f"   {game['date']} - Final: {game['away_score']}-{game['home_score']}")
    print("-" * 50)
    
    # Get home starter
    print(f"\nHome starter ({game['home_team_id']}):")
    home_starter_name = input("  Name: ").strip()
    if home_starter_name:
        home_starter = get_player_by_name(game['home_team_id'], home_starter_name)
        if home_starter:
            print(f"  Found: #{home_starter.get('number', '?')} {home_starter['name']}")
            
            # Get pitching line
            ip = input("  IP: ").strip()
            hits = input("  H: ").strip()
            runs = input("  R: ").strip()
            er = input("  ER: ").strip()
            bb = input("  BB: ").strip()
            so = input("  SO: ").strip()
            
            stats = {
                'ip': parse_innings_pitched(ip),
                'hits': int(hits) if hits else 0,
                'runs': int(runs) if runs else 0,
                'er': int(er) if er else 0,
                'bb': int(bb) if bb else 0,
                'so': int(so) if so else 0,
                'game_date': game['date']
            }
            
            update_player_pitching_stats(home_starter['id'], stats, is_starter=True)
            print("  ✓ Updated")
    
    # Similar for away starter...
    print(f"\nAway starter ({game['away_team_id']}):")
    away_starter_name = input("  Name: ").strip()
    # ... (same pattern)


def show_collection_status():
    """
    Show which games have box scores collected
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Games with pitching matchups recorded
    c.execute("""
        SELECT COUNT(*) FROM pitching_matchups
    """)
    matchups = c.fetchone()[0]
    
    # Total completed games
    c.execute("""
        SELECT COUNT(*) FROM games WHERE status = 'final'
    """)
    total = c.fetchone()[0]
    
    # Recent games without matchups
    c.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id
        FROM games g
        LEFT JOIN pitching_matchups pm ON g.id = pm.game_id
        WHERE g.status = 'final' AND pm.id IS NULL
        ORDER BY g.date DESC
        LIMIT 10
    """)
    missing = c.fetchall()
    
    conn.close()
    
    print("\n📊 Box Score Collection Status")
    print("-" * 40)
    print(f"  Games with matchups: {matchups}")
    print(f"  Total completed:     {total}")
    print(f"  Missing:             {total - matchups}")
    
    if missing:
        print("\n  Recent games needing box scores:")
        for row in missing:
            print(f"    {row['date']}: {row['away_team_id']} @ {row['home_team_id']}")


def main():
    if len(sys.argv) < 2:
        # Default: collect for today
        today = datetime.now().strftime('%Y-%m-%d')
        collect_for_date(today)
    elif sys.argv[1] == '--date' and len(sys.argv) > 2:
        collect_for_date(sys.argv[2])
    elif sys.argv[1] == '--recent':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        collect_recent(days)
    elif sys.argv[1] == '--game' and len(sys.argv) > 2:
        # Single game
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM games WHERE id = ?", (sys.argv[2],))
        game = c.fetchone()
        conn.close()
        if game:
            collect_box_score(dict(game))
        else:
            print(f"Game not found: {sys.argv[2]}")
    elif sys.argv[1] == '--manual' and len(sys.argv) > 2:
        manual_entry_game(sys.argv[2])
    elif sys.argv[1] == '--status':
        show_collection_status()
    else:
        print("Usage:")
        print("  python collect_box_scores.py                    # Today's games")
        print("  python collect_box_scores.py --date 2026-02-14  # Specific date")
        print("  python collect_box_scores.py --recent 7         # Last N days")
        print("  python collect_box_scores.py --game <game_id>   # Specific game")
        print("  python collect_box_scores.py --manual <game_id> # Manual entry")
        print("  python collect_box_scores.py --status           # Show status")


if __name__ == "__main__":
    main()
