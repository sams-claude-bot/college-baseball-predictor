#!/usr/bin/env python3
"""
Starting Pitcher Tracking

Tracks starting pitchers, days rest, and pitch counts for the prediction model.

Usage:
    python track_starters.py <team_id>              # Show team's rotation
    python track_starters.py set <game_id> <home> <away>  # Set starters
    python track_starters.py history <player_id>    # Show pitcher history
    python track_starters.py rest <team_id>         # Show days rest for pitchers
    python track_starters.py project <team_id>      # Project weekend rotation
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))

from database import get_connection
from player_stats import init_player_tables


def get_team_starters(team_id, min_starts=1):
    """
    Get all pitchers who have started games for a team
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT ps.*, 
               COUNT(DISTINCT pm.game_id) as actual_starts
        FROM player_stats ps
        LEFT JOIN pitching_matchups pm ON (
            (pm.home_starter_id = ps.id OR pm.away_starter_id = ps.id)
        )
        WHERE ps.team_id = ?
        AND (ps.is_starter = 1 OR ps.games_started > 0 OR ps.position LIKE '%P%')
        GROUP BY ps.id
        HAVING actual_starts >= ? OR ps.is_starter = 1 OR ps.games_started > 0
        ORDER BY ps.games_started DESC, ps.era ASC
    """, (team_id, min_starts))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_pitcher_starts(player_id, limit=10):
    """
    Get a pitcher's recent starts with game details
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT g.date, g.home_team_id, g.away_team_id, g.home_score, g.away_score,
               CASE WHEN pm.home_starter_id = ? THEN 'home' ELSE 'away' END as role,
               ps.last_start_pitches as pitches
        FROM pitching_matchups pm
        JOIN games g ON pm.game_id = g.id
        JOIN player_stats ps ON ps.id = ?
        WHERE pm.home_starter_id = ? OR pm.away_starter_id = ?
        ORDER BY g.date DESC
        LIMIT ?
    """, (player_id, player_id, player_id, player_id, limit))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_last_start_date(player_id):
    """
    Get the date of a pitcher's last start
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT MAX(g.date) as last_start
        FROM pitching_matchups pm
        JOIN games g ON pm.game_id = g.id
        WHERE pm.home_starter_id = ? OR pm.away_starter_id = ?
    """, (player_id, player_id))
    
    row = c.fetchone()
    conn.close()
    
    if row and row['last_start']:
        return datetime.strptime(row['last_start'], '%Y-%m-%d')
    return None


def calculate_days_rest(player_id, as_of_date=None):
    """
    Calculate days since last start
    """
    if as_of_date is None:
        as_of_date = datetime.now()
    elif isinstance(as_of_date, str):
        as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d')
    
    last_start = get_last_start_date(player_id)
    
    if last_start:
        return (as_of_date - last_start).days
    return None  # No recorded starts


def set_pitching_matchup(game_id, home_starter_name, away_starter_name):
    """
    Set the pitching matchup for a game by player names
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Get game info
    c.execute("""
        SELECT home_team_id, away_team_id, date FROM games WHERE id = ?
    """, (game_id,))
    game = c.fetchone()
    
    if not game:
        print(f"Game not found: {game_id}")
        conn.close()
        return False
    
    home_team_id = game['home_team_id']
    away_team_id = game['away_team_id']
    game_date = game['date']
    
    # Find home starter
    home_starter_id = None
    if home_starter_name:
        c.execute("""
            SELECT id FROM player_stats 
            WHERE team_id = ? AND LOWER(name) LIKE ?
        """, (home_team_id, f'%{home_starter_name.lower()}%'))
        row = c.fetchone()
        if row:
            home_starter_id = row['id']
    
    # Find away starter
    away_starter_id = None
    if away_starter_name:
        c.execute("""
            SELECT id FROM player_stats 
            WHERE team_id = ? AND LOWER(name) LIKE ?
        """, (away_team_id, f'%{away_starter_name.lower()}%'))
        row = c.fetchone()
        if row:
            away_starter_id = row['id']
    
    # Insert/update matchup
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
    
    print(f"✓ Set matchup for {game_id}:")
    print(f"  Home: {home_starter_name} (ID: {home_starter_id})")
    print(f"  Away: {away_starter_name} (ID: {away_starter_id})")
    
    return True


def get_team_rotation_status(team_id):
    """
    Get current rotation status with days rest
    """
    starters = get_team_starters(team_id, min_starts=0)
    today = datetime.now()
    
    rotation = []
    for pitcher in starters:
        days_rest = calculate_days_rest(pitcher['id'], today)
        
        rotation.append({
            'id': pitcher['id'],
            'name': pitcher['name'],
            'position': pitcher['position'],
            'era': pitcher.get('era', 0.0),
            'record': f"{pitcher.get('wins', 0)}-{pitcher.get('losses', 0)}",
            'starts': pitcher.get('games_started', 0),
            'innings': pitcher.get('innings_pitched', 0),
            'days_rest': days_rest,
            'avg_pitches': pitcher.get('avg_pitch_count', 0),
            'last_pitches': pitcher.get('last_start_pitches', 0)
        })
    
    return sorted(rotation, key=lambda x: (x['starts'] or 0), reverse=True)


def project_weekend_rotation(team_id, start_date=None):
    """
    Project likely starters for upcoming weekend series (Fri/Sat/Sun)
    """
    if start_date is None:
        # Find next Friday
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0 and today.hour >= 12:
            days_until_friday = 7
        start_date = today + timedelta(days=days_until_friday)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    rotation = get_team_rotation_status(team_id)
    
    # Sort by days rest (prefer well-rested) and experience (starts)
    def rotation_score(pitcher):
        rest = pitcher['days_rest'] if pitcher['days_rest'] is not None else 7
        starts = pitcher['starts'] or 0
        era = pitcher['era'] or 99
        
        # Score: prefer more rest (up to 7), more starts, lower ERA
        return (min(rest, 7), starts, -era)
    
    available = [p for p in rotation if p['days_rest'] is None or p['days_rest'] >= 3]
    available.sort(key=rotation_score, reverse=True)
    
    projection = {
        'friday': available[0] if len(available) > 0 else None,
        'saturday': available[1] if len(available) > 1 else None,
        'sunday': available[2] if len(available) > 2 else None,
    }
    
    return projection


def show_rotation(team_id):
    """
    Display team's rotation status
    """
    rotation = get_team_rotation_status(team_id)
    
    print(f"\n⚾ {team_id.upper()} ROTATION STATUS")
    print("=" * 70)
    print(f"{'Name':<25} {'Pos':<6} {'W-L':<6} {'ERA':<6} {'GS':<4} {'IP':<6} {'Rest':<5} {'PC':<4}")
    print("-" * 70)
    
    for p in rotation[:8]:  # Top 8 pitchers
        rest = str(p['days_rest']) + 'd' if p['days_rest'] is not None else '--'
        pc = int(p['avg_pitches']) if p['avg_pitches'] else '--'
        era = f"{p['era']:.2f}" if p['era'] else '--'
        ip = f"{p['innings']:.1f}" if p['innings'] else '0.0'
        
        print(f"{p['name']:<25} {p['position'] or '--':<6} {p['record']:<6} {era:<6} {p['starts'] or 0:<4} {ip:<6} {rest:<5} {pc}")
    
    # Project weekend
    print("\n📅 PROJECTED WEEKEND ROTATION")
    print("-" * 40)
    projection = project_weekend_rotation(team_id)
    
    for day, pitcher in projection.items():
        if pitcher:
            print(f"  {day.title()}: {pitcher['name']} ({pitcher['era']:.2f} ERA, {pitcher['days_rest'] or '?'}d rest)")
        else:
            print(f"  {day.title()}: TBD")


def show_pitcher_history(player_id):
    """
    Display a pitcher's start history
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT * FROM player_stats WHERE id = ?", (player_id,))
    pitcher = c.fetchone()
    conn.close()
    
    if not pitcher:
        print(f"Player not found: {player_id}")
        return
    
    pitcher = dict(pitcher)
    
    print(f"\n📊 PITCHER HISTORY: {pitcher['name']}")
    print("=" * 60)
    print(f"Team: {pitcher['team_id']}")
    print(f"Position: {pitcher['position']}")
    print(f"Record: {pitcher.get('wins', 0)}-{pitcher.get('losses', 0)}")
    print(f"ERA: {pitcher.get('era', 0):.2f}")
    print(f"Starts: {pitcher.get('games_started', 0)}")
    print(f"IP: {pitcher.get('innings_pitched', 0):.1f}")
    print(f"K: {pitcher.get('strikeouts_pitched', 0)}")
    print(f"BB: {pitcher.get('walks_allowed', 0)}")
    print(f"WHIP: {pitcher.get('whip', 0):.2f}")
    print(f"K/9: {pitcher.get('k_per_9', 0):.1f}")
    
    # Recent starts
    starts = get_pitcher_starts(player_id)
    if starts:
        print(f"\nRecent Starts:")
        print("-" * 40)
        for start in starts:
            opp = start['away_team_id'] if start['role'] == 'home' else start['home_team_id']
            loc = 'vs' if start['role'] == 'home' else '@'
            score = f"{start['home_score']}-{start['away_score']}"
            pitches = start['pitches'] or '?'
            print(f"  {start['date']}: {loc} {opp} ({score}) - {pitches} pitches")


def update_pitch_count(player_id, game_date, pitch_count):
    """
    Update pitch count for a specific game
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        UPDATE player_stats SET
            last_start_date = ?,
            last_start_pitches = ?,
            season_pitches = season_pitches + ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (game_date, pitch_count, pitch_count, player_id))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Updated pitch count: {pitch_count} pitches on {game_date}")


def show_days_rest_report(team_id=None):
    """
    Show days rest for all tracked starters
    """
    conn = get_connection()
    c = conn.cursor()
    
    query = """
        SELECT ps.id, ps.name, ps.team_id, ps.era, ps.games_started,
               MAX(g.date) as last_start
        FROM player_stats ps
        LEFT JOIN pitching_matchups pm ON (pm.home_starter_id = ps.id OR pm.away_starter_id = ps.id)
        LEFT JOIN games g ON pm.game_id = g.id
        WHERE ps.games_started > 0 OR ps.is_starter = 1
    """
    
    if team_id:
        query += " AND ps.team_id = ?"
        c.execute(query + " GROUP BY ps.id ORDER BY ps.team_id, last_start DESC", (team_id,))
    else:
        c.execute(query + " GROUP BY ps.id ORDER BY ps.team_id, last_start DESC")
    
    rows = c.fetchall()
    conn.close()
    
    print("\n📅 DAYS REST REPORT")
    print("=" * 70)
    print(f"{'Team':<18} {'Pitcher':<25} {'ERA':<6} {'GS':<4} {'Last Start':<12} {'Rest'}")
    print("-" * 70)
    
    today = datetime.now()
    current_team = None
    
    for row in rows:
        if row['team_id'] != current_team:
            if current_team:
                print()
            current_team = row['team_id']
        
        if row['last_start']:
            last = datetime.strptime(row['last_start'], '%Y-%m-%d')
            rest = (today - last).days
            rest_str = f"{rest}d"
        else:
            rest_str = "--"
        
        era = f"{row['era']:.2f}" if row['era'] else "--"
        last_str = row['last_start'] or '--'
        
        print(f"{row['team_id']:<18} {row['name']:<25} {era:<6} {row['games_started'] or 0:<4} {last_str:<12} {rest_str}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python track_starters.py <team_id>               # Show rotation")
        print("  python track_starters.py set <game> <home> <away> # Set matchup")
        print("  python track_starters.py history <player_id>     # Pitcher history")
        print("  python track_starters.py rest [team_id]          # Days rest report")
        print("  python track_starters.py project <team_id>       # Weekend projection")
        return
    
    cmd = sys.argv[1]
    
    if cmd == 'set' and len(sys.argv) >= 5:
        game_id = sys.argv[2]
        home_starter = sys.argv[3]
        away_starter = sys.argv[4]
        set_pitching_matchup(game_id, home_starter, away_starter)
    
    elif cmd == 'history' and len(sys.argv) >= 3:
        player_id = int(sys.argv[2])
        show_pitcher_history(player_id)
    
    elif cmd == 'rest':
        team_id = sys.argv[2] if len(sys.argv) > 2 else None
        show_days_rest_report(team_id)
    
    elif cmd == 'project' and len(sys.argv) >= 3:
        team_id = sys.argv[2]
        print(f"\n📅 WEEKEND PROJECTION: {team_id.upper()}")
        print("-" * 40)
        projection = project_weekend_rotation(team_id)
        for day, pitcher in projection.items():
            if pitcher:
                rest = pitcher['days_rest'] if pitcher['days_rest'] is not None else '?'
                print(f"  {day.title()}: {pitcher['name']} ({pitcher['era']:.2f} ERA, {rest}d rest)")
            else:
                print(f"  {day.title()}: TBD")
    
    elif cmd in ['--help', '-h']:
        main()  # Show usage
    
    else:
        # Assume it's a team_id
        show_rotation(cmd)


if __name__ == "__main__":
    main()
