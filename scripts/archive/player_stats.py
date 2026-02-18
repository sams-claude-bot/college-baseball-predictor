#!/usr/bin/env python3
"""
Track player stats for pitching and hitting matchups

Focus: Mississippi State roster with detailed stats
"""

import sys
import json
from pathlib import Path
from datetime import datetime

_scripts_dir = Path(__file__).parent
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from scripts.database import get_connection

def init_player_tables():
    """Ensure player tables exist with full stats"""
    conn = get_connection()
    c = conn.cursor()
    
    # Enhanced players table
    c.execute('''
        CREATE TABLE IF NOT EXISTS player_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            name TEXT NOT NULL,
            number INTEGER,
            position TEXT,
            year TEXT,
            bats TEXT,
            throws TEXT,
            height TEXT,
            weight INTEGER,
            hometown TEXT,
            -- Batting stats
            games INTEGER DEFAULT 0,
            at_bats INTEGER DEFAULT 0,
            runs INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0,
            doubles INTEGER DEFAULT 0,
            triples INTEGER DEFAULT 0,
            home_runs INTEGER DEFAULT 0,
            rbi INTEGER DEFAULT 0,
            walks INTEGER DEFAULT 0,
            strikeouts INTEGER DEFAULT 0,
            stolen_bases INTEGER DEFAULT 0,
            caught_stealing INTEGER DEFAULT 0,
            batting_avg REAL DEFAULT 0,
            obp REAL DEFAULT 0,
            slg REAL DEFAULT 0,
            ops REAL DEFAULT 0,
            -- Pitching stats
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            era REAL DEFAULT 0,
            games_pitched INTEGER DEFAULT 0,
            games_started INTEGER DEFAULT 0,
            saves INTEGER DEFAULT 0,
            innings_pitched REAL DEFAULT 0,
            hits_allowed INTEGER DEFAULT 0,
            runs_allowed INTEGER DEFAULT 0,
            earned_runs INTEGER DEFAULT 0,
            walks_allowed INTEGER DEFAULT 0,
            strikeouts_pitched INTEGER DEFAULT 0,
            whip REAL DEFAULT 0,
            k_per_9 REAL DEFAULT 0,
            bb_per_9 REAL DEFAULT 0,
            -- Metadata
            is_starter INTEGER DEFAULT 0,
            is_closer INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, name)
        )
    ''')
    
    # Game pitching matchups
    c.execute('''
        CREATE TABLE IF NOT EXISTS pitching_matchups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            home_starter_id INTEGER,
            away_starter_id INTEGER,
            home_starter_name TEXT,
            away_starter_name TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (home_starter_id) REFERENCES player_stats(id),
            FOREIGN KEY (away_starter_id) REFERENCES player_stats(id)
        )
    ''')
    
    conn.commit()
    conn.close()


def add_player(team_id, name, number=None, position=None, year=None, 
               bats=None, throws=None, **stats):
    """Add or update a player"""
    init_player_tables()
    conn = get_connection()
    c = conn.cursor()
    
    # Build dynamic update
    fields = ['team_id', 'name', 'number', 'position', 'year', 'bats', 'throws']
    values = [team_id, name, number, position, year, bats, throws]
    
    stat_fields = [
        'games', 'at_bats', 'runs', 'hits', 'doubles', 'triples', 'home_runs',
        'rbi', 'walks', 'strikeouts', 'stolen_bases', 'batting_avg', 'obp', 'slg', 'ops',
        'wins', 'losses', 'era', 'games_pitched', 'games_started', 'saves',
        'innings_pitched', 'hits_allowed', 'earned_runs', 'walks_allowed',
        'strikeouts_pitched', 'whip', 'k_per_9', 'is_starter', 'is_closer'
    ]
    
    for field in stat_fields:
        if field in stats:
            fields.append(field)
            values.append(stats[field])
    
    placeholders = ','.join(['?' for _ in values])
    field_str = ','.join(fields)
    
    # Upsert
    update_str = ','.join([f"{f}=excluded.{f}" for f in fields if f not in ['team_id', 'name']])
    
    c.execute(f'''
        INSERT INTO player_stats ({field_str})
        VALUES ({placeholders})
        ON CONFLICT(team_id, name) DO UPDATE SET
            {update_str},
            updated_at=CURRENT_TIMESTAMP
    ''', values)
    
    conn.commit()
    player_id = c.lastrowid
    conn.close()
    
    return player_id


def get_team_roster(team_id, position_filter=None):
    """Get team roster with stats"""
    conn = get_connection()
    c = conn.cursor()
    
    query = "SELECT * FROM player_stats WHERE team_id = ?"
    params = [team_id]
    
    if position_filter:
        if position_filter == 'pitchers':
            query += " AND (position LIKE '%P%' OR games_pitched > 0)"
        elif position_filter == 'hitters':
            query += " AND position NOT LIKE '%P%' AND at_bats > 0"
    
    query += " ORDER BY position, name"
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_starting_pitchers(team_id):
    """Get team's starting pitchers"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM player_stats 
        WHERE team_id = ? AND (is_starter = 1 OR games_started > 0)
        ORDER BY games_started DESC, wins DESC
    ''', (team_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_top_hitters(team_id, limit=9):
    """Get team's top hitters by batting average"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM player_stats 
        WHERE team_id = ? AND at_bats >= 10
        ORDER BY batting_avg DESC
        LIMIT ?
    ''', (team_id, limit))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def set_pitching_matchup(game_id, home_starter_name, away_starter_name, 
                          home_team_id=None, away_team_id=None):
    """Set the pitching matchup for a game"""
    init_player_tables()
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT OR REPLACE INTO pitching_matchups 
        (game_id, home_starter_name, away_starter_name)
        VALUES (?, ?, ?)
    ''', (game_id, home_starter_name, away_starter_name))
    
    conn.commit()
    conn.close()


def get_pitching_matchup(game_id):
    """Get pitching matchup for a game"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM pitching_matchups WHERE game_id = ?
    ''', (game_id,))
    
    row = c.fetchone()
    conn.close()
    
    return dict(row) if row else None


def seed_mississippi_state_roster():
    """Seed Mississippi State 2026 roster (sample data - update with real stats)"""
    
    # Sample pitchers (these would be updated with real 2026 data)
    pitchers = [
        {"name": "Jurrangelo Cijntje", "number": 18, "position": "RHP", "year": "Jr.", "throws": "R", 
         "is_starter": 1, "games_started": 0, "wins": 0, "losses": 0, "era": 0.00, "innings_pitched": 0,
         "strikeouts_pitched": 0, "walks_allowed": 0, "whip": 0.00},
        {"name": "Khal Stephen", "number": 22, "position": "RHP", "year": "Sr.", "throws": "R",
         "is_starter": 1, "games_started": 0, "wins": 0, "losses": 0, "era": 0.00},
        {"name": "Luke Hales", "number": 33, "position": "LHP", "year": "Jr.", "throws": "L",
         "is_starter": 1, "games_started": 0, "wins": 0, "losses": 0, "era": 0.00},
        {"name": "Brandon Smith", "number": 45, "position": "RHP", "year": "So.", "throws": "R",
         "is_closer": 1, "saves": 0, "era": 0.00},
        {"name": "Nate Dohm", "number": 29, "position": "RHP", "year": "Jr.", "throws": "R",
         "games_pitched": 0, "era": 0.00},
    ]
    
    # Sample hitters
    hitters = [
        {"name": "Dakota Jordan", "number": 7, "position": "OF", "year": "Sr.", "bats": "L",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0, "ops": .000},
        {"name": "Amani Larry", "number": 3, "position": "OF", "year": "Sr.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "Hunter Hines", "number": 24, "position": "1B", "year": "Jr.", "bats": "L",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "RJ Yeager", "number": 9, "position": "SS", "year": "Jr.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "Slate Alford", "number": 4, "position": "2B", "year": "So.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "Luke Hancock", "number": 20, "position": "3B", "year": "Sr.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "Logan Tanner", "number": 11, "position": "C", "year": "Gr.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "Matt Corder", "number": 8, "position": "OF", "year": "Gr.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
        {"name": "Kelsey Johnson", "number": 2, "position": "DH/UTL", "year": "Sr.", "bats": "R",
         "games": 0, "at_bats": 0, "batting_avg": .000, "home_runs": 0, "rbi": 0},
    ]
    
    team_id = "mississippi-state"
    
    print(f"\n⚾ Seeding Mississippi State roster...")
    
    for p in pitchers:
        add_player(team_id, **p)
        print(f"   + {p['name']} ({p['position']})")
    
    for h in hitters:
        add_player(team_id, **h)
        print(f"   + {h['name']} ({h['position']})")
    
    print(f"\n✓ Added {len(pitchers)} pitchers and {len(hitters)} hitters")


def print_matchup_report(team_id, opponent_id, game_date=None):
    """Print a matchup report for a game"""
    
    print(f"\n{'='*60}")
    print(f"⚾ MATCHUP REPORT")
    print(f"{'='*60}")
    
    # Get starters
    home_starters = get_starting_pitchers(team_id)
    away_starters = get_starting_pitchers(opponent_id)
    
    print(f"\n🎯 PROBABLE PITCHING MATCHUP")
    print("-" * 40)
    
    if home_starters:
        sp = home_starters[0]
        print(f"{team_id.upper()} STARTER:")
        print(f"   {sp['name']} ({sp['position']}, {sp['year']})")
        print(f"   Record: {sp['wins']}-{sp['losses']} | ERA: {sp['era']:.2f}")
        if sp['innings_pitched']:
            print(f"   IP: {sp['innings_pitched']:.1f} | K: {sp['strikeouts_pitched']} | WHIP: {sp['whip']:.2f}")
    else:
        print(f"{team_id.upper()}: TBA")
    
    print()
    
    if away_starters:
        sp = away_starters[0]
        print(f"{opponent_id.upper()} STARTER:")
        print(f"   {sp['name']} ({sp['position']}, {sp['year']})")
        print(f"   Record: {sp['wins']}-{sp['losses']} | ERA: {sp['era']:.2f}")
    else:
        print(f"{opponent_id.upper()}: TBA")
    
    # Top hitters
    print(f"\n🏏 TOP HITTERS")
    print("-" * 40)
    
    home_hitters = get_top_hitters(team_id, 5)
    if home_hitters:
        print(f"\n{team_id.upper()}:")
        print(f"{'Name':<20} {'AVG':<6} {'HR':<4} {'RBI':<4} {'OPS':<6}")
        for h in home_hitters:
            print(f"{h['name']:<20} {h['batting_avg']:.3f}  {h['home_runs']:<4} {h['rbi']:<4} {h['ops']:.3f}")
    
    away_hitters = get_top_hitters(opponent_id, 5)
    if away_hitters:
        print(f"\n{opponent_id.upper()}:")
        print(f"{'Name':<20} {'AVG':<6} {'HR':<4} {'RBI':<4} {'OPS':<6}")
        for h in away_hitters:
            print(f"{h['name']:<20} {h['batting_avg']:.3f}  {h['home_runs']:<4} {h['rbi']:<4} {h['ops']:.3f}")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python player_stats.py seed              - Seed MS State roster")
        print("  python player_stats.py roster <team>     - Show team roster")
        print("  python player_stats.py pitchers <team>   - Show pitchers")
        print("  python player_stats.py hitters <team>    - Show top hitters")
        print("  python player_stats.py matchup <home> <away> - Show matchup report")
        print("  python player_stats.py add <team> <name> <pos> [stats...]")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "seed":
        seed_mississippi_state_roster()
    
    elif cmd == "roster":
        team = sys.argv[2] if len(sys.argv) > 2 else "mississippi-state"
        roster = get_team_roster(team)
        print(f"\n{team.upper()} ROSTER ({len(roster)} players):")
        for p in roster:
            print(f"  #{p['number'] or '?'} {p['name']} - {p['position']} ({p['year']})")
    
    elif cmd == "pitchers":
        team = sys.argv[2] if len(sys.argv) > 2 else "mississippi-state"
        pitchers = get_starting_pitchers(team)
        print(f"\n{team.upper()} PITCHERS:")
        for p in pitchers:
            era = f"{p['era']:.2f}" if p['era'] else "-.--"
            print(f"  {p['name']} ({p['position']}) - {p['wins']}-{p['losses']}, {era} ERA")
    
    elif cmd == "hitters":
        team = sys.argv[2] if len(sys.argv) > 2 else "mississippi-state"
        hitters = get_top_hitters(team)
        print(f"\n{team.upper()} TOP HITTERS:")
        for h in hitters:
            print(f"  {h['name']} ({h['position']}) - .{int(h['batting_avg']*1000):03d}, {h['home_runs']} HR, {h['rbi']} RBI")
    
    elif cmd == "matchup":
        home = sys.argv[2] if len(sys.argv) > 2 else "mississippi-state"
        away = sys.argv[3] if len(sys.argv) > 3 else "hofstra"
        print_matchup_report(home, away)
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
