#!/usr/bin/env python3
"""
Build Pitching Infrastructure

Populates:
1. players table - Bridge player IDs from player_stats and game_pitching_stats
2. pitcher_game_log - Game-by-game pitching data with starter flags and rest days
"""

import re
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"


def normalize_name(name: str) -> str:
    """
    Normalize a player name to 'First Last' format.
    Handles:
    - "Last, First" -> "First Last"
    - "First Last (W, 1-0)" -> "First Last"
    - Extra whitespace
    """
    if not name:
        return ""
    
    # Remove decision markers like (W, 1-0), (L, 0-1), (S, 3), etc.
    name = re.sub(r'\s*\([WLSH],?\s*[\d-]+\)\s*$', '', name)
    
    # Handle "Last, First" format
    if ',' in name:
        parts = [p.strip() for p in name.split(',', 1)]
        if len(parts) == 2:
            name = f"{parts[1]} {parts[0]}"
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity ratio between two names."""
    n1 = normalize_name(name1).lower()
    n2 = normalize_name(name2).lower()
    return SequenceMatcher(None, n1, n2).ratio()


def extract_decision(name: str) -> str:
    """Extract decision (W, L, S, H) from name if present."""
    match = re.search(r'\(([WLSH]),?\s*[\d-]+\)', name)
    if match:
        return match.group(1)
    return None


def build_players_table(conn):
    """
    Populate the players table from player_stats and game_pitching_stats.
    Creates a stable ID mapping for pitchers.
    """
    c = conn.cursor()
    
    # Get all pitchers from player_stats (primary source)
    c.execute('''
        SELECT id, team_id, name, position, year, throws, 
               innings_pitched, era, whip, games_started, saves
        FROM player_stats
        WHERE innings_pitched > 0 OR position LIKE '%P%'
    ''')
    player_stats_pitchers = c.fetchall()
    
    print(f"Found {len(player_stats_pitchers)} pitchers in player_stats")
    
    # Get unique pitchers from game_pitching_stats
    c.execute('''
        SELECT DISTINCT team_id, player_name
        FROM game_pitching_stats
    ''')
    game_pitchers = c.fetchall()
    print(f"Found {len(game_pitchers)} unique pitcher appearances in game_pitching_stats")
    
    # Build team-based lookup from player_stats
    # {team_id: {normalized_name: player_stats_id}}
    team_pitcher_map = defaultdict(dict)
    for row in player_stats_pitchers:
        ps_id, team_id, name = row['id'], row['team_id'], row['name']
        normalized = normalize_name(name).lower()
        team_pitcher_map[team_id][normalized] = ps_id
        
        # Also add first initial + last name variant
        parts = normalized.split()
        if len(parts) >= 2:
            variant = f"{parts[0][0]} {' '.join(parts[1:])}"
            team_pitcher_map[team_id][variant] = ps_id
    
    # Track which game_pitching_stats entries we can match
    matched = 0
    unmatched = []
    gps_to_player_stats = {}  # (team_id, gps_name) -> player_stats_id
    
    for row in game_pitchers:
        team_id, gps_name = row['team_id'], row['player_name']
        normalized = normalize_name(gps_name).lower()
        
        # Direct match
        if normalized in team_pitcher_map.get(team_id, {}):
            gps_to_player_stats[(team_id, gps_name)] = team_pitcher_map[team_id][normalized]
            matched += 1
            continue
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        for ps_name, ps_id in team_pitcher_map.get(team_id, {}).items():
            score = name_similarity(normalized, ps_name)
            if score > best_score and score >= 0.85:  # 85% threshold
                best_score = score
                best_match = ps_id
        
        if best_match:
            gps_to_player_stats[(team_id, gps_name)] = best_match
            matched += 1
        else:
            unmatched.append((team_id, gps_name))
    
    print(f"Matched {matched}/{len(game_pitchers)} game pitchers to player_stats")
    
    if unmatched and len(unmatched) <= 20:
        print("Unmatched pitchers:")
        for team_id, name in unmatched[:20]:
            print(f"  {team_id}: {name}")
    elif unmatched:
        print(f"  ({len(unmatched)} unmatched pitchers)")
    
    # Clear existing players table
    c.execute('DELETE FROM players')
    
    # Insert pitchers into players table using player_stats as the source
    # We'll use player_stats.id as the player ID for consistency
    inserted = 0
    for row in player_stats_pitchers:
        c.execute('''
            INSERT OR IGNORE INTO players (
                id, team_id, name, position, year, throws,
                innings_pitched, earned_runs, strikeouts_pitched, walks_pitched, era, whip
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['id'], row['team_id'], row['name'], row['position'],
            row['year'], row['throws'], row['innings_pitched'],
            0, 0, 0,  # Will aggregate from game logs later
            row['era'], row['whip']
        ))
        if c.rowcount:
            inserted += 1
    
    conn.commit()
    print(f"Inserted {inserted} pitchers into players table")
    
    return gps_to_player_stats


def map_espn_ids_to_games(conn):
    """
    Map ESPN numeric game_ids in game_pitching_stats to our text game_ids.
    Returns a mapping dict: espn_id -> our_game_id
    """
    c = conn.cursor()
    
    # Find unmapped game_ids (numeric ESPN IDs)
    c.execute('''
        SELECT DISTINCT gps.game_id, gps.created_at
        FROM game_pitching_stats gps
        LEFT JOIN games g ON gps.game_id = g.id
        WHERE g.id IS NULL
    ''')
    unmapped = c.fetchall()
    
    if not unmapped:
        return {}
    
    print(f"  Mapping {len(unmapped)} ESPN game IDs to database game IDs...")
    
    mapping = {}
    for row in unmapped:
        espn_id = row['game_id']
        created_at = row['created_at']
        
        # Get teams for this ESPN game
        c.execute('''
            SELECT DISTINCT team_id FROM game_pitching_stats WHERE game_id = ?
        ''', (espn_id,))
        teams = [r['team_id'] for r in c.fetchall()]
        
        if len(teams) != 2:
            continue
        
        # Extract date from created_at (format: 2026-02-16 04:07:09)
        try:
            game_date = created_at.split()[0]
            # Also try the day before (games might be fetched after midnight)
            from datetime import datetime, timedelta
            dt = datetime.strptime(game_date, '%Y-%m-%d')
            prev_date = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
        except:
            continue
        
        # Try to find matching game
        c.execute('''
            SELECT id FROM games
            WHERE ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
            AND date IN (?, ?)
            AND status = 'final'
            LIMIT 1
        ''', (teams[0], teams[1], teams[1], teams[0], game_date, prev_date))
        
        result = c.fetchone()
        if result:
            mapping[espn_id] = result['id']
    
    print(f"  Mapped {len(mapping)} ESPN IDs successfully")
    return mapping


def build_pitcher_game_log(conn, gps_to_player_stats: dict):
    """
    Populate pitcher_game_log from game_pitching_stats.
    Determines starters and calculates rest days.
    """
    c = conn.cursor()
    
    # First, map ESPN IDs to our game IDs
    espn_mapping = map_espn_ids_to_games(conn)
    
    # Get all game pitching stats
    c.execute('''
        SELECT gps.*, COALESCE(g.date, substr(gps.created_at, 1, 10)) as date
        FROM game_pitching_stats gps
        LEFT JOIN games g ON gps.game_id = g.id
        ORDER BY date, gps.game_id, gps.team_id, gps.innings_pitched DESC, gps.id
    ''')
    all_appearances = c.fetchall()
    
    print(f"Processing {len(all_appearances)} pitching appearances")
    
    # Group by game + team to determine starter
    game_team_pitchers = defaultdict(list)
    for row in all_appearances:
        key = (row['game_id'], row['team_id'])
        game_team_pitchers[key].append(row)
    
    # Clear existing pitcher_game_log
    c.execute('DELETE FROM pitcher_game_log')
    
    # Track last appearance per player for rest calculation
    last_appearance = {}  # player_id -> date string
    
    # Process in date order
    entries_by_date = defaultdict(list)
    for key, pitchers in game_team_pitchers.items():
        game_id, team_id = key
        if pitchers:
            date = pitchers[0]['date']
            entries_by_date[date].append((key, pitchers))
    
    inserted = 0
    matched_entries = 0
    skipped_no_game = 0
    
    for date in sorted(entries_by_date.keys()):
        for (orig_game_id, team_id), pitchers in entries_by_date[date]:
            # Map ESPN ID to our game ID if needed
            game_id = espn_mapping.get(orig_game_id, orig_game_id)
            
            # Verify the game exists
            c.execute('SELECT 1 FROM games WHERE id = ?', (game_id,))
            if not c.fetchone():
                skipped_no_game += len(pitchers)
                continue
            
            # First pitcher (most IP) is the starter
            for idx, row in enumerate(pitchers):
                player_name = row['player_name']
                
                # Get player_stats_id if we matched this pitcher
                player_id = gps_to_player_stats.get((team_id, player_name))
                
                if player_id is None:
                    # Create a placeholder - use a hash-based ID
                    # We'll use negative IDs for unmatched pitchers
                    name_hash = hash(f"{team_id}:{normalize_name(player_name)}") % 100000
                    player_id = -name_hash
                else:
                    matched_entries += 1
                
                was_starter = 1 if idx == 0 else 0
                
                # Calculate rest days
                rest_days = None
                if player_id in last_appearance:
                    try:
                        last_date = datetime.strptime(last_appearance[player_id], '%Y-%m-%d')
                        current_date = datetime.strptime(date, '%Y-%m-%d')
                        rest_days = (current_date - last_date).days
                    except:
                        pass
                
                # Determine decision
                decision = extract_decision(player_name)
                if decision is None:
                    if row['win']:
                        decision = 'W'
                    elif row['loss']:
                        decision = 'L'
                    elif row['save']:
                        decision = 'S'
                
                c.execute('''
                    INSERT INTO pitcher_game_log (
                        game_id, player_id, team_id, was_starter,
                        innings_pitched, hits_allowed, runs_allowed, earned_runs,
                        walks, strikeouts, pitches, strikes, decision, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game_id, player_id, team_id, was_starter,
                    row['innings_pitched'], row['hits_allowed'], row['runs_allowed'],
                    row['earned_runs'], row['walks'], row['strikeouts'],
                    row['pitches'], row['strikes'], decision,
                    f"rest_days={rest_days}" if rest_days is not None else None
                ))
                inserted += 1
                
                # Update last appearance
                last_appearance[player_id] = date
    
    if skipped_no_game:
        print(f"  Skipped {skipped_no_game} appearances (game not found)")
    
    conn.commit()
    print(f"Inserted {inserted} pitcher_game_log entries")
    print(f"  {matched_entries} entries linked to player_stats")
    print(f"  {inserted - matched_entries} entries with temporary IDs (unmatched)")
    
    # Show starter detection stats
    c.execute('SELECT COUNT(*) FROM pitcher_game_log WHERE was_starter = 1')
    starters = c.fetchone()[0]
    print(f"  {starters} entries marked as starter")


def add_rest_days_column(conn):
    """Add rest_days column to pitcher_game_log if it doesn't exist."""
    c = conn.cursor()
    try:
        c.execute('ALTER TABLE pitcher_game_log ADD COLUMN rest_days INTEGER')
        conn.commit()
        print("Added rest_days column to pitcher_game_log")
    except sqlite3.OperationalError:
        pass  # Column already exists


def calculate_rest_days(conn):
    """Calculate and store rest_days for each pitcher appearance."""
    c = conn.cursor()
    
    # Get all appearances ordered by player and date
    c.execute('''
        SELECT pgl.id, pgl.player_id, g.date
        FROM pitcher_game_log pgl
        JOIN games g ON pgl.game_id = g.id
        ORDER BY pgl.player_id, g.date
    ''')
    appearances = c.fetchall()
    
    last_appearance = {}
    updates = []
    
    for row in appearances:
        log_id, player_id, date = row['id'], row['player_id'], row['date']
        
        rest_days = None
        if player_id in last_appearance:
            try:
                last_date = datetime.strptime(last_appearance[player_id], '%Y-%m-%d')
                current_date = datetime.strptime(date, '%Y-%m-%d')
                rest_days = (current_date - last_date).days
            except:
                pass
        
        if rest_days is not None:
            updates.append((rest_days, log_id))
        
        last_appearance[player_id] = date
    
    # Batch update
    c.executemany('UPDATE pitcher_game_log SET rest_days = ? WHERE id = ?', updates)
    conn.commit()
    print(f"Updated rest_days for {len(updates)} appearances")


def show_summary(conn):
    """Show summary statistics."""
    c = conn.cursor()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    c.execute('SELECT COUNT(*) FROM players')
    print(f"Players table: {c.fetchone()[0]} rows")
    
    c.execute('SELECT COUNT(*) FROM pitcher_game_log')
    print(f"Pitcher game log: {c.fetchone()[0]} rows")
    
    c.execute('SELECT COUNT(*) FROM pitcher_game_log WHERE was_starter = 1')
    print(f"  Starters: {c.fetchone()[0]}")
    
    c.execute('SELECT COUNT(*) FROM pitcher_game_log WHERE player_id > 0')
    print(f"  Linked to player_stats: {c.fetchone()[0]}")
    
    c.execute('''
        SELECT COUNT(DISTINCT team_id) as teams, 
               COUNT(DISTINCT game_id) as games,
               COUNT(DISTINCT player_id) as pitchers
        FROM pitcher_game_log
    ''')
    row = c.fetchone()
    print(f"  Teams: {row['teams']}, Games: {row['games']}, Unique pitchers: {row['pitchers']}")
    
    # Show sample starters
    print("\nSample recent starters:")
    c.execute('''
        SELECT g.date, pgl.team_id, ps.name as player_name, 
               pgl.innings_pitched, pgl.earned_runs, pgl.strikeouts
        FROM pitcher_game_log pgl
        JOIN games g ON pgl.game_id = g.id
        LEFT JOIN player_stats ps ON pgl.player_id = ps.id
        WHERE pgl.was_starter = 1 AND ps.name IS NOT NULL
        ORDER BY g.date DESC
        LIMIT 10
    ''')
    for row in c.fetchall():
        print(f"  {row['date']} {row['team_id']}: {row['player_name']} - {row['innings_pitched']}IP, {row['earned_runs']}ER, {row['strikeouts']}K")


def main():
    print("Building Pitching Infrastructure")
    print("="*50)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        # Step 1: Build players table
        print("\nStep 1: Building players table...")
        gps_to_player_stats = build_players_table(conn)
        
        # Step 2: Add rest_days column
        print("\nStep 2: Adding rest_days column...")
        add_rest_days_column(conn)
        
        # Step 3: Build pitcher_game_log
        print("\nStep 3: Building pitcher_game_log...")
        build_pitcher_game_log(conn, gps_to_player_stats)
        
        # Step 4: Calculate rest days
        print("\nStep 4: Calculating rest days...")
        calculate_rest_days(conn)
        
        # Summary
        show_summary(conn)
        
    finally:
        conn.close()
    
    print("\n✓ Pitching infrastructure build complete")


if __name__ == "__main__":
    main()
