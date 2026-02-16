#!/usr/bin/env python3
"""
Merge duplicate teams (Unknown conference) into their canonical versions.
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"
ESPN_IDS_PATH = Path(__file__).parent.parent / "data" / "espn_team_ids.json"

# Manual mapping for known duplicates
MANUAL_MAPPINGS = {
    "tbd": None,  # Not a real team
    "seuniv-fires": None,  # Not identifiable
    "gardnerwebb-runnin": "gardner-webb",
    "uncw": "unc-wilmington",
    "william-mary-tribe": "william-mary",
}

# Common mascot suffixes to strip
SUFFIXES = [
    "-bearcats", "-fighting-camels", "-golden-griffins", "-cougars", "-49ers",
    "-bluejays", "-blue-hens", "-blue-devils", "-pirates", "-colonels",
    "-phoenix", "-runnin", "-patriots", "-hoyas", "-tigers", "-crimson",
    "-pride", "-leopards", "-cardinals", "-beach", "-lancers", "-lions",
    "-bulldogs", "-delta-devils", "-privateers", "-demons", "-blue-hose",
    "-highlanders", "-spiders", "-fires", "-bulls", "-golden-eagles",
    "-horned-frogs", "-green-wave", "-blazers", "-seahawks", "-roadrunners",
    "-rams", "-cavaliers", "-hokies", "-keydets", "-shockers", "-tribe"
]


def get_canonical_id(dupe_id, existing_teams):
    """Try to find the canonical team ID for a duplicate."""
    if dupe_id in MANUAL_MAPPINGS:
        return MANUAL_MAPPINGS[dupe_id]
    
    for suffix in SUFFIXES:
        if dupe_id.endswith(suffix):
            base = dupe_id[:-len(suffix)]
            if base in existing_teams:
                return base
    return None


def merge_games(cur, dupe_id, canonical_id):
    """Merge games from dupe to canonical, avoiding conflicts."""
    
    # Get all games involving the dupe
    cur.execute("""
        SELECT id, date, home_team_id, away_team_id, home_score, away_score
        FROM games 
        WHERE home_team_id = ? OR away_team_id = ?
    """, (dupe_id, dupe_id))
    dupe_games = cur.fetchall()
    
    merged = 0
    deleted = 0
    
    for game in dupe_games:
        game_id, date, home_id, away_id, home_score, away_score = game
        
        # Determine what the canonical game would look like
        new_home = canonical_id if home_id == dupe_id else home_id
        new_away = canonical_id if away_id == dupe_id else away_id
        
        # Check if a game already exists with the canonical team on the same date vs same opponent
        opponent = new_away if new_home == canonical_id else new_home
        
        cur.execute("""
            SELECT id FROM games 
            WHERE date = ? 
            AND ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
        """, (date, canonical_id, opponent, opponent, canonical_id))
        
        existing = cur.fetchone()
        
        if existing:
            # Conflict - delete the dupe's game
            cur.execute("DELETE FROM games WHERE id = ?", (game_id,))
            deleted += 1
        else:
            # No conflict - update to canonical
            cur.execute("""
                UPDATE games 
                SET home_team_id = ?, away_team_id = ?
                WHERE id = ?
            """, (new_home, new_away, game_id))
            merged += 1
    
    return merged, deleted


def delete_from_table(cur, table, column, team_id):
    """Delete all rows from table where column matches team_id."""
    cur.execute(f"DELETE FROM {table} WHERE {column} = ?", (team_id,))
    return cur.rowcount


def merge_or_delete_from_table(cur, table, column, dupe_id, canonical_id):
    """Try to update records, delete if conflicts exist."""
    # For most tables, just delete dupe records - canonical is source of truth
    return delete_from_table(cur, table, column, dupe_id)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get all teams with real conferences
    cur.execute("SELECT id FROM teams WHERE conference != 'Unknown'")
    existing_teams = {row['id'] for row in cur.fetchall()}
    
    # Get all Unknown teams
    cur.execute("SELECT id, name FROM teams WHERE conference = 'Unknown'")
    unknown_teams = cur.fetchall()
    
    print(f"Found {len(unknown_teams)} teams with Unknown conference")
    
    # Build mappings
    mappings = {}
    to_delete = []
    
    for row in unknown_teams:
        dupe_id = row['id']
        canonical = get_canonical_id(dupe_id, existing_teams)
        
        if canonical is None:
            to_delete.append(dupe_id)
        else:
            mappings[dupe_id] = canonical
    
    # Stats
    total_games_merged = 0
    total_games_deleted = 0
    teams_merged = 0
    
    # Tables that reference team_id (excluding elo_ratings which we just delete)
    team_tables = [
        ("players", "team_id"),
        ("team_stats", "team_id"),
        ("player_stats", "team_id"),
        ("preseason_priors", "team_id"),
        ("pitcher_game_log", "team_id"),
        ("ncaa_team_stats", "team_id"),
        ("ncaa_individual_stats", "team_id"),
        ("player_boxscore_batting", "team_id"),
        ("player_boxscore_pitching", "team_id"),
        ("game_batting_stats", "team_id"),
        ("game_pitching_stats", "team_id"),
        ("team_aggregate_stats", "team_id"),
        ("team_sos", "team_id"),
        ("power_rankings", "team_id"),
        ("elo_history", "team_id"),
        ("rankings_history", "team_id"),
    ]
    
    # Tables with home/away team refs
    game_like_tables = [
        ("betting_lines", "home_team_id", "away_team_id"),
        ("model_predictions", "home_team_id", "away_team_id") if False else None,  # Check if exists
    ]
    
    print("\n=== MERGING TEAMS ===")
    
    for dupe_id, canonical_id in mappings.items():
        print(f"\n{dupe_id} -> {canonical_id}")
        
        # Merge games
        merged, deleted = merge_games(cur, dupe_id, canonical_id)
        total_games_merged += merged
        total_games_deleted += deleted
        print(f"  Games: {merged} merged, {deleted} deleted (conflicts)")
        
        # Handle betting_lines
        cur.execute("""
            UPDATE betting_lines SET home_team_id = ? WHERE home_team_id = ?
        """, (canonical_id, dupe_id))
        cur.execute("""
            UPDATE betting_lines SET away_team_id = ? WHERE away_team_id = ?
        """, (canonical_id, dupe_id))
        
        # Delete from elo_ratings (don't modify, just delete dupe)
        deleted_elo = delete_from_table(cur, "elo_ratings", "team_id", dupe_id)
        if deleted_elo:
            print(f"  Deleted {deleted_elo} elo_ratings entries")
        
        # Handle other team tables (delete dupe entries)
        for table, column in team_tables:
            try:
                deleted_count = delete_from_table(cur, table, column, dupe_id)
                if deleted_count:
                    print(f"  Deleted {deleted_count} rows from {table}")
            except sqlite3.OperationalError:
                pass  # Table might not exist
        
        # Delete the team itself
        cur.execute("DELETE FROM teams WHERE id = ?", (dupe_id,))
        teams_merged += 1
    
    print("\n=== DELETING INVALID TEAMS ===")
    
    for team_id in to_delete:
        print(f"\nDeleting {team_id}")
        
        # Delete games involving this team
        cur.execute("""
            DELETE FROM games WHERE home_team_id = ? OR away_team_id = ?
        """, (team_id, team_id))
        print(f"  Deleted {cur.rowcount} games")
        
        # Delete from elo_ratings
        delete_from_table(cur, "elo_ratings", "team_id", team_id)
        
        # Delete from other tables
        for table, column in team_tables:
            try:
                delete_from_table(cur, table, column, team_id)
            except sqlite3.OperationalError:
                pass
        
        # Delete from betting_lines
        cur.execute("DELETE FROM betting_lines WHERE home_team_id = ? OR away_team_id = ?", (team_id, team_id))
        
        # Delete the team
        cur.execute("DELETE FROM teams WHERE id = ?", (team_id,))
    
    # Commit changes
    conn.commit()
    
    # Verify
    print("\n=== VERIFICATION ===")
    
    cur.execute("SELECT COUNT(*) FROM teams WHERE conference = 'Unknown'")
    remaining_unknown = cur.fetchone()[0]
    print(f"Teams with Unknown conference: {remaining_unknown}")
    
    cur.execute("""
        SELECT DISTINCT home_team_id FROM games 
        WHERE home_team_id NOT IN (SELECT id FROM teams)
    """)
    orphaned_home = cur.fetchall()
    
    cur.execute("""
        SELECT DISTINCT away_team_id FROM games 
        WHERE away_team_id NOT IN (SELECT id FROM teams)
    """)
    orphaned_away = cur.fetchall()
    
    print(f"Orphaned home_team_id references: {len(orphaned_home)}")
    print(f"Orphaned away_team_id references: {len(orphaned_away)}")
    
    if orphaned_home:
        print(f"  Home: {[r[0] for r in orphaned_home]}")
    if orphaned_away:
        print(f"  Away: {[r[0] for r in orphaned_away]}")
    
    conn.close()
    
    print("\n=== SUMMARY ===")
    print(f"Teams merged: {teams_merged}")
    print(f"Teams deleted: {len(to_delete)}")
    print(f"Games reassigned: {total_games_merged}")
    print(f"Duplicate games deleted: {total_games_deleted}")
    
    return teams_merged, len(to_delete), total_games_merged


if __name__ == "__main__":
    main()
