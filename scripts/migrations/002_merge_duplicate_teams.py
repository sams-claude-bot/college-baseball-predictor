#!/usr/bin/env python3
"""
Migration: Merge duplicate teams into canonical IDs.

Duplicates to merge:
- charleston + college-of-charleston → college-of-charleston
- fau + florida-atlantic → florida-atlantic  
- gcu + grand-canyon → grand-canyon
- southeastern-la + southeastern-louisiana → southeastern-louisiana

Updates all references in: games, elo_ratings, betting_lines, and any other tables with team_id.
"""

import sqlite3
from pathlib import Path
import shutil
from datetime import datetime

DB_PATH = Path(__file__).parent.parent.parent / "data" / "baseball.db"

# Map of (duplicate_id -> canonical_id)
MERGES = [
    ('charleston', 'college-of-charleston'),
    ('fau', 'florida-atlantic'),
    ('gcu', 'grand-canyon'),
    ('southeastern-la', 'southeastern-louisiana'),
]

# Tables with team_id columns
TABLES_WITH_TEAM_ID = [
    ('players', ['team_id']),
    ('team_stats', ['team_id']),
    ('rankings_history', ['team_id']),
    ('elo_ratings', ['team_id']),
    ('player_stats', ['team_id']),
    ('preseason_priors', ['team_id']),  # May not exist
    ('pitcher_game_log', ['team_id']),
    ('ncaa_team_stats', ['team_id']),
    ('ncaa_individual_stats', ['team_id']),
]

# Tables with home_team_id/away_team_id
TABLES_WITH_HOME_AWAY = [
    'games',
    'betting_lines',
]

# Tables with game_id that might contain team names
TABLES_WITH_GAME_ID = [
    'model_predictions',
    'totals_predictions',
    'player_boxscore_batting',
    'player_boxscore_pitching',
    'game_batting_stats',
    'game_pitching_stats',
    'statbroadcast_boxscores',
]

def table_exists(cur, table_name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None

def column_exists(cur, table_name, column_name):
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cur.fetchall()]
    return column_name in columns

def run_migration():
    print(f"Opening database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    for dup_id, canonical_id in MERGES:
        print(f"\n{'='*60}")
        print(f"Merging: {dup_id} → {canonical_id}")
        print('='*60)
        
        # 1. Check if canonical team exists, if not rename the duplicate
        cur.execute("SELECT id, name FROM teams WHERE id = ?", (canonical_id,))
        canonical = cur.fetchone()
        
        cur.execute("SELECT id, name FROM teams WHERE id = ?", (dup_id,))
        duplicate = cur.fetchone()
        
        if not duplicate:
            print(f"  Duplicate team '{dup_id}' not found, skipping...")
            continue
            
        if not canonical:
            print(f"  Canonical team '{canonical_id}' not found, will rename {dup_id}")
            cur.execute("UPDATE teams SET id = ? WHERE id = ?", (canonical_id, dup_id))
            print(f"  ✓ Renamed team: {dup_id} → {canonical_id}")
            # Now update all references
        else:
            print(f"  Both teams exist:")
            print(f"    Duplicate: {duplicate}")
            print(f"    Canonical: {canonical}")
        
        # 2. Update tables with team_id column
        for table, columns in TABLES_WITH_TEAM_ID:
            if not table_exists(cur, table):
                continue
            for col in columns:
                if not column_exists(cur, table, col):
                    continue
                
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (dup_id,))
                count = cur.fetchone()[0]
                if count > 0:
                    # Check for conflicts (would create duplicates)
                    try:
                        cur.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (canonical_id, dup_id))
                        print(f"  ✓ Updated {count} rows in {table}.{col}")
                    except sqlite3.IntegrityError as e:
                        # Unique constraint violation - delete duplicates
                        cur.execute(f"DELETE FROM {table} WHERE {col} = ?", (dup_id,))
                        print(f"  ✓ Deleted {count} duplicate rows from {table}.{col}")
        
        # 3. Update games table (home_team_id, away_team_id)
        for table in TABLES_WITH_HOME_AWAY:
            if not table_exists(cur, table):
                continue
                
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE home_team_id = ?", (dup_id,))
            home_count = cur.fetchone()[0]
            if home_count > 0:
                cur.execute(f"UPDATE {table} SET home_team_id = ? WHERE home_team_id = ?", (canonical_id, dup_id))
                print(f"  ✓ Updated {home_count} rows in {table}.home_team_id")
            
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE away_team_id = ?", (dup_id,))
            away_count = cur.fetchone()[0]
            if away_count > 0:
                cur.execute(f"UPDATE {table} SET away_team_id = ? WHERE away_team_id = ?", (canonical_id, dup_id))
                print(f"  ✓ Updated {away_count} rows in {table}.away_team_id")
        
        # 4. Update game IDs that contain the duplicate team name
        for table in TABLES_WITH_GAME_ID:
            if not table_exists(cur, table):
                continue
            if not column_exists(cur, table, 'game_id'):
                continue
                
            # Find game_ids containing the duplicate team name
            cur.execute(f"SELECT DISTINCT game_id FROM {table} WHERE game_id LIKE ?", (f'%{dup_id}%',))
            game_ids = [row[0] for row in cur.fetchall()]
            
            for old_game_id in game_ids:
                new_game_id = old_game_id.replace(dup_id, canonical_id)
                try:
                    cur.execute(f"UPDATE {table} SET game_id = ? WHERE game_id = ?", (new_game_id, old_game_id))
                except sqlite3.IntegrityError:
                    # Duplicate - delete the old one
                    cur.execute(f"DELETE FROM {table} WHERE game_id = ?", (old_game_id,))
            
            if game_ids:
                print(f"  ✓ Updated {len(game_ids)} game_ids in {table}")
        
        # 5. Update game IDs in games table itself
        cur.execute("SELECT id FROM games WHERE id LIKE ?", (f'%{dup_id}%',))
        game_ids = [row[0] for row in cur.fetchall()]
        for old_id in game_ids:
            new_id = old_id.replace(dup_id, canonical_id)
            # Check if new_id already exists
            cur.execute("SELECT id FROM games WHERE id = ?", (new_id,))
            if cur.fetchone():
                # Delete the duplicate game
                cur.execute("DELETE FROM games WHERE id = ?", (old_id,))
                print(f"  ✓ Deleted duplicate game: {old_id}")
            else:
                cur.execute("UPDATE games SET id = ? WHERE id = ?", (new_id, old_id))
                print(f"  ✓ Renamed game: {old_id} → {new_id}")
        
        # 6. Delete the duplicate team record
        if canonical:
            cur.execute("DELETE FROM teams WHERE id = ?", (dup_id,))
            print(f"  ✓ Deleted duplicate team record: {dup_id}")
    
    conn.commit()
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    for dup_id, canonical_id in MERGES:
        cur.execute("SELECT COUNT(*) FROM teams WHERE id = ?", (dup_id,))
        dup_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM teams WHERE id = ?", (canonical_id,))
        can_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM games WHERE home_team_id = ? OR away_team_id = ?", (canonical_id, canonical_id))
        game_count = cur.fetchone()[0]
        
        status = "✓" if dup_count == 0 and can_count == 1 else "✗"
        print(f"{status} {canonical_id}: {game_count} games (duplicate '{dup_id}' removed: {dup_count == 0})")
    
    conn.close()
    print("\n✅ Migration complete!")

if __name__ == "__main__":
    run_migration()
