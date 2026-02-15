#!/usr/bin/env python3
"""
Migration: Clean up orphan teams with no games, players, or stats.

Conservative approach:
- Keep Big 12 teams (colorado, iowa-state) - games may be added later
- Merge umass duplicates (massachusetts → umass)
- Remove truly orphaned teams with no future use
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "baseball.db"

# Teams to keep (real D1 teams that may have games added later)
KEEP_TEAMS = {'colorado', 'iowa-state'}

# Teams to merge
MERGE_TEAMS = [
    ('massachusetts', 'umass'),  # Keep umass as canonical
]

# Teams to remove if they have no references anywhere
REMOVE_IF_ORPHAN = [
    'nebraska-omaha',
    'new-haven', 
    'nc-at',
    'southeastern-missouri',
    'umes',
]

def has_references(cur, team_id):
    """Check if team has any references in the database"""
    tables_to_check = [
        ("games", "home_team_id"),
        ("games", "away_team_id"),
        ("players", "team_id"),
        ("player_stats", "team_id"),
        ("betting_lines", "home_team_id"),
        ("betting_lines", "away_team_id"),
    ]
    
    for table, col in tables_to_check:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (team_id,))
            if cur.fetchone()[0] > 0:
                return True
        except:
            pass
    
    return False

def run_migration():
    print(f"Opening database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 1. Merge duplicate UMass teams
    print("\n=== Merging UMass duplicates ===")
    for dup_id, canonical_id in MERGE_TEAMS:
        cur.execute("SELECT id FROM teams WHERE id = ?", (dup_id,))
        if cur.fetchone():
            # Delete elo_ratings for duplicate (if exists)
            cur.execute("DELETE FROM elo_ratings WHERE team_id = ?", (dup_id,))
            # Delete the duplicate team
            cur.execute("DELETE FROM teams WHERE id = ?", (dup_id,))
            print(f"  ✓ Merged {dup_id} → {canonical_id}")
        else:
            print(f"  - {dup_id} not found, skipping")
    
    # 2. Remove truly orphaned teams
    print("\n=== Removing orphan teams ===")
    for team_id in REMOVE_IF_ORPHAN:
        cur.execute("SELECT id, name FROM teams WHERE id = ?", (team_id,))
        team = cur.fetchone()
        if not team:
            print(f"  - {team_id} not found, skipping")
            continue
        
        if has_references(cur, team_id):
            print(f"  ⚠ {team_id} has references, keeping")
            continue
        
        # Safe to remove
        cur.execute("DELETE FROM elo_ratings WHERE team_id = ?", (team_id,))
        cur.execute("DELETE FROM teams WHERE id = ?", (team_id,))
        print(f"  ✓ Removed orphan team: {team_id} ({team[1]})")
    
    # 3. Report on kept teams
    print("\n=== Teams kept (may have future games) ===")
    for team_id in KEEP_TEAMS:
        cur.execute("SELECT id, name, conference FROM teams WHERE id = ?", (team_id,))
        team = cur.fetchone()
        if team:
            print(f"  • {team[0]}: {team[1]} ({team[2]})")
    
    conn.commit()
    
    # Final verification
    print("\n=== Verification ===")
    cur.execute("""
        SELECT t.id, t.name,
            (SELECT COUNT(*) FROM games WHERE home_team_id = t.id OR away_team_id = t.id) as game_count
        FROM teams t
        LEFT JOIN games g ON t.id = g.home_team_id OR t.id = g.away_team_id
        GROUP BY t.id
        HAVING COUNT(g.id) = 0
        ORDER BY t.name
    """)
    orphans = cur.fetchall()
    print(f"Remaining teams with 0 games: {len(orphans)}")
    for team_id, name, game_count in orphans:
        print(f"  - {team_id}: {name}")
    
    conn.close()
    print("\n✅ Migration complete!")

if __name__ == "__main__":
    run_migration()
