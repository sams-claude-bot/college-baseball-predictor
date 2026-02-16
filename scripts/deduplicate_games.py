#!/usr/bin/env python3
"""
Deduplicate games in the database.
Handles ESPN format (_) vs schedule scraper format (-vs-).
Preserves actual doubleheaders (identified by _g1, _g2 suffixes).
"""

import sqlite3
import re
from collections import defaultdict

DB_PATH = "data/baseball.db"

# Tables that reference game_id
REFERENCING_TABLES = [
    "model_predictions",
    "betting_lines",
    "tracked_bets",
    "tracked_bets_spreads",
    "totals_predictions",
    "spread_predictions",
    "game_predictions",
    "game_boxscores",
    "game_batting_stats",
    "game_pitching_stats",
]

def is_espn_format(game_id):
    """ESPN format: 2026-02-13_away_home (underscores after date)"""
    if re.match(r'^\d{4}-\d{2}-\d{2}_', game_id):
        return True
    return False

def get_game_number(game_id):
    """
    Extract game number from ID if it's a doubleheader.
    Returns 1, 2, etc. or 0 if not specified (assumed game 1).
    """
    match = re.search(r'_g(\d+)$', game_id)
    if match:
        return int(match.group(1))
    return 0  # Assumed to be game 1 or standalone

def has_score(game):
    """Check if game has a score recorded"""
    return game['home_score'] is not None or game['away_score'] is not None

def pick_best_game(games):
    """
    Given a list of duplicate games (same date, home, away), 
    determine which to keep and which to delete.
    
    Returns list of (keep_id, [delete_ids]) tuples for each logical game.
    """
    # Group by game number (0 = no suffix, 1 = _g1, 2 = _g2, etc.)
    by_game_num = defaultdict(list)
    for g in games:
        game_num = get_game_number(g['id'])
        by_game_num[game_num].append(g)
    
    results = []
    
    # Handle each game number group
    for game_num, game_list in sorted(by_game_num.items()):
        if len(game_list) == 1:
            # Only one game for this game number, keep it
            continue
        
        # Multiple games for same game number - need to dedupe
        # Sort by preference: ESPN format first, has score second
        def sort_key(g):
            espn = 1 if is_espn_format(g['id']) else 0
            scored = 1 if has_score(g) else 0
            return (espn, scored, g['id'])  # id for stability
        
        sorted_games = sorted(game_list, key=sort_key, reverse=True)
        keep_game = sorted_games[0]
        delete_games = sorted_games[1:]
        
        results.append((keep_game['id'], [g['id'] for g in delete_games]))
    
    return results

def update_references(conn, old_id, new_id):
    """Update all references from old_id to new_id"""
    cursor = conn.cursor()
    total_updated = 0
    for table in REFERENCING_TABLES:
        try:
            # Check if any references exist
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE game_id = ?", (old_id,))
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"    Updating {count} references in {table}: {old_id} -> {new_id}")
                cursor.execute(f"UPDATE {table} SET game_id = ? WHERE game_id = ?", (new_id, old_id))
                total_updated += count
        except sqlite3.OperationalError:
            pass
    return total_updated

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get initial count
    cursor.execute("SELECT COUNT(*) FROM games")
    initial_count = cursor.fetchone()[0]
    print(f"Initial game count: {initial_count}")
    
    # Find all duplicate groups
    cursor.execute("""
        SELECT date, home_team_id, away_team_id, COUNT(*) as cnt 
        FROM games 
        GROUP BY date, home_team_id, away_team_id 
        HAVING COUNT(*) > 1
        ORDER BY date, home_team_id
    """)
    duplicate_groups = cursor.fetchall()
    print(f"Found {len(duplicate_groups)} duplicate game combinations\n")
    
    total_deleted = 0
    total_refs_updated = 0
    
    for group in duplicate_groups:
        date, home_id, away_id, cnt = group['date'], group['home_team_id'], group['away_team_id'], group['cnt']
        
        # Get all games in this group
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score, status, venue
            FROM games 
            WHERE date = ? AND home_team_id = ? AND away_team_id = ?
            ORDER BY id
        """, (date, home_id, away_id))
        games = [dict(row) for row in cursor.fetchall()]
        
        dedup_actions = pick_best_game(games)
        
        for keep_id, delete_ids in dedup_actions:
            if delete_ids:
                print(f"{date}: {away_id} @ {home_id}")
                print(f"  Keep: {keep_id}")
                for del_id in delete_ids:
                    print(f"  Delete: {del_id}")
                    refs = update_references(conn, del_id, keep_id)
                    total_refs_updated += refs
                    cursor.execute("DELETE FROM games WHERE id = ?", (del_id,))
                    total_deleted += 1
                print()
    
    conn.commit()
    
    # Verify no duplicates remain (excluding legitimate doubleheaders)
    cursor.execute("""
        SELECT date, home_team_id, away_team_id, COUNT(*) as cnt 
        FROM games 
        GROUP BY date, home_team_id, away_team_id 
        HAVING COUNT(*) > 1
    """)
    remaining_dups = cursor.fetchall()
    
    # Get final count
    cursor.execute("SELECT COUNT(*) FROM games")
    final_count = cursor.fetchone()[0]
    
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Initial games: {initial_count}")
    print(f"  Deleted: {total_deleted}")
    print(f"  Final games: {final_count}")
    print(f"  References updated: {total_refs_updated}")
    print(f"  Remaining duplicate groups: {len(remaining_dups)}")
    
    if remaining_dups:
        print("\nRemaining duplicates (should be doubleheaders with _g1/_g2):")
        for dup in remaining_dups[:20]:
            # Show the IDs to verify they're doubleheaders
            cursor.execute("""
                SELECT id, home_score, away_score FROM games 
                WHERE date = ? AND home_team_id = ? AND away_team_id = ?
                ORDER BY id
            """, (dup['date'], dup['home_team_id'], dup['away_team_id']))
            ids = cursor.fetchall()
            print(f"  {dup['date']}: {dup['away_team_id']} @ {dup['home_team_id']}")
            for g in ids:
                print(f"    - {g['id']}: {g['away_score']}-{g['home_score']}")
    
    conn.close()
    
    return len(remaining_dups) == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Some duplicates remain - likely doubleheaders. Review above.")
