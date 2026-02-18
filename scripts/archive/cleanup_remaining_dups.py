#!/usr/bin/env python3
"""
Second pass: Clean up remaining duplicates where a non-suffixed game
matches the score of a _g1 or _g2 suffixed game.
"""

import sqlite3

DB_PATH = "data/baseball.db"

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

def update_references(conn, old_id, new_id):
    cursor = conn.cursor()
    for table in REFERENCING_TABLES:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE game_id = ?", (old_id,))
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"    Updating {count} refs in {table}: {old_id} -> {new_id}")
                cursor.execute(f"UPDATE {table} SET game_id = ? WHERE game_id = ?", (new_id, old_id))
        except sqlite3.OperationalError:
            pass

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Find groups with 3+ entries (problematic cases)
    cursor.execute("""
        SELECT date, home_team_id, away_team_id, COUNT(*) as cnt 
        FROM games 
        GROUP BY date, home_team_id, away_team_id 
        HAVING COUNT(*) > 2
    """)
    problem_groups = cursor.fetchall()
    
    deleted = 0
    
    for group in problem_groups:
        date, home_id, away_id = group['date'], group['home_team_id'], group['away_team_id']
        
        cursor.execute("""
            SELECT id, home_score, away_score FROM games 
            WHERE date = ? AND home_team_id = ? AND away_team_id = ?
        """, (date, home_id, away_id))
        games = list(cursor.fetchall())
        
        # Identify games by type
        unsuffixed = []  # No _g1/_g2 suffix
        suffixed = {}    # game_num -> game
        
        for g in games:
            game_id = g['id']
            if '_g1' in game_id:
                suffixed[1] = g
            elif '_g2' in game_id:
                suffixed[2] = g
            else:
                unsuffixed.append(g)
        
        # For each unsuffixed game, check if its score matches any suffixed game
        for ug in unsuffixed:
            ug_score = (ug['home_score'], ug['away_score'])
            
            for gnum, sg in suffixed.items():
                sg_score = (sg['home_score'], sg['away_score'])
                
                if ug_score == sg_score:
                    # Duplicate found - delete the unsuffixed one
                    print(f"{date}: {away_id} @ {home_id}")
                    print(f"  Delete: {ug['id']} (score matches _g{gnum})")
                    print(f"  Keep: {sg['id']}")
                    update_references(conn, ug['id'], sg['id'])
                    cursor.execute("DELETE FROM games WHERE id = ?", (ug['id'],))
                    deleted += 1
                    break
    
    conn.commit()
    
    # Verify remaining duplicates
    cursor.execute("""
        SELECT date, home_team_id, away_team_id, COUNT(*) as cnt 
        FROM games 
        GROUP BY date, home_team_id, away_team_id 
        HAVING COUNT(*) > 1
    """)
    remaining = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*) FROM games")
    final_count = cursor.fetchone()[0]
    
    print(f"\nDeleted: {deleted}")
    print(f"Final game count: {final_count}")
    print(f"Remaining duplicate groups: {len(remaining)}")
    
    conn.close()

if __name__ == "__main__":
    main()
