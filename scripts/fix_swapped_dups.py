#!/usr/bin/env python3
"""
Fix games where home/away are swapped between two entries.
ESPN format (_away_home) is authoritative for home/away designation.
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

def is_espn_format(game_id):
    """ESPN format: 2026-02-13_away_home"""
    import re
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}_', game_id))

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
    
    # Find swapped duplicates: same date, teams reversed
    cursor.execute("""
        SELECT g1.id as id1, g2.id as id2,
               g1.date, g1.home_team_id as home1, g1.away_team_id as away1,
               g1.home_score as h1_score, g1.away_score as a1_score,
               g2.home_score as h2_score, g2.away_score as a2_score
        FROM games g1
        JOIN games g2 ON g1.date = g2.date 
            AND g1.home_team_id = g2.away_team_id 
            AND g1.away_team_id = g2.home_team_id
            AND g1.id < g2.id
        ORDER BY g1.date
    """)
    swapped = cursor.fetchall()
    
    print(f"Found {len(swapped)} swapped home/away pairs\n")
    
    deleted = 0
    
    for row in swapped:
        id1, id2 = row['id1'], row['id2']
        date = row['date']
        home1, away1 = row['home1'], row['away1']
        
        espn1 = is_espn_format(id1)
        espn2 = is_espn_format(id2)
        
        has_score1 = row['h1_score'] is not None
        has_score2 = row['h2_score'] is not None
        
        print(f"{date}: {away1} @ {home1}")
        print(f"  {id1}: {row['a1_score']}-{row['h1_score']} (ESPN={espn1})")
        print(f"  {id2}: {row['a2_score']}-{row['h2_score']} (ESPN={espn2})")
        
        # Priority: ESPN format wins for home/away designation
        # If ESPN version has no score but other does, we might want to update ESPN version
        
        if espn1 and not espn2:
            # Keep ESPN (id1), delete id2
            keep_id, delete_id = id1, id2
            # If ESPN has no score but other does, transfer score
            if not has_score1 and has_score2:
                # Note: scores are from opposite perspective, so swap them
                cursor.execute("""
                    UPDATE games SET home_score = ?, away_score = ? WHERE id = ?
                """, (row['a2_score'], row['h2_score'], keep_id))
                print(f"  Transferred score to {keep_id}")
        elif espn2 and not espn1:
            keep_id, delete_id = id2, id1
            if not has_score2 and has_score1:
                cursor.execute("""
                    UPDATE games SET home_score = ?, away_score = ? WHERE id = ?
                """, (row['a1_score'], row['h1_score'], keep_id))
                print(f"  Transferred score to {keep_id}")
        else:
            # Both ESPN or both non-ESPN - prefer one with score, or first one
            if has_score1 and not has_score2:
                keep_id, delete_id = id1, id2
            elif has_score2 and not has_score1:
                keep_id, delete_id = id2, id1
            else:
                keep_id, delete_id = id1, id2  # Arbitrary
        
        print(f"  Keep: {keep_id}")
        print(f"  Delete: {delete_id}")
        
        update_references(conn, delete_id, keep_id)
        cursor.execute("DELETE FROM games WHERE id = ?", (delete_id,))
        deleted += 1
        print()
    
    conn.commit()
    
    # Final verification
    cursor.execute("""
        SELECT g1.id, g2.id FROM games g1
        JOIN games g2 ON g1.date = g2.date 
            AND g1.home_team_id = g2.away_team_id 
            AND g1.away_team_id = g2.home_team_id
            AND g1.id < g2.id
    """)
    remaining = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*) FROM games")
    final_count = cursor.fetchone()[0]
    
    print(f"Deleted: {deleted}")
    print(f"Final game count: {final_count}")
    print(f"Remaining swapped pairs: {len(remaining)}")
    
    conn.close()

if __name__ == "__main__":
    main()
