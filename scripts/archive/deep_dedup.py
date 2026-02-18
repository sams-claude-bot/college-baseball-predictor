#!/usr/bin/env python3
"""Deep dedup: resolve all game conflicts following the rules."""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

def is_espn_format(game_id):
    """ESPN format uses underscores: 2026-02-13_away_home"""
    return '_' in game_id and '-vs-' not in game_id

def is_scraper_format(game_id):
    """Scraper format uses -vs-: 2026-02-13-team-vs-team"""
    return '-vs-' in game_id

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Check for foreign key references
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tables in DB: {len(tables)} tables")
    
    # Find all conflicts: team has 2+ games on same date with DIFFERENT opponents
    # The games table uses: id, home_team_id, away_team_id
    cur.execute("""
        WITH game_teams AS (
            SELECT id, date, away_team_id as team, home_team_id as opponent FROM games
            UNION ALL
            SELECT id, date, home_team_id as team, away_team_id as opponent FROM games
        )
        SELECT team, date, GROUP_CONCAT(id) as game_ids, 
               GROUP_CONCAT(opponent) as opponents, COUNT(*) as cnt
        FROM game_teams
        GROUP BY team, date
        HAVING COUNT(*) > 1
    """)
    
    conflicts = cur.fetchall()
    print(f"\nFound {len(conflicts)} team-date combinations with 2+ games")
    
    to_delete = set()
    kept = []
    
    for row in conflicts:
        team = row['team']
        date = row['date']
        game_ids = row['game_ids'].split(',')
        opponents = row['opponents'].split(',')
        
        # If all opponents are the same, not a conflict (same teams multiple times = already handled)
        unique_opponents = set(opponents)
        if len(unique_opponents) == 1:
            continue
        
        is_past = date < '2026-02-16'  # Games before today
        
        # Categorize games
        espn_games = [g for g in game_ids if is_espn_format(g)]
        scraper_games = [g for g in game_ids if is_scraper_format(g)]
        
        # Special case: portland-state is actually portland (scraper error)
        portland_state_games = [g for g in scraper_games if 'portland-state' in g]
        if portland_state_games:
            # Check if there's an ESPN game with portland
            portland_espn = [g for g in espn_games if '_portland_' in g or g.endswith('_portland')]
            if portland_espn:
                for g in portland_state_games:
                    to_delete.add(g)
                continue
        
        # Special case: UAB vs FSU (scraper) should be UAB vs Florida (ESPN)
        if team in ['uab', 'florida-state']:
            fsu_scraper = [g for g in scraper_games if 'florida-state-vs-uab' in g]
            if fsu_scraper:
                for g in fsu_scraper:
                    to_delete.add(g)
                continue
        
        if is_past:
            # PAST RULES:
            # If ESPN + SCRAPER with different opponents → delete SCRAPER
            if espn_games and scraper_games:
                for g in scraper_games:
                    to_delete.add(g)
            # If both ESPN → keep both (tournament doubleheader)
            elif len(espn_games) > 1 and not scraper_games:
                kept.append(f"PAST DH: {team} {date} - {len(espn_games)} ESPN games")
        else:
            # FUTURE RULES:
            # If ESPN + SCRAPER with different opponents → delete SCRAPER
            if espn_games and scraper_games:
                for g in scraper_games:
                    to_delete.add(g)
            # If both SCRAPER → delete BOTH (tournament bracket guess)
            elif len(scraper_games) > 1 and not espn_games:
                for g in scraper_games:
                    to_delete.add(g)
            # If both ESPN → keep both (could be legitimate DH)
            elif len(espn_games) > 1 and not scraper_games:
                kept.append(f"FUTURE DH: {team} {date} - {len(espn_games)} ESPN games")
    
    print(f"\nGames to delete: {len(to_delete)}")
    print(f"Doubleheaders kept: {len(kept)}")
    
    if kept:
        print("\nKept doubleheaders (showing unique):")
        seen_dates = set()
        for k in kept:
            date_key = k.split(' - ')[0]
            if date_key not in seen_dates:
                print(f"  {k}")
                seen_dates.add(date_key)
                if len(seen_dates) >= 15:
                    print(f"  ... and more")
                    break
    
    # Check for references before deleting
    delete_list = list(to_delete)
    if delete_list:
        # Check betting_lines
        if 'betting_lines' in tables:
            placeholders = ','.join('?' * len(delete_list))
            cur.execute(f"SELECT COUNT(*) FROM betting_lines WHERE game_id IN ({placeholders})", delete_list)
            betting_refs = cur.fetchone()[0]
            print(f"\nBetting line references to delete: {betting_refs}")
        
        # Check predictions
        if 'predictions' in tables:
            cur.execute(f"SELECT COUNT(*) FROM predictions WHERE game_id IN ({placeholders})", delete_list)
            pred_refs = cur.fetchone()[0]
            print(f"Prediction references to delete: {pred_refs}")
            
        # Check game_predictions
        if 'game_predictions' in tables:
            cur.execute(f"SELECT COUNT(*) FROM game_predictions WHERE game_id IN ({placeholders})", delete_list)
            gp_refs = cur.fetchone()[0]
            print(f"Game prediction references to delete: {gp_refs}")
    
    # Preview some deletions
    print("\nSample games to delete:")
    for g in sorted(to_delete)[:15]:
        print(f"  {g}")
    if len(to_delete) > 15:
        print(f"  ... and {len(to_delete) - 15} more")
    
    # Actually delete
    if delete_list:
        placeholders = ','.join('?' * len(delete_list))
        
        # Delete from child tables first (check each table)
        child_tables_with_game_id = ['betting_lines', 'predictions', 'game_predictions', 
                                      'totals_predictions', 'spread_predictions', 'game_boxscores',
                                      'player_boxscore_batting', 'player_boxscore_pitching',
                                      'statbroadcast_boxscores', 'game_batting_stats', 
                                      'game_pitching_stats', 'pitching_matchups']
        
        for table in child_tables_with_game_id:
            if table in tables:
                try:
                    cur.execute(f"DELETE FROM {table} WHERE game_id IN ({placeholders})", delete_list)
                    if cur.rowcount > 0:
                        print(f"Deleted {cur.rowcount} from {table}")
                except sqlite3.OperationalError as e:
                    # Table might not have game_id column
                    pass
        
        # Delete games
        cur.execute(f"DELETE FROM games WHERE id IN ({placeholders})", delete_list)
        print(f"\nDeleted {cur.rowcount} games")
        
        conn.commit()
    
    # Final count
    cur.execute("SELECT COUNT(*) FROM games")
    final_count = cur.fetchone()[0]
    print(f"\nFinal game count: {final_count}")
    
    # Verify no remaining conflicts
    cur.execute("""
        WITH game_teams AS (
            SELECT id, date, away_team_id as team, home_team_id as opponent FROM games
            UNION ALL
            SELECT id, date, home_team_id as team, away_team_id as opponent FROM games
        ),
        conflicts AS (
            SELECT team, date, GROUP_CONCAT(DISTINCT opponent) as opponents, 
                   COUNT(DISTINCT opponent) as opp_count,
                   GROUP_CONCAT(id) as game_ids
            FROM game_teams
            GROUP BY team, date
            HAVING COUNT(DISTINCT opponent) > 1
        )
        SELECT * FROM conflicts ORDER BY date LIMIT 20
    """)
    remaining = cur.fetchall()
    
    if remaining:
        print(f"\n⚠️  Remaining conflicts (showing up to 20):")
        for r in remaining:
            game_ids = r['game_ids'].split(',')
            espn_count = sum(1 for g in game_ids if is_espn_format(g))
            scraper_count = sum(1 for g in game_ids if is_scraper_format(g))
            print(f"  {r['date']} | {r['team']} vs {r['opponents']} | ESPN:{espn_count} Scraper:{scraper_count}")
    else:
        print("\n✅ No remaining conflicts with different opponents!")
    
    conn.close()

if __name__ == "__main__":
    main()
