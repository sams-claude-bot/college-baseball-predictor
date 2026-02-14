#!/usr/bin/env python3
"""
Comprehensive Database Cleanup Script
Removes tournament games, duplicates, and fixes scheduling conflicts.
"""

import sqlite3
import json
from datetime import datetime
from collections import defaultdict

DB_PATH = 'data/baseball.db'

# Tournament games to identify and remove
# These are neutral-site tournaments that cause scheduling conflicts
TOURNAMENT_PATTERNS = {
    'globe_life_field': {
        'name': 'Amegy Bank College Baseball Series / Globe Life Field events',
        'venue_patterns': ['globe life', 'arlington, texas', 'arlington, tx'],
        'date_ranges': [('2026-02-27', '2026-03-01')],  # Typical dates for these events
    },
    'shriners_showdown': {
        'name': "Shriners Children's College Showdown",
        'venue_patterns': ['minute maid', 'houston, tx'],
        'teams_involved': ['lsu', 'texas', 'texas-am', 'oklahoma', 'baylor'],
    },
    'puerto_rico_challenge': {
        'name': 'Puerto Rico Challenge',
        'venue_patterns': ['puerto rico', 'san juan'],
    },
    'desert_invitational': {
        'name': 'MLB Desert Invitational',
        'venue_patterns': ['surprise', 'phoenix', 'az', 'scottsdale'],
    },
    'las_vegas_classic': {
        'name': 'Las Vegas Classic / Events',
        'venue_patterns': ['las vegas', 'vegas'],
    },
    'jacksonville': {
        'name': 'Jacksonville Tournament',  
        'notes': 'LSU plays multiple teams (Indiana, Notre Dame, UCF) on same trip',
    },
    'fort_myers': {
        'name': 'Fort Myers Events',
        'venue_patterns': ['fort myers', 'terry park'],
    },
}

# Teams that appear in tournament matchups based on analysis
KNOWN_TOURNAMENT_GAMES = [
    # Globe Life Field / Arlington events (Feb 27 - Mar 1)
    # These are multi-team events where teams play different opponents
    ('2026-02-27', 'mississippi-state', 'ucla'),
    ('2026-02-28', 'mississippi-state', 'ucla'),
    ('2026-03-01', 'mississippi-state', 'ucla'),
    ('2026-02-27', 'texas-am', 'ucla'),
    ('2026-02-28', 'texas-am', 'ucla'),
    ('2026-03-01', 'texas-am', 'ucla'),
    ('2026-02-27', 'tennessee', 'ucla'),
    ('2026-02-28', 'tennessee', 'ucla'),
    ('2026-03-01', 'tennessee', 'ucla'),
    
    # Vanderbilt at Arlington (Shriners Showdown)
    ('2026-02-13', 'vanderbilt', 'tcu'),
    ('2026-02-14', 'vanderbilt', 'texas-tech'),
    ('2026-02-15', 'vanderbilt', 'oklahoma-state'),
    
    # Auburn at Arlington
    ('2026-02-20', 'auburn', 'kansas-state'),
    ('2026-02-21', 'auburn', 'florida-state'),
    ('2026-02-22', 'auburn', 'louisville'),
    
    # Las Vegas Classic (Vanderbilt)
    ('2026-02-27', 'vanderbilt', 'uc-irvine'),
    ('2026-02-28', 'vanderbilt', 'arizona'),
    ('2026-03-01', 'vanderbilt', 'oregon'),
    
    # Jacksonville Tournament (LSU playing multiple teams)
    ('2026-02-20', 'lsu', 'indiana'),
    ('2026-02-21', 'lsu', 'notre-dame'),
    ('2026-02-22', 'lsu', 'ucf'),
    
    # Missouri at Fort Myers
    ('2026-02-13', 'missouri', 'mount-st-marys'),
    ('2026-02-14', 'missouri', 'mount-st-marys'),
    ('2026-02-15', 'missouri', 'mount-st-marys'),
    ('2026-02-17', 'missouri', 'fau'),
    ('2026-02-20', 'missouri', 'new-haven'),
    ('2026-02-21', 'missouri', 'new-haven'),
    ('2026-02-22', 'missouri', 'new-haven'),
    
    # Ole Miss Bruce Bolt Classic
    ('2026-02-27', 'ole-miss', 'michigan'),
    ('2026-02-28', 'ole-miss', 'michigan'),
    ('2026-03-01', 'ole-miss', 'michigan'),
]


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def find_duplicate_games(conn):
    """Find exact duplicate game entries"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, home_team_id, away_team_id, COUNT(*) as cnt,
               GROUP_CONCAT(id) as game_ids
        FROM games 
        GROUP BY date, home_team_id, away_team_id 
        HAVING COUNT(*) > 1
        ORDER BY date
    """)
    
    duplicates = []
    for row in cursor.fetchall():
        duplicates.append({
            'date': row['date'],
            'home': row['home_team_id'],
            'away': row['away_team_id'],
            'count': row['cnt'],
            'ids': row['game_ids'].split(',')
        })
    
    return duplicates


def find_scheduling_conflicts(conn):
    """Find teams playing multiple games on the same day"""
    cursor = conn.cursor()
    
    # Get all games for each team on each date
    cursor.execute("""
        SELECT date, team_id, GROUP_CONCAT(game_id) as games, COUNT(*) as game_count
        FROM (
            SELECT date, home_team_id as team_id, id as game_id FROM games
            UNION ALL
            SELECT date, away_team_id as team_id, id as game_id FROM games
        )
        GROUP BY date, team_id
        HAVING COUNT(*) > 1
        ORDER BY date, team_id
    """)
    
    conflicts = []
    for row in cursor.fetchall():
        conflicts.append({
            'date': row['date'],
            'team': row['team_id'],
            'games': row['games'].split(','),
            'count': row['game_count']
        })
    
    return conflicts


def identify_tournament_games(conn):
    """Identify games that are part of multi-team tournaments"""
    cursor = conn.cursor()
    tournament_games = set()
    
    # Check known tournament games
    for date, team1, team2 in KNOWN_TOURNAMENT_GAMES:
        cursor.execute("""
            SELECT id FROM games 
            WHERE date = ? 
            AND ((home_team_id = ? AND away_team_id = ?)
                 OR (home_team_id = ? AND away_team_id = ?))
        """, (date, team1, team2, team2, team1))
        
        for row in cursor.fetchall():
            tournament_games.add(row['id'])
    
    # Also check venue patterns
    for pattern_key, pattern in TOURNAMENT_PATTERNS.items():
        if 'venue_patterns' in pattern:
            for venue_pat in pattern['venue_patterns']:
                cursor.execute("""
                    SELECT id FROM games 
                    WHERE LOWER(venue) LIKE ?
                """, (f'%{venue_pat}%',))
                for row in cursor.fetchall():
                    tournament_games.add(row['id'])
    
    # Check neutral site games that are in tournament date ranges
    cursor.execute("""
        SELECT id, date, home_team_id, away_team_id 
        FROM games 
        WHERE is_neutral_site = 1
    """)
    for row in cursor.fetchall():
        tournament_games.add(row['id'])
    
    return list(tournament_games)


def find_games_on_conflicts(conn, conflicts):
    """For each conflict, identify which games are likely tournament games"""
    cursor = conn.cursor()
    
    games_to_review = []
    
    for conflict in conflicts:
        date = conflict['date']
        team = conflict['team']
        game_ids = conflict['games']
        
        # Get full game details
        games = []
        for gid in game_ids:
            cursor.execute("""
                SELECT id, date, home_team_id, away_team_id, venue, is_neutral_site, tournament_id
                FROM games WHERE id = ?
            """, (gid,))
            row = cursor.fetchone()
            if row:
                games.append(dict(row))
        
        # If team plays multiple DIFFERENT opponents on same day, it's tournament
        opponents = set()
        for g in games:
            opp = g['away_team_id'] if g['home_team_id'] == team else g['home_team_id']
            opponents.add(opp)
        
        if len(opponents) > 1:
            # Playing different teams - definitely tournament games
            for g in games:
                games_to_review.append({
                    'game': g,
                    'reason': f"Team {team} plays {len(opponents)} different opponents on {date}",
                    'action': 'remove_tournament'
                })
        elif len(opponents) == 1:
            # Same opponent multiple times - could be doubleheader or duplicate
            if len(games) > 2:
                # More than 2 games vs same opponent - remove extras
                for g in games[2:]:
                    games_to_review.append({
                        'game': g,
                        'reason': f"More than 2 games vs same opponent on {date}",
                        'action': 'remove_excess'
                    })
    
    return games_to_review


def remove_game_and_related(conn, game_id, dry_run=False):
    """Remove a game and all related records"""
    cursor = conn.cursor()
    
    related_tables = [
        'game_predictions',
        'betting_lines', 
        'pitching_matchups',
        'pitcher_game_log',
        'model_predictions',
        'predictions'
    ]
    
    removed = {'game': 0, 'related': {}}
    
    for table in related_tables:
        if dry_run:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE game_id = ?", (game_id,))
            count = cursor.fetchone()[0]
        else:
            cursor.execute(f"DELETE FROM {table} WHERE game_id = ?", (game_id,))
            count = cursor.rowcount
        
        if count > 0:
            removed['related'][table] = count
    
    if dry_run:
        cursor.execute("SELECT COUNT(*) FROM games WHERE id = ?", (game_id,))
        removed['game'] = cursor.fetchone()[0]
    else:
        cursor.execute("DELETE FROM games WHERE id = ?", (game_id,))
        removed['game'] = cursor.rowcount
    
    return removed


def cleanup_duplicates(conn, dry_run=True):
    """Remove duplicate game entries, keeping the first one"""
    print("\n" + "="*60)
    print("STEP 1: REMOVING DUPLICATE GAME ENTRIES")
    print("="*60)
    
    duplicates = find_duplicate_games(conn)
    
    if not duplicates:
        print("No duplicate games found.")
        return 0
    
    print(f"Found {len(duplicates)} sets of duplicate games:")
    
    removed_count = 0
    for dup in duplicates:
        print(f"\n  {dup['date']}: {dup['away']} @ {dup['home']} - {dup['count']} copies")
        
        # Keep the first ID, remove the rest
        ids_to_remove = dup['ids'][1:]
        
        for gid in ids_to_remove:
            if dry_run:
                print(f"    [DRY RUN] Would remove: {gid}")
            else:
                result = remove_game_and_related(conn, gid)
                print(f"    Removed: {gid}")
                removed_count += 1
    
    return removed_count


def cleanup_tournament_games(conn, dry_run=True):
    """Remove identified tournament games"""
    print("\n" + "="*60)
    print("STEP 2: REMOVING TOURNAMENT GAMES")
    print("="*60)
    
    tournament_game_ids = identify_tournament_games(conn)
    
    if not tournament_game_ids:
        print("No tournament games identified.")
        return 0
    
    cursor = conn.cursor()
    
    print(f"Found {len(tournament_game_ids)} tournament games to remove:")
    
    removed_count = 0
    for gid in tournament_game_ids:
        cursor.execute("""
            SELECT id, date, home_team_id, away_team_id, venue 
            FROM games WHERE id = ?
        """, (gid,))
        row = cursor.fetchone()
        
        if row:
            if dry_run:
                print(f"  [DRY RUN] Would remove: {row['date']} - {row['away_team_id']} @ {row['home_team_id']}")
            else:
                result = remove_game_and_related(conn, gid)
                print(f"  Removed: {row['date']} - {row['away_team_id']} @ {row['home_team_id']}")
                removed_count += 1
    
    return removed_count


def cleanup_conflicts(conn, dry_run=True):
    """Review and clean up scheduling conflicts"""
    print("\n" + "="*60)
    print("STEP 3: RESOLVING SCHEDULING CONFLICTS")
    print("="*60)
    
    conflicts = find_scheduling_conflicts(conn)
    
    if not conflicts:
        print("No scheduling conflicts found.")
        return 0
    
    games_to_review = find_games_on_conflicts(conn, conflicts)
    
    if not games_to_review:
        print("No games identified for removal from conflicts.")
        return 0
    
    print(f"Found {len(games_to_review)} games to review from {len(conflicts)} conflicts:")
    
    removed_count = 0
    seen_ids = set()
    
    for item in games_to_review:
        game = item['game']
        gid = game['id']
        
        if gid in seen_ids:
            continue
        seen_ids.add(gid)
        
        if dry_run:
            print(f"  [DRY RUN] Would remove: {game['date']} - {game['away_team_id']} @ {game['home_team_id']}")
            print(f"            Reason: {item['reason']}")
        else:
            result = remove_game_and_related(conn, gid)
            print(f"  Removed: {game['date']} - {game['away_team_id']} @ {game['home_team_id']}")
            removed_count += 1
    
    return removed_count


def verify_cleanup(conn):
    """Verify the cleanup was successful"""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check for remaining duplicates
    duplicates = find_duplicate_games(conn)
    print(f"Remaining duplicate games: {len(duplicates)}")
    
    # Check for remaining conflicts
    conflicts = find_scheduling_conflicts(conn)
    print(f"Remaining scheduling conflicts: {len(conflicts)}")
    
    if conflicts:
        print("\nRemaining conflicts (may be legitimate double-headers):")
        for c in conflicts[:10]:
            print(f"  {c['date']}: {c['team']} plays {c['count']} games")
    
    # Game counts
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM games")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM games WHERE status = 'scheduled'")
    scheduled = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM games WHERE status = 'final'")
    final = cursor.fetchone()[0]
    
    print(f"\nGame counts:")
    print(f"  Total: {total}")
    print(f"  Scheduled: {scheduled}")
    print(f"  Final: {final}")
    
    return len(duplicates) == 0


def main():
    print("="*60)
    print("COMPREHENSIVE DATABASE CLEANUP")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    conn = get_db_connection()
    
    # First do a dry run
    print("\n" + "#"*60)
    print("DRY RUN - No changes will be made")
    print("#"*60)
    
    dup_count = cleanup_duplicates(conn, dry_run=True)
    tourn_count = cleanup_tournament_games(conn, dry_run=True)
    conflict_count = cleanup_conflicts(conn, dry_run=True)
    
    total_dry = dup_count + tourn_count + conflict_count
    print(f"\nDry run summary: Would remove {total_dry} games total")
    
    # Now do the actual cleanup
    print("\n" + "#"*60)
    print("ACTUAL CLEANUP - Making changes")
    print("#"*60)
    
    dup_count = cleanup_duplicates(conn, dry_run=False)
    tourn_count = cleanup_tournament_games(conn, dry_run=False)
    conflict_count = cleanup_conflicts(conn, dry_run=False)
    
    conn.commit()
    
    total_removed = dup_count + tourn_count + conflict_count
    print(f"\nActual cleanup: Removed {total_removed} games")
    print(f"  - Duplicates: {dup_count}")
    print(f"  - Tournament games: {tourn_count}")
    print(f"  - Conflict resolutions: {conflict_count}")
    
    # Verify
    success = verify_cleanup(conn)
    
    # Save cleanup log
    cleanup_log = {
        'timestamp': datetime.now().isoformat(),
        'duplicates_removed': dup_count,
        'tournament_games_removed': tourn_count,
        'conflicts_resolved': conflict_count,
        'total_removed': total_removed,
        'verification_passed': success
    }
    
    with open('data/comprehensive_cleanup_log.json', 'w') as f:
        json.dump(cleanup_log, f, indent=2)
    
    conn.close()
    
    print(f"\nCleanup complete. Log saved to data/comprehensive_cleanup_log.json")
    return success


if __name__ == "__main__":
    main()
