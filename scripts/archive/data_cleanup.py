#!/usr/bin/env python3
"""
Data Cleanup — Reusable tools for maintaining clean game data.

Usage:
    python3 scripts/data_cleanup.py dedup              # Remove exact duplicate game rows
    python3 scripts/data_cleanup.py purge-phantom       # Delete phantom games + orphaned predictions
    python3 scripts/data_cleanup.py audit               # Report data quality issues without fixing
    python3 scripts/data_cleanup.py all                 # Run full cleanup (audit + dedup + purge)
    python3 scripts/data_cleanup.py --dry-run <action>  # Preview without writing

College baseball context:
- Most teams play 3-game weekend series (Fri/Sat/Sun)
- Weather (especially rain) often moves Sunday games to Saturday doubleheaders
- ESPN scoreboard API caps at ~71 games/day; use espn_backfill.py to catch the rest
- Schedule loaders sometimes create phantom games from bad source data
- ESPN team name aliases (e.g. "Army Black Knights" vs "Army") can create duplicate rows
"""

import argparse
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "baseball.db"

# Known bad slugs that ESPN name resolution has produced as duplicates.
# Maps bad_slug -> canonical_slug. Add new ones here as discovered.
BAD_SLUG_ALIASES = {
    'army-black': 'army',
    'canisius-golden': 'canisius',
    'evansville-purple': 'evansville',
    'florida-international': 'fiu',
    'lehigh-mountain': 'lehigh',
    'maine-black': 'maine',
    'nebraska-omaha': 'omaha',
    'nevada-wolf': 'nevada',
    'niagara-purple': 'niagara',
    'olemiss': 'ole-miss',
    'san-jos-state': 'san-jose-state',
    'sc-upstate': 'south-carolina-upstate',
    'william-and-mary': 'william-mary',
    'miami': 'miami-fl',
    'tulane-green': 'tulane',
    'college-of-charleston': 'charleston',
    'queens': 'queens-university',
}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def id_quality(game_id):
    """Score a game ID — lower is better. Penalizes bad slugs and high suffixes."""
    penalty = sum(1 for slug in BAD_SLUG_ALIASES if slug in game_id)
    if '_g4' in game_id: penalty += 0.4
    elif '_g3' in game_id: penalty += 0.3
    elif '_g2' in game_id: penalty += 0.2
    return (penalty, len(game_id), game_id)


def audit(conn):
    """Report data quality issues without making changes."""
    cur = conn.cursor()
    
    print("=" * 60)
    print("DATA QUALITY AUDIT")
    print("=" * 60)
    
    # 1. Exact duplicate game rows
    dupes = cur.execute("""
        SELECT home_team_id, away_team_id, home_score, away_score, date, 
               GROUP_CONCAT(id) as ids, COUNT(*) c
        FROM games WHERE status='final'
        GROUP BY home_team_id, away_team_id, home_score, away_score, date
        HAVING c > 1
    """).fetchall()
    print(f"\n[1] Exact duplicate game rows: {len(dupes)}")
    for d in dupes:
        print(f"    {d['date']} | {d['away_team_id']} @ {d['home_team_id']} "
              f"= {d['away_score']}-{d['home_score']} | IDs: {d['ids']}")
    
    # 2. Phantom games (scheduled but past)
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    phantoms = cur.execute(
        "SELECT COUNT(*) FROM games WHERE status='scheduled' AND date < ?", (today,)
    ).fetchone()[0]
    print(f"\n[2] Stale 'scheduled' games (before today): {phantoms}")
    if phantoms:
        rows = cur.execute(
            "SELECT id, date FROM games WHERE status='scheduled' AND date < ? ORDER BY date LIMIT 20",
            (today,)
        ).fetchall()
        for r in rows:
            print(f"    {r['id']}")
    
    # 3. Games with status='phantom'
    phantom_count = cur.execute("SELECT COUNT(*) FROM games WHERE status='phantom'").fetchone()[0]
    print(f"\n[3] Games marked 'phantom': {phantom_count}")
    
    # 4. Orphan team IDs in games (team referenced but not in teams table)
    orphans = cur.execute("""
        SELECT DISTINCT t.id FROM (
            SELECT home_team_id as id FROM games
            UNION SELECT away_team_id as id FROM games
        ) t LEFT JOIN teams ON teams.id = t.id
        WHERE teams.id IS NULL
    """).fetchall()
    print(f"\n[4] Orphan team IDs (in games but not teams table): {len(orphans)}")
    for o in orphans:
        print(f"    {o['id']}")
    
    # 5. Game count by date (recent)
    print(f"\n[5] Recent game counts:")
    rows = cur.execute("""
        SELECT date, 
               SUM(CASE WHEN status='final' THEN 1 ELSE 0 END) as final,
               SUM(CASE WHEN status='scheduled' THEN 1 ELSE 0 END) as scheduled,
               SUM(CASE WHEN status NOT IN ('final','scheduled') THEN 1 ELSE 0 END) as other,
               COUNT(*) as total
        FROM games
        WHERE date >= date('now', '-7 days')
        GROUP BY date ORDER BY date
    """).fetchall()
    for r in rows:
        parts = [f"{r['final']} final"]
        if r['scheduled']: parts.append(f"{r['scheduled']} sched")
        if r['other']: parts.append(f"{r['other']} other")
        print(f"    {r['date']}: {', '.join(parts)}")
    
    print(f"\n{'=' * 60}")
    issues = len(dupes) + phantoms + phantom_count + len(orphans)
    print(f"Total issues: {issues}")
    return issues


def dedup(conn, dry_run=False):
    """Remove exact duplicate game rows, keeping the one with the cleanest ID.
    
    Duplicates arise when ESPN resolves team names to different slugs
    (e.g., 'army' vs 'army-black') creating separate game rows for
    the same actual game.
    """
    cur = conn.cursor()
    
    dupes = cur.execute("""
        SELECT home_team_id, away_team_id, home_score, away_score, date, 
               GROUP_CONCAT(id) as ids, COUNT(*) c
        FROM games WHERE status='final'
        GROUP BY home_team_id, away_team_id, home_score, away_score, date
        HAVING c > 1
    """).fetchall()
    
    if not dupes:
        print("No duplicate game rows found.")
        return 0
    
    print(f"Found {len(dupes)} duplicate groups")
    deleted = 0
    
    for d in dupes:
        ids = d['ids'].split(',')
        ids.sort(key=id_quality)
        keep = ids[0]
        remove = ids[1:]
        
        for bad_id in remove:
            if dry_run:
                print(f"  [DRY RUN] DEL: {bad_id} (keeping {keep})")
            else:
                # Remap references in prediction/bet tables
                for table in ['model_predictions', 'tracked_bets', 'tracked_bets_spreads', 'totals_predictions']:
                    try:
                        cur.execute(f"UPDATE {table} SET game_id=? WHERE game_id=?", (keep, bad_id))
                    except:
                        pass
                
                cur.execute("DELETE FROM games WHERE id=?", (bad_id,))
                print(f"  DEL: {bad_id} (kept {keep})")
            deleted += 1
    
    if not dry_run:
        conn.commit()
    
    print(f"Deleted {deleted} duplicate rows")
    return deleted


def purge_phantom(conn, dry_run=False):
    """Delete phantom games and all predictions/bets that reference them.
    
    Phantom games are entries created from bad schedule source data where
    the teams never actually played each other on that date. The backfill
    script (espn_backfill.py) marks these as status='phantom'.
    
    Also cleans up stale 'scheduled' games from past dates that were never
    resolved — these are also likely bad source data.
    """
    cur = conn.cursor()
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Collect IDs to purge: phantom status + stale scheduled
    phantom_ids = [r[0] for r in cur.execute(
        "SELECT id FROM games WHERE status='phantom'"
    ).fetchall()]
    
    stale_ids = [r[0] for r in cur.execute(
        "SELECT id FROM games WHERE status='scheduled' AND date < ?", (today,)
    ).fetchall()]
    
    all_ids = list(set(phantom_ids + stale_ids))
    
    if not all_ids:
        print("No phantom or stale games to purge.")
        return 0
    
    print(f"Games to purge: {len(phantom_ids)} phantom + {len(stale_ids)} stale scheduled = {len(all_ids)} total")
    
    if dry_run:
        for gid in all_ids:
            print(f"  [DRY RUN] Would delete: {gid}")
        return len(all_ids)
    
    # Clean references in prediction/bet tables
    placeholders = ','.join('?' * len(all_ids))
    for table in ['model_predictions', 'totals_predictions', 'tracked_bets', 'tracked_bets_spreads']:
        try:
            cleaned = cur.execute(
                f"DELETE FROM {table} WHERE game_id IN ({placeholders})", all_ids
            ).rowcount
            if cleaned:
                print(f"  Cleaned {cleaned} rows from {table}")
        except:
            pass
    
    # Delete the games
    cur.execute(f"DELETE FROM games WHERE id IN ({placeholders})", all_ids)
    conn.commit()
    
    print(f"Purged {len(all_ids)} games")
    return len(all_ids)


def cleanup_orphan_teams(conn, dry_run=False):
    """Remove teams that exist in the teams table but have zero game references.
    Only removes teams with bad slug aliases — keeps legitimately empty teams."""
    cur = conn.cursor()
    
    deleted = 0
    for bad_slug in BAD_SLUG_ALIASES:
        exists = cur.execute("SELECT id FROM teams WHERE id=?", (bad_slug,)).fetchone()
        if not exists:
            continue
        refs = cur.execute(
            "SELECT COUNT(*) FROM games WHERE home_team_id=? OR away_team_id=?",
            (bad_slug, bad_slug)
        ).fetchone()[0]
        if refs == 0:
            if dry_run:
                print(f"  [DRY RUN] Would delete orphan team: {bad_slug}")
            else:
                cur.execute("DELETE FROM teams WHERE id=?", (bad_slug,))
                cur.execute("DELETE FROM elo_ratings WHERE team_id=?", (bad_slug,))
                print(f"  Deleted orphan team: {bad_slug}")
            deleted += 1
    
    if not dry_run:
        conn.commit()
    
    if deleted:
        print(f"Removed {deleted} orphan teams")
    else:
        print("No orphan teams found")
    return deleted


def main():
    parser = argparse.ArgumentParser(description='College Baseball Data Cleanup')
    parser.add_argument('action', choices=['audit', 'dedup', 'purge-phantom', 'orphan-teams', 'all'],
                       help='Cleanup action to perform')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    args = parser.parse_args()
    
    conn = get_conn()
    
    if args.action == 'audit':
        audit(conn)
    elif args.action == 'dedup':
        dedup(conn, dry_run=args.dry_run)
    elif args.action == 'purge-phantom':
        purge_phantom(conn, dry_run=args.dry_run)
    elif args.action == 'orphan-teams':
        cleanup_orphan_teams(conn, dry_run=args.dry_run)
    elif args.action == 'all':
        print("Running full cleanup...\n")
        audit(conn)
        print()
        dedup(conn, dry_run=args.dry_run)
        print()
        purge_phantom(conn, dry_run=args.dry_run)
        print()
        cleanup_orphan_teams(conn, dry_run=args.dry_run)
    
    conn.close()


if __name__ == '__main__':
    main()
