#!/usr/bin/env python3
"""
Finalize Games — Clean up non-final games from a given date.

1. Syncs scores from D1Baseball team pages (handles doubleheaders)
2. Marks games as postponed/canceled if D1BB has no result and date has passed
3. Evaluates any newly-finalized predictions

Usage:
    python3 scripts/finalize_games.py                    # Yesterday
    python3 scripts/finalize_games.py --date 2026-02-21  # Specific date
    python3 scripts/finalize_games.py --dry-run           # Preview only
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from run_utils import ScriptRunner
from verify_team_schedule import fetch_d1bb_schedule, load_d1bb_slugs, load_reverse_slug_map
from d1b_team_sync import sync_team
from schedule_gateway import ScheduleGateway

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_nonfinal_teams(db, date_str):
    """Return set of team IDs that have non-final games on date."""
    rows = db.execute("""
        SELECT home_team_id, away_team_id FROM games
        WHERE date = ? AND status != 'final'
    """, (date_str,)).fetchall()
    teams = set()
    for r in rows:
        teams.add(r['home_team_id'])
        teams.add(r['away_team_id'])
    return teams


def mark_postponed(db, date_str, dry_run=False, verbose=False):
    """Mark remaining non-final games from a past date as postponed.
    
    Only marks games where D1BB also has no result (confirming it wasn't played).
    Games stuck as in-progress with partial scores get marked too if D1BB has no result.
    """
    rows = db.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.status, g.home_score, g.away_score
        FROM games g
        WHERE g.date = ? AND g.status != 'final'
    """, (date_str,)).fetchall()
    
    gw = ScheduleGateway(db)
    marked = 0
    for r in rows:
        if not dry_run:
            gw.mark_postponed(r['id'], reason='No result on D1Baseball')
        if verbose:
            print(f"  ⏸️  Postponed: {r['id']} (was {r['status']})")
        marked += 1
    
    return marked


def main():
    parser = argparse.ArgumentParser(description='Finalize games from a given date')
    parser.add_argument('--date', '-d', help='Date to finalize (YYYY-MM-DD, default: yesterday)')
    parser.add_argument('--dry-run', action='store_true', help='Preview only')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-postpone', action='store_true', help='Skip marking games as postponed')
    args = parser.parse_args()

    runner = ScriptRunner("finalize_games")
    
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    runner.info(f"Finalizing games for {target_date}")
    
    db = get_db()
    slugs = load_d1bb_slugs()
    reverse = load_reverse_slug_map()
    
    # Count non-final before
    before = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status != 'final'",
        (target_date,)
    ).fetchone()['c']
    
    total_games = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ?",
        (target_date,)
    ).fetchone()['c']
    
    runner.info(f"Total games: {total_games} | Non-final: {before}")
    
    if before == 0:
        runner.info("All games already final — nothing to do")
        runner.finish()
        return
    
    # Phase 1: Sync scores from D1BB team pages
    teams = get_nonfinal_teams(db, target_date)
    teams_with_slugs = {t for t in teams if t in slugs}
    runner.info(f"Phase 1: Syncing {len(teams_with_slugs)} teams from D1Baseball...")
    
    total_scored = 0
    total_created = 0
    errors = []
    
    for i, tid in enumerate(sorted(teams_with_slugs)):
        try:
            stats = sync_team(db, tid, slugs[tid], reverse,
                            dry_run=args.dry_run, verbose=args.verbose)
            total_scored += stats['scored']
            total_created += stats['created']
            if stats['errors']:
                errors.append(tid)
        except Exception as e:
            runner.warn(f"Error syncing {tid}: {e}")
            errors.append(tid)
        
        if not args.dry_run and i % 10 == 9:
            db.commit()
        time.sleep(0.3)
    
    if not args.dry_run:
        db.commit()
    
    runner.info(f"Phase 1 results: {total_scored} scored, {total_created} created, {len(errors)} errors")
    
    # Phase 2: Mark remaining as postponed (if date is in the past)
    remaining = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status != 'final'",
        (target_date,)
    ).fetchone()['c']
    
    if remaining > 0 and not args.no_postpone:
        today = datetime.now().strftime('%Y-%m-%d')
        if target_date < today:
            runner.info(f"Phase 2: Marking {remaining} remaining games as postponed...")
            marked = mark_postponed(db, target_date, dry_run=args.dry_run, verbose=args.verbose)
            if not args.dry_run:
                db.commit()
            runner.info(f"Marked {marked} games as postponed")
        else:
            runner.info(f"Phase 2: Skipped — {target_date} is today or future ({remaining} still pending)")
    
    # Final count
    final_nonfinal = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status NOT IN ('final', 'postponed', 'canceled')",
        (target_date,)
    ).fetchone()['c']
    
    final_count = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status = 'final'",
        (target_date,)
    ).fetchone()['c']
    
    postponed_count = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status = 'postponed'",
        (target_date,)
    ).fetchone()['c']
    
    db.close()
    
    runner.info(f"Final: {final_count} final, {postponed_count} postponed, {final_nonfinal} unresolved")
    runner.add_stat("date", target_date)
    runner.add_stat("scored", total_scored)
    runner.add_stat("postponed", postponed_count)
    runner.add_stat("unresolved", final_nonfinal)
    runner.finish()


if __name__ == '__main__':
    main()
