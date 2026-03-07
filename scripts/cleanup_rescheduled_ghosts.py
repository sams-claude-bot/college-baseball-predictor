#!/usr/bin/env python3
"""
One-time cleanup: merge postponed game ghosts that have replacements.

Finds all postponed games with no scores, checks for a replacement game
(same teams, within +7 days, not postponed), migrates FK data, and deletes
the ghost entry.

Usage:
    python3 scripts/cleanup_rescheduled_ghosts.py --dry-run    # Preview
    python3 scripts/cleanup_rescheduled_ghosts.py              # Execute
    python3 scripts/cleanup_rescheduled_ghosts.py --verbose    # With details
"""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from schedule_gateway import ScheduleGateway

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'


def find_ghost_pairs(db):
    """Find all postponed games that have a replacement on a nearby date."""
    postponed = db.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.date,
               g.home_score, g.away_score
        FROM games g
        WHERE g.status = 'postponed'
          AND (g.home_score IS NULL OR g.home_score = 0)
          AND (g.away_score IS NULL OR g.away_score = 0)
        ORDER BY g.date
    """).fetchall()

    pairs = []
    for p in postponed:
        replacement = db.execute("""
            SELECT id, date, status FROM games
            WHERE date > ?
              AND abs(julianday(date) - julianday(?)) <= 7
              AND status IN ('scheduled', 'final', 'in-progress')
              AND (
                (home_team_id = ? AND away_team_id = ?) OR
                (home_team_id = ? AND away_team_id = ?)
              )
              AND id != ?
            ORDER BY date
            LIMIT 1
        """, (p['date'], p['date'],
              p['home_team_id'], p['away_team_id'],
              p['away_team_id'], p['home_team_id'],
              p['id'])).fetchone()

        if replacement:
            pairs.append({
                'ghost_id': p['id'],
                'ghost_date': p['date'],
                'replacement_id': replacement['id'],
                'replacement_date': replacement['date'],
                'replacement_status': replacement['status'],
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Clean up postponed game ghosts that have replacements')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview only, do not modify database')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show details for each merge')
    args = parser.parse_args()

    db = sqlite3.connect(str(DB_PATH), timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")

    pairs = find_ghost_pairs(db)

    if not pairs:
        print("No ghost pairs found — nothing to clean up.")
        db.close()
        return

    print(f"Found {len(pairs)} ghost pairs to merge"
          f"{' (dry-run)' if args.dry_run else ''}:")

    gw = ScheduleGateway(db)
    total_fk = {'migrated': 0, 'deleted': 0}

    for pair in pairs:
        if args.verbose or args.dry_run:
            print(f"  {pair['ghost_id']} ({pair['ghost_date']}) "
                  f"-> {pair['replacement_id']} ({pair['replacement_date']}) "
                  f"[{pair['replacement_status']}]")

        if not args.dry_run:
            fk = gw.migrate_fk_rows(pair['ghost_id'], pair['replacement_id'])
            total_fk['migrated'] += fk['migrated']
            total_fk['deleted'] += fk['deleted']
            db.execute("DELETE FROM games WHERE id = ?", (pair['ghost_id'],))
            db.commit()

            if args.verbose:
                print(f"    FK: {fk}")

    action = "Would merge" if args.dry_run else "Merged"
    print(f"\n{action} {len(pairs)} ghost games.")
    if not args.dry_run:
        print(f"FK rows migrated: {total_fk['migrated']}, "
              f"deleted (conflicts): {total_fk['deleted']}")

    # Post-cleanup verification
    remaining = db.execute("""
        SELECT COUNT(*) as c FROM games
        WHERE status = 'postponed'
          AND (home_score IS NULL OR home_score = 0)
          AND (away_score IS NULL OR away_score = 0)
    """).fetchone()['c']
    print(f"Remaining postponed (no scores): {remaining}")

    db.close()


if __name__ == '__main__':
    main()
