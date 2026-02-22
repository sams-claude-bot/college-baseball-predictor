#!/usr/bin/env python3
"""
Integrity guard for games table consistency.

Checks:
1) score/winner conflicts on final games
2) game_id orientation mismatches vs away/home team ids
3) final games with null winner_id (non-tie)

Exit code:
- 0 when clean
- 1 when any issue found
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'baseball.db'


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1) score/winner conflicts
    score_winner_conflicts = c.execute('''
        SELECT id, date, home_team_id, away_team_id, home_score, away_score, winner_id
        FROM games
        WHERE status = 'final'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND winner_id IS NOT NULL
          AND (
                (home_score > away_score AND winner_id != home_team_id)
             OR (away_score > home_score AND winner_id != away_team_id)
          )
        ORDER BY date, id
    ''').fetchall()

    # 2) game_id orientation mismatches (ignore DH suffix)
    rows = c.execute('''
        SELECT id, date, home_team_id, away_team_id, status
        FROM games
    ''').fetchall()

    id_orientation_mismatches = []
    for r in rows:
        base = re.sub(r'_(gm|g)\d+$', '', r['id'])
        m = re.match(r'^(\d{4}-\d{2}-\d{2})_(.+)_(.+)$', base)
        if not m:
            continue
        _date, id_away, id_home = m.groups()
        if id_away != r['away_team_id'] or id_home != r['home_team_id']:
            id_orientation_mismatches.append(r)

    # 3) final games missing winner_id where not tie
    null_winner_finals = c.execute('''
        SELECT id, date, home_team_id, away_team_id, home_score, away_score
        FROM games
        WHERE status = 'final'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND home_score != away_score
          AND winner_id IS NULL
        ORDER BY date, id
    ''').fetchall()

    conn.close()

    total_issues = (
        len(score_winner_conflicts)
        + len(id_orientation_mismatches)
        + len(null_winner_finals)
    )

    print('=== Integrity Guard ===')
    print(f'score/winner conflicts: {len(score_winner_conflicts)}')
    print(f'ID orientation mismatches: {len(id_orientation_mismatches)}')
    print(f'final non-ties with null winner_id: {len(null_winner_finals)}')

    if score_winner_conflicts:
        print('\n--- score/winner conflicts (first 10) ---')
        for r in score_winner_conflicts[:10]:
            print(f"{r['date']} {r['id']} | {r['away_team_id']} {r['away_score']} @ {r['home_team_id']} {r['home_score']} | winner={r['winner_id']}")

    if id_orientation_mismatches:
        print('\n--- ID orientation mismatches (first 10) ---')
        for r in id_orientation_mismatches[:10]:
            print(f"{r['date']} {r['id']} | db={r['away_team_id']} @ {r['home_team_id']} ({r['status']})")

    if null_winner_finals:
        print('\n--- final non-ties with null winner_id (first 10) ---')
        for r in null_winner_finals[:10]:
            print(f"{r['date']} {r['id']} | {r['away_team_id']} {r['away_score']} @ {r['home_team_id']} {r['home_score']}")

    if total_issues:
        print(f"\n❌ Integrity check FAILED: {total_issues} issue(s) found")
        return 1

    print('\n✅ Integrity check passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
