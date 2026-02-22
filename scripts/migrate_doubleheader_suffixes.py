#!/usr/bin/env python3
"""Migrate legacy _gN game IDs to canonical _gmN IDs.

- _g1 -> unsuffixed base game ID
- _gN (N>=2) -> _gmN
Also migrates known game_id FK references.
"""

import re
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / 'data' / 'baseball.db'

FK_TABLES = [
    'model_predictions', 'betting_lines', 'game_weather',
    'tracked_bets', 'tracked_bets_spreads', 'tracked_confident_bets',
    'totals_predictions', 'spread_predictions', 'game_predictions',
    'pitching_matchups', 'game_boxscores', 'game_batting_stats',
    'game_pitching_stats', 'player_boxscore_batting', 'player_boxscore_pitching',
    'statbroadcast_boxscores', 'elo_history'
]


def target_id(old_id: str):
    m = re.search(r'^(.*)_g(\d+)$', old_id)
    if not m:
        return None
    base, n = m.group(1), int(m.group(2))
    if n == 1:
        return base
    return f"{base}_gm{n}"


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    rows = c.execute("SELECT id FROM games WHERE id GLOB '*_g[0-9]*' ORDER BY id").fetchall()
    planned = []
    for r in rows:
        old = r['id']
        new = target_id(old)
        if not new or new == old:
            continue
        planned.append((old, new))

    print(f"found_legacy={len(planned)}")

    migrated = 0
    skipped = 0

    for old, new in planned:
        exists = c.execute("SELECT 1 FROM games WHERE id = ?", (new,)).fetchone()
        if exists:
            print(f"SKIP exists: {old} -> {new}")
            skipped += 1
            continue

        for t in FK_TABLES:
            try:
                c.execute(f"UPDATE {t} SET game_id=? WHERE game_id=?", (new, old))
            except Exception:
                pass

        c.execute("UPDATE games SET id=? WHERE id=?", (new, old))
        print(f"MIGRATE {old} -> {new}")
        migrated += 1

    conn.commit()
    conn.close()

    print(f"migrated={migrated} skipped={skipped}")


if __name__ == '__main__':
    main()
