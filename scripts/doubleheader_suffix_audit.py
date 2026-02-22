#!/usr/bin/env python3
"""Audit _gN vs _gmN suffix drift and DH consistency."""

import re
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / 'data' / 'baseball.db'


def parse_suffix(game_id: str):
    m = re.search(r'_(gm|g)(\d+)$', game_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    rows = c.execute("SELECT id, date, home_team_id, away_team_id, status FROM games").fetchall()

    legacy_g = []
    bad_gm1 = []
    malformed = []
    dh_pair_bases = {}  # base -> set(nums)

    for r in rows:
        gid = r['id']
        style, num = parse_suffix(gid)

        if style == 'g':
            legacy_g.append(r)
        if style == 'gm' and num == 1:
            bad_gm1.append(r)
        if style in ('g', 'gm') and (num is None or num < 1):
            malformed.append(r)

        base = re.sub(r'_(gm|g)\d+$', '', gid)
        if style in ('g', 'gm') and num:
            dh_pair_bases.setdefault(base, set()).add(num)

    # bases that have game2+ without game1 represented in any form
    missing_g1 = []
    for base, nums in dh_pair_bases.items():
        if any(n >= 2 for n in nums) and 1 not in nums:
            # verify unsuffixed game1 exists
            exists_unsuffixed = c.execute("SELECT 1 FROM games WHERE id=?", (base,)).fetchone()
            if not exists_unsuffixed:
                missing_g1.append((base, sorted(nums)))

    print('=== Doubleheader Suffix Audit ===')
    print(f'legacy _gN rows: {len(legacy_g)}')
    print(f'non-canonical _gm1 rows: {len(bad_gm1)}')
    print(f'malformed suffix rows: {len(malformed)}')
    print(f'bases with 2+ but missing game1: {len(missing_g1)}')

    if legacy_g:
        print('\n--- legacy _gN (first 20) ---')
        for r in legacy_g[:20]:
            print(f"{r['date']} {r['id']} ({r['status']})")

    if bad_gm1:
        print('\n--- non-canonical _gm1 (first 20) ---')
        for r in bad_gm1[:20]:
            print(f"{r['date']} {r['id']} ({r['status']})")

    if missing_g1:
        print('\n--- missing game1 base (first 20) ---')
        for base, nums in missing_g1[:20]:
            print(f"{base} has suffixes {nums} but no unsuffixed game1")

    conn.close()

    issues = len(legacy_g) + len(bad_gm1) + len(malformed) + len(missing_g1)
    if issues:
        print(f"\n❌ DH suffix audit FAILED: {issues} issue(s)")
        raise SystemExit(1)

    print('\n✅ DH suffix audit passed')


if __name__ == '__main__':
    main()
