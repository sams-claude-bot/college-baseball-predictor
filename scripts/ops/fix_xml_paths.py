#!/usr/bin/env python3
"""One-time fix: verify and correct xml_file paths in statbroadcast_events."""
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from statbroadcast_client import StatBroadcastClient

DB = Path(__file__).parent.parent.parent / 'data' / 'baseball.db'

def main():
    conn = sqlite3.connect(str(DB), timeout=30)
    client = StatBroadcastClient()

    rows = conn.execute('''
        SELECT sb_event_id, xml_file, group_id, game_id, game_date
        FROM statbroadcast_events
        WHERE completed = 0 OR game_date >= "2026-02-25"
        ORDER BY game_date DESC
    ''').fetchall()

    print(f"Checking {len(rows)} recent/active events...")

    mismatches = []
    checked = 0
    for r in rows:
        eid, db_xml, db_group, game_id, game_date = r
        try:
            info = client.get_event_info(eid)
            if info and info.get('xml_file') and info['xml_file'] != db_xml:
                mismatches.append((eid, db_xml, info['xml_file'], info.get('group_id', ''), game_id, game_date))
        except Exception:
            pass
        checked += 1
        if checked % 50 == 0:
            print(f"  {checked}/{len(rows)}...")
        time.sleep(0.02)

    print(f"\nChecked: {checked}")
    print(f"Mismatched: {len(mismatches)}")

    if mismatches:
        print("\nMismatches:")
        for eid, old, new, grp, gid, date in mismatches:
            print(f"  {date} {gid or '?'} | {old} -> {new}")
            conn.execute(
                'UPDATE statbroadcast_events SET xml_file = ?, group_id = ? WHERE sb_event_id = ?',
                (new, grp, eid)
            )
        conn.commit()
        print(f"\nFixed {len(mismatches)} events!")
    else:
        print("\nAll xml_files are correct!")

    conn.close()

if __name__ == '__main__':
    main()
