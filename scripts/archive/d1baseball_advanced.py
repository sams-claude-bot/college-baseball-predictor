#!/usr/bin/env python3
"""
D1Baseball Advanced Stats — DB updater.

Takes JSON from stdin (or --file) with advanced stats extracted from D1Baseball,
and updates player_stats + player_stats_snapshots in the database.

JSON format:
{
  "team_slug": "arkansas",
  "adv_batting": [...],
  "bb_batting": [...],
  "adv_pitching": [...],
  "bb_pitching": [...]
}

This script handles DB operations only — browser extraction is done by OpenClaw.
Can also be used as a library: import and call update_team_advanced(db, team_id, data).
"""

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

SLUG_TO_TEAM_ID = {
    'gatech': 'georgia-tech',
    'vandy': 'vanderbilt',
}

# Load full slug map from config if available
_slugs_file = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
if _slugs_file.exists():
    import json as _json
    _data = _json.loads(_slugs_file.read_text())
    for _tid, _d1slug in _data.get('team_id_to_d1bb_slug', {}).items():
        if _d1slug != _tid and _d1slug not in SLUG_TO_TEAM_ID:
            SLUG_TO_TEAM_ID[_d1slug] = _tid


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def resolve_team_id(db, slug):
    mapped = SLUG_TO_TEAM_ID.get(slug, slug)
    row = db.execute("SELECT id FROM teams WHERE id = ?", (mapped,)).fetchone()
    if row:
        return row['id']
    name_guess = mapped.replace('-', ' ').title()
    row = db.execute("SELECT id FROM teams WHERE name LIKE ?", (f'%{name_guess}%',)).fetchone()
    return row['id'] if row else None


def safe_float(val):
    """Parse a stat value to float. Handles %, INF, dashes, etc."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ('', '—', '-', 'INF', 'inf', 'NaN', ''):
        return None
    if s.endswith('%'):
        try:
            return round(float(s[:-1]) / 100.0, 4)
        except ValueError:
            return None
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def safe_kbb(val):
    """Parse K:BB ratio."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ('', 'INF', 'inf', '—', '-'):
        return None
    try:
        return round(float(s), 3)
    except ValueError:
        return None


def update_player(db, team_id, name, updates):
    """Update a player's stats in player_stats table."""
    filtered = {k: v for k, v in updates.items() if v is not None}
    if not filtered:
        return False
    set_clause = ', '.join(f"{k} = :{k}" for k in filtered)
    params = dict(filtered)
    params['team_id'] = team_id
    params['name'] = name
    result = db.execute(f"""
        UPDATE player_stats SET {set_clause}, updated_at = CURRENT_TIMESTAMP
        WHERE team_id = :team_id AND name = :name
    """, params)
    return result.rowcount > 0


def update_snapshot(db, team_id, name, snapshot_date, updates):
    """Update advanced columns in existing snapshot row."""
    filtered = {k: v for k, v in updates.items() if v is not None}
    if not filtered:
        return
    set_clause = ', '.join(f"{k} = :{k}" for k in filtered)
    params = dict(filtered)
    params['snapshot_date'] = snapshot_date
    params['team_id'] = team_id
    params['player_name'] = name
    db.execute(f"""
        UPDATE player_stats_snapshots SET {set_clause}
        WHERE snapshot_date = :snapshot_date AND team_id = :team_id AND player_name = :player_name
    """, params)


def update_team_advanced(db, team_id, data, dry_run=False):
    """Update a team's advanced stats from extracted D1Baseball data."""
    today = date.today().isoformat()
    stats = {'updated': 0, 'skipped': 0}

    # Advanced Batting
    for row in data.get('adv_batting', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents'):
            continue
        updates = {
            'k_pct': safe_float(row.get('K%')),
            'bb_pct': safe_float(row.get('BB%')),
            'k_bb_ratio': safe_kbb(row.get('K:BB')),
            'iso': safe_float(row.get('ISO')),
            'babip': safe_float(row.get('BABIP')),
            'woba': safe_float(row.get('wOBA')),
            'wrc': safe_float(row.get('wRC')),
            'wraa': safe_float(row.get('wRAA')),
            'wrc_plus': safe_float(row.get('wRC+')),
        }
        if dry_run:
            print(f"  [BAT ADV] {name}: wRC+={updates.get('wrc_plus')} wOBA={updates.get('woba')} ISO={updates.get('iso')}")
        else:
            if update_player(db, team_id, name, updates):
                stats['updated'] += 1
                update_snapshot(db, team_id, name, today, updates)
            else:
                stats['skipped'] += 1

    # Batted Ball Batting
    for row in data.get('bb_batting', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents'):
            continue
        updates = {
            'gb_pct': safe_float(row.get('GB%')),
            'ld_pct': safe_float(row.get('LD%')),
            'fb_pct': safe_float(row.get('FB%')),
            'pu_pct': safe_float(row.get('PU%')),
            'hr_fb_pct': safe_float(row.get('HR/FB%')),
        }
        if not dry_run:
            update_player(db, team_id, name, updates)
            update_snapshot(db, team_id, name, today, {
                'gb_pct': updates['gb_pct'],
                'ld_pct': updates['ld_pct'],
                'fb_pct': updates['fb_pct'],
                'pu_pct': updates['pu_pct'],
                'hr_fb_pct': updates['hr_fb_pct'],
            })

    # Advanced Pitching
    for row in data.get('adv_pitching', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents'):
            continue
        updates = {
            'fip': safe_float(row.get('FIP')),
            'xfip': safe_float(row.get('xFIP')),
            'siera': safe_float(row.get('SIERA')),
            'lob_pct': safe_float(row.get('LOB%')),
            'k_pct_pitch': safe_float(row.get('K%')),
            'bb_pct_pitch': safe_float(row.get('BB%')),
            'k_bb_pct_pitch': safe_float(row.get('K-BB%')),
            'k_bb_ratio_pitch': safe_kbb(row.get('K:BB')),
            'babip_pitch': safe_float(row.get('BABIP')),
            'obp_against': safe_float(row.get('OBP')),
            'slg_against': safe_float(row.get('SLG')),
            'ops_against': safe_float(row.get('OPS')),
        }
        if dry_run:
            print(f"  [PIT ADV] {name}: FIP={updates.get('fip')} xFIP={updates.get('xfip')} SIERA={updates.get('siera')}")
        else:
            if update_player(db, team_id, name, updates):
                stats['updated'] += 1
                update_snapshot(db, team_id, name, today, {
                    'fip': updates['fip'], 'xfip': updates['xfip'],
                    'siera': updates['siera'], 'lob_pct': updates['lob_pct'],
                    'k_pct_pitch': updates['k_pct_pitch'],
                    'bb_pct_pitch': updates['bb_pct_pitch'],
                    'babip_pitch': updates['babip_pitch'],
                    'obp_against': updates['obp_against'],
                    'slg_against': updates['slg_against'],
                    'ops_against': updates['ops_against'],
                })
            else:
                stats['skipped'] += 1

    # Batted Ball Pitching
    for row in data.get('bb_pitching', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents'):
            continue
        updates = {
            'gb_pct_pitch': safe_float(row.get('GB%')),
            'ld_pct_pitch': safe_float(row.get('LD%')),
            'fb_pct_pitch': safe_float(row.get('FB%')),
            'pu_pct_pitch': safe_float(row.get('PU%')),
            'hr_fb_pct_pitch': safe_float(row.get('HR/FB%')),
        }
        if not dry_run:
            update_player(db, team_id, name, updates)
            update_snapshot(db, team_id, name, today, {
                'gb_pct_pitch': updates['gb_pct_pitch'],
                'ld_pct_pitch': updates['ld_pct_pitch'],
                'fb_pct_pitch': updates['fb_pct_pitch'],
                'pu_pct_pitch': updates['pu_pct_pitch'],
                'hr_fb_pct_pitch': updates['hr_fb_pct_pitch'],
            })

    if not dry_run:
        db.commit()

    return stats


def main():
    parser = argparse.ArgumentParser(description='Update D1Baseball advanced stats in DB')
    parser.add_argument('--file', '-f', help='JSON file (default: stdin)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            payload = json.load(f)
    else:
        payload = json.load(sys.stdin)

    # Support single team or list
    teams = payload if isinstance(payload, list) else [payload]

    db = get_db()
    total_updated = 0
    total_skipped = 0

    for team_data in teams:
        slug = team_data.get('team_slug', '')
        team_id = resolve_team_id(db, slug)
        if not team_id:
            print(f"ERROR: No team_id for slug '{slug}'")
            continue

        print(f"\n=== {slug} (team_id={team_id}) ===")
        print(f"  adv_batting: {len(team_data.get('adv_batting', []))} rows")
        print(f"  bb_batting: {len(team_data.get('bb_batting', []))} rows")
        print(f"  adv_pitching: {len(team_data.get('adv_pitching', []))} rows")
        print(f"  bb_pitching: {len(team_data.get('bb_pitching', []))} rows")

        stats = update_team_advanced(db, team_id, team_data, dry_run=args.dry_run)
        total_updated += stats['updated']
        total_skipped += stats['skipped']
        print(f"  Updated: {stats['updated']}, Skipped (not in DB): {stats['skipped']}")

    db.close()
    print(f"\nTotal: {total_updated} updated, {total_skipped} skipped")


if __name__ == '__main__':
    main()
