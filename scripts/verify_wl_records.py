#!/usr/bin/env python3
"""
Verify W-L records against D1Baseball conference standings.

Fetches D1BB conference standings pages, parses HTML for team slugs and
overall W-L records, compares against our computed W-L from the games table.

Usage:
    python3 scripts/verify_wl_records.py                    # All conferences
    python3 scripts/verify_wl_records.py --conference SEC   # Single conference
    python3 scripts/verify_wl_records.py --verbose          # Show all teams
    python3 scripts/verify_wl_records.py --json             # JSON output
    python3 scripts/verify_wl_records.py --delay 0.5        # Custom delay
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
CONF_SLUGS_PATH = PROJECT_DIR / 'config' / 'd1bb_conference_slugs.json'
D1B_SLUG_MAPPING_PATH = PROJECT_DIR / 'data' / 'd1b_slug_mapping.json'


def load_conference_slugs():
    """Load conference name -> D1BB URL slug mapping."""
    with open(CONF_SLUGS_PATH) as f:
        return json.load(f)


def load_d1bb_slug_map():
    """Load D1BB slug -> our team_id mapping."""
    with open(D1B_SLUG_MAPPING_PATH) as f:
        return json.load(f)


def fetch_conference_standings(conf_slug, year=2026):
    """Fetch and parse D1BB conference standings page.

    Returns list of dicts with: d1bb_slug, wins, losses, conf_record, streak
    """
    url = f"https://d1baseball.com/conference/{conf_slug}/{year}/"
    resp = requests.get(url, timeout=15, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; BaseballBot/1.0)'
    })
    resp.raise_for_status()

    results = []
    table = re.search(r'<table[^>]*>(.*?)</table>', resp.text, re.DOTALL)
    if not table:
        return results

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table.group(1), re.DOTALL)
    for row in rows:
        slug_m = re.search(r'/team/([^/]+)/', row)
        tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        tds_clean = [re.sub(r'<[^>]+>', '', td).strip() for td in tds]
        if slug_m and len(tds_clean) >= 5:
            overall = tds_clean[4]  # "13-2" format
            parts = overall.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                w, l = int(parts[0]), int(parts[1])
                results.append({
                    'd1bb_slug': slug_m.group(1),
                    'wins': w, 'losses': l,
                    'conf_record': tds_clean[1] if len(tds_clean) > 1 else '',
                    'streak': tds_clean[6] if len(tds_clean) > 6 else ''
                })
    return results


def compute_db_wl(db, team_id):
    """Compute W-L record from our games table."""
    row = db.execute("""
        SELECT
            COUNT(CASE WHEN winner_id = ? THEN 1 END) AS wins,
            COUNT(CASE WHEN winner_id IS NOT NULL AND winner_id != ? THEN 1 END) AS losses
        FROM games
        WHERE (home_team_id = ? OR away_team_id = ?)
          AND status = 'final'
    """, (team_id, team_id, team_id, team_id)).fetchone()
    return row['wins'], row['losses']


def main():
    parser = argparse.ArgumentParser(
        description='Verify W-L records against D1Baseball')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all teams, not just mismatches')
    parser.add_argument('--conference', '-c',
                        help='Check a single conference (e.g., SEC)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between HTTP requests (default: 0.3s)')
    args = parser.parse_args()

    conf_slugs = load_conference_slugs()
    d1bb_slug_map = load_d1bb_slug_map()

    db = sqlite3.connect(str(DB_PATH), timeout=30)
    db.row_factory = sqlite3.Row

    # Filter conferences if specified
    if args.conference:
        if args.conference not in conf_slugs:
            print(f"Unknown conference: {args.conference}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(conf_slugs.keys()))}", file=sys.stderr)
            sys.exit(1)
        conferences = {args.conference: conf_slugs[args.conference]}
    else:
        conferences = conf_slugs

    today = datetime.now().strftime('%Y-%m-%d')
    total_checked = 0
    total_match = 0
    mismatches = []
    errors = []

    for conf_name, conf_slug in sorted(conferences.items()):
        try:
            standings = fetch_conference_standings(conf_slug)
        except Exception as e:
            errors.append({'conference': conf_name, 'error': str(e)})
            if not args.json:
                print(f"  WARNING: Failed to fetch {conf_name}: {e}",
                      file=sys.stderr)
            continue

        for team in standings:
            d1bb_slug = team['d1bb_slug']
            team_id = d1bb_slug_map.get(d1bb_slug)

            if not team_id:
                errors.append({
                    'conference': conf_name,
                    'd1bb_slug': d1bb_slug,
                    'error': 'No team_id mapping'
                })
                if not args.json:
                    print(f"  WARNING: No mapping for D1BB slug '{d1bb_slug}' "
                          f"in {conf_name}", file=sys.stderr)
                continue

            db_wins, db_losses = compute_db_wl(db, team_id)
            d1bb_wins, d1bb_losses = team['wins'], team['losses']
            total_checked += 1

            if db_wins == d1bb_wins and db_losses == d1bb_losses:
                total_match += 1
                if args.verbose and not args.json:
                    print(f"  {team_id}: DB={db_wins}-{db_losses}, "
                          f"D1BB={d1bb_wins}-{d1bb_losses} OK")
            else:
                mismatch = {
                    'team_id': team_id,
                    'conference': conf_name,
                    'd1bb_slug': d1bb_slug,
                    'db_wins': db_wins, 'db_losses': db_losses,
                    'd1bb_wins': d1bb_wins, 'd1bb_losses': d1bb_losses,
                    'win_diff': db_wins - d1bb_wins,
                    'loss_diff': db_losses - d1bb_losses,
                }
                mismatches.append(mismatch)

        time.sleep(args.delay)

    db.close()

    # Output
    if args.json:
        result = {
            'date': today,
            'total_checked': total_checked,
            'total_match': total_match,
            'mismatches': mismatches,
            'errors': errors,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== W-L Verification ({today}) ===")
        print(f"Checked {total_checked} teams across "
              f"{len(conferences)} conferences")
        print(f"{total_match} teams match D1Baseball")

        if mismatches:
            print(f"{len(mismatches)} mismatches:")
            for m in sorted(mismatches, key=lambda x: x['team_id']):
                print(f"  {m['team_id']:30s} DB={m['db_wins']}-{m['db_losses']}, "
                      f"D1BB={m['d1bb_wins']}-{m['d1bb_losses']} "
                      f"(diff: W{m['win_diff']:+d} L{m['loss_diff']:+d}) "
                      f"[{m['conference']}]")
        else:
            print("No mismatches found!")

        if errors:
            print(f"\n{len(errors)} warnings:")
            for e in errors:
                print(f"  {e}")


if __name__ == '__main__':
    main()
