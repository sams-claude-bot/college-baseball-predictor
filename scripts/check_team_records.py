#!/usr/bin/env python3
"""
Check each team's overall W-L record in our DB against D1Baseball's "Overall" record.
Optionally auto-fix mismatches by running targeted team syncs, then re-check.

Usage:
  python3 scripts/check_team_records.py
  python3 scripts/check_team_records.py --team lsu
  python3 scripts/check_team_records.py --show-matches
  python3 scripts/check_team_records.py --auto-fix
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / "data" / "baseball.db"
SLUGS_PATH = PROJECT_DIR / "config" / "d1bb_slugs.json"

sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from d1b_team_sync import sync_team  # noqa: E402
from verify_team_schedule import load_reverse_slug_map  # noqa: E402

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BaseballBot/1.0; +record-check)"
}


def load_team_slugs():
    data = json.loads(SLUGS_PATH.read_text())
    return data.get("team_id_to_d1bb_slug", {})


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def db_record(conn, team_id):
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN status='final' AND winner_id=? THEN 1 ELSE 0 END), 0) AS wins,
          COALESCE(SUM(CASE WHEN status='final' AND winner_id IS NOT NULL AND winner_id!=? THEN 1 ELSE 0 END), 0) AS losses
        FROM games
        WHERE home_team_id=? OR away_team_id=?
        """,
        (team_id, team_id, team_id, team_id),
    ).fetchone()
    return int(row["wins"]), int(row["losses"])


def fetch_d1_overall_record(slug):
    url = f"https://d1baseball.com/team/{slug}/"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    html = r.text

    # Try explicit "Overall" + W-L first
    m = re.search(r"Overall\s*</[^>]+>\s*<[^>]+>\s*(\d+)\s*-\s*(\d+)", html, re.IGNORECASE)
    if not m:
        # Fallback: look for "Overall" then nearby "NN-NN"
        anchor = re.search(r"Overall", html, re.IGNORECASE)
        if anchor:
            block = html[anchor.start(): anchor.start() + 1200]
            m = re.search(r"(\d+)\s*-\s*(\d+)", block)
    if not m:
        raise ValueError("Could not parse D1 overall record")

    return int(m.group(1)), int(m.group(2))


def run_check(teams, slugs, delay=0.2, show_matches=False):
    conn = get_db()
    mismatches = []
    parse_errors = []

    print(f"Checking {len(teams)} teams against D1Baseball Overall records...")

    for i, team_id in enumerate(teams, start=1):
        slug = slugs[team_id]
        try:
            db_w, db_l = db_record(conn, team_id)
            d1_w, d1_l = fetch_d1_overall_record(slug)
            ok = (db_w, db_l) == (d1_w, d1_l)
            if ok:
                if show_matches:
                    print(f"✅ {team_id:30s} DB {db_w}-{db_l} | D1 {d1_w}-{d1_l}")
            else:
                mismatches.append((team_id, slug, db_w, db_l, d1_w, d1_l))
                print(f"❌ {team_id:30s} DB {db_w}-{db_l} | D1 {d1_w}-{d1_l}")
        except Exception as e:
            parse_errors.append((team_id, slug, str(e)))
            print(f"⚠️  {team_id:30s} parse error: {e}")

        if i % 25 == 0 or i == len(teams):
            print(f"Progress: {i}/{len(teams)}")

        if delay > 0 and i < len(teams):
            time.sleep(delay)

    conn.close()
    return mismatches, parse_errors


def print_summary(label, teams_count, mismatches, parse_errors):
    print(f"\n=== SUMMARY ({label}) ===")
    print(f"Teams checked: {teams_count}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Parse errors: {len(parse_errors)}")

    if mismatches:
        print("\nMISMATCHES:")
        for t, s, dbw, dbl, d1w, d1l in mismatches:
            print(f"- {t} ({s}): DB {dbw}-{dbl} vs D1 {d1w}-{d1l}")

    if parse_errors:
        print("\nPARSE ERRORS:")
        for t, s, e in parse_errors:
            print(f"- {t} ({s}): {e}")


def auto_fix_mismatches(mismatches, slugs, delay=0.2):
    if not mismatches:
        print("\nNo mismatches to auto-fix.")
        return

    print(f"\nAuto-fix enabled: syncing {len(mismatches)} mismatched teams...")
    reverse = load_reverse_slug_map()
    conn = get_db()

    fixed = 0
    failed = 0
    for i, (team_id, _slug, *_rest) in enumerate(mismatches, start=1):
        slug = slugs[team_id]
        try:
            stats = sync_team(conn, team_id, slug, reverse, dry_run=False, verbose=False)
            conn.commit()
            fixed += 1
            print(f"🔧 {team_id:30s} sync ok (created={stats['created']}, scored={stats['scored']}, unchanged={stats['skipped']})")
        except Exception as e:
            conn.rollback()
            failed += 1
            print(f"💥 {team_id:30s} sync failed: {e}")

        if delay > 0 and i < len(mismatches):
            time.sleep(delay)

    conn.close()
    print(f"Auto-fix done: {fixed} synced, {failed} failed")


def main():
    ap = argparse.ArgumentParser(description="Check team W-L vs D1Baseball overall records")
    ap.add_argument("--team", help="Single team_id to check (e.g. lsu)")
    ap.add_argument("--delay", type=float, default=0.2, help="Delay between requests (default: 0.2s)")
    ap.add_argument("--show-matches", action="store_true", help="Also print teams that match")
    ap.add_argument("--auto-fix", action="store_true", help="Sync only mismatched teams, then re-check")
    args = ap.parse_args()

    slugs = load_team_slugs()
    teams = sorted(slugs.keys())
    if args.team:
        if args.team not in slugs:
            print(f"ERROR: team '{args.team}' not found in d1bb_slugs.json")
            return 2
        teams = [args.team]

    mismatches, parse_errors = run_check(
        teams=teams,
        slugs=slugs,
        delay=args.delay,
        show_matches=args.show_matches,
    )
    print_summary("initial", len(teams), mismatches, parse_errors)

    if args.auto_fix and mismatches:
        auto_fix_mismatches(mismatches, slugs=slugs, delay=args.delay)
        print("\nRe-checking after auto-fix...")
        mismatches, parse_errors = run_check(
            teams=teams,
            slugs=slugs,
            delay=args.delay,
            show_matches=args.show_matches,
        )
        print_summary("after auto-fix", len(teams), mismatches, parse_errors)

    # non-zero if anything failed/mismatched for easy debugger visibility
    return 1 if mismatches or parse_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
