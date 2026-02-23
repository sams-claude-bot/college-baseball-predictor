#!/usr/bin/env python3
"""
FanDuel NCAA Baseball Odds — DB loader.

Same pattern as dk_odds_scraper.py. The cron agent opens the FanDuel page
visually and extracts odds into a JSON file. This script loads that JSON into
the betting_lines table with book='fanduel'.

Usage:
    python3 scripts/fd_odds_scraper.py load <json_file>   # Load odds from JSON
    python3 scripts/fd_odds_scraper.py load --stdin        # Read JSON from stdin
    python3 scripts/fd_odds_scraper.py status              # Show today's lines

JSON format (array of games):
[
  {
    "away": "Nebraska Cornhuskers",
    "home": "Louisville Cardinals",
    "away_ml": 136,
    "home_ml": -174,
    "spread": -1.5,
    "away_spread_odds": -136,
    "home_spread_odds": 106,
    "over_under": 12.5,
    "over_odds": -108,
    "under_odds": -118
  }
]

Only "away", "home", and at least one of (away_ml, home_ml) are required.
Spread and O/U fields are optional — omit if not on the page.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_utils import ScriptRunner
from team_resolver import resolve_team

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
BOOK = 'fanduel'


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_team_name(name):
    """Resolve a FanDuel display name to our team_id.
    
    FanDuel uses 'Team Mascot' format (e.g., 'Nebraska Cornhuskers').
    The team_resolver handles stripping mascots via aliases.
    """
    if not name:
        return None
    result = resolve_team(name)
    if result:
        return result
    # Fallback: slugify
    import re
    slug = name.lower().strip().replace(' ', '-').replace("'", '').replace('&', 'and')
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    return slug


def validate_home_away(db, date_str, home_id, away_id, runner=None):
    """
    Validate home/away orientation against the games table.
    
    Returns:
        (home_id, away_id, swapped) - corrected IDs and whether a swap occurred
    """
    warn = runner.warn if runner else print
    
    # Check for exact match first
    row = db.execute("""
        SELECT id, home_team_id, away_team_id 
        FROM games 
        WHERE date = ? AND home_team_id = ? AND away_team_id = ?
    """, (date_str, home_id, away_id)).fetchone()
    
    if row:
        return home_id, away_id, False  # Correct orientation
    
    # Check if teams are swapped
    row = db.execute("""
        SELECT id, home_team_id, away_team_id 
        FROM games 
        WHERE date = ? AND home_team_id = ? AND away_team_id = ?
    """, (date_str, away_id, home_id)).fetchone()
    
    if row:
        warn(f"  ⚠️  HOME/AWAY SWAP DETECTED: {away_id} @ {home_id} -> correcting to {home_id} @ {away_id}")
        return away_id, home_id, True  # Return corrected: actual_home, actual_away
    
    # No matching game found
    return home_id, away_id, False


def load_odds(games_json, date_str=None, runner=None):
    """Load parsed odds into betting_lines table.
    
    Args:
        games_json: list of dicts with away/home/odds fields
        date_str: date string (default: today)
        runner: ScriptRunner for logging
    
    Returns:
        (added, updated, failed) counts
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    log = runner.info if runner else print

    db = get_db()
    added = 0
    updated = 0
    failed = 0
    swapped = 0

    for g in games_json:
        away_name = g.get('away', '').strip()
        home_name = g.get('home', '').strip()

        if not away_name or not home_name:
            if runner:
                runner.warn(f"Skipping game with missing team name: {g}")
            failed += 1
            continue

        away_id = resolve_team_name(away_name)
        home_id = resolve_team_name(home_name)

        if not away_id or not home_id:
            if runner:
                runner.warn(f"Could not resolve: {away_name} -> {away_id}, {home_name} -> {home_id}")
            failed += 1
            continue

        # Validate home/away orientation against games table
        home_id, away_id, was_swapped = validate_home_away(db, date_str, home_id, away_id, runner)
        
        # If swapped, we also need to swap the odds
        if was_swapped:
            swapped += 1
            # Swap ML
            g['away_ml'], g['home_ml'] = g.get('home_ml'), g.get('away_ml')
            # Swap spread (flip sign and swap odds)
            if g.get('spread') is not None:
                g['spread'] = -g['spread']
            if g.get('home_spread') is not None:
                g['home_spread'] = -g['home_spread']
            g['away_spread_odds'], g['home_spread_odds'] = g.get('home_spread_odds'), g.get('away_spread_odds')
            # Note: O/U stays the same

        game_id = f"{date_str}_{away_id}_{home_id}"

        # Parse odds values (handle string or int)
        def parse_ml(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val).replace('−', '-').replace('–', '-').replace('\u2212', '-')
            try:
                return int(s)
            except (ValueError, TypeError):
                return None

        away_ml = parse_ml(g.get('away_ml'))
        home_ml = parse_ml(g.get('home_ml'))
        over_under = g.get('over_under')
        over_odds = parse_ml(g.get('over_odds'))
        under_odds = parse_ml(g.get('under_odds'))
        
        # Spread
        home_spread = g.get('spread') or g.get('home_spread')
        away_spread = -home_spread if home_spread else None
        home_spread_odds = parse_ml(g.get('home_spread_odds'))
        away_spread_odds = parse_ml(g.get('away_spread_odds'))

        try:
            existing = db.execute(
                "SELECT id FROM betting_lines WHERE game_id = ? AND book = ?", (game_id, BOOK)
            ).fetchone()

            if existing:
                db.execute("""
                    UPDATE betting_lines SET
                        home_ml = COALESCE(?, home_ml),
                        away_ml = COALESCE(?, away_ml),
                        home_spread = COALESCE(?, home_spread),
                        home_spread_odds = COALESCE(?, home_spread_odds),
                        away_spread = COALESCE(?, away_spread),
                        away_spread_odds = COALESCE(?, away_spread_odds),
                        over_under = COALESCE(?, over_under),
                        over_odds = COALESCE(?, over_odds),
                        under_odds = COALESCE(?, under_odds),
                        captured_at = CURRENT_TIMESTAMP
                    WHERE game_id = ? AND book = ?
                """, (home_ml, away_ml, home_spread, home_spread_odds,
                      away_spread, away_spread_odds, over_under, over_odds,
                      under_odds, game_id, BOOK))
                updated += 1
            else:
                db.execute("""
                    INSERT INTO betting_lines
                        (game_id, date, home_team_id, away_team_id, book,
                         home_ml, away_ml, home_spread, home_spread_odds,
                         away_spread, away_spread_odds, over_under, over_odds, under_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, date_str, home_id, away_id, BOOK,
                      home_ml, away_ml, home_spread, home_spread_odds,
                      away_spread, away_spread_odds, over_under, over_odds, under_odds))
                added += 1

            log(f"  {'UPD' if existing else 'NEW'}: {away_name} @ {home_name} | ML: {away_ml}/{home_ml}" +
                (f" | O/U: {over_under}" if over_under else "") +
                (f" | Spread: {home_spread}" if home_spread else ""))

        except Exception as e:
            if runner:
                runner.error(f"DB error for {away_name} @ {home_name}: {e}")
            failed += 1

    db.commit()
    db.close()

    return added, updated, failed, swapped


def show_status(runner):
    """Show today's FanDuel betting lines."""
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')

    rows = db.execute("""
        SELECT bl.*, h.name as home_name, a.name as away_name
        FROM betting_lines bl
        LEFT JOIN teams h ON bl.home_team_id = h.id
        LEFT JOIN teams a ON bl.away_team_id = a.id
        WHERE bl.date = ? AND bl.book = ?
        ORDER BY bl.captured_at
    """, (today, BOOK)).fetchall()

    runner.info(f"FanDuel lines for {today}: {len(rows)} games")
    
    ml_count = sum(1 for r in rows if r['home_ml'] is not None)
    ou_count = sum(1 for r in rows if r['over_under'] is not None)
    sp_count = sum(1 for r in rows if r['home_spread'] is not None)

    for r in rows:
        line = f"  {r['away_name'] or r['away_team_id']} @ {r['home_name'] or r['home_team_id']}"
        if r['away_ml']:
            line += f" | ML: {r['away_ml']}/{r['home_ml']}"
        if r['over_under']:
            line += f" | O/U: {r['over_under']}"
        if r['home_spread']:
            line += f" | Spread: {r['home_spread']}"
        runner.info(line)

    runner.add_stat("total_lines", len(rows))
    runner.add_stat("with_moneyline", ml_count)
    runner.add_stat("with_over_under", ou_count)
    runner.add_stat("with_spread", sp_count)

    db.close()


def main():
    parser = argparse.ArgumentParser(description='FanDuel NCAA Baseball Odds Loader')
    parser.add_argument('command', choices=['load', 'status'], help='Command to run')
    parser.add_argument('file', nargs='?', help='JSON file to load (or --stdin)')
    parser.add_argument('--stdin', action='store_true', help='Read JSON from stdin')
    parser.add_argument('--date', help='Override date (YYYY-MM-DD)')
    args = parser.parse_args()

    runner = ScriptRunner("fd_odds")

    if args.command == 'status':
        show_status(runner)
        runner.finish()

    elif args.command == 'load':
        # Read JSON
        if args.stdin:
            raw = sys.stdin.read()
        elif args.file:
            raw = Path(args.file).read_text()
        else:
            runner.error("Must specify a JSON file or --stdin")
            runner.finish()

        try:
            games = json.loads(raw)
        except json.JSONDecodeError as e:
            runner.error(f"Invalid JSON: {e}")
            runner.finish()

        if not isinstance(games, list):
            runner.error(f"Expected JSON array, got {type(games).__name__}")
            runner.finish()

        runner.info(f"Loading {len(games)} games from {'stdin' if args.stdin else args.file}")

        added, updated, failed, swapped = load_odds(games, date_str=args.date, runner=runner)

        runner.add_stat("games_in_json", len(games))
        runner.add_stat("added", added)
        runner.add_stat("updated", updated)
        runner.add_stat("failed", failed)
        if swapped > 0:
            runner.add_stat("home_away_swaps_corrected", swapped)

        if failed > 0 and (added + updated) == 0:
            runner.error("All games failed to load")

        runner.finish()


if __name__ == '__main__':
    main()
