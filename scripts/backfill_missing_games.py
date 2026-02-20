#!/usr/bin/env python3
"""
Backfill missing games by cross-referencing D1Baseball team schedules.

For each team with fewer final games than expected, checks their D1BB schedule
and adds/scores any games we're missing.

Usage:
    python3 scripts/backfill_missing_games.py                # Check teams with < 3 finals
    python3 scripts/backfill_missing_games.py --threshold 4  # Check teams with < 4 finals
    python3 scripts/backfill_missing_games.py --all          # Check ALL teams
    python3 scripts/backfill_missing_games.py --dry-run      # Preview only
"""
import sys
import time
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from scripts.verify_team_schedule import fetch_d1bb_schedule
from scripts.database import get_connection

SLUGS_PATH = PROJECT_DIR / 'config' / 'd1bb_slugs.json'


def get_teams_needing_check(db, threshold=3):
    """Get teams with fewer final games than threshold."""
    cursor = db.execute('''
        SELECT t.id, t.name,
            (SELECT COUNT(*) FROM games g 
             WHERE (g.home_team_id=t.id OR g.away_team_id=t.id) 
             AND g.status='final') as finals,
            (SELECT COUNT(*) FROM games g 
             WHERE (g.home_team_id=t.id OR g.away_team_id=t.id)) as total
        FROM teams t
        GROUP BY t.id
        HAVING finals < ?
        ORDER BY finals ASC, t.name
    ''', (threshold,))
    return cursor.fetchall()


def get_all_teams(db):
    """Get all teams."""
    cursor = db.execute('''
        SELECT t.id, t.name,
            (SELECT COUNT(*) FROM games g 
             WHERE (g.home_team_id=t.id OR g.away_team_id=t.id) 
             AND g.status='final') as finals,
            (SELECT COUNT(*) FROM games g 
             WHERE (g.home_team_id=t.id OR g.away_team_id=t.id)) as total
        FROM teams t
        ORDER BY t.name
    ''')
    return cursor.fetchall()


def load_slugs():
    if SLUGS_PATH.exists():
        data = json.loads(SLUGS_PATH.read_text())
        return data.get('team_id_to_d1bb_slug', {})
    return {}


def backfill_team(db, team_id, team_name, slug, dry_run=False):
    """Check D1BB for this team and backfill missing games/scores."""
    try:
        d1bb_games = fetch_d1bb_schedule(slug)
    except Exception as e:
        print(f"    ERROR fetching {slug}: {e}")
        return 0, 0, 0

    today = datetime.now().strftime('%Y-%m-%d')
    added = 0
    scored = 0
    canceled = 0

    for d1g in d1bb_games:
        date = d1g.get('date')
        if not date or date > today:
            continue

        opp_id = d1g.get('opponent_team_id')
        if not opp_id:
            continue

        is_home = d1g.get('is_home', True)
        result = d1g.get('result')

        # Check if canceled
        if d1g.get('canceled'):
            # Remove from DB if exists
            if is_home:
                game_id = f"{date}_{opp_id}_{team_id}"
            else:
                game_id = f"{date}_{team_id}_{opp_id}"
            existing = db.execute('SELECT id FROM games WHERE id = ?', (game_id,)).fetchone()
            if existing:
                if not dry_run:
                    db.execute('DELETE FROM games WHERE id = ?', (game_id,))
                canceled += 1
            continue

        # Determine home/away
        if is_home:
            home_id, away_id = team_id, opp_id
        else:
            home_id, away_id = opp_id, team_id

        game_id = f"{date}_{away_id}_{home_id}"

        # Check if game exists
        existing = db.execute('SELECT id, status, home_score FROM games WHERE id = ?', (game_id,)).fetchone()

        if not existing:
            # Also try alternate ID format
            alt_id = f"{date}_{team_id}_{opp_id}" if is_home else f"{date}_{opp_id}_{team_id}"
            existing = db.execute('SELECT id, status, home_score FROM games WHERE id = ?', (alt_id,)).fetchone()
            if existing:
                game_id = alt_id

        if not existing:
            # Check by date + teams (ID format might differ)
            existing = db.execute('''
                SELECT id, status, home_score FROM games 
                WHERE date = ? AND (
                    (home_team_id = ? AND away_team_id = ?) OR
                    (home_team_id = ? AND away_team_id = ?)
                )
            ''', (date, home_id, away_id, away_id, home_id)).fetchone()
            if existing:
                game_id = existing[0]

        if not existing:
            # Missing game — add it
            home_score = away_score = winner_id = None
            status = 'scheduled'
            if result:
                if is_home:
                    home_score = result['team_score']
                    away_score = result['opp_score']
                else:
                    home_score = result['opp_score']
                    away_score = result['team_score']
                winner_id = home_id if home_score > away_score else away_id
                status = 'final'

            if not dry_run:
                db.execute('''
                    INSERT OR IGNORE INTO games (id, date, home_team_id, away_team_id, home_score, away_score, winner_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_id, date, home_id, away_id, home_score, away_score, winner_id, status))
            score_str = f" ({home_score}-{away_score})" if home_score is not None else ""
            print(f"    ➕ {game_id}{score_str}")
            added += 1

        elif existing and existing[2] is None and result:
            # Game exists but no score — backfill
            if is_home:
                hs, aws = result['team_score'], result['opp_score']
            else:
                hs, aws = result['opp_score'], result['team_score']

            # Need to know actual home/away from DB
            db_game = db.execute('SELECT home_team_id, away_team_id FROM games WHERE id = ?', (game_id,)).fetchone()
            if db_game:
                winner = db_game[0] if hs > aws else db_game[1]
                if not dry_run:
                    db.execute('''
                        UPDATE games SET home_score=?, away_score=?, winner_id=?, status='final'
                        WHERE id=?
                    ''', (hs, aws, winner, game_id))
                print(f"    📝 {game_id}: {hs}-{aws}")
                scored += 1

    return added, scored, canceled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=3,
                        help='Check teams with fewer than N final games')
    parser.add_argument('--all', action='store_true', help='Check all teams')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--team', type=str, help='Check specific team')
    args = parser.parse_args()

    db = get_connection()
    slugs = load_slugs()

    if args.team:
        team = db.execute('SELECT id, name FROM teams WHERE id = ?', (args.team,)).fetchone()
        if not team:
            print(f"Team not found: {args.team}")
            return
        teams = [(team[0], team[1], 0, 0)]
    elif args.all:
        teams = get_all_teams(db)
    else:
        teams = get_teams_needing_check(db, args.threshold)

    print(f"Checking {len(teams)} teams against D1Baseball...")
    if args.dry_run:
        print("(DRY RUN — no changes)")
    print()

    total_added = 0
    total_scored = 0
    total_canceled = 0
    errors = 0

    for team_id, name, finals, total in teams:
        slug = slugs.get(team_id)
        if not slug:
            errors += 1
            continue

        print(f"  {name} ({team_id}): {finals} finals in DB")
        added, scored, canceled = backfill_team(db, team_id, name, slug, dry_run=args.dry_run)
        total_added += added
        total_scored += scored
        total_canceled += canceled

        if added == 0 and scored == 0 and canceled == 0:
            print(f"    ✅ up to date")

        time.sleep(0.5)  # Rate limit D1BB requests

    if not args.dry_run:
        db.commit()

    db.close()
    print(f"\n{'DRY RUN ' if args.dry_run else ''}SUMMARY:")
    print(f"  Teams checked: {len(teams)}")
    print(f"  Games added: {total_added}")
    print(f"  Scores backfilled: {total_scored}")
    print(f"  Canceled removed: {total_canceled}")
    print(f"  Teams without D1BB slug: {errors}")


if __name__ == '__main__':
    main()
