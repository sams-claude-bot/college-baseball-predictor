#!/usr/bin/env python3
"""
D1Baseball Team Schedule Sync (No Browser Required)

Iterates all teams with D1BB slugs, fetches their schedule page via plain HTTP,
and syncs games + scores to the database. Replaces browser-based d1bb_schedule.py
for the nightly schedule sync.

Usage:
    python3 scripts/d1bb_team_sync.py                   # All teams
    python3 scripts/d1bb_team_sync.py --teams-with-games-on 2026-02-21  # Only teams playing that day
    python3 scripts/d1bb_team_sync.py --team mississippi-state  # Single team
    python3 scripts/d1bb_team_sync.py --dry-run          # Preview only
    python3 scripts/d1bb_team_sync.py --delay 0.5        # Custom delay between requests
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from run_utils import ScriptRunner
from verify_team_schedule import fetch_d1bb_schedule, load_d1bb_slugs, load_reverse_slug_map

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_all_teams_with_slugs():
    """Return list of (team_id, d1bb_slug) for all mapped teams."""
    slugs = load_d1bb_slugs()
    return list(slugs.items())


def get_teams_with_games_on(db, date_str):
    """Return set of team_ids that have games on a given date."""
    rows = db.execute(
        "SELECT home_team_id, away_team_id FROM games WHERE date = ?",
        (date_str,)
    ).fetchall()
    teams = set()
    for r in rows:
        teams.add(r['home_team_id'])
        teams.add(r['away_team_id'])
    return teams


def sync_team(db, team_id, d1bb_slug, reverse_slugs, dry_run=False, verbose=False):
    """Sync a single team's schedule from D1Baseball.
    
    Returns dict with counts: created, updated, scored, skipped, errors
    """
    stats = {'created': 0, 'updated': 0, 'scored': 0, 'skipped': 0, 'errors': 0}
    
    for attempt in range(3):
        try:
            d1bb_games = fetch_d1bb_schedule(d1bb_slug)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            if verbose:
                print(f"  ❌ Failed to fetch {d1bb_slug} after 3 attempts: {e}")
            stats['errors'] = 1
            return stats
    
    for g in d1bb_games:
        opp_id = g['opponent_team_id']
        date = g['date']
        
        # Skip future games beyond 14 days (we only care about near-term accuracy)
        # Actually no — Sam wants 100% schedule accuracy, keep all games
        
        if g['is_home']:
            home_id, away_id = team_id, opp_id
        else:
            home_id, away_id = opp_id, team_id
        
        game_num = g.get('game_num', 1)
        dh_suffix = f"_gm{game_num}" if game_num > 1 else ""
        game_id = f"{date}_{away_id}_{home_id}{dh_suffix}"
        
        # Check if game exists
        existing = db.execute(
            "SELECT id, home_score, away_score, status, home_team_id, away_team_id FROM games WHERE id = ?",
            (game_id,)
        ).fetchone()
        
        # Also check with swapped home/away (ESPN sometimes gets this wrong)
        if not existing:
            alt_id = f"{date}_{home_id}_{away_id}{dh_suffix}"
            existing = db.execute(
                "SELECT id, home_score, away_score, status, home_team_id, away_team_id FROM games WHERE id = ?",
                (alt_id,)
            ).fetchone()
        
        # Try fuzzy match: same date, same teams in any order
        # For doubleheaders (game_num > 1), only match games with the same suffix
        if not existing:
            if game_num == 1:
                # For game 1, match any non-DH game on this date with these teams
                existing = db.execute("""
                    SELECT id, home_score, away_score, status, home_team_id, away_team_id FROM games
                    WHERE date = ? AND id NOT LIKE '%_gm%' AND (
                        (home_team_id = ? AND away_team_id = ?) OR
                        (home_team_id = ? AND away_team_id = ?)
                    )
                """, (date, home_id, away_id, away_id, home_id)).fetchone()
            else:
                # For game 2+, only match games with matching _gm suffix
                existing = db.execute("""
                    SELECT id, home_score, away_score, status, home_team_id, away_team_id FROM games
                    WHERE date = ? AND id LIKE ? AND (
                        (home_team_id = ? AND away_team_id = ?) OR
                        (home_team_id = ? AND away_team_id = ?)
                    )
                """, (date, f'%_gm{game_num}', home_id, away_id, away_id, home_id)).fetchone()
        
        if existing:
            # Game exists — check if we need to update scores
            if existing['status'] == 'final' and existing['home_score'] is not None:
                stats['skipped'] += 1
                continue
            
            if g.get('result'):
                r = g['result']
                # Map scores correctly based on home/away
                if g['is_home']:
                    home_score = r['team_score']
                    away_score = r['opp_score']
                else:
                    home_score = r['opp_score']
                    away_score = r['team_score']
                
                # Account for possible home/away swap in existing record
                ex_home = existing['home_team_id']
                ex_away = existing['away_team_id']
                if ex_home == home_id:
                    hs, aws = home_score, away_score
                else:
                    hs, aws = away_score, home_score
                
                winner = ex_home if hs > aws else ex_away
                
                if not dry_run:
                    db.execute("""
                        UPDATE games SET home_score=?, away_score=?, winner_id=?, status='final'
                        WHERE id=?
                    """, (hs, aws, winner, existing['id']))
                
                if verbose:
                    print(f"  📝 Scored: {existing['id']} → {hs}-{aws}")
                stats['scored'] += 1
            else:
                stats['skipped'] += 1
            continue
        
        # New game — insert it
        home_score = away_score = winner_id = None
        status = 'scheduled'
        
        if g.get('result'):
            r = g['result']
            if g['is_home']:
                home_score = r['team_score']
                away_score = r['opp_score']
            else:
                home_score = r['opp_score']
                away_score = r['team_score']
            winner_id = home_id if home_score > away_score else away_id
            status = 'final'
        
        if not dry_run:
            try:
                db.execute("""
                    INSERT INTO games (id, date, home_team_id, away_team_id, home_score, away_score, winner_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, date, home_id, away_id, home_score, away_score, winner_id, status))
                stats['created'] += 1
                if verbose:
                    score_str = f" ({home_score}-{away_score})" if home_score is not None else ""
                    print(f"  ➕ {game_id}{score_str}")
            except sqlite3.IntegrityError:
                # Duplicate — race condition or slight ID mismatch
                stats['skipped'] += 1
        else:
            score_str = f" ({home_score}-{away_score})" if home_score is not None else ""
            print(f"  [DRY] Would create: {game_id}{score_str}")
            stats['created'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Sync schedules from D1Baseball team pages (no browser)')
    parser.add_argument('--team', '-t', help='Single team ID to sync')
    parser.add_argument('--teams-with-games-on', help='Only sync teams with games on this date (YYYY-MM-DD)')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between HTTP requests (default: 0.3s)')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, no DB writes')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    runner = ScriptRunner("d1bb_team_sync")
    db = get_db()
    
    all_slugs = load_d1bb_slugs()  # team_id -> d1bb_slug
    reverse_slugs = load_reverse_slug_map()
    
    # Determine which teams to sync
    if args.team:
        if args.team in all_slugs:
            teams = [(args.team, all_slugs[args.team])]
        else:
            # Maybe they passed a slug
            for tid, slug in all_slugs.items():
                if slug == args.team:
                    teams = [(tid, slug)]
                    break
            else:
                runner.error(f"Team '{args.team}' not found in d1bb_slugs.json")
                runner.finish()
                return
    elif args.teams_with_games_on:
        game_teams = get_teams_with_games_on(db, args.teams_with_games_on)
        teams = [(tid, all_slugs[tid]) for tid in game_teams if tid in all_slugs]
        runner.info(f"Teams with games on {args.teams_with_games_on}: {len(teams)}")
    else:
        teams = list(all_slugs.items())
    
    runner.info(f"Syncing {len(teams)} teams (delay={args.delay}s)")
    
    totals = {'created': 0, 'updated': 0, 'scored': 0, 'skipped': 0, 'errors': 0}
    failed_teams = []
    
    for i, (team_id, d1bb_slug) in enumerate(teams):
        if i % 25 == 0 or i == len(teams) - 1:
            runner.info(f"Progress: {i+1}/{len(teams)}")
        
        try:
            stats = sync_team(db, team_id, d1bb_slug, reverse_slugs,
                             dry_run=args.dry_run, verbose=args.verbose)
        except Exception as e:
            runner.warn(f"Unexpected error on {team_id}: {e}")
            failed_teams.append(team_id)
            continue
        
        for k in totals:
            totals[k] += stats[k]
        
        if stats['errors']:
            failed_teams.append(team_id)
        
        # Commit every 25 teams to avoid huge transactions
        if not args.dry_run and i % 25 == 24:
            db.commit()
        
        if args.delay > 0 and i < len(teams) - 1:
            time.sleep(args.delay)
    
    if not args.dry_run:
        db.commit()
    
    db.close()
    
    runner.info(f"Results: {totals['created']} created, {totals['scored']} scored, "
                f"{totals['skipped']} unchanged, {totals['errors']} errors")
    
    if failed_teams:
        runner.warn(f"Failed teams ({len(failed_teams)}): {', '.join(failed_teams[:20])}")
    
    runner.add_stat("teams_synced", len(teams) - len(failed_teams))
    runner.add_stat("created", totals['created'])
    runner.add_stat("scored", totals['scored'])
    runner.add_stat("errors", totals['errors'])
    runner.finish()


if __name__ == '__main__':
    main()
