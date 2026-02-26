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
from schedule_gateway import ScheduleGateway

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
    """Sync a single team's schedule from D1Baseball via ScheduleGateway.
    
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
    
    gw = ScheduleGateway(db)
    
    for g in d1bb_games:
        opp_id = g['opponent_team_id']
        date = g['date']
        
        if g['is_home']:
            home_id, away_id = team_id, opp_id
        else:
            home_id, away_id = opp_id, team_id
        
        game_num = g.get('game_num', 1)
        
        # Determine scores from D1BB result
        home_score = away_score = None
        status = None
        
        if g.get('result'):
            r = g['result']
            if g['is_home']:
                home_score = r['team_score']
                away_score = r['opp_score']
            else:
                home_score = r['opp_score']
                away_score = r['team_score']
            
            # D1BB can occasionally flip home/away labeling (neutral-site events).
            # Use W/L outcome from the team page as a consistency check.
            outcome = r.get('outcome')
            if outcome in ('W', 'L'):
                team_is_home = (team_id == home_id)
                team_score = home_score if team_is_home else away_score
                opp_score = away_score if team_is_home else home_score
                should_win = (outcome == 'W')
                is_win_now = team_score > opp_score
                if is_win_now != should_win:
                    # Swap scores to match the W/L outcome
                    home_score, away_score = away_score, home_score
            
            status = 'final'
        
        if dry_run:
            canon = gw.canonical_game_id(date, away_id, home_id, game_num)
            score_str = f" ({home_score}-{away_score})" if home_score is not None else ""
            existing = gw.find_existing_game(date, away_id, home_id, game_num)
            action = "update" if existing else "create"
            print(f"  [DRY] Would {action}: {canon}{score_str}")
            stats['created' if not existing else 'scored'] += 1
            continue
        
        action = gw.upsert_game(
            date, away_id, home_id, game_num,
            home_score=home_score, away_score=away_score,
            status=status, source='d1bb_team_sync'
        )
        
        if action == 'created':
            stats['created'] += 1
            if verbose:
                canon = gw.canonical_game_id(date, away_id, home_id, game_num)
                score_str = f" ({home_score}-{away_score})" if home_score is not None else ""
                print(f"  ➕ {canon}{score_str}")
        elif action in ('updated', 'replaced'):
            stats['scored'] += 1
            if verbose:
                canon = gw.canonical_game_id(date, away_id, home_id, game_num)
                print(f"  📝 {action}: {canon} → {home_score}-{away_score}")
        elif action == 'unchanged':
            stats['skipped'] += 1
        elif action == 'rejected':
            stats['errors'] += 1
    
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
