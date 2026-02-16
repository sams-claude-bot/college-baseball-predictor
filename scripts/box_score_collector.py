#!/usr/bin/env python3
"""
ESPN Box Score Collector - Collects game-by-game batting and pitching stats.

Uses ESPN's public API to collect box scores for completed games.

Usage:
    python3 scripts/box_score_collector.py --date 2026-02-15
    python3 scripts/box_score_collector.py --yesterday
    python3 scripts/box_score_collector.py --game-id 401847509
"""

import argparse
import json
import sys
import time
import sqlite3
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('box_score_collector')

# ESPN team name -> our team_id mapping (built dynamically)
ESPN_TEAM_MAP = {}

def normalize_player_name(name):
    """Convert 'Last, First' to 'First Last' format."""
    if not name:
        return name
    if ',' in name:
        parts = name.split(',', 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()
REQUEST_DELAY = 1


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def build_team_map(db):
    """Build mapping from ESPN team names to our team IDs."""
    global ESPN_TEAM_MAP
    teams = db.execute("SELECT id, name, nickname FROM teams").fetchall()
    for t in teams:
        ESPN_TEAM_MAP[t['name'].lower()] = t['id']
        if t['nickname']:
            ESPN_TEAM_MAP[t['nickname'].lower()] = t['id']
        # Add common variations
        ESPN_TEAM_MAP[f"{t['name']} {t['nickname']}".lower()] = t['id']


def fetch_json(url):
    """Fetch JSON from URL."""
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '15', url],
            capture_output=True, text=True, timeout=20
        )
        return json.loads(result.stdout) if result.stdout else None
    except Exception as e:
        log.error(f"Fetch failed: {e}")
        return None


def resolve_team_id(espn_name, espn_id=None):
    """Resolve ESPN team display name to our team_id. Auto-creates if needed."""
    name_lower = espn_name.lower()

    # Direct match in local cache
    if name_lower in ESPN_TEAM_MAP:
        return ESPN_TEAM_MAP[name_lower]

    # Partial match
    for key, team_id in ESPN_TEAM_MAP.items():
        if key in name_lower or name_lower in key:
            return team_id

    # Try just the school name (remove mascot)
    parts = espn_name.split()
    for i in range(len(parts), 0, -1):
        partial = ' '.join(parts[:i]).lower()
        if partial in ESPN_TEAM_MAP:
            return ESPN_TEAM_MAP[partial]

    # Auto-create via espn_sync if we have an ESPN ID
    if espn_id:
        try:
            from scripts.espn_sync import resolve_team, load_espn_id_map, get_conn
            load_espn_id_map()
            conn = get_conn()
            team_id = resolve_team(espn_id, espn_name, "", conn)
            conn.close()
            ESPN_TEAM_MAP[name_lower] = team_id
            return team_id
        except Exception:
            pass

    # Fallback: generate slug from display name
    try:
        from scripts.espn_sync import espn_display_to_slug
        return espn_display_to_slug(espn_name)
    except Exception:
        return None


def get_games_for_date(date_str):
    """Get completed game IDs for a date (YYYYMMDD format)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={date_str}&limit=200"
    data = fetch_json(url)
    if not data:
        return []

    games = []
    for event in data.get('events', []):
        comp = event['competitions'][0]
        status = comp['status']['type']['name']
        if status != 'STATUS_FINAL':
            continue

        # Check if at least one team is P4
        competitors = comp.get('competitors', [])
        teams = []
        for c in competitors:
            team_name = c['team'].get('displayName', '')
            espn_id = c['team'].get('id', '')
            team_id = resolve_team_id(team_name, espn_id=espn_id)
            teams.append({
                'espn_name': team_name,
                'team_id': team_id,
                'home_away': c.get('homeAway', ''),
                'score': c.get('score', '0'),
            })

        # Collect all D1 games (both teams must resolve)
        resolved_teams = [t for t in teams if t['team_id']]
        if len(resolved_teams) >= 2:
            games.append({
                'game_id': event['id'],
                'teams': teams,
                'date': date_str,
            })

    return games


def collect_box_score(game_id, db):
    """Collect batting and pitching box score for a game."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/summary?event={game_id}"
    data = fetch_json(url)
    if not data or 'boxscore' not in data:
        log.error(f"  No box score data for game {game_id}")
        return False

    bs = data['boxscore']
    batting_count = 0
    pitching_count = 0

    for team_data in bs.get('players', []):
        team_info = team_data.get('team', {})
        espn_name = team_info.get('displayName', '')
        team_id = resolve_team_id(espn_name)

        if not team_id:
            # Try auto-create
            espn_id = team_info.get('id', '')
            team_id = resolve_team_id(espn_name, espn_id=espn_id)
            if not team_id:
                log.warning(f"  Could not resolve team: {espn_name}")
                continue

        for stat_group in team_data.get('statistics', []):
            stat_type = stat_group.get('type', '')
            labels = stat_group.get('labels', [])

            for athlete_data in stat_group.get('athletes', []):
                athlete = athlete_data.get('athlete', {})
                player_name = normalize_player_name(athlete.get('displayName', ''))
                stats = athlete_data.get('stats', [])

                if not player_name or not stats:
                    continue

                # Create stat dict from labels and values
                stat_dict = dict(zip(labels, stats))

                if stat_type == 'batting':
                    insert_game_batting(db, game_id, team_id, player_name, stat_dict)
                    batting_count += 1
                elif stat_type == 'pitching':
                    insert_game_pitching(db, game_id, team_id, player_name, stat_dict)
                    pitching_count += 1

    db.commit()
    return batting_count > 0 or pitching_count > 0


def safe_int(val, default=0):
    if not val or val == '-':
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    if not val or val == '-':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def insert_game_batting(db, game_id, team_id, player_name, stats):
    """Insert a batting line into game_batting_stats."""
    db.execute("""
        INSERT OR REPLACE INTO game_batting_stats (
            game_id, team_id, player_name, position,
            at_bats, runs, hits, rbi, home_runs, walks, strikeouts,
            stolen_bases, batting_avg, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        str(game_id), team_id, player_name, None,
        safe_int(stats.get('AB')),
        safe_int(stats.get('R')),
        safe_int(stats.get('H')),
        safe_int(stats.get('RBI')),
        safe_int(stats.get('HR')),
        safe_int(stats.get('BB')),
        safe_int(stats.get('K')),
        0,  # SB not always in ESPN labels
        safe_float(stats.get('AVG')),
    ))


def insert_game_pitching(db, game_id, team_id, player_name, stats):
    """Insert a pitching line into game_pitching_stats."""
    # Parse IP (format: "6.0" or "5.2")
    ip = safe_float(stats.get('IP', '0'))

    # Parse W/L/S from decision column if present
    win = 1 if stats.get('DEC', '') == 'W' else 0
    loss = 1 if stats.get('DEC', '') == 'L' else 0
    save = 1 if stats.get('DEC', '') == 'S' else 0

    db.execute("""
        INSERT OR REPLACE INTO game_pitching_stats (
            game_id, team_id, player_name,
            innings_pitched, hits_allowed, runs_allowed, earned_runs,
            walks, strikeouts, home_runs_allowed, pitches, strikes,
            era, win, loss, save, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        str(game_id), team_id, player_name,
        ip,
        safe_int(stats.get('H')),
        safe_int(stats.get('R')),
        safe_int(stats.get('ER')),
        safe_int(stats.get('BB')),
        safe_int(stats.get('K')),
        safe_int(stats.get('HR')),
        safe_int(stats.get('#P', stats.get('P', '0'))),
        0,  # strikes not always available
        safe_float(stats.get('ERA')),
        win, loss, save,
    ))


def main():
    parser = argparse.ArgumentParser(description='Collect ESPN box scores')
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')
    parser.add_argument('--yesterday', action='store_true')
    parser.add_argument('--game-id', help='Single ESPN game ID')
    args = parser.parse_args()

    if not any([args.date, args.yesterday, args.game_id]):
        parser.print_help()
        sys.exit(1)

    db = get_db()
    build_team_map(db)

    if args.game_id:
        log.info(f"Collecting box score for game {args.game_id}")
        success = collect_box_score(args.game_id, db)
        log.info(f"{'Success' if success else 'Failed'}")
        db.close()
        return

    if args.yesterday:
        date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        date = args.date

    date_fmt = date.replace('-', '')
    log.info(f"Collecting box scores for {date}")

    games = get_games_for_date(date_fmt)
    log.info(f"Found {len(games)} completed P4 games")

    success = 0
    failed = 0
    for i, game in enumerate(games, 1):
        teams_str = ' vs '.join(t['espn_name'] for t in game['teams'])
        log.info(f"[{i}/{len(games)}] Game {game['game_id']}: {teams_str}")

        if collect_box_score(game['game_id'], db):
            success += 1
        else:
            failed += 1

        if i < len(games):
            time.sleep(REQUEST_DELAY)

    log.info(f"\nResults: {success} collected, {failed} failed out of {len(games)} games")
    db.close()


if __name__ == '__main__':
    main()
