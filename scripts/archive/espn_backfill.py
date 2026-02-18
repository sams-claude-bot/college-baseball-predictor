#!/usr/bin/env python3
"""
ESPN Backfill — Fill in missing scores using per-team schedule endpoints.

The scoreboard API caps at ~71 games/day. This script catches the rest by
querying individual team schedules for any games still marked 'scheduled'
that should have been played already.

Also handles date corrections — if a game was moved to a different day
(weather, doubleheader reshuffling), it will match by opponent within a
±1 day window and update the date + score.

Usage:
    python3 scripts/espn_backfill.py                    # Backfill all missing
    python3 scripts/espn_backfill.py --date 2026-02-15  # Backfill specific date
    python3 scripts/espn_backfill.py --dry-run           # Preview without writing
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "baseball.db"
ESPN_TEAMS_PATH = BASE_DIR / "data" / "espn_team_ids.json"

TEAM_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams/{espn_id}/schedule"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def api_get(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"    API error: {e}")
        return None


def load_espn_id_map():
    """Returns {our_team_id: espn_id_str}"""
    if ESPN_TEAMS_PATH.exists():
        with open(ESPN_TEAMS_PATH) as f:
            return json.load(f)
    return {}


def extract_score(competitor):
    """Extract integer score from competitor object. Handles both formats."""
    score = competitor.get('score')
    if score is None:
        return None
    if isinstance(score, dict):
        val = score.get('value')
        if val is not None:
            return int(val)
        dv = score.get('displayValue')
        if dv is not None:
            return int(dv)
    if isinstance(score, (int, float)):
        return int(score)
    if isinstance(score, str) and score.isdigit():
        return int(score)
    return None


def espn_date_to_local(date_str):
    """Convert ESPN UTC datetime to CT date string (YYYY-MM-DD).
    ESPN dates like '2026-02-14T23:00Z' are 5PM CT on Feb 14."""
    try:
        # Parse UTC time
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        # Convert to CT (UTC-6, close enough — DST offset doesn't matter for date)
        ct = dt - timedelta(hours=6)
        return ct.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        return None


def fetch_team_schedule(espn_id, season=2026):
    """Fetch a team's full schedule from ESPN."""
    url = f"{TEAM_SCHEDULE_URL.format(espn_id=espn_id)}?season={season}"
    return api_get(url)


def find_matching_game(cur, our_team_id, opponent_id, target_date, date_window=1):
    """Find a game in our DB matching team vs opponent within ±date_window days.
    Returns the game row or None."""
    target = datetime.strptime(target_date, '%Y-%m-%d')
    
    for offset in range(0, date_window + 1):
        for delta in ([0] if offset == 0 else [offset, -offset]):
            check_date = (target + timedelta(days=delta)).strftime('%Y-%m-%d')
            
            # Check both home/away orientations and suffixes (_g2, _g3, etc.)
            for query in [
                "SELECT id, date, status, home_score, away_score, home_team_id, away_team_id FROM games WHERE date=? AND home_team_id=? AND away_team_id=? AND status='scheduled'",
                "SELECT id, date, status, home_score, away_score, home_team_id, away_team_id FROM games WHERE date=? AND home_team_id=? AND away_team_id=? AND status='scheduled'",
            ]:
                # Try: our_team is home, opponent is away
                cur.execute(
                    "SELECT id, date, status, home_score, away_score, home_team_id, away_team_id "
                    "FROM games WHERE date=? AND home_team_id=? AND away_team_id=? AND status='scheduled'",
                    (check_date, our_team_id, opponent_id)
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
                
                # Try: our_team is away, opponent is home
                cur.execute(
                    "SELECT id, date, status, home_score, away_score, home_team_id, away_team_id "
                    "FROM games WHERE date=? AND home_team_id=? AND away_team_id=? AND status='scheduled'",
                    (check_date, opponent_id, our_team_id)
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
    
    return None


def backfill(target_date=None, dry_run=False):
    """Main backfill logic.
    
    Three-pass approach:
    1. For each team with 'scheduled' games, fetch ESPN team schedule
    2. Match final ESPN games to our scheduled games (±1 day)
    3. For games with NO ESPN match at all, mark as 'phantom' (bad source data)
    """
    conn = get_conn()
    cur = conn.cursor()
    espn_map = load_espn_id_map()  # {our_id: espn_id}
    reverse_map = {v: k for k, v in espn_map.items()}  # {espn_id: our_id}
    
    # Find all games still 'scheduled' that should have been played
    if target_date:
        where_clause = "date = ? AND status = 'scheduled'"
        params = (target_date,)
    else:
        today = datetime.now().strftime('%Y-%m-%d')
        where_clause = "date < ? AND status = 'scheduled'"
        params = (today,)
    
    # Get all missing games
    missing_games = cur.execute(
        f"SELECT id, date, home_team_id, away_team_id FROM games WHERE {where_clause} ORDER BY date",
        params
    ).fetchall()
    
    if not missing_games:
        print("No games need backfilling!")
        return {"updated": 0, "date_fixed": 0, "not_found": 0, "phantom": 0}
    
    print(f"Games needing backfill: {len(missing_games)}")
    
    # Collect unique teams to query
    teams_to_query = set()
    for g in missing_games:
        teams_to_query.add(g['home_team_id'])
        teams_to_query.add(g['away_team_id'])
    
    print(f"Teams to query: {len(teams_to_query)}")
    
    # PASS 1: Fetch all team schedules and build a lookup of ESPN results
    # Key: (home_id, away_id, espn_ct_date) -> {score, innings, status}
    espn_results = {}  # (home_id, away_id, date) -> list of {home_score, away_score, innings, status}
    espn_team_opponents = {}  # team_id -> set of opponent_ids they play at all in 2026
    queried_teams = set()
    stats = {"updated": 0, "date_fixed": 0, "not_found": 0, "skipped": 0, 
             "cancelled": 0, "postponed": 0, "phantom": 0}
    
    for team_id in sorted(teams_to_query):
        if team_id in queried_teams:
            continue
        
        espn_id = espn_map.get(team_id)
        if not espn_id:
            print(f"  {team_id}: no ESPN ID mapping, skipping")
            stats["skipped"] += 1
            continue
        
        queried_teams.add(team_id)
        
        data = fetch_team_schedule(espn_id)
        if not data:
            print(f"  {team_id}: failed to fetch schedule")
            stats["skipped"] += 1
            time.sleep(1)
            continue
        
        events = data.get('events', [])
        opponents = set()
        
        for ev in events:
            comp = ev['competitions'][0]
            status_name = comp.get('status', {}).get('type', {}).get('name', '')
            
            home_c = next((c for c in comp['competitors'] if c['homeAway'] == 'home'), None)
            away_c = next((c for c in comp['competitors'] if c['homeAway'] == 'away'), None)
            if not home_c or not away_c:
                continue
            
            h_espn = str(home_c['team']['id'])
            a_espn = str(away_c['team']['id'])
            h_id = reverse_map.get(h_espn)
            a_id = reverse_map.get(a_espn)
            
            if not h_id or not a_id:
                continue
            
            # Track all opponents this team plays
            opp = a_id if h_id == team_id else h_id
            opponents.add(opp)
            
            espn_date = espn_date_to_local(ev.get('date', ''))
            if not espn_date:
                continue
            
            key = (h_id, a_id, espn_date)
            
            if status_name == 'STATUS_FINAL':
                home_score = extract_score(home_c)
                away_score = extract_score(away_c)
                innings = 9
                if home_c.get('linescores'):
                    innings = len(home_c['linescores'])
                elif away_c.get('linescores'):
                    innings = len(away_c['linescores'])
                
                if key not in espn_results:
                    espn_results[key] = []
                espn_results[key].append({
                    'home_score': home_score, 'away_score': away_score,
                    'innings': innings, 'status': 'final',
                    'home_id': h_id, 'away_id': a_id
                })
            elif status_name == 'STATUS_POSTPONED':
                if key not in espn_results:
                    espn_results[key] = []
                espn_results[key].append({
                    'status': 'postponed', 'home_id': h_id, 'away_id': a_id
                })
            elif status_name == 'STATUS_CANCELED':
                if key not in espn_results:
                    espn_results[key] = []
                espn_results[key].append({
                    'status': 'cancelled', 'home_id': h_id, 'away_id': a_id
                })
        
        espn_team_opponents[team_id] = opponents
        time.sleep(0.3)
    
    print(f"\nESPN results cached: {len(espn_results)} matchup-dates")
    
    # PASS 2: Match our scheduled games to ESPN results
    processed_game_ids = set()
    
    for game in missing_games:
        game_id = game['id']
        if game_id in processed_game_ids:
            continue
        
        home = game['home_team_id']
        away = game['away_team_id']
        game_date = game['date']
        
        # Search ESPN results within ±1 day, both home/away orientations
        matched = False
        target = datetime.strptime(game_date, '%Y-%m-%d')
        
        for day_offset in [0, -1, 1]:
            check_date = (target + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            
            # Try both orientations
            for h, a in [(home, away), (away, home)]:
                key = (h, a, check_date)
                results = espn_results.get(key, [])
                
                for result in results:
                    if result['status'] == 'final':
                        date_changed = check_date != game_date
                        winner_id = result['home_id'] if result['home_score'] > result['away_score'] else result['away_id']
                        
                        if dry_run:
                            action = "SCORE" if not date_changed else f"DATE_FIX ({game_date} -> {check_date})"
                            print(f"  [DRY RUN] {action}: {game_id} -> {result['away_id']} {result['away_score']} @ {result['home_id']} {result['home_score']}")
                        else:
                            if date_changed:
                                new_id = game_id.replace(game_date, check_date)
                                # Check for ID conflicts
                                cur.execute("SELECT id FROM games WHERE id=?", (new_id,))
                                if cur.fetchone():
                                    for s in range(2, 6):
                                        candidate = f"{new_id}_g{s}"
                                        cur.execute("SELECT id FROM games WHERE id=?", (candidate,))
                                        if not cur.fetchone():
                                            new_id = candidate
                                            break
                                
                                cur.execute("""
                                    UPDATE games SET 
                                        id=?, date=?, home_team_id=?, away_team_id=?,
                                        home_score=?, away_score=?, winner_id=?,
                                        status='final', innings=?, updated_at=CURRENT_TIMESTAMP
                                    WHERE id=?
                                """, (new_id, check_date, result['home_id'], result['away_id'],
                                      result['home_score'], result['away_score'], winner_id,
                                      result['innings'], game_id))
                                print(f"  DATE_FIX + SCORE: {game_id} -> {new_id} | {result['away_id']} {result['away_score']} @ {result['home_id']} {result['home_score']}")
                                stats["date_fixed"] += 1
                            else:
                                cur.execute("""
                                    UPDATE games SET 
                                        home_team_id=?, away_team_id=?,
                                        home_score=?, away_score=?, winner_id=?,
                                        status='final', innings=?, updated_at=CURRENT_TIMESTAMP
                                    WHERE id=?
                                """, (result['home_id'], result['away_id'],
                                      result['home_score'], result['away_score'], winner_id,
                                      result['innings'], game_id))
                                print(f"  SCORE: {game_id} | {result['away_id']} {result['away_score']} @ {result['home_id']} {result['home_score']}")
                            
                            stats["updated"] += 1
                        
                        # Remove this result so it's not matched again (doubleheader handling)
                        results.remove(result)
                        matched = True
                        processed_game_ids.add(game_id)
                        break
                    
                    elif result['status'] == 'postponed':
                        if dry_run:
                            print(f"  [DRY RUN] POSTPONED: {game_id}")
                        else:
                            cur.execute("UPDATE games SET status='postponed', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                        (game_id,))
                            print(f"  POSTPONED: {game_id}")
                        stats["postponed"] += 1
                        results.remove(result)
                        matched = True
                        processed_game_ids.add(game_id)
                        break
                    
                    elif result['status'] == 'cancelled':
                        if dry_run:
                            print(f"  [DRY RUN] CANCELLED: {game_id}")
                        else:
                            cur.execute("UPDATE games SET status='cancelled', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                        (game_id,))
                            print(f"  CANCELLED: {game_id}")
                        stats["cancelled"] += 1
                        results.remove(result)
                        matched = True
                        processed_game_ids.add(game_id)
                        break
                
                if matched:
                    break
            if matched:
                break
        
        if not matched:
            # PASS 3: Check if these teams even play each other in 2026
            # If neither team has the other on their ESPN schedule, it's phantom data
            home_opps = espn_team_opponents.get(home, set())
            away_opps = espn_team_opponents.get(away, set())
            
            plays_each_other = away in home_opps or home in away_opps
            
            if not plays_each_other:
                if dry_run:
                    print(f"  [DRY RUN] PHANTOM (teams don't play each other): {game_id}")
                else:
                    cur.execute("UPDATE games SET status='phantom', notes='Not on ESPN schedule - bad source data', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                (game_id,))
                    print(f"  PHANTOM: {game_id} (teams don't play each other in 2026)")
                stats["phantom"] += 1
            else:
                # They do play, but not on this date — maybe rescheduled to a future date
                if dry_run:
                    print(f"  [DRY RUN] NOT_FOUND (teams play later): {game_id}")
                else:
                    cur.execute("UPDATE games SET status='phantom', notes='Misdated - teams play on different date per ESPN', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                (game_id,))
                    print(f"  MISDATED: {game_id} (matchup exists but not on this date)")
                stats["not_found"] += 1
    
    if not dry_run:
        conn.commit()
    
    # Count remaining
    cur.execute(f"SELECT COUNT(*) FROM games WHERE {where_clause}", params)
    remaining = cur.fetchone()[0]
    
    conn.close()
    
    print(f"\n{'='*50}")
    print(f"Scores filled: {stats['updated']} ({stats['date_fixed']} with date fixes)")
    print(f"Postponed: {stats['postponed']}, Cancelled: {stats['cancelled']}")
    print(f"Phantom (bad data): {stats['phantom']}, Misdated: {stats['not_found']}")
    print(f"Skipped (no mapping): {stats['skipped']}")
    print(f"Still 'scheduled': {remaining}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='ESPN Backfill — fill missing scores via team schedules')
    parser.add_argument('--date', help='Backfill specific date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    args = parser.parse_args()
    
    backfill(target_date=args.date, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
