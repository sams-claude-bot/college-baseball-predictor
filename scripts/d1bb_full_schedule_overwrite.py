#!/usr/bin/env python3
"""
D1Baseball Full Schedule Overwrite

Scrapes complete season schedules from D1Baseball team schedule pages
and overwrites existing ESPN-seeded schedule data for all fully tracked teams.

Usage:
    python3 scripts/d1bb_full_schedule_overwrite.py [--dry-run] [--verbose] [--limit N]
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
SITES_FILE = PROJECT_DIR / 'config' / 'team_sites.json'

sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from team_resolver import resolve_team as db_resolve_team

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_config():
    """Load slug mappings and determine fully tracked teams."""
    slugs_data = json.loads(SLUGS_FILE.read_text())
    team_to_slug = slugs_data.get('team_id_to_d1bb_slug', {})
    slug_to_team = {v: k for k, v in team_to_slug.items()}
    
    # Fully tracked = teams in team_sites.json that also have d1bb slugs
    sites = json.loads(SITES_FILE.read_text())
    fully_tracked = {tid: team_to_slug[tid] for tid in sites if tid in team_to_slug}
    
    return fully_tracked, team_to_slug, slug_to_team


def resolve_opponent(name, slug, slug_to_team):
    """Resolve opponent to our team ID."""
    # Direct slug lookup
    if slug in slug_to_team:
        return slug_to_team[slug]
    
    # Try db resolver
    resolved = db_resolve_team(name)
    if resolved:
        return resolved
    
    return None


def parse_date_from_link(href, year=2026):
    """Parse date from D1BB score link like /scores/?date=20260213."""
    m = re.search(r'date=(\d{4})(\d{2})(\d{2})', href)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def scrape_team_schedule(team_id, d1bb_slug, slug_to_team, verbose=False):
    """Scrape full schedule from a team's D1Baseball schedule page."""
    url = f"https://d1baseball.com/team/{d1bb_slug}/schedule/"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        if verbose:
            print(f"  ERROR fetching {url}: {e}")
        return [], [f"HTTP error for {team_id}: {e}"]
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    games = []
    unresolved = []
    
    # Find schedule rows - D1BB uses table rows or schedule entries
    # Look for rows with date links and opponent links
    
    # Strategy: find all links to /scores/?date= and then parse surrounding context
    # The page structure has rows with: date | vs/@ | opponent | score | venue
    
    # Try finding schedule table rows
    rows = soup.select('tr, .schedule-row, .game-row')
    if not rows:
        # Fallback: parse the whole page text structure
        rows = soup.select('li, .game-item, .schedule-item')
    
    # D1BB schedule pages use <tr> rows with: date | vs/@ | opponent | result | venue
    schedule_rows = soup.select('tr[data-schedule-id]')
    if not schedule_rows:
        # Fallback: find rows containing date links
        schedule_rows = []
        for dl in soup.find_all('a', href=re.compile(r'/scores/\?date=')):
            tr = dl.find_parent('tr')
            if tr:
                schedule_rows.append(tr)
    
    for row in schedule_rows:
        # Extract date
        date_link = row.find('a', href=re.compile(r'/scores/\?date='))
        if not date_link:
            continue
        date_str = parse_date_from_link(date_link['href'])
        if not date_str:
            continue
        
        # Extract opponent
        opp_link = row.find('a', href=re.compile(r'/team/[^/]+'))
        if not opp_link:
            continue
        opp_slug_match = re.search(r'/team/([^/]+)', opp_link['href'])
        if not opp_slug_match:
            continue
        opp_slug = opp_slug_match.group(1)
        opp_name = opp_link.get_text(strip=True)
        opp_name = re.sub(r'^\d+\s*', '', opp_name).split('(')[0].strip()
        
        opp_id = resolve_opponent(opp_name, opp_slug, slug_to_team)
        if not opp_id:
            unresolved.append(f"{opp_name} ({opp_slug})")
            continue
        
        # Determine home/away: look for a <td> containing just "vs" or "@"
        tds = row.find_all('td')
        is_away = False
        for td in tds:
            td_text = td.get_text(strip=True)
            if td_text == '@':
                is_away = True
                break
            elif td_text == 'vs':
                is_away = False
                break
        
        if is_away:
            home_id = opp_id
            away_id = team_id
        else:
            home_id = team_id
            away_id = opp_id
        
        # Extract score from result cell
        home_score = None
        away_score = None
        status = 'scheduled'
        
        result_td = row.find('td', class_=re.compile(r'result'))
        if result_td:
            result_text = result_td.get_text(strip=True)
            # Patterns: "W 15 - 5", "L 3 - 5"
            score_match = re.search(r'([WL])\s*(\d+)\s*[-–]\s*(\d+)', result_text)
            if score_match:
                wl = score_match.group(1)
                s1 = int(score_match.group(2))
                s2 = int(score_match.group(3))
                # W means team_id won, L means team_id lost
                # s1 is always the winning score, s2 the losing
                if wl == 'W':
                    team_score, opp_score = s1, s2
                else:
                    team_score, opp_score = s2, s1
                
                if home_id == team_id:
                    home_score = team_score
                    away_score = opp_score
                else:
                    away_score = team_score
                    home_score = opp_score
                status = 'final'
        
        game = {
            'date': date_str,
            'home_id': home_id,
            'away_id': away_id,
            'home_score': home_score,
            'away_score': away_score,
            'status': status,
        }
        games.append(game)
    
    return games, unresolved


def get_team_aliases(db, team_id):
    """Get all known IDs for a team."""
    ids = {team_id}
    try:
        canonical = db.execute(
            "SELECT team_id FROM team_aliases WHERE alias = ?", (team_id,)
        ).fetchone()
        canon_id = canonical[0] if canonical else team_id
        ids.add(canon_id)
        rows = db.execute(
            "SELECT alias FROM team_aliases WHERE team_id = ?", (canon_id,)
        ).fetchall()
        for row in rows:
            ids.add(row[0])
    except Exception:
        pass
    return ids


def find_existing_game(db, date, home_id, away_id):
    """Find existing game handling aliases and home/away swaps."""
    home_ids = get_team_aliases(db, home_id)
    away_ids = get_team_aliases(db, away_id)
    
    combos = []
    for h in home_ids:
        for a in away_ids:
            combos.append((h, a))
            combos.append((a, h))  # swapped
    
    for h, a in combos:
        row = db.execute(
            "SELECT id, home_score, away_score, status, home_team_id, away_team_id FROM games WHERE date = ? AND home_team_id = ? AND away_team_id = ?",
            (date, h, a)
        ).fetchone()
        if row:
            return row
    return None


def upsert_game(db, game, verbose=False):
    """Insert or update a game from D1BB schedule data."""
    date = game['date']
    home_id = game['home_id']
    away_id = game['away_id']
    home_score = game['home_score']
    away_score = game['away_score']
    status = game['status']
    
    game_id = f"{date}_{away_id}_{home_id}"
    
    # Check exact ID
    existing = db.execute("SELECT id, home_score, away_score, status FROM games WHERE id = ?", (game_id,)).fetchone()
    
    if existing:
        # Update schedule fields; preserve historical scores if D1 doesn't have them
        updates = []
        params = []
        
        if status == 'final' and home_score is not None:
            # Only overwrite scores if existing doesn't have them or they differ
            if existing['home_score'] is None or existing['status'] != 'final':
                updates.extend(["home_score = ?", "away_score = ?", "status = ?"])
                params.extend([home_score, away_score, 'final'])
                winner = home_id if home_score > away_score else away_id if away_score > home_score else None
                if winner:
                    updates.append("winner_id = ?")
                    params.append(winner)
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        
        if len(updates) > 1:  # More than just updated_at
            params.append(game_id)
            db.execute(f"UPDATE games SET {', '.join(updates)} WHERE id = ?", params)
            return 'updated'
        return 'unchanged'
    
    # Check for ESPN ghost with different ID
    ghost = find_existing_game(db, date, home_id, away_id)
    if ghost:
        old_id = ghost['id']
        # Preserve existing scores if D1 doesn't provide them
        if home_score is None and ghost['home_score'] is not None:
            home_score = ghost['home_score']
            away_score = ghost['away_score']
            status = ghost['status'] or status
        
        # Migrate FK references
        fk_tables = [
            'model_predictions', 'betting_lines', 'game_weather',
            'tracked_bets', 'tracked_bets_spreads', 'tracked_confident_bets',
            'totals_predictions', 'spread_predictions', 'game_predictions',
            'pitching_matchups', 'game_boxscores', 'game_batting_stats',
            'game_pitching_stats', 'player_boxscore_batting', 'player_boxscore_pitching',
            'statbroadcast_boxscores',
        ]
        for table in fk_tables:
            try:
                db.execute(f"UPDATE {table} SET game_id = ? WHERE game_id = ?", (game_id, old_id))
            except Exception:
                try:
                    db.execute(f"DELETE FROM {table} WHERE game_id = ?", (old_id,))
                except Exception:
                    pass
        
        db.execute("DELETE FROM games WHERE id = ?", (old_id,))
        
        winner = None
        if home_score is not None and away_score is not None:
            winner = home_id if home_score > away_score else away_id if away_score > home_score else None
        
        db.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, home_score, away_score, winner_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, date, home_id, away_id, home_score, away_score, winner, status))
        
        if verbose:
            print(f"  MERGED: {old_id} -> {game_id}")
        return 'merged'
    
    # Truly new game
    winner = None
    if home_score is not None and away_score is not None:
        winner = home_id if home_score > away_score else away_id if away_score > home_score else None
    
    db.execute("""
        INSERT INTO games (id, date, home_team_id, away_team_id, home_score, away_score, winner_id, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (game_id, date, home_id, away_id, home_score, away_score, winner, status))
    
    return 'inserted'


def main():
    parser = argparse.ArgumentParser(description='D1BB Full Schedule Overwrite')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--limit', type=int, help='Limit number of teams to process')
    args = parser.parse_args()
    
    print("=" * 60)
    print("D1Baseball Full Schedule Overwrite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    fully_tracked, team_to_slug, slug_to_team = load_config()
    print(f"\nFully tracked teams with D1BB slugs: {len(fully_tracked)}")
    
    if args.limit:
        items = list(fully_tracked.items())[:args.limit]
        fully_tracked = dict(items)
        print(f"  (limited to {args.limit})")
    
    db = get_db()
    
    # Pre-scrape game count
    pre_count = db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    
    stats = {
        'teams_targeted': len(fully_tracked),
        'teams_ok': 0,
        'teams_err': 0,
        'inserted': 0,
        'updated': 0,
        'merged': 0,
        'unchanged': 0,
        'unresolved_teams': [],
        'errors': [],
    }
    
    # Track which game IDs we touch
    touched_ids = set()
    
    for i, (team_id, d1bb_slug) in enumerate(sorted(fully_tracked.items())):
        print(f"\n[{i+1}/{len(fully_tracked)}] {team_id} ({d1bb_slug})")
        
        try:
            games, unresolved = scrape_team_schedule(team_id, d1bb_slug, slug_to_team, verbose=args.verbose)
            print(f"  Found {len(games)} games, {len(unresolved)} unresolved opponents")
            
            if unresolved:
                for u in unresolved:
                    if u not in [x[1] for x in stats['unresolved_teams']]:
                        stats['unresolved_teams'].append((team_id, u))
            
            for game in games:
                # Dedup: skip if we've already processed this exact matchup from the other team's page
                dedup_key = f"{game['date']}_{min(game['home_id'], game['away_id'])}_{max(game['home_id'], game['away_id'])}"
                if dedup_key in touched_ids:
                    continue
                touched_ids.add(dedup_key)
                
                if not args.dry_run:
                    result = upsert_game(db, game, verbose=args.verbose)
                    stats[result] = stats.get(result, 0) + 1
                    if args.verbose and result not in ('unchanged',):
                        print(f"  {result.upper()}: {game['away_id']} @ {game['home_id']} ({game['date']})")
                else:
                    print(f"  DRY: {game['away_id']} @ {game['home_id']} ({game['date']})")
            
            stats['teams_ok'] += 1
            
        except Exception as e:
            stats['teams_err'] += 1
            stats['errors'].append(f"{team_id}: {e}")
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Rate limit
        time.sleep(0.5)
    
    if not args.dry_run:
        db.commit()
    
    # Post-scrape counts
    post_count = db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    next30 = db.execute("SELECT COUNT(*) FROM games WHERE date >= '2026-02-18' AND date <= '2026-03-20'").fetchone()[0]
    
    # Unique unresolved
    unique_unresolved = list(set(u for _, u in stats['unresolved_teams']))
    
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Teams targeted: {stats['teams_targeted']}")
    print(f"Teams processed successfully: {stats['teams_ok']}")
    print(f"Teams with errors: {stats['teams_err']}")
    print(f"Games inserted: {stats['inserted']}")
    print(f"Games updated/overwritten: {stats['updated']}")
    print(f"Duplicates merged: {stats['merged']}")
    print(f"Unchanged: {stats.get('unchanged', 0)}")
    print(f"Unresolved opponent mappings: {len(unique_unresolved)}")
    if unique_unresolved:
        for u in sorted(unique_unresolved)[:30]:
            print(f"  - {u}")
        if len(unique_unresolved) > 30:
            print(f"  ... and {len(unique_unresolved) - 30} more")
    print(f"\nVerification:")
    print(f"  Games before run: {pre_count}")
    print(f"  Games after run: {post_count}")
    print(f"  Net new: {post_count - pre_count}")
    print(f"  Games in next 30 days: {next30}")
    print(f"  Game IDs touched this run: {len(touched_ids)}")
    if stats['errors']:
        print(f"\nErrors:")
        for e in stats['errors'][:10]:
            print(f"  - {e}")
    
    db.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
