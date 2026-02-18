#!/usr/bin/env python3
"""
D1Baseball Schedule Scraper

Scrapes upcoming games from D1Baseball scoreboard and updates the database.
Handles new games, time changes, and cancellations.

Usage:
    python3 scripts/d1bb_schedule.py --days 3       # Next 3 days
    python3 scripts/d1bb_schedule.py --date 2026-02-20
    python3 scripts/d1bb_schedule.py --today
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
OPENCLAW_USER_DATA = Path.home() / '.openclaw' / 'browser' / 'openclaw' / 'user-data'

sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from team_resolver import resolve_team as db_resolve_team


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_slug_reverse_map():
    """Load D1BB slug -> team_id mapping."""
    if SLUGS_FILE.exists():
        data = json.loads(SLUGS_FILE.read_text())
        return {v: k for k, v in data.get('team_id_to_d1bb_slug', {}).items()}
    return {}


def resolve_team(db, name, slug_map):
    """Resolve team name to our database ID."""
    if not name or not name.strip():
        return None
    
    # Try slug map first
    name_lower = name.lower().strip()
    name_slug = re.sub(r'[^a-z0-9]+', '', name_lower)
    
    if not name_slug:  # Don't match empty slugs
        return None
    
    for slug, team_id in slug_map.items():
        if slug == name_slug or (len(name_slug) >= 3 and name_slug in slug):
            return team_id
    
    # Try database resolver
    resolved = db_resolve_team(name)
    if resolved:
        return resolved
    
    return None


def extract_games_for_date(page, date_str, verbose=False):
    """Extract games from D1Baseball scoreboard for a specific date."""
    
    url = f"https://d1baseball.com/scores/?date={date_str}"
    if verbose:
        print(f"  Loading {url}")
    
    page.goto(url, wait_until='domcontentloaded', timeout=45000)
    time.sleep(2)  # Let content load
    
    # Extract game data from the page (D1BB shows each game twice, once per conference)
    games = page.evaluate("""() => {
        const seen = new Set();  // Track seen matchups to dedupe
        const results = [];
        
        // D1Baseball structure: .d1-score-tile containers with .team-1 (away) and .team-2 (home)
        const gameContainers = document.querySelectorAll('.d1-score-tile');
        
        for (const container of gameContainers) {
            try {
                const game = {};
                
                // Find team-1 (away) and team-2 (home) elements
                const team1El = container.querySelector('.team-1, .team.team-1');
                const team2El = container.querySelector('.team-2, .team.team-2');
                
                if (!team1El || !team2El) continue;
                
                // Get team links
                const link1 = team1El.querySelector('a[href*="/team/"]');
                const link2 = team2El.querySelector('a[href*="/team/"]');
                
                if (link1 && link2) {
                    const match1 = link1.getAttribute('href').match(/\\/team\\/([^\\/]+)/);
                    const match2 = link2.getAttribute('href').match(/\\/team\\/([^\\/]+)/);
                    
                    if (match1 && match2) {
                        game.away_slug = match1[1];
                        // Remove rank prefix from name
                        game.away_name = link1.textContent.trim().replace(/^\\d+\\s*/, '').split('(')[0].trim();
                        game.home_slug = match2[1];
                        game.home_name = link2.textContent.trim().replace(/^\\d+\\s*/, '').split('(')[0].trim();
                    }
                }
                
                // Get scores (D1BB uses .score-runs for the actual run total)
                const score1 = team1El.querySelector('.score-runs');
                const score2 = team2El.querySelector('.score-runs');
                if (score1 && score2) {
                    const awayScore = parseInt(score1.textContent.trim());
                    const homeScore = parseInt(score2.textContent.trim());
                    if (!isNaN(awayScore) && !isNaN(homeScore)) {
                        game.away_score = awayScore;
                        game.home_score = homeScore;
                    }
                }
                
                // Get game time/status from the tile - time is first line
                const tileText = container.innerText.trim();
                const firstLine = tileText.split(String.fromCharCode(10))[0].trim();
                // Check if first line looks like a time (e.g., "2:00 PM", "11:30 AM")
                if (firstLine.includes(':') && (firstLine.toUpperCase().includes('AM') || firstLine.toUpperCase().includes('PM'))) {
                    game.time_text = firstLine;
                } else if (firstLine.toUpperCase() === 'FINAL') {
                    game.status = 'final';
                } else if (firstLine.match(/^(Top|Bottom|Middle)/i)) {
                    game.status = 'in-progress';
                }
                
                // Check container class for game status
                if (container.className.includes('final')) {
                    game.status = 'final';
                } else if (container.className.includes('in-progress')) {
                    game.status = 'in-progress';
                }
                
                if (game.away_slug && game.home_slug) {
                    // Dedupe: D1BB shows each game twice (once per conference)
                    const key = [game.away_slug, game.home_slug].sort().join('_');
                    if (!seen.has(key)) {
                        seen.add(key);
                        results.push(game);
                    }
                }
            } catch (e) {
                // Skip malformed
            }
        }
        
        return results;
    }""")
    
    return games


def parse_time(time_text, date_str):
    """Parse time text to datetime."""
    if not time_text:
        return None
    
    # Clean up the text
    time_text = time_text.strip().upper()
    
    # Try various formats
    for fmt in ['%I:%M %p', '%I:%M%p', '%H:%M']:
        try:
            t = datetime.strptime(time_text, fmt)
            d = datetime.strptime(date_str, '%Y-%m-%d')
            return d.replace(hour=t.hour, minute=t.minute).strftime('%H:%M')
        except:
            continue
    
    return None


def _replace_espn_ghost(db, old_id, new_id, date, home_id, away_id, time, home_score, away_score, status):
    """Replace an ESPN-sourced game with the D1BB version.
    
    Migrates predictions/betting_lines/weather to the new game ID where possible,
    deletes orphaned FK rows, then removes the old game and inserts the new one.
    """
    # Tables with game_id foreign keys
    fk_tables = [
        'model_predictions', 'betting_lines', 'game_weather',
        'tracked_bets', 'tracked_bets_spreads', 'tracked_confident_bets',
        'totals_predictions', 'spread_predictions', 'game_predictions',
        'pitching_matchups', 'game_boxscores', 'game_batting_stats',
        'game_pitching_stats', 'player_boxscore_batting', 'player_boxscore_pitching',
        'statbroadcast_boxscores',
    ]
    migrated = 0
    deleted_fk = 0
    for table in fk_tables:
        try:
            # Try to migrate rows to new ID
            n = db.execute(f"UPDATE {table} SET game_id = ? WHERE game_id = ?", (new_id, old_id)).rowcount
            migrated += n
        except Exception:
            # Table might not exist or have unique constraint conflicts — just delete
            try:
                n = db.execute(f"DELETE FROM {table} WHERE game_id = ?", (old_id,)).rowcount
                deleted_fk += n
            except Exception:
                pass

    # Delete old game
    db.execute("DELETE FROM games WHERE id = ?", (old_id,))

    # Insert new game
    winner = None
    if home_score is not None and away_score is not None:
        if home_score > away_score:
            winner = home_id
        elif away_score > home_score:
            winner = away_id

    db.execute("""
        INSERT INTO games (id, date, time, home_team_id, away_team_id, home_score, away_score, winner_id, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (new_id, date, time, home_id, away_id, home_score, away_score, winner,
          status or ('final' if home_score is not None else 'scheduled')))

    print(f"  Replaced ESPN ghost {old_id} -> {new_id} (migrated {migrated}, deleted {deleted_fk} FK rows)")
    return 'replaced'


def _get_team_aliases(db, team_id):
    """Get all known IDs for a team using the team_aliases table.
    
    Resolves team_id to its canonical form, then finds all aliases that map
    to the same canonical ID. Handles ESPN/D1BB slug mismatches like
    se-louisiana / southeastern-louisiana.
    """
    ids = {team_id}
    
    # Find canonical ID: what does this team_id resolve to?
    canonical = db.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ?", (team_id,)
    ).fetchone()
    canon_id = canonical[0] if canonical else team_id
    ids.add(canon_id)
    
    # Find all aliases that resolve to the same canonical ID
    rows = db.execute(
        "SELECT alias FROM team_aliases WHERE team_id = ?", (canon_id,)
    ).fetchall()
    for row in rows:
        ids.add(row[0])
    
    return ids


def _find_existing_game(db, date, home_id, away_id):
    """Find an existing game by date + teams, handling ID mismatches and home/away swaps.
    
    Returns the existing game row (dict) or None.
    Checks:
      1. Exact match on date + home_team_id + away_team_id
      2. Team alias variants (ESPN vs D1BB slugs)
      3. Swapped home/away (ESPN sometimes gets this wrong)
    """
    home_ids = _get_team_aliases(db, home_id)
    away_ids = _get_team_aliases(db, away_id)
    
    # Build all combinations of home/away aliases + swapped
    combos = []
    for h in home_ids:
        for a in away_ids:
            combos.append((h, a))   # normal
            combos.append((a, h))   # swapped
    
    for h, a in combos:
        row = db.execute(
            "SELECT id, home_score, away_score, home_team_id, away_team_id FROM games WHERE date = ? AND home_team_id = ? AND away_team_id = ?",
            (date, h, a)
        ).fetchone()
        if row:
            return row

    return None


def upsert_game(db, date, home_id, away_id, time=None, home_score=None, away_score=None, status=None):
    """Insert or update a game. Handles ESPN ghost dedup."""
    
    # Generate game ID - match existing format: YYYY-MM-DD_away_home
    game_id = f"{date}_{away_id}_{home_id}"
    
    # Check if exact ID exists
    cursor = db.execute("SELECT id, home_score, away_score FROM games WHERE id = ?", (game_id,))
    existing = cursor.fetchone()
    
    if existing:
        # Update if we have new info
        updates = []
        params = []
        
        if time:
            updates.append("time = ?")
            params.append(time)
        
        # Update scores if we have them and existing doesn't (or game is in-progress/final)
        if home_score is not None and (existing['home_score'] is None or status in ('final', 'in-progress')):
            updates.append("home_score = ?")
            params.append(home_score)
            updates.append("away_score = ?")
            params.append(away_score)
            # Update status for in-progress and final games
            if status in ('final', 'in-progress'):
                updates.append("status = ?")
                params.append(status)
                if status == 'final':
                    if home_score > away_score:
                        updates.append("winner_id = ?")
                        params.append(home_id)
                    elif away_score > home_score:
                        updates.append("winner_id = ?")
                        params.append(away_id)
        
        if updates:
            params.append(game_id)
            db.execute(f"UPDATE games SET {', '.join(updates)} WHERE id = ?", params)
            return 'updated'
        return 'unchanged'
    
    # No exact ID match — check for ESPN ghost with different ID but same matchup
    ghost = _find_existing_game(db, date, home_id, away_id)
    if ghost and ghost['id'] != game_id:
        # Found an ESPN-sourced game with different ID — replace it
        return _replace_espn_ghost(db, ghost['id'], game_id, date, home_id, away_id,
                                   time, home_score, away_score, status)
    
    # Truly new game
    db.execute("""
        INSERT INTO games (id, date, time, home_team_id, away_team_id, home_score, away_score, winner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_id, date, time, home_id, away_id, home_score, away_score,
        home_id if home_score and away_score and home_score > away_score else
        away_id if home_score and away_score and away_score > home_score else None
    ))
    return 'created'


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball schedule')
    parser.add_argument('--date', '-d', help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=3, help='Number of days to scrape (default: 3)')
    parser.add_argument('--today', action='store_true', help='Just today')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    # Determine dates to scrape
    dates = []
    if args.date:
        dates = [args.date]
    elif args.today:
        dates = [datetime.now().strftime('%Y-%m-%d')]
    else:
        base = datetime.now()
        dates = [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(args.days)]
    
    print(f"Scraping schedules for: {', '.join(dates)}")
    
    db = get_db()
    slug_map = load_slug_reverse_map()
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright required")
        sys.exit(1)
    
    stats = {'created': 0, 'updated': 0, 'unchanged': 0, 'unresolved': 0}
    
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(OPENCLAW_USER_DATA),
            headless=True,
        )
        page = browser.new_page()
        
        for date_str in dates:
            print(f"\n=== {date_str} ===")
            
            try:
                games = extract_games_for_date(page, date_str, verbose=args.verbose)
                print(f"  Found {len(games)} games")
                
                for game in games:
                    # Resolve teams
                    home_id = resolve_team(db, game.get('home_name', ''), slug_map)
                    away_id = resolve_team(db, game.get('away_name', ''), slug_map)
                    
                    # Try slug if name didn't work
                    if not home_id:
                        home_id = slug_map.get(game.get('home_slug'))
                    if not away_id:
                        away_id = slug_map.get(game.get('away_slug'))
                    
                    if not home_id or not away_id:
                        if args.verbose:
                            print(f"  SKIP: Could not resolve {game.get('away_name')} @ {game.get('home_name')}")
                        stats['unresolved'] += 1
                        continue
                    
                    # Parse time
                    game_time = parse_time(game.get('time_text'), date_str)
                    
                    if not args.dry_run:
                        result = upsert_game(
                            db, date_str, home_id, away_id,
                            time=game_time,
                            home_score=game.get('home_score'),
                            away_score=game.get('away_score'),
                            status=game.get('status')
                        )
                        stats[result] += 1
                        
                        if args.verbose and result != 'unchanged':
                            print(f"  {result.upper()}: {away_id} @ {home_id}")
                    else:
                        print(f"  {away_id} @ {home_id} ({game_time or 'TBD'})")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(1)
        
        browser.close()
    
    if not args.dry_run:
        db.commit()
    db.close()
    
    print(f"\n{'='*40}")
    print(f"Created: {stats['created']}, Updated: {stats['updated']}, Unchanged: {stats['unchanged']}")
    print(f"Unresolved teams: {stats['unresolved']}")


if __name__ == '__main__':
    main()
