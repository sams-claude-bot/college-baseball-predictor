#!/usr/bin/env python3
"""
NCAA Team Stats — DB loader.

This script does NOT scrape NCAA itself (Akamai blocks headless browsers).
The OpenClaw cron job opens stats.ncaa.org visually and extracts stats into
a JSON file. This script loads that JSON into the ncaa_team_stats table.

Usage:
    python3 scripts/ncaa_stats_scraper.py load <json_file>   # Load stats from JSON
    python3 scripts/ncaa_stats_scraper.py load --stdin       # Read JSON from stdin
    python3 scripts/ncaa_stats_scraper.py status             # Show stats summary
    python3 scripts/ncaa_stats_scraper.py status --team tennessee  # Stats for a team
    python3 scripts/ncaa_stats_scraper.py status --season 2025     # Stats for a season

JSON format (array of stat categories):
[
  {
    "stat_name": "era",
    "season": 2026,
    "teams": [
      {"rank": 1, "team": "Clemson (ACC)", "games": 4, "record": "4-0", "value": 0.55},
      {"rank": 2, "team": "UC Irvine (Big West)", "games": 4, "record": "4-0", "value": 1.25}
    ]
  }
]

NCAA stat codes for reference:
  210 = batting_avg, 211 = era, 212 = fielding_pct, 213 = scoring,
  319 = win_pct, 321/327 = slugging, 323 = hr_per_game, 325 = triples_per_game,
  326 = sb_per_game, 328 = dp_per_game, 425 = k_per_9, 496 = team_bb,
  504/589 = obp, 506 = hits_per_9, 509 = bb_per_9, 591 = k_bb_ratio,
  593 = hit_batters, 597 = whip
"""

import argparse
import json
import re
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

# Stat name normalization map
STAT_ALIASES = {
    # Pitching
    'team_era': 'era',
    'team era': 'era',
    'earned run average': 'era',
    'team_whip': 'whip',
    'team whip': 'whip',
    'strikeouts per 9': 'k_per_9',
    'k/9': 'k_per_9',
    'so/9': 'k_per_9',
    'walks per 9': 'bb_per_9',
    'bb/9': 'bb_per_9',
    'hits allowed per 9': 'hits_per_9',
    'h/9': 'hits_per_9',
    'k/bb': 'k_bb_ratio',
    'strikeout to walk': 'k_bb_ratio',
    
    # Batting
    'team_batting_avg': 'batting_avg',
    'team batting avg': 'batting_avg',
    'batting average': 'batting_avg',
    'avg': 'batting_avg',
    'team_obp': 'obp',
    'team obp': 'obp',
    'on base percentage': 'obp',
    'on-base percentage': 'obp',
    'team_slugging': 'slugging',
    'team slugging': 'slugging',
    'slg': 'slugging',
    'slugging percentage': 'slugging',
    'home runs per game': 'hr_per_game',
    'hr/g': 'hr_per_game',
    'runs per game': 'scoring',
    'scoring offense': 'scoring',
    
    # Fielding
    'team fielding pct': 'fielding_pct',
    'fielding percentage': 'fielding_pct',
    'fpct': 'fielding_pct',
    'double plays per game': 'dp_per_game',
    'dp/g': 'dp_per_game',
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_stat_name(name: str) -> str:
    """Normalize stat name to canonical form."""
    if not name:
        return name
    key = name.lower().strip()
    return STAT_ALIASES.get(key, key.replace(' ', '_').replace('-', '_'))


def slugify_team(name: str) -> str:
    """Convert a team name to a slug ID as fallback."""
    slug = name.lower().strip()
    # Remove conference suffix like "(ACC)" or "(Big West)"
    slug = re.sub(r'\s*\([^)]+\)\s*$', '', slug)
    slug = slug.replace("'", '').replace('&', 'and').replace(' ', '-')
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


def resolve_team_name(name: str) -> tuple:
    """
    Resolve an NCAA team name to our team_id.
    
    Returns (team_id, resolved_bool).
    If not found in resolver, returns slugified fallback.
    """
    if not name:
        return None, False
    
    # Try resolver first
    result = resolve_team(name)
    if result:
        return result, True
    
    # Try without conference suffix
    clean = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
    result = resolve_team(clean)
    if result:
        return result, True
    
    # Fallback to slug
    return slugify_team(name), False


def parse_record(record: str) -> tuple:
    """Parse W-L record string into (wins, losses)."""
    if not record:
        return None, None
    
    # Handle various formats: "4-0", "4 - 0", "4-0-1" (with ties)
    match = re.match(r'(\d+)\s*[-–]\s*(\d+)', str(record))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def load_stats(stats_json: list, season_override: int = None, runner: ScriptRunner = None):
    """
    Load parsed NCAA stats into ncaa_team_stats table.
    
    Args:
        stats_json: list of stat category dicts with 'stat_name', 'season', 'teams'
        season_override: override season from JSON
        runner: ScriptRunner for logging
    
    Returns:
        (added, updated, failed, unresolved_teams) counts
    """
    log = runner.info if runner else print
    warn = runner.warn if runner else print
    
    db = get_db()
    added = 0
    updated = 0
    failed = 0
    unresolved_teams = set()
    
    for stat_entry in stats_json:
        stat_name = normalize_stat_name(stat_entry.get('stat_name', ''))
        season = season_override or stat_entry.get('season') or datetime.now().year
        teams = stat_entry.get('teams', [])
        
        if not stat_name:
            if runner:
                runner.warn(f"Skipping entry with no stat_name: {stat_entry}")
            failed += 1
            continue
        
        log(f"Loading {stat_name} ({len(teams)} teams) for {season}")
        
        for team_data in teams:
            ncaa_name = team_data.get('team', '').strip()
            if not ncaa_name:
                failed += 1
                continue
            
            team_id, resolved = resolve_team_name(ncaa_name)
            if not resolved:
                unresolved_teams.add(ncaa_name)
            
            ranking = team_data.get('rank')
            games = team_data.get('games')
            value = team_data.get('value')
            record = team_data.get('record', '')
            wins, losses = parse_record(record)
            
            # Parse value if string
            if isinstance(value, str):
                try:
                    value = float(value.replace(',', ''))
                except (ValueError, TypeError):
                    value = None
            
            try:
                # UPSERT: INSERT OR REPLACE
                existing = db.execute(
                    "SELECT id FROM ncaa_team_stats WHERE team_id = ? AND season = ? AND stat_name = ?",
                    (team_id, season, stat_name)
                ).fetchone()
                
                if existing:
                    db.execute("""
                        UPDATE ncaa_team_stats SET
                            ncaa_team_name = ?,
                            stat_value = ?,
                            games_played = ?,
                            wins = ?,
                            losses = ?,
                            ranking = ?,
                            scraped_at = CURRENT_TIMESTAMP
                        WHERE team_id = ? AND season = ? AND stat_name = ?
                    """, (ncaa_name, value, games, wins, losses, ranking,
                          team_id, season, stat_name))
                    updated += 1
                else:
                    db.execute("""
                        INSERT INTO ncaa_team_stats
                            (team_id, ncaa_team_name, season, stat_name, stat_value,
                             games_played, wins, losses, ranking)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (team_id, ncaa_name, season, stat_name, value,
                          games, wins, losses, ranking))
                    added += 1
                    
            except Exception as e:
                if runner:
                    runner.error(f"DB error for {ncaa_name}: {e}")
                failed += 1
    
    db.commit()
    db.close()
    
    # Log unresolved teams
    if unresolved_teams and runner:
        runner.warn(f"Unresolved teams ({len(unresolved_teams)}): stored with slugified IDs")
        for name in sorted(unresolved_teams)[:10]:  # Show first 10
            slug = slugify_team(name)
            runner.info(f"  {name} → {slug} (unresolved)")
        if len(unresolved_teams) > 10:
            runner.info(f"  ... and {len(unresolved_teams) - 10} more")
    
    return added, updated, failed, len(unresolved_teams)


def show_status(runner: ScriptRunner, team_filter: str = None, season_filter: int = None):
    """Show current stats summary."""
    db = get_db()
    
    # Determine season
    if season_filter is None:
        season_filter = datetime.now().year
    
    if team_filter:
        # Show stats for a specific team
        team_id, _ = resolve_team_name(team_filter)
        
        # Also try direct match
        rows = db.execute("""
            SELECT * FROM ncaa_team_stats 
            WHERE (team_id = ? OR team_id LIKE ? OR ncaa_team_name LIKE ?)
            AND season = ?
            ORDER BY stat_name
        """, (team_id, f"%{team_filter}%", f"%{team_filter}%", season_filter)).fetchall()
        
        if not rows:
            runner.info(f"No stats found for '{team_filter}' in {season_filter}")
            runner.finish(exit_on_error=False)
            return
        
        runner.info(f"Stats for {rows[0]['ncaa_team_name'] or team_id} ({season_filter}):")
        for r in rows:
            rank_str = f"#{r['ranking']}" if r['ranking'] else ""
            runner.info(f"  {r['stat_name']}: {r['stat_value']} {rank_str}")
        
        runner.add_stat("team", team_id)
        runner.add_stat("stats_found", len(rows))
        
    else:
        # Show overall summary
        summary = db.execute("""
            SELECT 
                stat_name,
                COUNT(*) as team_count,
                MIN(scraped_at) as oldest,
                MAX(scraped_at) as newest
            FROM ncaa_team_stats 
            WHERE season = ?
            GROUP BY stat_name
            ORDER BY stat_name
        """, (season_filter,)).fetchall()
        
        total_teams = db.execute(
            "SELECT COUNT(DISTINCT team_id) FROM ncaa_team_stats WHERE season = ?",
            (season_filter,)
        ).fetchone()[0]
        
        runner.info(f"NCAA Team Stats for {season_filter}:")
        runner.info(f"  Total unique teams: {total_teams}")
        runner.info(f"  Stats loaded: {len(summary)}")
        
        if summary:
            runner.info("")
            runner.info("Stats by category:")
            for s in summary:
                runner.info(f"  {s['stat_name']}: {s['team_count']} teams")
        
        runner.add_stat("season", season_filter)
        runner.add_stat("total_teams", total_teams)
        runner.add_stat("stat_categories", len(summary))
    
    db.close()


def main():
    parser = argparse.ArgumentParser(description='NCAA Team Stats Loader')
    parser.add_argument('command', choices=['load', 'status'], help='Command to run')
    parser.add_argument('file', nargs='?', help='JSON file to load (or --stdin)')
    parser.add_argument('--stdin', action='store_true', help='Read JSON from stdin')
    parser.add_argument('--season', type=int, help='Override season year')
    parser.add_argument('--team', help='Filter status by team name')
    args = parser.parse_args()
    
    runner = ScriptRunner("ncaa_stats")
    
    if args.command == 'status':
        show_status(runner, team_filter=args.team, season_filter=args.season)
        runner.finish(exit_on_error=False)
    
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
            stats = json.loads(raw)
        except json.JSONDecodeError as e:
            runner.error(f"Invalid JSON: {e}")
            runner.finish()
        
        if not isinstance(stats, list):
            runner.error(f"Expected JSON array, got {type(stats).__name__}")
            runner.finish()
        
        runner.info(f"Loading {len(stats)} stat categories from {'stdin' if args.stdin else args.file}")
        
        added, updated, failed, unresolved = load_stats(
            stats, 
            season_override=args.season, 
            runner=runner
        )
        
        runner.add_stat("categories_in_json", len(stats))
        runner.add_stat("rows_added", added)
        runner.add_stat("rows_updated", updated)
        runner.add_stat("rows_failed", failed)
        runner.add_stat("unresolved_teams", unresolved)
        
        if failed > 0 and (added + updated) == 0:
            runner.error("All entries failed to load")
        
        runner.finish()


if __name__ == '__main__':
    main()
