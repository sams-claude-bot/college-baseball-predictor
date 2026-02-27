#!/usr/bin/env python3
"""
D1Baseball Combined Stats Scraper (Basic + Advanced)

Uses Playwright with the openclaw browser profile (which has D1Baseball login cookies)
to extract ALL stats in one pass: basic (AVG, ERA) + advanced (wOBA, FIP, batted ball).

Usage:
    python3 scripts/d1b_scraper.py --team mississippi-state
    python3 scripts/d1b_scraper.py --conference SEC
    python3 scripts/d1b_scraper.py --all-d1      # All 311 D1 teams
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from run_utils import ScriptRunner

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
OPENCLAW_USER_DATA = Path.home() / '.openclaw' / 'browser' / 'openclaw' / 'user-data'


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def load_slug_map():
    """Load team_id to D1BB slug mapping."""
    if SLUGS_FILE.exists():
        data = json.loads(SLUGS_FILE.read_text())
        return data.get('team_id_to_d1bb_slug', {})
    return {}


def get_conference_teams(db, conference):
    """Get team IDs for a specific conference."""
    cursor = db.execute("SELECT id FROM teams WHERE conference = ? ORDER BY id", (conference,))
    return [row[0] for row in cursor.fetchall()]


def get_all_d1_teams(slug_map):
    """Get all D1 team IDs that have slug mappings."""
    return sorted(slug_map.keys())


def safe_float(val):
    """Convert value to float, handling various formats."""
    if val is None:
        return None
    s = str(val).strip().replace('%', '')
    if s in ('', '-', '—', 'INF', 'inf', '.123'):
        return None
    try:
        return round(float(s), 3)
    except ValueError:
        return None


def safe_int(val):
    """Convert value to int."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ('', '-', '—'):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def resolve_team_id(db, slug):
    """Resolve D1BB slug to our team_id."""
    # Load reverse mapping
    if SLUGS_FILE.exists():
        data = json.loads(SLUGS_FILE.read_text())
        slug_to_id = {v: k for k, v in data.get('team_id_to_d1bb_slug', {}).items()}
        if slug in slug_to_id:
            return slug_to_id[slug]
    
    # Try direct match
    cursor = db.execute("SELECT id FROM teams WHERE id = ?", (slug,))
    row = cursor.fetchone()
    if row:
        return row[0]
    
    return None


def extract_team_stats(page, team_slug, verbose=False):
    """
    Extract ALL stats from a D1Baseball team stats page.
    Returns dict with basic_batting, basic_pitching, adv_batting, adv_pitching, bb_batting, bb_pitching.
    """
    url = f"https://d1baseball.com/team/{team_slug}/stats/"
    if verbose:
        print(f"  Loading {url}...")
    
    page.goto(url, wait_until='domcontentloaded', timeout=30000)
    time.sleep(1.5)  # Let JS tables render
    
    # Check if page loaded properly
    if "404" in page.title() or "Nothing Found" in page.content():
        print(f"  ERROR: Page not found for slug '{team_slug}'")
        return None
    
    # Extract all tables with headers
    result = page.evaluate("""() => {
        const tables = document.querySelectorAll('table');
        const extracted = [];
        
        for (let t = 0; t < tables.length; t++) {
            const headers = Array.from(tables[t].querySelectorAll('thead th, tr:first-child th'))
                .map(h => h.textContent.trim());
            if (headers.length === 0) continue;
            
            const rows = [];
            const tbody = tables[t].querySelector('tbody') || tables[t];
            const trs = tbody.querySelectorAll('tr');
            
            for (const tr of trs) {
                const cells = tr.querySelectorAll('td');
                if (cells.length === 0) continue;
                
                const row = {};
                Array.from(cells).forEach((cell, i) => {
                    if (i < headers.length) {
                        row[headers[i]] = cell.textContent.trim();
                    }
                });
                if (Object.keys(row).length > 0) {
                    rows.push(row);
                }
            }
            
            if (rows.length > 0) {
                extracted.push({
                    tableIndex: t,
                    headers: headers,
                    rowCount: rows.length,
                    rows: rows
                });
            }
        }
        return extracted;
    }""")
    
    # Classify tables by their headers
    data = {
        'team_slug': team_slug,
        'basic_batting': [],
        'basic_pitching': [],
        'adv_batting': [],
        'adv_pitching': [],
        'bb_batting': [],
        'bb_pitching': []
    }
    
    for table in result:
        headers = table['headers']
        rows = table['rows']
        header_set = set(headers)
        
        # Skip ADVANCED tables with placeholder data (non-subscriber)
        # Only check tables that actually have wOBA or FIP columns
        is_adv_table = 'wOBA' in header_set or 'FIP' in header_set
        has_placeholder = rows and any(
            r.get('wOBA') in ('.123', '1.23', '1.230') or r.get('FIP') in ('.123', '1.23', '1.230')
            for r in rows[:3]
        )
        if is_adv_table and has_placeholder:
            if verbose:
                print(f"  WARNING: Table {table['tableIndex']} has placeholder data - not logged in?")
            continue
        
        # Basic Batting: has BA, OBP, SLG but NOT wOBA
        if 'BA' in header_set and 'OBP' in header_set and 'SLG' in header_set and 'wOBA' not in header_set:
            data['basic_batting'] = rows
            if verbose:
                print(f"  Found Basic Batting: {len(rows)} players")
        
        # Basic Pitching: has ERA, W, L but NOT FIP
        elif 'ERA' in header_set and ('W' in header_set or 'IP' in header_set) and 'FIP' not in header_set:
            data['basic_pitching'] = rows
            if verbose:
                print(f"  Found Basic Pitching: {len(rows)} players")
        
        # Advanced Batting: has wOBA, wRC+
        elif 'wOBA' in header_set and 'wRC+' in header_set:
            data['adv_batting'] = rows
            if verbose:
                print(f"  Found Advanced Batting: {len(rows)} players")
        
        # Advanced Pitching: has FIP, xFIP
        elif 'FIP' in header_set and 'xFIP' in header_set:
            data['adv_pitching'] = rows
            if verbose:
                print(f"  Found Advanced Pitching: {len(rows)} players")
        
        # Batted Ball tables
        elif 'GB%' in header_set and 'LD%' in header_set and 'FB%' in header_set:
            if 'HR/FB%' in header_set and len(data['bb_batting']) == 0:
                data['bb_batting'] = rows
                if verbose:
                    print(f"  Found Batted Ball Batting: {len(rows)} players")
            elif len(data['bb_pitching']) == 0:
                data['bb_pitching'] = rows
                if verbose:
                    print(f"  Found Batted Ball Pitching: {len(rows)} players")
    
    return data


def upsert_player(db, team_id, name, position, year, stats):
    """Insert or update a player's stats."""
    # Check if player exists
    cursor = db.execute(
        "SELECT id FROM player_stats WHERE team_id = ? AND name = ?",
        (team_id, name)
    )
    existing = cursor.fetchone()
    
    # Build update dict
    all_stats = {
        'position': position,
        'year': year,
        **{k: v for k, v in stats.items() if v is not None}
    }
    
    if existing:
        # Update
        set_parts = [f"{k} = ?" for k in all_stats.keys()]
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        values = list(all_stats.values()) + [team_id, name]
        db.execute(
            f"UPDATE player_stats SET {', '.join(set_parts)} WHERE team_id = ? AND name = ?",
            values
        )
    else:
        # Insert
        all_stats['team_id'] = team_id
        all_stats['name'] = name
        cols = list(all_stats.keys())
        placeholders = ', '.join(['?'] * len(cols))
        db.execute(
            f"INSERT INTO player_stats ({', '.join(cols)}) VALUES ({placeholders})",
            list(all_stats.values())
        )
    
    return True


def save_team_stats(db, team_id, data, snapshot_date=None):
    """Save all stats for a team to the database."""
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()
    
    stats = {'batters': 0, 'pitchers': 0}
    
    # Process Basic Batting
    for row in data.get('basic_batting', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents', 'Team'):
            continue
        
        position = row.get('POS', row.get('Pos', ''))
        year = row.get('Class', row.get('Yr', ''))
        
        player_stats = {
            'games': safe_int(row.get('GP', row.get('G'))),
            'at_bats': safe_int(row.get('AB')),
            'runs': safe_int(row.get('R')),
            'hits': safe_int(row.get('H')),
            'doubles': safe_int(row.get('2B')),
            'triples': safe_int(row.get('3B')),
            'home_runs': safe_int(row.get('HR')),
            'rbi': safe_int(row.get('RBI')),
            'walks': safe_int(row.get('BB')),
            'strikeouts': safe_int(row.get('SO', row.get('K'))),
            'stolen_bases': safe_int(row.get('SB')),
            'batting_avg': safe_float(row.get('BA', row.get('AVG'))),
            'obp': safe_float(row.get('OBP')),
            'slg': safe_float(row.get('SLG')),
            'ops': safe_float(row.get('OPS')),
        }
        
        upsert_player(db, team_id, name, position, year, player_stats)
        stats['batters'] += 1
    
    # Process Advanced Batting (merge with existing)
    for row in data.get('adv_batting', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents', 'Team'):
            continue
        
        position = row.get('POS', row.get('Pos', ''))
        year = row.get('Class', row.get('Yr', ''))
        
        adv_stats = {
            'k_pct': safe_float(row.get('K%')),
            'bb_pct': safe_float(row.get('BB%')),
            'iso': safe_float(row.get('ISO')),
            'babip': safe_float(row.get('BABIP')),
            'woba': safe_float(row.get('wOBA')),
            'wrc_plus': safe_float(row.get('wRC+')),
        }
        
        # Only update if we have meaningful data
        if any(v is not None for v in adv_stats.values()):
            upsert_player(db, team_id, name, position, year, adv_stats)
    
    # Process Batted Ball Batting
    for row in data.get('bb_batting', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents', 'Team'):
            continue
        
        bb_stats = {
            'gb_pct': safe_float(row.get('GB%')),
            'ld_pct': safe_float(row.get('LD%')),
            'fb_pct': safe_float(row.get('FB%')),
            'hr_fb_pct': safe_float(row.get('HR/FB%')),
        }
        
        if any(v is not None for v in bb_stats.values()):
            upsert_player(db, team_id, name, '', '', bb_stats)
    
    # Process Basic Pitching
    for row in data.get('basic_pitching', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents', 'Team'):
            continue
        
        position = row.get('POS', 'P')
        year = row.get('Class', row.get('Yr', ''))
        
        pitch_stats = {
            'games_pitched': safe_int(row.get('APP', row.get('G'))),
            'games_started': safe_int(row.get('GS')),
            'wins': safe_int(row.get('W')),
            'losses': safe_int(row.get('L')),
            'saves': safe_int(row.get('SV')),
            'innings_pitched': safe_float(row.get('IP')),
            'hits_allowed': safe_int(row.get('H')),
            'runs_allowed': safe_int(row.get('R')),
            'earned_runs': safe_int(row.get('ER')),
            'walks_allowed': safe_int(row.get('BB')),
            'strikeouts_pitched': safe_int(row.get('SO', row.get('K'))),
            'era': safe_float(row.get('ERA')),
            'whip': safe_float(row.get('WHIP')),
        }
        
        upsert_player(db, team_id, name, position, year, pitch_stats)
        stats['pitchers'] += 1
    
    # Process Advanced Pitching
    for row in data.get('adv_pitching', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents', 'Team'):
            continue
        
        position = row.get('POS', 'P')
        year = row.get('Class', row.get('Yr', ''))
        
        adv_pitch = {
            'fip': safe_float(row.get('FIP')),
            'xfip': safe_float(row.get('xFIP')),
            'siera': safe_float(row.get('SIERA')),
            'k_pct': safe_float(row.get('K%')),
            'bb_pct': safe_float(row.get('BB%')),
        }
        
        if any(v is not None for v in adv_pitch.values()):
            upsert_player(db, team_id, name, position, year, adv_pitch)
    
    # Process Batted Ball Pitching
    for row in data.get('bb_pitching', []):
        name = row.get('Player', '').strip()
        if not name or name in ('Totals', 'Opponents', 'Team'):
            continue
        
        bb_pitch = {
            'gb_pct': safe_float(row.get('GB%')),
            'ld_pct': safe_float(row.get('LD%')),
            'fb_pct': safe_float(row.get('FB%')),
        }
        
        if any(v is not None for v in bb_pitch.values()):
            upsert_player(db, team_id, name, 'P', '', bb_pitch)
    
    db.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball stats (basic + advanced)')
    parser.add_argument('--team', '-t', help='Single team ID to scrape')
    parser.add_argument('--conference', '-c', help='Conference to scrape (e.g., SEC)')
    parser.add_argument('--all-d1', action='store_true', help='Scrape all D1 teams (~311)')
    parser.add_argument('--dry-run', action='store_true', help='Extract but do not save to DB')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--delay', type=float, default=2.0, help='Seconds between teams (default: 2)')
    args = parser.parse_args()
    
    if not any([args.team, args.conference, args.all_d1]):
        parser.error("Must specify --team, --conference, or --all-d1")
    
    runner = ScriptRunner("d1b_scraper")
    
    # Load slug mapping
    slug_map = load_slug_map()
    
    # Get list of teams to process
    db = get_db()
    
    if args.team:
        teams = [args.team]
    elif args.conference:
        teams = get_conference_teams(db, args.conference)
    elif args.all_d1:
        teams = get_all_d1_teams(slug_map)
    
    runner.info(f"Processing {len(teams)} teams...")
    
    # Initialize Playwright with openclaw profile
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        runner.error("playwright not installed. Run: pip install playwright && playwright install chromium")
        runner.finish()
    
    totals = {'success': 0, 'failed': 0, 'batters': 0, 'pitchers': 0}
    failed_teams = []
    
    with sync_playwright() as p:
        if not OPENCLAW_USER_DATA.exists():
            runner.warn(f"OpenClaw browser profile not found at {OPENCLAW_USER_DATA}")
        
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(OPENCLAW_USER_DATA),
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        page = browser.new_page()
        
        for team_id in teams:
            slug = slug_map.get(team_id, team_id)
            runner.info(f"=== {team_id} (slug: {slug}) ===")
            
            try:
                data = extract_team_stats(page, slug, verbose=args.verbose)
                
                if data:
                    basic_bat = len(data.get('basic_batting', []))
                    basic_pit = len(data.get('basic_pitching', []))
                    adv_bat = len(data.get('adv_batting', []))
                    adv_pit = len(data.get('adv_pitching', []))
                    
                    runner.info(f"  Parsed: {basic_bat} batters, {basic_pit} pitchers (adv: {adv_bat}/{adv_pit})")
                    
                    if not args.dry_run:
                        resolved = resolve_team_id(db, slug)
                        if resolved:
                            stats = save_team_stats(db, resolved, data)
                            totals['batters'] += stats['batters']
                            totals['pitchers'] += stats['pitchers']
                            runner.info(f"  Saved: {stats['batters']} batters, {stats['pitchers']} pitchers")
                        else:
                            runner.warn(f"Could not resolve team_id for {slug}")
                    
                    totals['success'] += 1
                else:
                    totals['failed'] += 1
                    failed_teams.append(team_id)
                    
            except Exception as e:
                runner.error(f"Error scraping {team_id}: {e}")
                import traceback
                traceback.print_exc()
                totals['failed'] += 1
                failed_teams.append(team_id)
            
            time.sleep(args.delay)
        
        browser.close()
    
    db.close()
    
    runner.add_stat("teams_processed", len(teams))
    runner.add_stat("teams_success", totals['success'])
    runner.add_stat("teams_failed", totals['failed'])
    runner.add_stat("batters", totals['batters'])
    runner.add_stat("pitchers", totals['pitchers'])
    
    if failed_teams:
        runner.add_stat("failed_teams", ", ".join(failed_teams[:10]))
    
    runner.finish()


if __name__ == '__main__':
    main()
