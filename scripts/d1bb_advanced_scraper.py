#!/usr/bin/env python3
"""
D1Baseball Advanced Stats Scraper

Uses Playwright with the openclaw browser profile (which has D1Baseball login cookies)
to extract advanced stats (wOBA, wRC+, FIP, xFIP, batted ball data) and store in DB.

Usage:
    python3 scripts/d1bb_advanced_scraper.py --team mississippi-state
    python3 scripts/d1bb_advanced_scraper.py --conference SEC
    python3 scripts/d1bb_advanced_scraper.py --all-p4
    python3 scripts/d1bb_advanced_scraper.py --all-p4 --dry-run
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
OPENCLAW_USER_DATA = Path.home() / '.openclaw' / 'browser' / 'openclaw' / 'user-data'

# Import the update function from the existing script
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from d1baseball_advanced import update_team_advanced, get_db, resolve_team_id


def load_slug_map():
    """Load team_id to D1BB slug mapping."""
    if SLUGS_FILE.exists():
        data = json.loads(SLUGS_FILE.read_text())
        return data.get('team_id_to_d1bb_slug', {})
    return {}


def get_p4_teams(db):
    """Get all Power 4 team IDs."""
    cursor = db.execute("""
        SELECT id FROM teams 
        WHERE conference IN ('SEC', 'Big Ten', 'ACC', 'Big 12')
        ORDER BY id
    """)
    return [row[0] for row in cursor.fetchall()]


def get_conference_teams(db, conference):
    """Get team IDs for a specific conference."""
    cursor = db.execute("""
        SELECT id FROM teams WHERE conference = ? ORDER BY id
    """, (conference,))
    return [row[0] for row in cursor.fetchall()]


def extract_team_stats(page, team_slug, verbose=False):
    """
    Extract all advanced stats from a D1Baseball team stats page.
    Returns dict with adv_batting, bb_batting, adv_pitching, bb_pitching lists.
    """
    url = f"https://d1baseball.com/team/{team_slug}/stats/"
    if verbose:
        print(f"  Loading {url}...")
    
    page.goto(url, wait_until='networkidle', timeout=30000)
    time.sleep(1)  # Let JS tables render
    
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
        'adv_batting': [],
        'bb_batting': [],
        'adv_pitching': [],
        'bb_pitching': []
    }
    
    for table in result:
        headers = table['headers']
        rows = table['rows']
        
        # Skip tables with placeholder data (non-subscriber)
        if rows and all(r.get('wOBA') == '.123' or r.get('FIP') == '.123' for r in rows[:3] if 'wOBA' in r or 'FIP' in r):
            print(f"  WARNING: Table {table['tableIndex']} has placeholder data - not logged in?")
            continue
        
        if 'wOBA' in headers and 'wRC+' in headers:
            # Advanced Batting
            data['adv_batting'] = rows
            if verbose:
                print(f"  Found Advanced Batting: {len(rows)} players")
        elif 'FIP' in headers and 'xFIP' in headers:
            # Advanced Pitching  
            data['adv_pitching'] = rows
            if verbose:
                print(f"  Found Advanced Pitching: {len(rows)} players")
        elif 'GB%' in headers and 'LD%' in headers and 'FB%' in headers:
            # Batted Ball - determine if batting or pitching by other columns
            if 'HR/FB%' in headers and len(data['bb_batting']) == 0:
                # First GB% table is usually batting
                data['bb_batting'] = rows
                if verbose:
                    print(f"  Found Batted Ball Batting: {len(rows)} players")
            elif len(data['bb_pitching']) == 0:
                data['bb_pitching'] = rows
                if verbose:
                    print(f"  Found Batted Ball Pitching: {len(rows)} players")
    
    # Validate we got something useful
    total = len(data['adv_batting']) + len(data['adv_pitching'])
    if total == 0:
        print(f"  WARNING: No advanced stats extracted for {team_slug}")
        return None
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball advanced stats')
    parser.add_argument('--team', '-t', help='Single team ID to scrape')
    parser.add_argument('--conference', '-c', help='Conference to scrape (e.g., SEC)')
    parser.add_argument('--all-p4', action='store_true', help='Scrape all Power 4 teams')
    parser.add_argument('--dry-run', action='store_true', help='Extract but do not save to DB')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-json', '-o', help='Save extracted data to JSON file')
    args = parser.parse_args()
    
    if not any([args.team, args.conference, args.all_p4]):
        parser.error("Must specify --team, --conference, or --all-p4")
    
    # Load slug mapping
    slug_map = load_slug_map()
    
    # Get list of teams to process
    db = get_db()
    
    if args.team:
        teams = [args.team]
    elif args.conference:
        teams = get_conference_teams(db, args.conference)
    elif args.all_p4:
        teams = get_p4_teams(db)
    
    print(f"Processing {len(teams)} teams...")
    
    # Initialize Playwright with openclaw profile
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)
    
    all_data = []
    stats = {'success': 0, 'failed': 0, 'updated': 0, 'skipped': 0}
    
    with sync_playwright() as p:
        # Use persistent context to get the openclaw browser cookies
        if not OPENCLAW_USER_DATA.exists():
            print(f"WARNING: OpenClaw browser profile not found at {OPENCLAW_USER_DATA}")
            print("Make sure you've logged into D1Baseball via the openclaw browser profile.")
        
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(OPENCLAW_USER_DATA),
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        page = browser.new_page()
        
        for team_id in teams:
            # Get D1BB slug for this team
            slug = slug_map.get(team_id, team_id)
            print(f"\n=== {team_id} (slug: {slug}) ===")
            
            try:
                data = extract_team_stats(page, slug, verbose=args.verbose)
                if data:
                    all_data.append(data)
                    stats['success'] += 1
                    
                    if not args.dry_run:
                        resolved_id = resolve_team_id(db, slug)
                        if resolved_id:
                            result = update_team_advanced(db, resolved_id, data)
                            stats['updated'] += result['updated']
                            stats['skipped'] += result['skipped']
                            print(f"  DB Updated: {result['updated']} players, {result['skipped']} skipped")
                        else:
                            print(f"  ERROR: Could not resolve team_id for {slug}")
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"  ERROR: {e}")
                stats['failed'] += 1
            
            # Rate limit
            time.sleep(1)
        
        browser.close()
    
    db.close()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Teams processed: {stats['success']} success, {stats['failed']} failed")
    if not args.dry_run:
        print(f"Players updated: {stats['updated']}, skipped: {stats['skipped']}")
    
    # Save JSON output if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"JSON saved to: {args.output_json}")


if __name__ == '__main__':
    main()
