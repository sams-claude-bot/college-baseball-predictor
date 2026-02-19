#!/usr/bin/env python3
"""
DraftKings NCAA Baseball Odds Scraper

Robust scraper with:
- Human-like delays (5-10s between actions)
- Retry logic with exponential backoff
- Saves snapshots for debugging
- Proper error handling and logging
- Uses existing browser profile (no login needed)

Usage:
    python3 scripts/dk_odds_scraper.py                    # Scrape today's odds
    python3 scripts/dk_odds_scraper.py --dry-run          # Parse only, don't save to DB
    python3 scripts/dk_odds_scraper.py --snapshot-only    # Just save snapshot, don't parse

IMPORTANT: Be respectful to DraftKings servers. This script is designed to be run
at most 2-3 times per day, with human-like delays between actions.
"""

import argparse
import json
import logging
import random
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
sys.path.insert(0, str(PROJECT_DIR))

from team_resolver import resolve_team as db_resolve_team
from config.logging_config import get_logger

# Constants
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SNAPSHOT_DIR = PROJECT_DIR / 'data' / 'snapshots'
OPENCLAW_USER_DATA = Path.home() / '.openclaw' / 'browser' / 'openclaw' / 'user-data'
RATE_LIMIT_FILE = PROJECT_DIR / 'data' / '.dk_last_scrape'

# DraftKings NCAA baseball URL
DK_URL = "https://sportsbook.draftkings.com/leagues/baseball/ncaa-baseball"

# Timing (be respectful!)
MIN_DELAY = 5.0   # Minimum seconds between actions
MAX_DELAY = 10.0  # Maximum seconds between actions
PAGE_LOAD_WAIT = 8.0  # Seconds to wait for page to fully load
MAX_RETRIES = 3   # Maximum retry attempts
RETRY_BACKOFF = 2.0  # Exponential backoff multiplier
MIN_SCRAPE_INTERVAL = 3600  # Minimum seconds between scrapes (1 hour)

log = get_logger(__name__)


def human_delay(min_sec=None, max_sec=None):
    """Sleep for a random human-like duration."""
    min_sec = min_sec or MIN_DELAY
    max_sec = max_sec or MAX_DELAY
    delay = random.uniform(min_sec, max_sec)
    log.debug(f"Waiting {delay:.1f}s...")
    time.sleep(delay)


def resolve_team(name):
    """Resolve a DraftKings team name to our team_id."""
    if not name:
        return None
    result = db_resolve_team(name)
    if result:
        return result
    # Fallback: slugify
    slug = name.lower().strip().replace(' ', '-').replace("'", '').replace('&', 'and')
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    return slug


def parse_odds(text):
    """Parse odds string like '+105' or '−135' to integer."""
    if not text:
        return None
    text = text.replace('−', '-').replace('–', '-').replace('\u2212', '-')
    match = re.search(r'([+-]?\d+)', text)
    return int(match.group(1)) if match else None


def parse_snapshot_text(text):
    """
    Parse DraftKings snapshot text into game data.
    
    Looks for patterns like:
    - Team names as links
    - Odds buttons with spread, total, ML
    """
    games = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for " at " pattern indicating a matchup
        if ' at ' in line.lower() or (i + 1 < len(lines) and lines[i + 1].strip().lower() == 'at'):
            away_name = None
            home_name = None
            
            # Look for team link patterns
            for j in range(max(0, i - 5), min(i + 10, len(lines))):
                l = lines[j].strip()
                
                # Pattern: link "Team Name"
                match = re.match(r'link "([^"]+)"', l)
                if match:
                    team_name = match.group(1)
                    # Skip non-team links
                    if any(skip in team_name.lower() for skip in ['more bets', 'sgp', 'live', 'parlay']):
                        continue
                    
                    if not away_name:
                        away_name = team_name
                    elif team_name != away_name:
                        home_name = team_name
                        break
            
            if not away_name or not home_name:
                i += 1
                continue
            
            # Find odds buttons (6 buttons: away_spread, over, away_ml, home_spread, under, home_ml)
            buttons = []
            for j in range(i, min(i + 30, len(lines))):
                l = lines[j].strip()
                # Pattern: button "spread odds" or button "O/U odds" or button "ML"
                if l.startswith('button "'):
                    match = re.match(r'button "([^"]+)"', l)
                    if match:
                        btn_text = match.group(1)
                        # Filter to odds-like buttons
                        if any(c in btn_text for c in ['+', '-', '−', 'O ', 'U ']):
                            buttons.append(btn_text)
                
                if len(buttons) >= 6:
                    break
            
            if len(buttons) >= 6:
                game = {
                    'away': away_name,
                    'home': home_name,
                    'away_id': resolve_team(away_name),
                    'home_id': resolve_team(home_name),
                }
                
                try:
                    # Parse away spread: "+1.5 −145" or "-1.5 +114"
                    sp = re.match(r'([+-]?\d+\.?\d*)\s+([+-−–]\d+)', 
                                  buttons[0].replace('−', '-').replace('–', '-'))
                    if sp:
                        game['away_spread'] = float(sp.group(1))
                        game['away_spread_odds'] = int(sp.group(2))
                    
                    # Parse over: "O 13.5 −115"
                    ov = re.match(r'O\s+(\d+\.?\d*)\s+([+-−–]\d+)', 
                                  buttons[1].replace('−', '-').replace('–', '-'))
                    if ov:
                        game['over_under'] = float(ov.group(1))
                        game['over_odds'] = int(ov.group(2))
                    
                    # Away ML
                    game['away_ml'] = parse_odds(buttons[2])
                    
                    # Parse home spread
                    sp2 = re.match(r'([+-]?\d+\.?\d*)\s+([+-−–]\d+)', 
                                   buttons[3].replace('−', '-').replace('–', '-'))
                    if sp2:
                        game['home_spread'] = float(sp2.group(1))
                        game['home_spread_odds'] = int(sp2.group(2))
                    
                    # Parse under: "U 13.5 −115"
                    un = re.match(r'U\s+(\d+\.?\d*)\s+([+-−–]\d+)', 
                                  buttons[4].replace('−', '-').replace('–', '-'))
                    if un:
                        game['under_odds'] = int(un.group(2))
                    
                    # Home ML
                    game['home_ml'] = parse_odds(buttons[5])
                    
                    games.append(game)
                    log.debug(f"Parsed: {away_name} @ {home_name}")
                    
                except Exception as e:
                    log.warning(f"Failed to parse odds for {away_name} @ {home_name}: {e}")
        
        i += 1
    
    return games


def save_to_db(games, date_str, dry_run=False):
    """Save parsed games to betting_lines table."""
    if dry_run:
        log.info("Dry run - not saving to database")
        for g in games:
            log.info(f"  Would save: {g['away']} @ {g['home']} | "
                    f"ML: {g.get('away_ml')}/{g.get('home_ml')} | "
                    f"O/U: {g.get('over_under')}")
        return len(games)
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    added = 0
    updated = 0
    
    for g in games:
        game_id = f"{date_str}_{g['away_id']}_{g['home_id']}"
        
        existing = conn.execute(
            "SELECT id FROM betting_lines WHERE game_id=?", (game_id,)
        ).fetchone()
        
        try:
            if existing:
                conn.execute("""
                    UPDATE betting_lines SET 
                        home_ml=?, away_ml=?, home_spread=?, home_spread_odds=?,
                        away_spread=?, away_spread_odds=?, over_under=?, over_odds=?, under_odds=?,
                        captured_at=CURRENT_TIMESTAMP
                    WHERE game_id=?
                """, (g.get('home_ml'), g.get('away_ml'), 
                      g.get('home_spread'), g.get('home_spread_odds'),
                      g.get('away_spread'), g.get('away_spread_odds'), 
                      g.get('over_under'), g.get('over_odds'), g.get('under_odds'), 
                      game_id))
                updated += 1
                log.info(f"  Updated: {g['away']} @ {g['home']}")
            else:
                conn.execute("""
                    INSERT INTO betting_lines (game_id, date, home_team_id, away_team_id, book,
                        home_ml, away_ml, home_spread, home_spread_odds, away_spread, away_spread_odds,
                        over_under, over_odds, under_odds)
                    VALUES (?, ?, ?, ?, 'draftkings', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, date_str, g['home_id'], g['away_id'],
                      g.get('home_ml'), g.get('away_ml'), 
                      g.get('home_spread'), g.get('home_spread_odds'),
                      g.get('away_spread'), g.get('away_spread_odds'), 
                      g.get('over_under'), g.get('over_odds'), g.get('under_odds')))
                added += 1
                log.info(f"  Added: {g['away']} @ {g['home']}")
                
        except Exception as e:
            log.error(f"  DB error for {g['away']} @ {g['home']}: {e}")
    
    conn.commit()
    conn.close()
    
    log.info(f"DraftKings: {added} added, {updated} updated for {date_str}")
    return added + updated


def check_rate_limit(force=False):
    """Check if we've scraped too recently. Returns True if OK to proceed."""
    if force:
        return True
    
    if RATE_LIMIT_FILE.exists():
        try:
            last_scrape = float(RATE_LIMIT_FILE.read_text().strip())
            elapsed = time.time() - last_scrape
            if elapsed < MIN_SCRAPE_INTERVAL:
                remaining = int(MIN_SCRAPE_INTERVAL - elapsed)
                log.warning(f"Rate limit: last scrape was {int(elapsed)}s ago. "
                           f"Wait {remaining}s or use --force")
                return False
        except (ValueError, OSError):
            pass
    
    return True


def update_rate_limit():
    """Record that we just scraped."""
    try:
        RATE_LIMIT_FILE.write_text(str(time.time()))
    except OSError as e:
        log.warning(f"Could not update rate limit file: {e}")


def scrape_with_playwright(dry_run=False, snapshot_only=False, force=False):
    """
    Scrape DraftKings using Playwright with the openclaw browser profile.
    """
    # Check rate limit
    if not check_rate_limit(force=force):
        return None
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    
    log.info(f"Starting DraftKings scrape for {date_str}")
    log.info(f"URL: {DK_URL}")
    
    for attempt in range(MAX_RETRIES):
        try:
            with sync_playwright() as p:
                log.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Launching browser...")
                
                # Use existing openclaw profile (has cookies, looks like real user)
                browser = p.chromium.launch_persistent_context(
                    user_data_dir=str(OPENCLAW_USER_DATA),
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                    ]
                )
                
                page = browser.new_page()
                
                # Set realistic viewport
                page.set_viewport_size({"width": 1920, "height": 1080})
                
                # Navigate with human-like timing
                log.info("Navigating to DraftKings...")
                page.goto(DK_URL, wait_until='domcontentloaded', timeout=30000)
                
                # Wait for page to fully load
                log.info(f"Waiting {PAGE_LOAD_WAIT}s for page to load...")
                time.sleep(PAGE_LOAD_WAIT)
                
                # Scroll down slowly to load lazy content (human-like)
                log.info("Scrolling to load content...")
                for _ in range(3):
                    page.evaluate("window.scrollBy(0, 500)")
                    human_delay(1, 2)
                
                # Scroll back up
                page.evaluate("window.scrollTo(0, 0)")
                human_delay(2, 3)
                
                # Take snapshot
                log.info("Taking snapshot...")
                
                # Get accessibility tree (better for parsing than HTML)
                snapshot = page.accessibility.snapshot()
                snapshot_text = json.dumps(snapshot, indent=2) if snapshot else ""
                
                # Also get text content
                text_content = page.inner_text('body')
                
                # Save snapshot for debugging
                snapshot_file = SNAPSHOT_DIR / f"dk_{date_str}_{datetime.now().strftime('%H%M%S')}.txt"
                with open(snapshot_file, 'w') as f:
                    f.write(f"=== URL: {DK_URL} ===\n")
                    f.write(f"=== Date: {datetime.now().isoformat()} ===\n\n")
                    f.write("=== ACCESSIBILITY TREE ===\n")
                    f.write(snapshot_text)
                    f.write("\n\n=== TEXT CONTENT ===\n")
                    f.write(text_content)
                log.info(f"Snapshot saved: {snapshot_file}")
                
                browser.close()
                
                if snapshot_only:
                    log.info("Snapshot-only mode, not parsing")
                    return 0
                
                # Parse the snapshot
                log.info("Parsing snapshot...")
                
                # Try parsing both formats
                games = parse_snapshot_text(snapshot_text)
                if not games:
                    games = parse_snapshot_text(text_content)
                
                if not games:
                    log.warning("No games found in snapshot. Page may have changed structure.")
                    log.info(f"Check snapshot file: {snapshot_file}")
                    
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_BACKOFF ** attempt * 30
                        log.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    return 0
                
                log.info(f"Found {len(games)} games")
                
                # Update rate limit
                if not dry_run and not snapshot_only:
                    update_rate_limit()
                
                # Save to database
                return save_to_db(games, date_str, dry_run=dry_run)
        
        except Exception as e:
            log.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF ** attempt * 30
                log.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                log.error("All retry attempts failed")
                raise
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Scrape DraftKings NCAA baseball odds')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Parse only, do not save to database')
    parser.add_argument('--snapshot-only', action='store_true',
                        help='Just save snapshot, do not parse')
    parser.add_argument('--file', type=str,
                        help='Parse from existing snapshot file instead of scraping')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Ignore rate limit (use sparingly!)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.file:
        # Parse existing file (no rate limit needed)
        log.info(f"Parsing from file: {args.file}")
        with open(args.file) as f:
            text = f.read()
        games = parse_snapshot_text(text)
        
        if games:
            log.info(f"Found {len(games)} games")
            date_str = datetime.now().strftime('%Y-%m-%d')
            save_to_db(games, date_str, dry_run=args.dry_run)
        else:
            log.warning("No games found in file")
    else:
        # Scrape live
        result = scrape_with_playwright(
            dry_run=args.dry_run, 
            snapshot_only=args.snapshot_only,
            force=args.force
        )
        
        if result:
            log.info(f"✅ Successfully processed {result} games")
        else:
            log.warning("⚠️ No games processed")


if __name__ == '__main__':
    main()
