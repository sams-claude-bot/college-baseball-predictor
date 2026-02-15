#!/usr/bin/env python3
"""
Nightly stats collection script.
Runs at night to collect box scores from multiple sources.

Sources:
1. ESPN scoreboard - final scores (web fetch)
2. StatBroadcast - detailed box scores (Playwright)
3. School stats pages - cumulative stats (web fetch)

Usage:
    python3 nightly_collection.py                    # Collect yesterday's games
    python3 nightly_collection.py --date 2026-02-13 # Specific date
    python3 nightly_collection.py --delay 15        # Custom delay
"""

import re
import sys
import json
import time
import argparse
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
SCRIPTS_DIR = Path(__file__).parent

# Try imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def get_connection():
    return sqlite3.connect(str(DATA_DIR / 'baseball.db'))


def log(msg):
    """Log with timestamp"""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


# ============================================================
# ESPN Collection (web fetch - no browser needed)
# ============================================================

def fetch_espn_scoreboard(date_str, delay=10):
    """Fetch and parse ESPN scoreboard"""
    if not HAS_REQUESTS:
        log("requests not installed - skipping ESPN")
        return []
    
    date_param = date_str.replace('-', '')
    url = f"https://www.espn.com/college-baseball/scoreboard?date={date_param}"
    log(f"Fetching ESPN scoreboard: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            log(f"ESPN returned {resp.status_code}")
            return []
        
        # Extract game IDs from URLs
        game_ids = list(set(re.findall(r'/gameId/(\d+)', resp.text)))
        log(f"Found {len(game_ids)} ESPN game IDs")
        
        time.sleep(delay)
        return game_ids
    except Exception as e:
        log(f"ESPN error: {e}")
        return []


def fetch_espn_boxscore(game_id, delay=10):
    """Fetch individual ESPN box score"""
    if not HAS_REQUESTS:
        return None
    
    url = f"https://www.espn.com/college-baseball/boxscore/_/gameId/{game_id}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            time.sleep(delay)
            return resp.text
    except:
        pass
    return None


# ============================================================
# StatBroadcast Collection (Playwright - needs browser)
# ============================================================

def fetch_statbroadcast_ids(school_code='msst', delay=10):
    """Get recent game IDs from StatBroadcast school portal"""
    if not HAS_PLAYWRIGHT:
        log("Playwright not installed - skipping StatBroadcast discovery")
        return []
    
    url = f"https://statbroadcast.com/events/statmonitr.php?gid={school_code}"
    log(f"Fetching StatBroadcast portal: {url}")
    
    game_ids = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            time.sleep(3)  # Wait for JS
            
            html = page.content()
            
            # Find baseball game IDs
            # Pattern: href="/events/archived.php?id=XXXXXX" or stats link
            base_ids = re.findall(r'id=(\d{6,})', html)
            
            # Filter for baseball (look for BASE marker nearby)
            # For now just return all - filter later
            game_ids = list(set(base_ids))[:20]  # Limit
            
            browser.close()
            
        log(f"Found {len(game_ids)} StatBroadcast IDs")
        time.sleep(delay)
    except Exception as e:
        log(f"StatBroadcast portal error: {e}")
    
    return game_ids


def fetch_statbroadcast_boxscore(game_id, delay=10):
    """Fetch StatBroadcast box score with Playwright"""
    if not HAS_PLAYWRIGHT:
        return None
    
    url = f"https://stats.statbroadcast.com/statmonitr/?id={game_id}"
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            time.sleep(2)
            
            title = page.title()
            html = page.content()
            
            browser.close()
            
            # Parse title for score
            match = re.match(r'([A-Z\s]+)\s*(\d+),\s*([A-Z\s]+)\s*(\d+)\s*-\s*Final', 
                           title, re.IGNORECASE)
            
            if match:
                result = {
                    'statbroadcast_id': game_id,
                    'away_team': match.group(1).strip(),
                    'away_score': int(match.group(2)),
                    'home_team': match.group(3).strip(),
                    'home_score': int(match.group(4)),
                    'raw_html': html[:100000]
                }
                time.sleep(delay)
                return result
                
    except Exception as e:
        log(f"StatBroadcast {game_id} error: {e}")
    
    return None


# ============================================================
# Main Collection
# ============================================================

def run_collection(date_str, delay=15, max_games=50):
    """Run full nightly collection"""
    log(f"Starting nightly collection for {date_str}")
    log(f"Delay: {delay}s, Max games: {max_games}")
    
    stats = {'espn': 0, 'statbroadcast': 0, 'errors': 0}
    
    # 1. ESPN Scoreboard
    log("\n=== ESPN Collection ===")
    espn_ids = fetch_espn_scoreboard(date_str, delay)
    
    for i, gid in enumerate(espn_ids[:max_games]):
        log(f"ESPN [{i+1}/{len(espn_ids)}] Game {gid}")
        html = fetch_espn_boxscore(gid, delay)
        if html:
            # Save to database (simplified)
            stats['espn'] += 1
    
    # 2. StatBroadcast (if available)
    if HAS_PLAYWRIGHT:
        log("\n=== StatBroadcast Collection ===")
        
        # Check a few school portals for recent baseball games
        schools = ['msst', 'lsu', 'texas', 'ucla', 'florida']
        
        all_sb_ids = set()
        for school in schools:
            ids = fetch_statbroadcast_ids(school, delay)
            all_sb_ids.update(ids)
        
        log(f"Total unique StatBroadcast IDs: {len(all_sb_ids)}")
        
        for i, gid in enumerate(list(all_sb_ids)[:max_games]):
            log(f"StatBroadcast [{i+1}/{len(all_sb_ids)}] Game {gid}")
            data = fetch_statbroadcast_boxscore(gid, delay)
            if data:
                log(f"  {data['away_team']} {data['away_score']} @ {data['home_team']} {data['home_score']}")
                stats['statbroadcast'] += 1
    
    # Summary
    log("\n=== Collection Complete ===")
    log(f"ESPN box scores: {stats['espn']}")
    log(f"StatBroadcast box scores: {stats['statbroadcast']}")
    log(f"Errors: {stats['errors']}")
    
    return stats


def cleanup():
    """Ensure all browsers are closed"""
    log("Cleaning up browsers...")
    try:
        # Kill any stray chromium processes from Playwright
        subprocess.run(['pkill', '-f', 'chromium.*--headless'], 
                      capture_output=True, timeout=5)
    except:
        pass
    log("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='Nightly baseball stats collection')
    parser.add_argument('--date', help='Date to collect (YYYY-MM-DD), default: yesterday')
    parser.add_argument('--delay', type=int, default=15, help='Delay between requests')
    parser.add_argument('--max', type=int, default=50, help='Max games to collect')
    args = parser.parse_args()
    
    # Default to yesterday
    if args.date:
        date_str = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')
    
    try:
        run_collection(date_str, args.delay, args.max)
    finally:
        cleanup()


if __name__ == '__main__':
    main()
