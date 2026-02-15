#!/usr/bin/env python3
"""
Headless Playwright script to collect box scores from StatBroadcast.
Runs without GUI - perfect for night collection.

Usage:
    python3 playwright_boxscore.py --id 633250           # Single game
    python3 playwright_boxscore.py --date 2026-02-13    # All games for date
    python3 playwright_boxscore.py --delay 10           # Custom delay between games
"""

import re
import sys
import json
import time
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

# Check for playwright
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

def get_connection():
    return sqlite3.connect(str(DATA_DIR / 'baseball.db'))


def init_tables():
    """Ensure box score tables exist"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS statbroadcast_boxscores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL UNIQUE,
            statbroadcast_id TEXT,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            innings TEXT,
            home_batting_json TEXT,
            away_batting_json TEXT,
            home_pitching_json TEXT,
            away_pitching_json TEXT,
            play_by_play_json TEXT,
            raw_html TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def fetch_boxscore_playwright(statbroadcast_id, headless=True, timeout=30000):
    """
    Fetch box score using Playwright headless browser.
    Returns parsed data dict or None on failure.
    """
    url = f"https://stats.statbroadcast.com/statmonitr/?id={statbroadcast_id}"
    print(f"  Fetching: {url}")
    
    result = {
        'statbroadcast_id': statbroadcast_id,
        'url': url,
        'home_team': None,
        'away_team': None,
        'home_score': None,
        'away_score': None,
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
        'raw_html': None,
    }
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            # Navigate and wait for content
            page.goto(url, timeout=timeout)
            
            # Wait for stats to load (look for common elements)
            try:
                page.wait_for_selector('.boxscore, .batting, .pitching, #linescore', 
                                       timeout=10000)
            except:
                # Try waiting for any table
                try:
                    page.wait_for_selector('table', timeout=5000)
                except:
                    pass
            
            # Give extra time for dynamic content
            time.sleep(2)
            
            # Get page content
            html = page.content()
            result['raw_html'] = html
            
            # Parse from title: "AWAY #, HOME # - Final"
            title = page.title()
            title_match = re.match(r'([A-Z\s]+)\s*(\d+),\s*([A-Z\s]+)\s*(\d+)\s*-\s*Final', title, re.IGNORECASE)
            if title_match:
                result['away_team'] = title_match.group(1).strip()
                result['away_score'] = int(title_match.group(2))
                result['home_team'] = title_match.group(3).strip()
                result['home_score'] = int(title_match.group(4))
            
            # Try to get batting stats from tables
            try:
                tables = page.query_selector_all('table')
                for table in tables:
                    table_html = table.inner_html()
                    # Look for batting tables (have AB, R, H columns)
                    if 'AB' in table_html and ('R' in table_html or 'H' in table_html):
                        rows = table.query_selector_all('tr')
                        for row in rows:
                            cells = row.query_selector_all('td')
                            if len(cells) >= 5:
                                # Likely a player stat row
                                player_data = [c.inner_text() for c in cells]
                                # Store raw for now
                                if result['home_team'] and result['home_team'].upper() in str(player_data):
                                    result['home_batting'].append(player_data)
                                else:
                                    result['away_batting'].append(player_data)
            except Exception as e:
                print(f"  Warning parsing tables: {e}")
            
            # Clean up
            browser.close()
            
            print(f"  Got: {result['away_team']} {result['away_score']} @ {result['home_team']} {result['home_score']}")
            return result
            
    except Exception as e:
        print(f"  Error: {e}")
        return None


def save_boxscore(data, date_str):
    """Save box score to database"""
    if not data:
        return False
    
    conn = get_connection()
    cur = conn.cursor()
    
    game_id = f"sb_{data['statbroadcast_id']}"
    
    try:
        cur.execute('''
            INSERT OR REPLACE INTO statbroadcast_boxscores
            (game_id, statbroadcast_id, date, home_team, away_team, 
             home_score, away_score, home_batting_json, away_batting_json,
             home_pitching_json, away_pitching_json, raw_html)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id,
            data['statbroadcast_id'],
            date_str,
            data.get('home_team'),
            data.get('away_team'),
            data.get('home_score'),
            data.get('away_score'),
            json.dumps(data.get('home_batting', [])),
            json.dumps(data.get('away_batting', [])),
            json.dumps(data.get('home_pitching', [])),
            json.dumps(data.get('away_pitching', [])),
            data.get('raw_html', '')[:100000]  # Limit size
        ))
        conn.commit()
        print(f"  Saved to database")
        return True
    except Exception as e:
        print(f"  Error saving: {e}")
        return False
    finally:
        conn.close()


def get_game_ids_from_school_portal(school_code='msst'):
    """
    Get StatBroadcast game IDs from a school's portal.
    Returns list of (game_id, date, sport, description) tuples.
    """
    url = f"https://statbroadcast.com/events/statmonitr.php?gid={school_code}"
    
    # This would need Playwright too since it's JS-rendered
    # For now, return empty - we'll populate IDs manually or from other sources
    return []


def main():
    parser = argparse.ArgumentParser(description='Collect StatBroadcast box scores with Playwright')
    parser.add_argument('--id', help='StatBroadcast game ID')
    parser.add_argument('--ids', help='Comma-separated list of game IDs')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'), 
                        help='Date for games (YYYY-MM-DD)')
    parser.add_argument('--delay', type=int, default=10, help='Delay between games (seconds)')
    parser.add_argument('--visible', action='store_true', help='Show browser (not headless)')
    args = parser.parse_args()
    
    init_tables()
    
    game_ids = []
    if args.id:
        game_ids = [args.id]
    elif args.ids:
        game_ids = [gid.strip() for gid in args.ids.split(',')]
    else:
        print("No game IDs provided. Use --id or --ids")
        print("Example: python3 playwright_boxscore.py --id 633250")
        return
    
    print(f"\nCollecting {len(game_ids)} box scores")
    print(f"Date: {args.date}, Delay: {args.delay}s, Headless: {not args.visible}")
    
    success = 0
    for i, gid in enumerate(game_ids):
        print(f"\n[{i+1}/{len(game_ids)}] Game {gid}")
        
        data = fetch_boxscore_playwright(gid, headless=not args.visible)
        if data and save_boxscore(data, args.date):
            success += 1
        
        if i < len(game_ids) - 1:
            print(f"  Waiting {args.delay}s...")
            time.sleep(args.delay)
    
    print(f"\n=== Complete ===")
    print(f"Collected {success}/{len(game_ids)} box scores")


if __name__ == '__main__':
    main()
