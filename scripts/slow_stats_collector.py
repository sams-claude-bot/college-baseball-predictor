#!/usr/bin/env python3
"""
Slow stats collector - pulls NCAA.com stats one category at a time
with delays to avoid rate limits.

Usage:
    python3 slow_stats_collector.py           # Run all categories slowly
    python3 slow_stats_collector.py --team    # Team stats only
    python3 slow_stats_collector.py --indiv   # Individual stats only
    python3 slow_stats_collector.py --delay 20  # Custom delay (seconds)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scrape_ncaa_stats import (
    TEAM_STAT_IDS, INDIVIDUAL_STAT_IDS,
    init_ncaa_stats_table, normalize_team_id,
    parse_stats_page, NCAA_BASE
)
from scripts.database import get_connection

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Progress file to resume if interrupted
PROGRESS_FILE = Path(__file__).parent.parent / 'data' / 'collection_progress.json'


def load_progress():
    """Load collection progress from file"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {'completed_team': [], 'completed_indiv': [], 'last_run': None}


def save_progress(progress):
    """Save collection progress to file"""
    progress['last_run'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def fetch_with_retry(url, retries=3, delay=5):
    """Fetch URL with retries"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                print(f"  Rate limited! Waiting {delay * 3}s...")
                time.sleep(delay * 3)
            else:
                print(f"  HTTP {resp.status_code}, retrying...")
        except Exception as e:
            print(f"  Error: {e}, retrying...")
        time.sleep(delay)
    return None


def collect_team_stat(stat_name, stat_path, delay=15):
    """Collect a single team stat category"""
    url = f"{NCAA_BASE}/{stat_path}"
    print(f"Fetching team {stat_name} from {url}")
    
    html = fetch_with_retry(url)
    if not html:
        print(f"  Failed to fetch {stat_name}")
        return 0
    
    # Parse the stats
    stats = parse_stats_page(html, stat_name, is_team=True)
    if not stats:
        print(f"  No data parsed for {stat_name}")
        return 0
    
    # Save to database
    conn = get_connection()
    cur = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    count = 0
    for stat in stats:
        team_id = normalize_team_id(stat.get('team', ''))
        cur.execute('''
            INSERT INTO ncaa_team_stats 
            (team_name, team_id, stat_category, stat_value, rank, games, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            stat.get('team', ''),
            team_id,
            stat_name,
            stat.get('value'),
            stat.get('rank'),
            stat.get('games'),
            today
        ))
        count += 1
    
    conn.commit()
    conn.close()
    print(f"  Saved {count} team {stat_name} records")
    
    time.sleep(delay)  # Rate limit delay
    return count


def collect_individual_stat(stat_name, stat_path, delay=15):
    """Collect a single individual stat category"""
    url = f"{NCAA_BASE}/{stat_path}"
    print(f"Fetching individual {stat_name} from {url}")
    
    html = fetch_with_retry(url)
    if not html:
        print(f"  Failed to fetch {stat_name}")
        return 0
    
    # Parse the stats
    stats = parse_stats_page(html, stat_name, is_team=False)
    if not stats:
        print(f"  No data parsed for {stat_name}")
        return 0
    
    # Save to database
    conn = get_connection()
    cur = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    count = 0
    for stat in stats:
        team_id = normalize_team_id(stat.get('team', ''))
        cur.execute('''
            INSERT INTO ncaa_individual_stats 
            (player_name, team_name, team_id, stat_category, stat_value, rank, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            stat.get('player', ''),
            stat.get('team', ''),
            team_id,
            stat_name,
            stat.get('value'),
            stat.get('rank'),
            today
        ))
        count += 1
    
    conn.commit()
    conn.close()
    print(f"  Saved {count} individual {stat_name} records")
    
    time.sleep(delay)  # Rate limit delay
    return count


def main():
    parser = argparse.ArgumentParser(description='Slowly collect NCAA stats')
    parser.add_argument('--team', action='store_true', help='Team stats only')
    parser.add_argument('--indiv', action='store_true', help='Individual stats only')
    parser.add_argument('--delay', type=int, default=15, help='Delay between requests (seconds)')
    parser.add_argument('--resume', action='store_true', help='Resume from last progress')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start fresh')
    args = parser.parse_args()
    
    if not HAS_REQUESTS:
        print("Error: requests library required")
        sys.exit(1)
    
    # Initialize tables
    init_ncaa_stats_table()
    
    # Load/reset progress
    if args.reset:
        progress = {'completed_team': [], 'completed_indiv': [], 'last_run': None}
        save_progress(progress)
        print("Progress reset")
    else:
        progress = load_progress()
        if args.resume and progress['completed_team']:
            print(f"Resuming from last run: {progress['last_run']}")
            print(f"  Team stats completed: {progress['completed_team']}")
            print(f"  Individual stats completed: {progress['completed_indiv']}")
    
    total = 0
    do_team = args.team or (not args.team and not args.indiv)
    do_indiv = args.indiv or (not args.team and not args.indiv)
    
    # Collect team stats
    if do_team:
        print(f"\n=== Collecting Team Stats (delay={args.delay}s) ===")
        for stat_name, stat_path in TEAM_STAT_IDS.items():
            if args.resume and stat_name in progress['completed_team']:
                print(f"Skipping {stat_name} (already done)")
                continue
            count = collect_team_stat(stat_name, stat_path, args.delay)
            total += count
            progress['completed_team'].append(stat_name)
            save_progress(progress)
    
    # Collect individual stats
    if do_indiv:
        print(f"\n=== Collecting Individual Stats (delay={args.delay}s) ===")
        for stat_name, stat_path in INDIVIDUAL_STAT_IDS.items():
            if args.resume and stat_name in progress['completed_indiv']:
                print(f"Skipping {stat_name} (already done)")
                continue
            count = collect_individual_stat(stat_name, stat_path, args.delay)
            total += count
            progress['completed_indiv'].append(stat_name)
            save_progress(progress)
    
    print(f"\n=== Collection Complete ===")
    print(f"Total records collected: {total}")
    
    # Show summary
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM ncaa_team_stats')
    print(f"Total team stat records: {cur.fetchone()[0]}")
    cur.execute('SELECT COUNT(*) FROM ncaa_individual_stats')
    print(f"Total individual stat records: {cur.fetchone()[0]}")
    conn.close()


if __name__ == '__main__':
    main()
