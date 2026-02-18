#!/usr/bin/env python3
"""
Discover SIDEARM stats URLs for all D1 baseball teams.

Tries common athletics domain patterns and verifies SIDEARM JSON payload exists.

Usage:
    python3 scripts/discover_d1_stats_urls.py --test          # Test existing URLs
    python3 scripts/discover_d1_stats_urls.py --discover      # Find new URLs
    python3 scripts/discover_d1_stats_urls.py --batch 0 50    # Discover batch
"""

import json
import sys
import time
import sqlite3
import urllib.request
import urllib.error
import logging
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
URLS_FILE = PROJECT_DIR / 'data' / 'd1_team_urls.json'
P4_URLS_FILE = PROJECT_DIR / 'data' / 'p4_team_urls.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('discover')

# Common athletics domain patterns for D1 schools
# Format: team_id -> athletics domain
KNOWN_DOMAINS = {
    # SEC
    'alabama': 'rolltide.com',
    'arkansas': 'arkansasrazorbacks.com',
    'auburn': 'auburntigers.com',
    'florida': 'floridagators.com',
    'georgia': 'georgiadogs.com',
    'kentucky': 'ukathletics.com',
    'lsu': 'lsusports.net',
    'mississippi-state': 'hailstate.com',
    'missouri': 'mutigers.com',
    'ole-miss': 'olemisssports.com',
    'oklahoma': 'soonersports.com',
    'south-carolina': 'gamecocksonline.com',
    'tennessee': 'utsports.com',
    'texas': 'texassports.com',
    'texas-a&m': '12thman.com',
    'vanderbilt': 'vucommodores.com',
    # Big Ten
    'illinois': 'fightingillini.com',
    'indiana': 'iuhoosiers.com',
    'iowa': 'hawkeyesports.com',
    'maryland': 'umterps.com',
    'michigan': 'mgoblue.com',
    'michigan-state': 'msuspartans.com',
    'minnesota': 'gophersports.com',
    'nebraska': 'huskers.com',
    'northwestern': 'nusports.com',
    'ohio-state': 'ohiostatebuckeyes.com',
    'oregon': 'goducks.com',
    'oregon-state': 'osubeavers.com',
    'penn-state': 'gopsusports.com',
    'purdue': 'purduesports.com',
    'rutgers': 'scarletknights.com',
    'ucla': 'uclabruins.com',
    'usc': 'usctrojans.com',
    'washington': 'gohuskies.com',
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def check_sidearm_url(url, timeout=10):
    """Check if a URL is a valid SIDEARM stats page. Returns True/False."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read(50000).decode('utf-8', errors='ignore')
            # SIDEARM pages have Nuxt payload or specific markers
            if '__NUXT' in html or 'sidearm' in html.lower() or 'batting_leaders' in html.lower():
                return True, 'sidearm'
            elif '<table' in html and ('AVG' in html or 'ERA' in html):
                return True, 'html_table'
            return False, 'no_stats'
    except urllib.error.HTTPError as e:
        return False, f'http_{e.code}'
    except Exception as e:
        return False, str(e)[:50]


def load_existing_urls():
    """Load existing URL mappings."""
    urls = {}
    if P4_URLS_FILE.exists():
        with open(P4_URLS_FILE) as f:
            p4 = json.load(f)
            for tid, url in p4.get('teams', {}).items():
                urls[tid] = url if isinstance(url, str) else url.get('stats_url', '')
    if URLS_FILE.exists():
        with open(URLS_FILE) as f:
            d1 = json.load(f)
            urls.update(d1.get('teams', {}))
    return urls


def guess_athletics_domains(team_id, team_name):
    """Generate possible athletics domain guesses for a team."""
    if team_id in KNOWN_DOMAINS:
        return [KNOWN_DOMAINS[team_id]]
    
    # Common patterns
    guesses = []
    slug = team_id.replace('-', '')
    name_parts = team_name.lower().split() if team_name else []
    
    # Try common patterns
    patterns = [
        f'{slug}sports.com',
        f'{slug}athletics.com',
        f'go{slug}.com',
        f'{slug}.com',
    ]
    
    # School-specific pattern with "go" prefix
    if name_parts:
        first = name_parts[0]
        patterns.extend([
            f'go{first}.com',
            f'{first}sports.com',
            f'{first}athletics.com',
        ])
    
    return patterns


def discover_urls(teams, existing_urls, start=0, end=None):
    """Try to discover stats URLs for teams without them."""
    if end is None:
        end = len(teams)
    
    found = {}
    failed = []
    
    for i, (team_id, team_name) in enumerate(teams[start:end], start):
        if team_id in existing_urls:
            continue
        
        log.info(f'[{i+1}/{len(teams)}] {team_id} ({team_name})')
        
        # Try known domain first
        domains = guess_athletics_domains(team_id, team_name)
        
        url_found = False
        for domain in domains:
            url = f'https://{domain}/sports/baseball/stats/2026'
            ok, reason = check_sidearm_url(url)
            if ok:
                log.info(f'  ✅ Found: {url} ({reason})')
                found[team_id] = url
                url_found = True
                break
            time.sleep(0.5)
        
        if not url_found:
            failed.append(team_id)
            log.info(f'  ❌ No URL found')
        
        time.sleep(1)  # Be nice
    
    return found, failed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test existing URLs')
    parser.add_argument('--discover', action='store_true', help='Discover new URLs')
    parser.add_argument('--batch', nargs=2, type=int, help='Start and end index')
    args = parser.parse_args()
    
    db = get_db()
    teams = db.execute("""
        SELECT id, name FROM teams 
        WHERE id NOT IN ('colorado','iowa-state','smu','syracuse')
        ORDER BY conference DESC, id
    """).fetchall()
    db.close()
    
    existing = load_existing_urls()
    print(f'Total teams in DB: {len(teams)}')
    print(f'Existing URLs: {len(existing)}')
    print(f'Missing: {len(teams) - len(existing)}')
    
    if args.test:
        print('\nTesting existing URLs...')
        for tid, url in sorted(existing.items()):
            ok, reason = check_sidearm_url(url)
            status = '✅' if ok else '❌'
            print(f'  {status} {tid}: {reason}')
            time.sleep(0.5)
    
    elif args.discover or args.batch:
        start = args.batch[0] if args.batch else 0
        end = args.batch[1] if args.batch else len(teams)
        
        team_list = [(t['id'], t['name']) for t in teams]
        found, failed = discover_urls(team_list, existing, start, end)
        
        print(f'\n✅ Found: {len(found)}')
        print(f'❌ Failed: {len(failed)}')
        
        if found:
            # Merge with existing
            all_urls = {**existing, **found}
            with open(URLS_FILE, 'w') as f:
                json.dump({'teams': all_urls, 'updated': time.strftime('%Y-%m-%d')}, f, indent=2)
            print(f'Saved to {URLS_FILE}')


if __name__ == '__main__':
    main()
