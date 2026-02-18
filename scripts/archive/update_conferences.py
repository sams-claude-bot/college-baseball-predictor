#!/usr/bin/env python3
"""
Update conference assignments from ESPN team detail API.

Usage:
    python3 scripts/update_conferences.py                # Update all missing
    python3 scripts/update_conferences.py --batch 0 20   # Batch of 20
    python3 scripts/update_conferences.py --show         # Show current assignments
"""

import json
import re
import sys
import time
import sqlite3
import urllib.request
import urllib.error
import logging
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
ESPN_IDS_FILE = PROJECT_DIR / 'data' / 'espn_team_ids.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('conferences')

# Conference name normalization
CONF_MAP = {
    'SEC': 'SEC',
    'Southeastern Conference': 'SEC',
    'Big Ten': 'Big Ten',
    'Big Ten Conference': 'Big Ten',
    'ACC': 'ACC',
    'Atlantic Coast Conference': 'ACC',
    'Big 12': 'Big 12',
    'Big 12 Conference': 'Big 12',
    'AAC': 'AAC',
    'American Athletic Conference': 'AAC',
    'Sun Belt': 'Sun Belt',
    'Sun Belt Conference': 'Sun Belt',
    'Conference USA': 'C-USA',
    'C-USA': 'C-USA',
    'Mountain West': 'MWC',
    'MWC': 'MWC',
    'WCC': 'WCC',
    'West Coast Conference': 'WCC',
    'Big East': 'Big East',
    'Big East Conference': 'Big East',
    'A-10': 'A-10',
    'Atlantic 10 Conference': 'A-10',
    'Atlantic 10': 'A-10',
    'Colonial Athletic Association': 'CAA',
    'CAA': 'CAA',
    'Missouri Valley Conference': 'MVC',
    'Missouri Valley': 'MVC',
    'MVC': 'MVC',
    'Southern Conference': 'SoCon',
    'SoCon': 'SoCon',
    'ASUN': 'ASUN',
    'ASUN Conference': 'ASUN',
    'Southland Conference': 'Southland',
    'Southland': 'Southland',
    'Ohio Valley Conference': 'OVC',
    'OVC': 'OVC',
    'Patriot League': 'Patriot',
    'Patriot': 'Patriot',
    'Ivy League': 'Ivy',
    'Ivy': 'Ivy',
    'MAAC': 'MAAC',
    'Metro Atlantic Athletic Conference': 'MAAC',
    'Northeast Conference': 'NEC',
    'NEC': 'NEC',
    'Horizon League': 'Horizon',
    'Horizon': 'Horizon',
    'Big West Conference': 'Big West',
    'Big West': 'Big West',
    'Summit League': 'Summit',
    'Summit': 'Summit',
    'MEAC': 'MEAC',
    'Mid-Eastern Athletic Conference': 'MEAC',
    'SWAC': 'SWAC',
    'Southwestern Athletic Conference': 'SWAC',
    'WAC': 'WAC',
    'Western Athletic Conference': 'WAC',
    'MAC': 'MAC',
    'Mid-American Conference': 'MAC',
    'America East Conference': 'America East',
    'America East': 'America East',
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def parse_conference(standing_summary):
    """Parse conference name from ESPN standingSummary like '1st in SEC - West'."""
    if not standing_summary:
        return None
    
    m = re.search(r'in (.+?)(?:\s*-\s*\w+)?$', standing_summary)
    if m:
        raw = m.group(1).strip()
        return CONF_MAP.get(raw, raw)
    return None


def fetch_team_conference(espn_id):
    """Get conference from ESPN team detail endpoint."""
    url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams/{espn_id}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        
        team = data.get('team', {})
        standing = team.get('standingSummary', '')
        conf = parse_conference(standing)
        return conf
    except Exception as e:
        log.warning(f'  Failed for ESPN {espn_id}: {e}')
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', nargs=2, type=int, help='Start and count')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    
    db = get_db()
    
    if args.show:
        rows = db.execute("""
            SELECT conference, COUNT(*) as cnt FROM teams 
            WHERE conference != '' AND conference IS NOT NULL
            GROUP BY conference ORDER BY cnt DESC
        """).fetchall()
        for r in rows:
            print(f"  {r['conference']:<15} {r['cnt']}")
        total = db.execute("SELECT COUNT(*) FROM teams WHERE conference = '' OR conference IS NULL").fetchone()[0]
        print(f"\n  No conference: {total}")
        db.close()
        return
    
    # Load ESPN ID map
    with open(ESPN_IDS_FILE) as f:
        espn_ids = json.load(f)
    
    # Get teams needing conference updates
    teams = db.execute("""
        SELECT t.id, t.name FROM teams t
        WHERE (t.conference IS NULL OR t.conference = '')
        ORDER BY t.id
    """).fetchall()
    
    start = args.batch[0] if args.batch else 0
    count = args.batch[1] if args.batch else len(teams)
    batch = teams[start:start + count]
    
    log.info(f'Updating conferences for {len(batch)} teams (batch {start}-{start+count})')
    
    updated = 0
    skipped = 0
    for i, t in enumerate(batch):
        espn_id = espn_ids.get(t['id'])
        if not espn_id:
            skipped += 1
            continue
        
        conf = fetch_team_conference(espn_id)
        if conf:
            db.execute("UPDATE teams SET conference = ? WHERE id = ?", (conf, t['id']))
            log.info(f'  [{i+1}/{len(batch)}] {t["id"]}: {conf}')
            updated += 1
        else:
            log.info(f'  [{i+1}/{len(batch)}] {t["id"]}: no conference found')
        
        time.sleep(1.5)  # Be nice to ESPN
    
    db.commit()
    log.info(f'✅ Updated {updated}, skipped {skipped} (no ESPN ID)')
    db.close()


if __name__ == '__main__':
    main()
