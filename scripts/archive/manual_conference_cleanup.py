#!/usr/bin/env python3
"""
Simple conference cleanup script that processes teams one by one.
This is designed to be run manually with search results provided via input.
"""

import sqlite3
import json
import logging
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
PROGRESS_FILE = PROJECT_DIR / 'data' / 'manual_cleanup_progress.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('manual_cleanup')

# Conference mappings for common names
CONFERENCE_MAPPINGS = {
    'Southeastern Conference': 'SEC',
    'Atlantic Coast Conference': 'ACC', 
    'Big Ten Conference': 'Big Ten',
    'Big 12 Conference': 'Big 12',
    'American Athletic Conference': 'AAC',
    'Sun Belt Conference': 'Sun Belt',
    'Conference USA': 'C-USA',
    'Mountain West Conference': 'MWC',
    'West Coast Conference': 'WCC',
    'Big East Conference': 'Big East',
    'Atlantic 10 Conference': 'A-10',
    'Atlantic 10': 'A-10',
    'Colonial Athletic Association': 'CAA',
    'Missouri Valley Conference': 'MVC',
    'Southern Conference': 'SoCon',
    'ASUN Conference': 'ASUN',
    'Southland Conference': 'Southland',
    'Ohio Valley Conference': 'OVC',
    'Patriot League': 'Patriot',
    'Ivy League': 'Ivy',
    'Metro Atlantic Athletic Conference': 'MAAC',
    'Northeast Conference': 'NEC',
    'Horizon League': 'Horizon',
    'Big West Conference': 'Big West',
    'Summit League': 'Summit',
    'Mid-Eastern Athletic Conference': 'MEAC',
    'Southwestern Athletic Conference': 'SWAC',
    'Western Athletic Conference': 'WAC',
    'Mid-American Conference': 'MAC',
    'America East Conference': 'America East',
    'Big South Conference': 'Big South',
}


def load_progress():
    """Load progress from JSON file if it exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed': []}


def save_progress(progress):
    """Save progress to JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_unknown_teams():
    """Get all teams with unknown conferences."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    teams = conn.execute("""
        SELECT id, name, nickname 
        FROM teams 
        WHERE conference IS NULL OR conference = '' OR conference = 'Unknown'
        ORDER BY id
    """).fetchall()
    
    conn.close()
    return teams


def update_teams_batch(updates):
    """Batch update multiple teams."""
    conn = sqlite3.connect(str(DB_PATH))
    
    for team_id, conference, athletics_url in updates:
        if athletics_url:
            conn.execute("""
                UPDATE teams 
                SET conference = ?, athletics_url = ? 
                WHERE id = ?
            """, (conference, athletics_url, team_id))
        else:
            conn.execute("""
                UPDATE teams 
                SET conference = ? 
                WHERE id = ?
            """, (conference, team_id))
    
    conn.commit()
    conn.close()
    log.info(f"Updated {len(updates)} teams in batch")


def normalize_conference(conference_name):
    """Normalize conference name to standard abbreviation."""
    if not conference_name:
        return None
        
    # Direct mapping
    normalized = CONFERENCE_MAPPINGS.get(conference_name, conference_name)
    
    # Handle some common cases
    if 'NAIA' in conference_name.upper() or 'Division II' in conference_name or 'Division III' in conference_name:
        return 'Non-D1'
    
    return normalized


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='Show teams to process')
    parser.add_argument('--resume', action='store_true', help='Resume from where left off')
    args = parser.parse_args()
    
    teams = get_unknown_teams()
    progress = load_progress()
    
    if args.resume:
        processed_ids = set(progress.get('processed', []))
        teams = [t for t in teams if t['id'] not in processed_ids]
        log.info(f"Resuming: {len(processed_ids)} already processed, {len(teams)} remaining")
    
    if args.show:
        print(f"\nTeams to process ({len(teams)}):")
        for i, team in enumerate(teams[:20]):  # Show first 20
            print(f"{i+1:3d}. {team['id']:<30} | {team['name']}")
        if len(teams) > 20:
            print(f"... and {len(teams) - 20} more")
        return
    
    print(f"Ready to process {len(teams)} teams")
    print("Use this script as a helper - the main cleanup will be done interactively")


if __name__ == '__main__':
    main()