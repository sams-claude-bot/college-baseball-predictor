#!/usr/bin/env python3
"""
Build StatBroadcast school → gid mapping from official SB index data.

Reads the scraped SB school data and matches to our team IDs using TeamResolver.
Outputs scripts/sb_group_ids.json with correct mappings.

Usage:
    python3 scripts/build_sb_mapping.py
    python3 scripts/build_sb_mapping.py --add-aliases  # Also add missing aliases to DB
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from team_resolver import TeamResolver

DB_PATH = PROJECT_ROOT / 'data' / 'baseball.db'
SB_SCHOOLS_PATH = PROJECT_ROOT / 'data' / 'sb_all_schools.json'
OUTPUT_PATH = PROJECT_ROOT / 'scripts' / 'sb_group_ids.json'

# Manual overrides for SB names that don't auto-resolve
# Value is our team_id, or None if not a D1 baseball school
MANUAL_OVERRIDES = {
    # Abbreviations
    "App State": "appalachian-state",
    "ECU": "east-carolina",
    "EKU": "eastern-kentucky",
    "ETSU": "east-tennessee-state",
    "EWU": "eastern-washington",
    "FGCU": "florida-gulf-coast",
    "FIU": "florida-international",
    "GCU": "grand-canyon",
    "LBSU": "long-beach-state",
    "MTSU": "middle-tennessee",
    "NAU": "northern-arizona",
    "NDSU": "north-dakota-state",
    "SDSU": "san-diego-state",
    "SE Missouri": "southeast-missouri",
    "SIU": "southern-illinois",
    "SMU": "smu",
    "TCU": "tcu",
    "UAB": "uab",
    "UCF": "ucf",
    "UCLA": "ucla",
    "UConn": "uconn",
    "UCSB": "uc-santa-barbara",
    "UIC": "uic",
    "ULM": "ul-monroe",
    "UNCG": "unc-greensboro",
    "UNCW": "unc-wilmington",
    "UNLV": "unlv",
    "VCU": "vcu",
    "WKU": "western-kentucky",
    "Valpo": "valparaiso",
    "Utah Valley": "utah-valley",
    "UTRGV": "utrgv",
    
    # State schools
    "Charlotte": "charlotte",
    "Coastal Carolina": "coastal-carolina",
    "Colorado State": "colorado-state",
    "Fresno State": "fresno-state",
    "Georgia Southern": "georgia-southern",
    "Georgia State": "georgia-state",
    "Idaho State": "idaho-state",
    "Iowa State": None,  # No D1 baseball
    "Jacksonville State": "jacksonville-state",
    "James Madison": "james-madison",
    "Kansas State": "kansas-state",
    "Michigan State": "michigan-state",
    "North Alabama": "north-alabama",
    "North Carolina Central": "north-carolina-central",
    "North Dakota": None,  # No D1 baseball
    "North Texas": "north-texas",
    "Northwestern University": "northwestern",
    "Ohio State": "ohio-state",
    "Old Dominion": "old-dominion",
    "Penn State": "penn-state",
    "South Alabama": "south-alabama",
    "South Dakota State": "south-dakota-state",
    "South Florida": "south-florida",
    "Southern Miss": "southern-miss",
    "Southern Utah": "southern-utah",
    "Texas A&M-Corpus Christi": "texas-aandm-corpus-christi",
    "Utah State": "utah-state",
    "Virginia Tech": "virginia-tech",
    "West Virginia": "west-virginia",
    "Western Carolina": "western-carolina",
    "Western Michigan": "western-michigan",
    
    # Others
    "Dayton": "dayton",
    "Fairfield": "fairfield",
    "Furman": "furman",
    "Hofstra": "hofstra",
    "Lipscomb": "lipscomb",
    "Marist": "marist",
    "Navy": "navy",
    "Pitt": "pittsburgh",
    "Providence": "providence",
    "Queens (N.C.)": "queens",
    "Rhode Island": "rhode-island",
    "Rice": "rice",
    "Richmond": "richmond",
    "Rutgers": "rutgers",
    "Saint Louis": "saint-louis",
    "St. Johns": "st-johns",
    "Stetson": "stetson",
    "Tulsa": "tulsa",
    "Xavier": "xavier",
    "Columbia": "columbia",
    "Cincinnati": "cincinnati",
    "Maryland": "maryland",
    "Penn": "penn",
    "Ohio": "ohio",
    "Oregon": "oregon",
    "Kansas": "kansas",
    "Utah": "utah",
    "Nevada": "nevada",
    "New Mexico": "new-mexico",
    "Washington": "washington",
    "Milwaukee": "milwaukee",
    "Michigan": "michigan",
    "Oakland": "oakland",
    "Delaware": "delaware",
    "Indiana": "indiana",
    "Indiana State": "indiana-state",
    "Purdue": "purdue",
    "Stanford": "stanford",
    
    # No D1 baseball / Not in our DB
    "Colorado": None,
    "DePaul": None,
    "East Texas A&M": None,  # Transitioning, not in DB yet
    "Marquette": None,
    "North Dakota": None,
    "UNI": None,  # Northern Iowa - no D1 baseball
    "Wisconsin": None,
    "Wyoming": None,
    "UFL": None,  # Not a school
    "Test": None,
    "Intersport": None,
    "SCLSU": None,
}


def load_sb_schools():
    """Load the scraped StatBroadcast school data."""
    with open(SB_SCHOOLS_PATH) as f:
        data = json.load(f)
    return data['schools']


def get_db_teams():
    """Get set of all team IDs in our database."""
    conn = sqlite3.connect(str(DB_PATH))
    teams = set(r[0] for r in conn.execute('SELECT id FROM teams').fetchall())
    conn.close()
    return teams


def add_alias(alias: str, team_id: str, source: str = "statbroadcast"):
    """Add an alias to the team_aliases table."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO team_aliases (alias, team_id, source) VALUES (?, ?, ?)",
            (alias.lower(), team_id, source)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"  Warning: Could not add alias '{alias}' -> {team_id}: {e}")
        return False
    finally:
        conn.close()


def build_mapping(add_aliases=False):
    """Build the team_id -> gid mapping."""
    sb_schools = load_sb_schools()
    db_teams = get_db_teams()
    resolver = TeamResolver()
    
    mapping = {}  # team_id -> gid
    extra_schools = {}  # sb_name -> gid (for non-D1 schools)
    stats = {
        'auto_matched': 0,
        'manual_matched': 0,
        'extra_schools': 0,
        'aliases_added': 0,
    }
    
    for sb_name, gid in sb_schools.items():
        # Skip non-schools
        if sb_name in ('Test', 'Intersport', 'SCLSU'):
            continue
        
        team_id = None
        
        # 1. Check manual overrides first
        if sb_name in MANUAL_OVERRIDES:
            team_id = MANUAL_OVERRIDES[sb_name]
            if team_id is None:
                # Explicitly marked as not D1 baseball
                extra_schools[sb_name] = gid
                stats['extra_schools'] += 1
                continue
            elif team_id not in db_teams:
                # Manual override points to team not in DB
                print(f"  Warning: Manual override {sb_name} -> {team_id} not in DB")
                extra_schools[sb_name] = gid
                stats['extra_schools'] += 1
                continue
            else:
                mapping[team_id] = gid
                stats['manual_matched'] += 1
                # Add alias for future auto-resolution
                if add_aliases:
                    if add_alias(sb_name, team_id):
                        stats['aliases_added'] += 1
                continue
        
        # 2. Try auto-resolution via TeamResolver
        team_id = resolver.resolve(sb_name)
        if team_id and team_id in db_teams:
            mapping[team_id] = gid
            stats['auto_matched'] += 1
            continue
        
        # 3. Not matched - put in extra schools
        extra_schools[sb_name] = gid
        stats['extra_schools'] += 1
    
    return mapping, extra_schools, stats


def main():
    parser = argparse.ArgumentParser(description='Build SB school mapping')
    parser.add_argument('--add-aliases', action='store_true',
                        help='Add missing aliases to team_aliases table')
    parser.add_argument('--dry-run', action='store_true',
                        help="Don't write output file")
    args = parser.parse_args()
    
    print("Building StatBroadcast school mapping...")
    print(f"  Source: {SB_SCHOOLS_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print()
    
    mapping, extra_schools, stats = build_mapping(add_aliases=args.add_aliases)
    
    # Build output structure
    output = dict(sorted(mapping.items()))
    output['_extra_schools'] = dict(sorted(extra_schools.items()))
    
    if not args.dry_run:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Wrote mapping to {OUTPUT_PATH}")
    else:
        print("(dry run - no file written)")
    
    print()
    print("=== Summary ===")
    print(f"Auto-matched:    {stats['auto_matched']}")
    print(f"Manual-matched:  {stats['manual_matched']}")
    print(f"Extra schools:   {stats['extra_schools']}")
    print(f"Total mapped:    {stats['auto_matched'] + stats['manual_matched']}")
    if args.add_aliases:
        print(f"Aliases added:   {stats['aliases_added']}")
    
    # Spot check
    print()
    print("=== Spot Check ===")
    for team in ['alabama', 'lsu', 'florida', 'texas', 'vanderbilt']:
        if team in mapping:
            print(f"  {team} -> {mapping[team]}")
        else:
            print(f"  {team} -> NOT FOUND")


if __name__ == '__main__':
    main()
