#!/usr/bin/env python3
"""Helper script for NCAA stats scraping."""

import json
import sys
from pathlib import Path

STAT_CODES = {
    'era': 211,
    'whip': 597,
    'k_per_9': 425,
    'bb_per_9': 509,
    'hits_per_9': 506,
    'batting_avg': 210,
    'obp': 504,
    'slugging': 327,
    'hr_per_game': 323,
    'scoring': 213,
    'fielding_pct': 212,
    'dp_per_game': 328,
    'k_bb_ratio': 591
}

def add_stat(data_file: str, stat_name: str, teams_json: str):
    """Add scraped stat data to the raw file."""
    data_path = Path(data_file)
    
    # Load existing data
    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
    else:
        data = {"season": 2025, "ranking_period": 104, "stats": {}}
    
    # Parse teams data
    teams = json.loads(teams_json)
    
    # Add stat
    data["stats"][stat_name] = teams
    
    # Save
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added {stat_name}: {len(teams)} teams")
    return len(teams)

def convert_to_final(raw_file: str, output_file: str, season: int):
    """Convert raw scraped data to final format for loading."""
    with open(raw_file) as f:
        raw = json.load(f)
    
    final = []
    for stat_name, teams in raw.get("stats", {}).items():
        final.append({
            "stat_name": stat_name,
            "season": season,
            "teams": teams
        })
    
    with open(output_file, 'w') as f:
        json.dump(final, f, indent=2)
    
    print(f"Converted {len(final)} stats to {output_file}")
    return len(final)

def main():
    if len(sys.argv) < 2:
        print("Usage: ncaa_scrape_helper.py <command> [args]")
        print("Commands:")
        print("  add <data_file> <stat_name> <teams_json>")
        print("  convert <raw_file> <output_file> <season>")
        print("  codes - List stat codes")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "add":
        if len(sys.argv) < 5:
            print("Usage: add <data_file> <stat_name> <teams_json>")
            sys.exit(1)
        add_stat(sys.argv[2], sys.argv[3], sys.argv[4])
    
    elif cmd == "convert":
        if len(sys.argv) < 5:
            print("Usage: convert <raw_file> <output_file> <season>")
            sys.exit(1)
        convert_to_final(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    
    elif cmd == "codes":
        for name, code in STAT_CODES.items():
            print(f"  {name}: {code}")
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
