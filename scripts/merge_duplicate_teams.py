#!/usr/bin/env python3
"""
Merge duplicate teams created by schedule fetcher.

This script finds teams with mascot suffixes (e.g., coastal-carolina-chanticleers)
and merges them into the base team (e.g., coastal-carolina).
"""

import sqlite3
import re
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'

# Common mascot suffixes to remove
MASCOTS = [
    'aggies', 'anteaters', 'aztecs', 'badgers', 'banana-slugs', 'bears', 'beavers',
    'bearcats', 'bengals', 'billikens', 'bison', 'black-bears', 'blazers', 'blue-devils',
    'blue-hens', 'bobcats', 'boilermakers', 'braves', 'broncos', 'bruins', 'buccaneers',
    'buckeyes', 'buffaloes', 'bulldogs', 'bulls', 'cadets', 'camels', 'cardinals',
    'catamounts', 'cavaliers', 'chanticleers', 'chippewas', 'colonels', 'commodores',
    'cornhuskers', 'cougars', 'cowboys', 'crimson', 'crimson-tide', 'crusaders',
    'cyclones', 'deacons', 'demon-deacons', 'dirtbags', 'dolphins', 'dons', 'ducks',
    'dukes', 'eagles', 'explorers', 'falcons', 'fighting', 'fighting-camels',
    'fighting-illini', 'fighting-irish', 'flames', 'flyers', 'friars', 'gaels',
    'gamecocks', 'gators', 'golden', 'golden-bears', 'golden-eagles', 'golden-flash',
    'golden-flashes', 'golden-grizzlies', 'golden-hurricanes', 'governors', 'great-danes',
    'green-wave', 'grizzlies', 'hawks', 'hilltoppers', 'hokies', 'hoosiers', 'hornets',
    'hoyas', 'huskies', 'hurricanes', 'illini', 'irish', 'islanders', 'jaguars',
    'jaspers', 'jayhawks', 'knights', 'lancers', 'leathernecks', 'lions', 'lobos',
    'longhorns', 'lumberjacks', 'matadors', 'mavericks', 'mean-green', 'midshipmen',
    'miners', 'minutemen', 'monarchs', 'mountaineers', 'musketeers', 'mustangs',
    'nittany-lions', 'norse', 'orange', 'orangemen', 'owls', 'paladins', 'panthers',
    'patriots', 'peacocks', 'penguins', 'phoenix', 'pilots', 'pioneers', 'pirates',
    'privateers', 'purple', 'purple-aces', 'purple-eagles', 'quakers', 'raiders',
    'ragin-cajuns', 'rainbow-warriors', 'rams', 'rattlers', 'razorbacks', 'rebels',
    'red', 'red-raiders', 'red-storm', 'red-wolves', 'redhawks', 'retrievers',
    'revolutionaries', 'roadrunners', 'rockets', 'royals', 'running-rebels',
    'salukis', 'scarlet', 'scarlet-knights', 'seahawks', 'seawolves', 'seminoles',
    'shockers', 'skyhawks', 'sooners', 'spartans', 'spiders', 'stags', 'sun-devils',
    'sun-tigers', 'tar-heels', 'terrapins', 'terriers', 'texans', 'thundering-herd',
    'tigers', 'titans', 'toreros', 'trojans', 'tribe', 'tritons', 'utes',
    'vandals', 'vaqueros', 'vikings', 'volunteers', 'warhawks', 'warriors', 'waves',
    'wildcats', 'wolfpack', 'wolverines', 'wolves', 'yellow-jackets', 'zips'
]

def find_base_team(team_id, conn):
    """Find the base team ID without mascot suffix."""
    # Try removing each mascot suffix
    for mascot in sorted(MASCOTS, key=len, reverse=True):  # Longer first
        suffix = f'-{mascot}'
        if team_id.endswith(suffix):
            base_id = team_id[:-len(suffix)]
            # Check if base team exists
            result = conn.execute(
                "SELECT id FROM teams WHERE id = ?", (base_id,)
            ).fetchone()
            if result:
                return base_id
    return None


def merge_teams():
    """Find and merge duplicate teams."""
    conn = sqlite3.connect(str(DB_PATH))
    
    # Find all teams with Unknown conference (likely duplicates)
    duplicates = conn.execute("""
        SELECT id, name FROM teams 
        WHERE conference = 'Unknown'
        ORDER BY id
    """).fetchall()
    
    print(f"Found {len(duplicates)} potential duplicate teams")
    
    merged = 0
    for dup_id, dup_name in duplicates:
        base_id = find_base_team(dup_id, conn)
        
        if base_id:
            print(f"  Merging {dup_id} → {base_id}")
            
            # Update games to use base team
            conn.execute(
                "UPDATE games SET home_team_id = ? WHERE home_team_id = ?",
                (base_id, dup_id)
            )
            conn.execute(
                "UPDATE games SET away_team_id = ? WHERE away_team_id = ?",
                (base_id, dup_id)
            )
            
            # Delete the duplicate team
            conn.execute("DELETE FROM teams WHERE id = ?", (dup_id,))
            merged += 1
    
    conn.commit()
    conn.close()
    
    print(f"\nMerged {merged} duplicate teams")
    return merged


def main():
    merge_teams()
    
    # Verify results
    conn = sqlite3.connect(str(DB_PATH))
    remaining = conn.execute("""
        SELECT COUNT(*) FROM teams WHERE conference = 'Unknown'
    """).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    conn.close()
    
    print(f"\nResults: {total} total teams, {remaining} still with Unknown conference")


if __name__ == '__main__':
    main()
