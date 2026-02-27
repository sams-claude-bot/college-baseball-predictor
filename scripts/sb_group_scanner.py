#!/usr/bin/env python3
"""
StatBroadcast Group ID Scanner
Scans event IDs to build a mapping of StatBroadcast group IDs to team names.
"""

import urllib.request
import codecs
import base64
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add scripts dir to path for team_resolver
sys.path.insert(0, str(Path(__file__).parent))
from team_resolver import TeamResolver

OUTPUT_FILE = Path(__file__).parent / "sb_group_ids.json"

def decode_sb_response(raw: str) -> str:
    """Decode StatBroadcast ROT13+base64 response."""
    rot13 = codecs.decode(raw, 'rot_13')
    return base64.b64decode(rot13).decode('utf-8')

def fetch_event(event_id: int):
    """Fetch and parse a single event. Returns (groupid, homename, visitorname) or None."""
    url = f"https://stats.statbroadcast.com/interface/webservice/event/{event_id}?data=dHlwZT1zdGF0YnJvYWRjYXN0"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode('utf-8')
            xml = decode_sb_response(raw)
            
            # Only process baseball events (bsgame + gender M)
            if '<sport>bsgame</sport>' not in xml or '<gender>M</gender>' not in xml:
                return None
            
            # Extract groupid
            gid_match = re.search(r'groupid="([^"]+)"', xml)
            home_match = re.search(r'homename="([^"]+)"', xml)
            visitor_match = re.search(r'visitorname="([^"]+)"', xml)
            
            if gid_match and home_match:
                groupid = gid_match.group(1)
                homename = home_match.group(1)
                visitorname = visitor_match.group(1) if visitor_match else None
                return (groupid, homename, visitorname)
    except Exception:
        pass
    return None

def scan_range(start: int, end: int, batch_size: int = 15, delay: float = 0.1):
    """Scan a range of event IDs and collect groupid → team name mappings."""
    mappings = {}  # groupid → set of team names seen
    processed = 0
    baseball_found = 0
    
    print(f"Scanning events {start} to {end}...")
    
    event_ids = list(range(start, end))
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_event, eid): eid for eid in event_ids}
        
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            
            if result:
                groupid, homename, visitorname = result
                baseball_found += 1
                
                if groupid not in mappings:
                    mappings[groupid] = set()
                mappings[groupid].add(homename)
                # Also track visitor for cross-reference
                if visitorname:
                    # Visitor team might appear as home in other events
                    pass
                
            if processed % 500 == 0:
                print(f"  Progress: {processed}/{len(event_ids)} events, {baseball_found} baseball, {len(mappings)} unique groups")
    
    print(f"  Done: {processed} events, {baseball_found} baseball, {len(mappings)} unique groups")
    return mappings

def main():
    print("=== StatBroadcast Group ID Scanner ===\n")
    
    # Load existing mappings
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing mappings")
    
    # Initialize resolver
    resolver = TeamResolver()
    
    # Scan multiple ranges - recent first
    all_gid_to_names = {}
    
    ranges = [
        (652000, 654500),   # Most recent (Feb 2026)
        (649000, 652000),   # Earlier Feb 2026
        (645000, 649000),   # Jan/Feb 2026
        (640000, 645000),   # Dec/Jan
        (635000, 640000),   # Fall 2025
        (630000, 635000),   # Earlier fall
        (620000, 630000),   # Summer/early fall
        (610000, 620000),   # Spring 2025
        (600000, 610000),   # Earlier 2025
    ]
    
    for start, end in ranges:
        mappings = scan_range(start, end)
        for gid, names in mappings.items():
            if gid not in all_gid_to_names:
                all_gid_to_names[gid] = set()
            all_gid_to_names[gid].update(names)
        
        # Brief pause between ranges
        time.sleep(0.5)
    
    print(f"\n=== Results ===")
    print(f"Found {len(all_gid_to_names)} unique StatBroadcast groups")
    
    # Resolve to team IDs
    new_mappings = {}  # team_id → gid
    unresolved = []    # (gid, team_names)
    
    for gid, names in all_gid_to_names.items():
        resolved = False
        for name in names:
            team_id = resolver.resolve(name)
            if team_id:
                new_mappings[team_id] = gid
                resolved = True
                break
        if not resolved:
            unresolved.append((gid, list(names)))
    
    print(f"Resolved {len(new_mappings)} groups to team IDs")
    print(f"Unresolved: {len(unresolved)} groups")
    
    # Merge with existing (new mappings take precedence, but keep existing that aren't overwritten)
    final = existing.copy()
    for team_id, gid in new_mappings.items():
        final[team_id] = gid
    
    # Sort alphabetically
    final = dict(sorted(final.items()))
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final, f, indent=4)
    
    print(f"\nSaved {len(final)} mappings to {OUTPUT_FILE}")
    
    # Report unresolved
    if unresolved:
        print(f"\n=== Unresolved Groups (need manual alias or not D1) ===")
        for gid, names in sorted(unresolved, key=lambda x: x[0]):
            print(f"  {gid}: {names}")
    
    # Check for unmapped teams
    with open('/tmp/all_teams.txt') as f:
        all_teams = [line.strip() for line in f if line.strip()]
    
    unmapped = [t for t in all_teams if t not in final]
    print(f"\n=== Unmapped Teams ({len(unmapped)}/{len(all_teams)}) ===")
    for t in unmapped:
        print(f"  {t}")
    
    # Verify key mappings haven't changed
    print(f"\n=== Verification ===")
    checks = {
        'washington-state': 'wsu',
        'byu': 'byu',
        'california-baptist': 'calb',
        'san-francisco': 'sanf',
    }
    for team_id, expected in checks.items():
        actual = final.get(team_id)
        status = "✓" if actual == expected else f"✗ (got {actual})"
        print(f"  {team_id}: {status}")

if __name__ == "__main__":
    main()
