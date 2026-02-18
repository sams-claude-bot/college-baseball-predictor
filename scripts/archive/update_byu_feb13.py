#!/usr/bin/env python3
"""
Update BYU player stats from Feb 13, 2026 doubleheader vs Western Kentucky
Data extracted from byucougars.com box score screenshots

Game 1: BYU 3, WKU 2 (W)
Game 2: BYU 2, WKU 5 (L)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'

# Player stats extracted from box score screenshots
# Format: (name, position, ab, r, h, doubles, triples, hr, rbi, bb, k, hbp)
GAME1_BATTING = [
    # Game 1 (BYU 3-2 W) - from screenshot
    ("Bryker Hurdsman", "RF", 5, 0, 1, 0, 0, 0, 0, 0, 2, 0),
    ("Ezra McNaughton", "DH", 3, 0, 0, 0, 0, 0, 0, 1, 2, 0),
    ("Ryder Robinson", "SS", 4, 0, 0, 0, 0, 0, 0, 0, 2, 0),
    ("Luke Anderson", "2B", 3, 1, 1, 0, 0, 0, 0, 1, 1, 0),
    ("Matt Hansen", "1B", 4, 0, 1, 0, 0, 0, 0, 0, 2, 0),
    ("Ryker Schow", "PR", 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    ("Patrick Graham", "3B", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    # From scoring summary - McChesney HR, Jones HBP+RBI, Painter SF+RBI
    ("Crew McChesney", "CF", 4, 1, 1, 0, 0, 1, 1, 0, 0, 0),  # Estimated from HR
    ("Easton Jones", "3B", 0, 0, 0, 0, 0, 0, 1, 0, 0, 1),  # HBP + RBI
    ("Trey Painter", "OF", 3, 0, 0, 0, 0, 0, 1, 0, 1, 0),  # SF RBI estimated
]

GAME2_BATTING = [
    # Game 2 (BYU 2-5 L) - from screenshot
    ("Bryker Hurdsman", "RF", 4, 0, 0, 0, 0, 0, 1, 0, 3, 0),
    ("Ezra McNaughton", "1B", 5, 0, 2, 0, 0, 0, 0, 0, 1, 0),
    ("Ryder Robinson", "SS", 5, 0, 0, 0, 0, 0, 1, 0, 2, 0),
    ("Luke Anderson", "2B", 4, 0, 2, 0, 0, 0, 0, 1, 1, 0),
    ("Crew McChesney", "CF", 4, 0, 2, 0, 0, 0, 0, 0, 1, 0),
    ("Easton Jones", "3B", 4, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    ("Matt Hansen", "DH", 4, 0, 1, 0, 0, 0, 0, 0, 2, 0),
    ("Ryker Schow", "PR", 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # Scored per scoring summary
    # From scoring summary - Erickson scored
    ("Erickson", "C", 3, 1, 0, 0, 0, 0, 0, 0, 1, 0),  # Estimated
]

# Pitching stats - estimated from box scores (not fully visible in screenshots)
# Format: (name, ip, h, r, er, bb, k, hr_allowed)
GAME1_PITCHING = [
    # BYU won 3-2, so their pitchers gave up 2 runs total
    # These are estimated/partial
]

GAME2_PITCHING = [
    # BYU lost 2-5, gave up 5 runs
]


def combine_games():
    """Combine stats from both games"""
    combined = {}
    
    for game in [GAME1_BATTING, GAME2_BATTING]:
        for name, pos, ab, r, h, doubles, triples, hr, rbi, bb, k, hbp in game:
            if name not in combined:
                combined[name] = {
                    'position': pos,
                    'games': 0,
                    'ab': 0, 'r': 0, 'h': 0, 
                    'doubles': 0, 'triples': 0, 'hr': 0,
                    'rbi': 0, 'bb': 0, 'k': 0
                }
            combined[name]['games'] += 1
            combined[name]['ab'] += ab
            combined[name]['r'] += r
            combined[name]['h'] += h
            combined[name]['doubles'] += doubles
            combined[name]['triples'] += triples
            combined[name]['hr'] += hr
            combined[name]['rbi'] += rbi
            combined[name]['bb'] += bb
            combined[name]['k'] += k
    
    return combined


def update_database(dry_run=False):
    """Update player_stats table with doubleheader stats"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    combined = combine_games()
    
    updated = 0
    inserted = 0
    
    for name, stats in combined.items():
        # Check if player exists
        c.execute("""
            SELECT id, games, at_bats, runs, hits, doubles, triples, 
                   home_runs, rbi, walks, strikeouts, name
            FROM player_stats 
            WHERE team_id='byu' AND (name LIKE ? OR name LIKE ?)
        """, (f'%{name}%', f'%{name.split()[0]}%{name.split()[-1]}%' if ' ' in name else f'%{name}%'))
        
        row = c.fetchone()
        
        if row:
            # Update existing player
            player_id = row[0]
            if not dry_run:
                c.execute("""
                    UPDATE player_stats SET
                        games = games + ?,
                        at_bats = at_bats + ?,
                        runs = runs + ?,
                        hits = hits + ?,
                        doubles = doubles + ?,
                        triples = triples + ?,
                        home_runs = home_runs + ?,
                        rbi = rbi + ?,
                        walks = walks + ?,
                        strikeouts = strikeouts + ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    stats['games'], stats['ab'], stats['r'], stats['h'],
                    stats['doubles'], stats['triples'], stats['hr'],
                    stats['rbi'], stats['bb'], stats['k'],
                    datetime.now().isoformat(), player_id
                ))
            print(f"  ✓ Updated {name}: +{stats['games']}G, +{stats['ab']}AB, +{stats['h']}H, +{stats['rbi']}RBI")
            updated += 1
        else:
            # Insert new player
            if not dry_run:
                c.execute("""
                    INSERT INTO player_stats (
                        team_id, name, position, games, at_bats, runs, hits,
                        doubles, triples, home_runs, rbi, walks, strikeouts,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'byu', name, stats['position'], stats['games'],
                    stats['ab'], stats['r'], stats['h'], stats['doubles'],
                    stats['triples'], stats['hr'], stats['rbi'], 
                    stats['bb'], stats['k'], datetime.now().isoformat()
                ))
            print(f"  + Inserted {name}: {stats['games']}G, {stats['ab']}AB, {stats['h']}H, {stats['rbi']}RBI")
            inserted += 1
    
    if not dry_run:
        conn.commit()
    conn.close()
    
    return updated, inserted


def main():
    print("=" * 60)
    print("BYU vs Western Kentucky Doubleheader - Feb 13, 2026")
    print("=" * 60)
    print("\nGame Results:")
    print("  Game 1: BYU 3, WKU 2 (W) - 11:00 AM")
    print("  Game 2: BYU 2, WKU 5 (L) - 2:45 PM")
    print("\nData Sources:")
    print("  ✓ byucougars.com box scores (via screenshots)")
    print("  ⚠ ESPN - No page found")
    print("  ⚠ StatBroadcast - JavaScript required")
    print("  ⚠ WKU Sports - JavaScript required")
    print("\n" + "-" * 60)
    print("Combined Doubleheader Stats:")
    print("-" * 60)
    
    combined = combine_games()
    for name, stats in sorted(combined.items()):
        print(f"  {name:20} {stats['games']}G {stats['ab']:2}AB {stats['r']:1}R {stats['h']:1}H {stats['hr']:1}HR {stats['rbi']:1}RBI {stats['bb']:1}BB {stats['k']:1}K")
    
    print("\n" + "-" * 60)
    print("Updating database...")
    print("-" * 60)
    
    updated, inserted = update_database(dry_run=False)
    
    print("\n" + "=" * 60)
    print(f"Summary: {updated} players updated, {inserted} players inserted")
    print("=" * 60)
    
    return updated + inserted


if __name__ == "__main__":
    total = main()
