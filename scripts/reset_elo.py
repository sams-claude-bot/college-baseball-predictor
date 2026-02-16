#!/usr/bin/env python3
"""
Reset and recalculate Elo ratings from scratch with conference-tiered starting values.

Tiers based on historical D1 baseball conference strength:
  Tier 1 (P4): SEC, ACC, Big 12, Big Ten — 1550-1600
  Tier 2 (Strong mid-major): AAC, Sun Belt, C-USA, MWC, Big East, WCC — 1480-1520
  Tier 3 (Mid-major): A-10, CAA, MVC, SoCon, ASUN, Big West, MAC — 1440-1480
  Tier 4 (Lower): OVC, Southland, Summit, WAC, Big South, NEC, Patriot, Horizon, America East — 1400-1440
  Tier 5 (HBCU/Ivy): MEAC, SWAC, Ivy — 1360-1400
  Unknown/Independent: 1450

Usage:
    python3 scripts/reset_elo.py              # Reset and recalculate
    python3 scripts/reset_elo.py --dry-run    # Show what would happen
"""

import sys
import math
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'

# Conference -> Starting Elo
CONFERENCE_ELO = {
    # Tier 1: Power 4
    'SEC': 1600,
    'ACC': 1580,
    'Big 12': 1560,
    'Big Ten': 1550,
    
    # Tier 2: Strong mid-major
    'AAC': 1520,
    'Sun Belt': 1510,
    'C-USA': 1500,
    'MWC': 1490,
    'Big East': 1490,
    'WCC': 1480,
    
    # Tier 3: Mid-major
    'A-10': 1470,
    'CAA': 1470,
    'MVC': 1460,
    'SoCon': 1460,
    'ASUN': 1460,
    'Big West': 1460,
    'MAC': 1450,
    
    # Tier 4: Lower D1
    'OVC': 1430,
    'Southland': 1430,
    'Summit': 1420,
    'Summit League': 1420,
    'WAC': 1420,
    'Big South': 1420,
    'NEC': 1410,
    'Patriot': 1410,
    'Horizon': 1410,
    'America East': 1400,
    
    # Tier 5: HBCU / Ivy
    'MEAC': 1380,
    'SWAC': 1370,
    'Ivy': 1400,
    
    # Independent
    'Independent': 1450,
}

DEFAULT_ELO = 1450  # Unknown teams

# Elo parameters
K_FACTOR = 32
HOME_ADVANTAGE = 50


def get_starting_elo(conference):
    """Get starting Elo for a conference."""
    return CONFERENCE_ELO.get(conference, DEFAULT_ELO)


def update_elo(home_rating, away_rating, home_won, margin, home_advantage=HOME_ADVANTAGE):
    """Calculate new Elo ratings after a game."""
    expected_home = 1.0 / (1.0 + 10 ** (-(home_rating - away_rating + home_advantage) / 400))
    actual_home = 1.0 if home_won else 0.0
    
    # Margin of victory multiplier (capped)
    mov_mult = math.log(max(margin, 1) + 1) * 0.8
    mov_mult = min(mov_mult, 2.0)
    
    k = K_FACTOR * mov_mult if margin > 0 else K_FACTOR
    
    delta = k * (actual_home - expected_home)
    return home_rating + delta, away_rating - delta


def main():
    dry_run = '--dry-run' in sys.argv
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # Step 1: Initialize all teams with conference-tiered Elo
    teams = conn.execute("""
        SELECT id, conference FROM teams
    """).fetchall()
    
    ratings = {}
    for t in teams:
        conf = t['conference'] or ''
        ratings[t['id']] = get_starting_elo(conf)
    
    # Count by tier
    tier_counts = {}
    for t in teams:
        elo = ratings[t['id']]
        tier_counts[elo] = tier_counts.get(elo, 0) + 1
    
    print(f"Initialized {len(ratings)} teams with tiered Elo")
    print(f"  P4 range: 1550-1600 ({sum(1 for t in teams if ratings[t['id']] >= 1550)} teams)")
    print(f"  Mid-major range: 1450-1520 ({sum(1 for t in teams if 1450 <= ratings[t['id']] < 1550)} teams)")
    print(f"  Lower range: 1370-1450 ({sum(1 for t in teams if ratings[t['id']] < 1450)} teams)")
    
    # Step 2: Get all completed games in chronological order
    games = conn.execute("""
        SELECT id, date, home_team_id, away_team_id, home_score, away_score, is_neutral_site
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date ASC, id ASC
    """).fetchall()
    
    print(f"\nProcessing {len(games)} completed games chronologically...")
    
    # Step 3: Process each game
    for g in games:
        home = g['home_team_id']
        away = g['away_team_id']
        home_score = g['home_score']
        away_score = g['away_score']
        neutral = g['is_neutral_site'] or 0
        
        # Initialize unknown teams
        if home not in ratings:
            ratings[home] = DEFAULT_ELO
        if away not in ratings:
            ratings[away] = DEFAULT_ELO
        
        home_won = home_score > away_score
        margin = abs(home_score - away_score)
        ha = 0 if neutral else HOME_ADVANTAGE
        
        new_home, new_away = update_elo(ratings[home], ratings[away], home_won, margin, ha)
        ratings[home] = new_home
        ratings[away] = new_away
    
    # Step 4: Show results
    # Get top/bottom teams
    team_names = {t['id']: t for t in conn.execute("SELECT id, name, conference FROM teams").fetchall()}
    sorted_teams = sorted(ratings.items(), key=lambda x: -x[1])
    
    print(f"\nTop 25 Elo ratings:")
    print(f"{'#':>3} {'Team':<25} {'Conf':<10} {'Elo':>6} {'Δ':>6}")
    print("-" * 55)
    for i, (tid, elo) in enumerate(sorted_teams[:25], 1):
        t = team_names.get(tid)
        name = t['name'] if t else tid
        conf = t['conference'] if t else '?'
        start = get_starting_elo(conf) if conf else DEFAULT_ELO
        delta = elo - start
        print(f"{i:>3} {name:<25} {conf:<10} {elo:>6.0f} {delta:>+6.0f}")
    
    print(f"\nConference averages:")
    conf_elos = {}
    for tid, elo in ratings.items():
        t = team_names.get(tid)
        if t and t['conference']:
            conf_elos.setdefault(t['conference'], []).append(elo)
    
    for conf, elos in sorted(conf_elos.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(elos) / len(elos)
        print(f"  {conf:<15} {avg:>6.0f} ({len(elos)} teams)")
    
    if dry_run:
        print("\n[DRY RUN] No changes written.")
        conn.close()
        return
    
    # Step 5: Write to database
    conn.execute("DELETE FROM elo_ratings")
    for tid, elo in ratings.items():
        conn.execute("""
            INSERT OR REPLACE INTO elo_ratings (team_id, rating, games_played, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (tid, round(elo, 1), 0))
    
    conn.commit()
    print(f"\n✅ Wrote {len(ratings)} Elo ratings to database")
    conn.close()


if __name__ == '__main__':
    main()
