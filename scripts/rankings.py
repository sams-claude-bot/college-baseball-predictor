#!/usr/bin/env python3
"""
Manage Top 25 rankings

Teams entering the Top 25 are automatically tracked.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Import database-backed team resolver
sys.path.insert(0, str(Path(__file__).parent))
from team_resolver import resolve_team as db_resolve_team

from scripts.database import (
    get_connection, add_ranking, get_current_top_25, 
    get_ranking_history, init_rankings_table, add_team
)

# Team aliases are now in the database (team_aliases table)
# Use: python3 scripts/team_resolver.py --add 'alias' team-id d1baseball

def normalize_team_id(name):
    """Convert team name to ID format using database resolver"""
    name_lower = name.lower().strip()
    
    # Use database resolver
    result = db_resolve_team(name_lower)
    if result:
        return result
    
    # Fallback: slugify
    return name_lower.replace(" ", "-").replace("(", "").replace(")", "")

def set_rankings(rankings_list, poll="d1baseball", week=None):
    """
    Set full Top 25 rankings
    
    rankings_list: List of team names in rank order (1-25)
    """
    init_rankings_table()
    date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📊 Setting {poll} Top 25 (Week {week or 'Preseason'}):")
    print("-" * 40)
    
    for rank, team_name in enumerate(rankings_list, 1):
        team_id = normalize_team_id(team_name)
        
        # Ensure team exists with proper name
        add_team(team_id, team_name)
        
        # Add ranking
        add_ranking(team_id, rank, poll, week, date)
        print(f"  {rank:2}. {team_name}")
    
    print("-" * 40)
    print(f"✓ Set {len(rankings_list)} rankings")

def add_single_ranking(team_name, rank, poll="d1baseball", week=None):
    """Add or update a single team's ranking"""
    init_rankings_table()
    
    team_id = normalize_team_id(team_name)
    add_team(team_id, team_name)
    add_ranking(team_id, rank, poll, week)
    
    print(f"✓ {team_name} ranked #{rank}")

def show_top_25():
    """Display current Top 25"""
    teams = get_current_top_25()
    
    if not teams:
        print("\n⚠️  No rankings set yet")
        print("Use: python rankings.py set <team1> <team2> ... (in order)")
        return
    
    print("\n📊 Current Top 25:")
    print("-" * 40)
    for t in teams:
        conf = f" ({t['conference']})" if t.get('conference') else ""
        print(f"  {t['current_rank']:2}. {t['name']}{conf}")

def show_team_history(team_name):
    """Show ranking history for a team"""
    team_id = normalize_team_id(team_name)
    history = get_ranking_history(team_id)
    
    if not history:
        print(f"\n⚠️  No ranking history for {team_name}")
        return
    
    print(f"\n📈 Ranking History: {team_name}")
    print("-" * 40)
    for h in history:
        week_str = f"Week {h['week']}" if h.get('week') else "Preseason"
        print(f"  {h['date']}: #{h['rank']} ({h['poll']} {week_str})")

def fetch_and_set_preseason():
    """
    Preseason 2026 D1Baseball Top 25 (manually entered based on available info)
    Update this when we can scrape or get official rankings
    """
    # Based on NCAA.com article mentioning UCLA as #1
    # This is a placeholder - update with actual preseason rankings
    preseason_2026 = [
        "UCLA",
        "Texas A&M", 
        "Florida",
        "Arkansas",
        "LSU",
        "Tennessee",
        "Georgia",
        "Texas",
        "Vanderbilt",
        "Florida State",
        "Wake Forest",
        "Virginia",
        "Oregon State",
        "NC State",
        "Stanford",
        "Ole Miss",
        "Clemson",
        "Miami (FL)",
        "Louisville",
        "Mississippi State",
        "South Carolina",
        "TCU",
        "North Carolina",
        "Arizona",
        "Kentucky"
    ]
    
    set_rankings(preseason_2026, poll="d1baseball", week=0)
    print("\n⚠️  Note: These are estimated preseason rankings.")
    print("    Update when official rankings are available.")

def main():
    if len(sys.argv) < 2:
        show_top_25()
        print("\nUsage:")
        print("  python rankings.py                    - Show current Top 25")
        print("  python rankings.py preseason          - Set preseason rankings")
        print("  python rankings.py set <t1> <t2> ...  - Set Top 25 in order")
        print("  python rankings.py add <team> <rank>  - Add single ranking")
        print("  python rankings.py history <team>     - Show team's ranking history")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "preseason":
        fetch_and_set_preseason()
    
    elif cmd == "set":
        if len(sys.argv) < 4:
            print("Usage: python rankings.py set <team1> <team2> ... (up to 25)")
            return
        teams = sys.argv[2:]
        set_rankings(teams)
    
    elif cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: python rankings.py add <team_name> <rank>")
            return
        team = sys.argv[2]
        rank = int(sys.argv[3])
        add_single_ranking(team, rank)
    
    elif cmd == "history":
        if len(sys.argv) < 3:
            print("Usage: python rankings.py history <team_name>")
            return
        show_team_history(sys.argv[2])
    
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
