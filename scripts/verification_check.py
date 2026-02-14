#!/usr/bin/env python3
"""
Post-Collection Verification Script

Compares our database against ESPN scoreboard to catch:
- Missing games (ESPN has them, we don't)
- Missing scores (game completed but we show pending)
- Score mismatches (our score != ESPN score)

Run after any collection to verify data integrity.

Usage:
    python3 verification_check.py           # Check today
    python3 verification_check.py 2026-02-13  # Check specific date
    python3 verification_check.py --fix     # Auto-fix found issues
"""

import json
import re
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.database import get_connection


def fetch_espn_scoreboard(date_str: str) -> list:
    """
    Fetch completed games from ESPN for a given date.
    Returns list of game dicts with teams and scores.
    """
    # ESPN uses YYYYMMDD format
    espn_date = date_str.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={espn_date}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch ESPN scoreboard: {e}")
        return []
    
    games = []
    for event in data.get("events", []):
        game = {
            "espn_id": event.get("id"),
            "status": event.get("status", {}).get("type", {}).get("name", "unknown"),
            "home_team": None,
            "away_team": None,
            "home_score": None,
            "away_score": None,
            "completed": False
        }
        
        # Check if game is final
        status_type = event.get("status", {}).get("type", {})
        game["completed"] = status_type.get("completed", False)
        
        # Get team info
        competitions = event.get("competitions", [{}])
        if competitions:
            for competitor in competitions[0].get("competitors", []):
                team_name = competitor.get("team", {}).get("displayName", "")
                score = competitor.get("score", "")
                home_away = competitor.get("homeAway", "")
                
                # Try to get numeric score
                try:
                    score_num = int(score) if score else None
                except ValueError:
                    score_num = None
                
                if home_away == "home":
                    game["home_team"] = team_name
                    game["home_score"] = score_num
                else:
                    game["away_team"] = team_name
                    game["away_score"] = score_num
        
        games.append(game)
    
    return games


def normalize_team_name(name: str) -> str:
    """Normalize team name for comparison."""
    if not name:
        return ""
    # Remove common suffixes
    name = re.sub(r'\s+(Bulldogs?|Tigers?|Gators?|Crimson Tide|Volunteers?|Razorbacks?|Wildcats?|Bears?|Commodores?|Rebels?|Aggies|Gamecocks?|Hogs|Cardinals?|Blue Devils?|Cavaliers?|Yellow Jackets?|Hokies|Demon Deacons?|Wolfpack|Tar Heels?|Seminoles?|Hurricanes?|Fighting Irish|Panthers?|Orange|Mountaineers?|Spartans?|Buckeyes?|Badgers?|Hawkeyes?|Golden Gophers?|Nittany Lions?|Boilermakers?|Hoosiers?|Illini|Cornhuskers?|Terrapins?|Scarlet Knights?|Wolverines?|Jayhawks?|Longhorns?|Sooners?|Cyclones?|Cowboys?|Horned Frogs?|Red Raiders?|Beavers?|Ducks?|Cougars?|Huskies?|Trojans?|Sun Devils?|Wildcats?|Bruins?|Utes?|Buffaloes?)$', '', name, flags=re.IGNORECASE)
    
    # Common abbreviations
    replacements = {
        "Mississippi State": "Miss State",
        "Texas A&M": "Texas A&M",
        "South Carolina": "S Carolina",
        "North Carolina": "UNC",
    }
    
    return name.strip().lower()


def match_teams(espn_name: str, db_name: str) -> bool:
    """Check if ESPN team name matches our DB team name."""
    if not espn_name or not db_name:
        return False
    
    espn_norm = normalize_team_name(espn_name)
    db_norm = normalize_team_name(db_name)
    
    # Exact match
    if espn_norm == db_norm:
        return True
    
    # One contains the other
    if espn_norm in db_norm or db_norm in espn_norm:
        return True
    
    # Check for common variations
    variations = {
        "mississippi state": ["miss state", "ms state", "msu", "mississippi st"],
        "mississippi": ["ole miss"],
        "florida": ["uf", "gators"],
        "alabama": ["bama", "ua"],
        "georgia": ["uga"],
        "lsu": ["louisiana state"],
        "texas a&m": ["tamu", "a&m"],
        "auburn": ["au"],
        "tennessee": ["vols", "ut"],
        "kentucky": ["uk"],
        "arkansas": ["ark"],
        "south carolina": ["s carolina", "sc", "usc"],
        "vanderbilt": ["vandy"],
        "texas": ["ut", "longhorns"],
    }
    
    for canonical, alts in variations.items():
        if espn_norm == canonical or espn_norm in alts:
            if db_norm == canonical or db_norm in alts:
                return True
    
    return False


def get_db_games_for_date(date_str: str) -> list:
    """Get all games from our database for a date."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT g.id, g.date, g.status, 
               ht.name as home_team, at.name as away_team,
               g.home_score, g.away_score
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.date = ?
    """, (date_str,))
    
    games = []
    for row in cursor.fetchall():
        games.append({
            "id": row[0],
            "date": row[1],
            "status": row[2],
            "home_team": row[3],
            "away_team": row[4],
            "home_score": row[5],
            "away_score": row[6]
        })
    
    conn.close()
    return games


def get_all_tracked_teams() -> set:
    """Get all team names we track in our database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM teams")
    teams = {row[0].lower() for row in cursor.fetchall()}
    conn.close()
    return teams


def is_tracked_team(team_name: str, tracked_teams: set) -> bool:
    """Check if a team is in our tracking list."""
    if not team_name:
        return False
    norm = normalize_team_name(team_name).lower()
    
    # Direct match
    if norm in tracked_teams:
        return True
    
    # Check if any tracked team contains this name or vice versa
    for tracked in tracked_teams:
        if norm in tracked or tracked in norm:
            return True
    
    return False


def verify_date(date_str: str, auto_fix: bool = False, verbose: bool = True, tracked_only: bool = True) -> dict:
    """
    Verify our database against ESPN for a given date.
    
    Returns dict with:
    - total_espn: Number of completed games on ESPN
    - total_db: Number of games we have
    - missing_games: Games on ESPN but not in our DB
    - missing_scores: Games we have as pending but ESPN shows final
    - score_mismatches: Games where our score differs from ESPN
    - verified_ok: Games that match correctly
    """
    results = {
        "date": date_str,
        "total_espn_completed": 0,
        "total_db": 0,
        "missing_games": [],
        "missing_scores": [],
        "score_mismatches": [],
        "verified_ok": 0,
        "issues_found": 0,
        "auto_fixed": 0
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Verification Check: {date_str}")
        print('='*60)
    
    # Fetch ESPN data
    espn_games = fetch_espn_scoreboard(date_str)
    completed_espn = [g for g in espn_games if g["completed"]]
    
    # If tracking only our teams, get the list
    tracked_teams = set()
    if tracked_only:
        tracked_teams = get_all_tracked_teams()
        # Filter ESPN games to only those involving tracked teams
        completed_espn = [
            g for g in completed_espn 
            if (is_tracked_team(g["home_team"], tracked_teams) or 
                is_tracked_team(g["away_team"], tracked_teams))
        ]
    
    results["total_espn_completed"] = len(completed_espn)
    
    if verbose:
        print(f"ESPN completed games (tracked teams): {len(completed_espn)}")
    
    # Fetch our DB data
    db_games = get_db_games_for_date(date_str)
    results["total_db"] = len(db_games)
    
    if verbose:
        print(f"Database games: {len(db_games)}")
    
    # Check each ESPN completed game against our DB
    for espn_game in completed_espn:
        espn_home = espn_game["home_team"]
        espn_away = espn_game["away_team"]
        
        # Find matching game in DB
        db_match = None
        for db_game in db_games:
            if (match_teams(espn_home, db_game["home_team"]) and 
                match_teams(espn_away, db_game["away_team"])):
                db_match = db_game
                break
            # Also check reversed (sometimes home/away is swapped)
            if (match_teams(espn_home, db_game["away_team"]) and 
                match_teams(espn_away, db_game["home_team"])):
                db_match = db_game
                # Note: Home/away might be swapped
                break
        
        if not db_match:
            # Missing game entirely
            results["missing_games"].append({
                "espn": f"{espn_away} @ {espn_home}",
                "score": f"{espn_game['away_score']}-{espn_game['home_score']}"
            })
            results["issues_found"] += 1
            if verbose:
                print(f"❌ MISSING: {espn_away} @ {espn_home} ({espn_game['away_score']}-{espn_game['home_score']})")
        
        elif db_match["status"] not in ("completed", "final"):
            # We have the game but don't show it as completed
            results["missing_scores"].append({
                "game": f"{db_match['away_team']} @ {db_match['home_team']}",
                "db_status": db_match["status"],
                "espn_score": f"{espn_game['away_score']}-{espn_game['home_score']}",
                "db_id": db_match["id"]
            })
            results["issues_found"] += 1
            if verbose:
                print(f"⚠️  PENDING BUT COMPLETED: {db_match['away_team']} @ {db_match['home_team']}")
                print(f"    ESPN: {espn_game['away_score']}-{espn_game['home_score']}, DB status: {db_match['status']}")
            
            # Auto-fix if requested
            if auto_fix and espn_game["home_score"] is not None:
                fix_game_score(
                    db_match["id"], 
                    espn_game["home_score"], 
                    espn_game["away_score"]
                )
                results["auto_fixed"] += 1
                if verbose:
                    print(f"    ✓ AUTO-FIXED")
        
        elif (db_match["home_score"] != espn_game["home_score"] or 
              db_match["away_score"] != espn_game["away_score"]):
            # Score mismatch
            results["score_mismatches"].append({
                "game": f"{db_match['away_team']} @ {db_match['home_team']}",
                "db_score": f"{db_match['away_score']}-{db_match['home_score']}",
                "espn_score": f"{espn_game['away_score']}-{espn_game['home_score']}",
                "db_id": db_match["id"]
            })
            results["issues_found"] += 1
            if verbose:
                print(f"⚠️  SCORE MISMATCH: {db_match['away_team']} @ {db_match['home_team']}")
                print(f"    DB: {db_match['away_score']}-{db_match['home_score']}")
                print(f"    ESPN: {espn_game['away_score']}-{espn_game['home_score']}")
            
            # Auto-fix if requested
            if auto_fix:
                fix_game_score(
                    db_match["id"],
                    espn_game["home_score"],
                    espn_game["away_score"]
                )
                results["auto_fixed"] += 1
                if verbose:
                    print(f"    ✓ AUTO-FIXED")
        
        else:
            results["verified_ok"] += 1
    
    # Summary
    if verbose:
        print(f"\n{'─'*40}")
        print(f"Summary:")
        print(f"  ✓ Verified OK: {results['verified_ok']}")
        print(f"  ❌ Missing games: {len(results['missing_games'])}")
        print(f"  ⚠️  Missing scores: {len(results['missing_scores'])}")
        print(f"  ⚠️  Score mismatches: {len(results['score_mismatches'])}")
        if auto_fix:
            print(f"  🔧 Auto-fixed: {results['auto_fixed']}")
        print(f"  Total issues: {results['issues_found']}")
        
        if results["issues_found"] == 0:
            print("\n✅ All games verified successfully!")
        else:
            print(f"\n⚠️  {results['issues_found']} issue(s) found - review above")
    
    return results


def fix_game_score(game_id: int, home_score: int, away_score: int):
    """Update a game with correct scores from ESPN."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE games 
        SET home_score = ?, away_score = ?, status = 'completed'
        WHERE id = ?
    """, (home_score, away_score, game_id))
    
    conn.commit()
    conn.close()


def verify_recent(days: int = 3, auto_fix: bool = False) -> dict:
    """Verify multiple recent days."""
    all_results = []
    total_issues = 0
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        result = verify_date(date, auto_fix=auto_fix)
        all_results.append(result)
        total_issues += result["issues_found"]
    
    print(f"\n{'='*60}")
    print(f"Overall: {total_issues} total issues across {days} days")
    
    return {"days": all_results, "total_issues": total_issues}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify college baseball data against ESPN")
    parser.add_argument("date", nargs="?", help="Date to check (YYYY-MM-DD), default=today")
    parser.add_argument("--fix", action="store_true", help="Auto-fix found issues")
    parser.add_argument("--recent", type=int, metavar="DAYS", help="Check last N days")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--all", action="store_true", help="Show all ESPN games (not just tracked teams)")
    
    args = parser.parse_args()
    
    if args.recent:
        verify_recent(args.recent, auto_fix=args.fix)
    else:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        verify_date(date_str, auto_fix=args.fix, verbose=not args.quiet, tracked_only=not args.all)
