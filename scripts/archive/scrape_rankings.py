#!/usr/bin/env python3
"""
Weekly Top 25 Rankings Scraper

Scrapes rankings from multiple sources:
1. D1Baseball.com
2. Baseball America
3. USA Today Coaches Poll
4. NCAA.com

Features:
- Auto-detect new teams entering Top 25
- Track rankings movement week-over-week
- Initialize Elo ratings for new entrants
- Generate rankings summary for reports
"""

import sys
import json
import re
import requests
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent
# sys.path.insert(0, str(BASE_DIR / "scripts"))  # Removed by cleanup
# sys.path.insert(0, str(BASE_DIR / "models"))  # Removed by cleanup

from scripts.database import (
    get_connection, add_team, add_ranking, get_current_top_25,
    get_ranking_history, init_rankings_table
)
from rankings import normalize_team_id, TEAM_ALIASES

# Season start date for week calculation
SEASON_START = datetime(2026, 2, 13)

# Headers for web requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Elo rating initialization by rank range
def get_initial_elo(rank):
    """Get initial Elo rating based on ranking position"""
    if rank <= 5:
        return 1680 + (5 - rank) * 4  # 1680-1700
    elif rank <= 10:
        return 1620 + (10 - rank) * 12  # 1620-1680
    elif rank <= 15:
        return 1580 + (15 - rank) * 8  # 1580-1620
    elif rank <= 20:
        return 1540 + (20 - rank) * 8  # 1540-1580
    else:
        return 1500 + (25 - rank) * 8  # 1500-1540


def get_current_week():
    """Calculate current week number from season start"""
    today = datetime.now()
    if today < SEASON_START:
        return 0  # Preseason
    delta = today - SEASON_START
    return max(1, (delta.days // 7) + 1)


def get_previous_rankings(poll="d1baseball"):
    """Get most recent rankings from database"""
    conn = get_connection()
    c = conn.cursor()
    
    # Get the most recent date with rankings for this poll
    c.execute('''
        SELECT DISTINCT date FROM rankings_history 
        WHERE poll = ?
        ORDER BY date DESC
        LIMIT 1
    ''', (poll,))
    
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    
    last_date = row[0]
    
    # Get all rankings from that date
    c.execute('''
        SELECT team_id, rank FROM rankings_history
        WHERE poll = ? AND date = ?
    ''', (poll, last_date))
    
    rankings = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    
    return rankings


def team_exists(team_id):
    """Check if team exists in database"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM teams WHERE id = ?", (team_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def initialize_team_elo(team_id, rank):
    """Initialize Elo rating for a new team based on their ranking"""
    rating = get_initial_elo(rank)
    
    conn = get_connection()
    c = conn.cursor()
    
    # Ensure elo_ratings table exists
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='elo_ratings'
    """)
    
    if not c.fetchone():
        c.execute('''
            CREATE TABLE elo_ratings (
                team_id TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500,
                games_played INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    # Insert or update rating
    c.execute('''
        INSERT INTO elo_ratings (team_id, rating, games_played)
        VALUES (?, ?, 0)
        ON CONFLICT(team_id) DO UPDATE SET
            rating = excluded.rating,
            updated_at = CURRENT_TIMESTAMP
    ''', (team_id, rating))
    
    conn.commit()
    conn.close()
    
    return rating


def flag_for_roster_scrape(team_id):
    """Flag a team for roster scraping"""
    flag_file = BASE_DIR / "data" / "teams" / "pending_rosters.json"
    flag_file.parent.mkdir(parents=True, exist_ok=True)
    
    pending = []
    if flag_file.exists():
        with open(flag_file) as f:
            pending = json.load(f)
    
    if team_id not in pending:
        pending.append(team_id)
        with open(flag_file, 'w') as f:
            json.dump(pending, f, indent=2)


# =============================================================================
# SCRAPERS
# =============================================================================

def scrape_d1baseball():
    """
    Scrape D1Baseball.com Top 25
    
    Returns list of (team_name, rank) tuples or None on failure
    """
    url = "https://d1baseball.com/top-25/"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rankings = []
        
        # D1Baseball typically uses a table or list structure
        # Look for ranking entries
        ranking_rows = soup.find_all('tr', class_=re.compile(r'ranking|row'))
        
        if not ranking_rows:
            # Try alternative selectors
            ranking_rows = soup.select('.rankings-table tr, .top-25-list li, article.ranking-item')
        
        for row in ranking_rows:
            # Try to extract rank and team name
            rank_elem = row.find(class_=re.compile(r'rank|position|number'))
            team_elem = row.find(class_=re.compile(r'team|name|school'))
            
            if rank_elem and team_elem:
                try:
                    rank = int(re.search(r'\d+', rank_elem.get_text()).group())
                    team = team_elem.get_text().strip()
                    if 1 <= rank <= 25:
                        rankings.append((team, rank))
                except (ValueError, AttributeError):
                    continue
        
        # Sort by rank and return
        if len(rankings) >= 20:  # At least 20 teams found
            rankings.sort(key=lambda x: x[1])
            return rankings[:25]
            
    except Exception as e:
        print(f"  ✗ D1Baseball scrape failed: {e}")
    
    return None


def scrape_ncaa_rankings():
    """
    Scrape NCAA.com baseball rankings
    
    Returns list of (team_name, rank) tuples or None on failure
    """
    url = "https://www.ncaa.com/rankings/baseball/d1/ncaa-baseball-division-i"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rankings = []
        
        # NCAA.com typically uses tables
        table = soup.find('table', class_=re.compile(r'rankings|polls'))
        if table:
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    try:
                        rank_text = cells[0].get_text().strip()
                        rank = int(re.search(r'\d+', rank_text).group())
                        team = cells[1].get_text().strip()
                        if 1 <= rank <= 25:
                            rankings.append((team, rank))
                    except (ValueError, AttributeError):
                        continue
        
        if len(rankings) >= 20:
            rankings.sort(key=lambda x: x[1])
            return rankings[:25]
            
    except Exception as e:
        print(f"  ✗ NCAA scrape failed: {e}")
    
    return None


def scrape_espn_rankings():
    """
    Scrape ESPN college baseball rankings
    
    Returns list of (team_name, rank) tuples or None on failure
    """
    url = "https://www.espn.com/college-baseball/rankings"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rankings = []
        
        # ESPN uses various table formats
        rows = soup.select('tr.Table__TR, tbody tr')
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                try:
                    rank_text = cells[0].get_text().strip()
                    rank = int(re.search(r'\d+', rank_text).group())
                    team_cell = cells[1]
                    # Team name might be in a link
                    team_link = team_cell.find('a')
                    team = team_link.get_text().strip() if team_link else team_cell.get_text().strip()
                    if 1 <= rank <= 25 and team:
                        rankings.append((team, rank))
                except (ValueError, AttributeError):
                    continue
        
        if len(rankings) >= 20:
            rankings.sort(key=lambda x: x[1])
            return rankings[:25]
            
    except Exception as e:
        print(f"  ✗ ESPN scrape failed: {e}")
    
    return None


def scrape_baseball_america():
    """
    Scrape Baseball America Top 25
    (Often behind paywall, may not work)
    """
    url = "https://www.baseballamerica.com/rankings/"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rankings = []
        
        # Try various selectors
        items = soup.select('.ranking-item, .poll-item, ol li, .rankings li')
        
        for i, item in enumerate(items[:25], 1):
            team_text = item.get_text().strip()
            # Extract team name, removing rank numbers and extra info
            team = re.sub(r'^\d+[\.\)]\s*', '', team_text)
            team = re.sub(r'\s*\(\d+-\d+\).*$', '', team)
            team = team.strip()
            if team:
                rankings.append((team, i))
        
        if len(rankings) >= 20:
            return rankings[:25]
            
    except Exception as e:
        print(f"  ✗ Baseball America scrape failed: {e}")
    
    return None


def get_fallback_rankings():
    """
    Return manually-maintained preseason rankings as fallback
    Updated: 2026 preseason (based on available projections)
    """
    # This is the fallback when scrapers fail
    # Keep updated with latest known rankings
    preseason_2026 = [
        ("UCLA", 1),
        ("Texas A&M", 2),
        ("Florida", 3),
        ("Arkansas", 4),
        ("LSU", 5),
        ("Tennessee", 6),
        ("Georgia", 7),
        ("Texas", 8),
        ("Vanderbilt", 9),
        ("Florida State", 10),
        ("Wake Forest", 11),
        ("Virginia", 12),
        ("Oregon State", 13),
        ("NC State", 14),
        ("Stanford", 15),
        ("Ole Miss", 16),
        ("Clemson", 17),
        ("Miami (FL)", 18),
        ("Louisville", 19),
        ("Mississippi State", 20),
        ("South Carolina", 21),
        ("TCU", 22),
        ("North Carolina", 23),
        ("Arizona", 24),
        ("Kentucky", 25),
    ]
    return preseason_2026


# =============================================================================
# MAIN SCRAPER LOGIC
# =============================================================================

def scrape_rankings(poll="d1baseball", force=False):
    """
    Scrape current Top 25 rankings
    
    Tries multiple sources in order, falls back to manual list if all fail.
    
    Args:
        poll: Which poll to scrape (d1baseball, baseball_america, coaches, espn)
        force: If True, scrape even if not Monday
        
    Returns:
        dict with:
            - rankings: list of (team_id, team_name, rank)
            - source: where the data came from
            - week: current week number
            - new_teams: list of newly added teams
            - dropped_teams: list of teams that dropped out
            - movements: rankings movements from previous week
    """
    init_rankings_table()
    
    week = get_current_week()
    today = datetime.now()
    is_monday = today.weekday() == 0
    
    if not force and not is_monday:
        print(f"  ℹ️  Not Monday (day={today.strftime('%A')}), skipping rankings scrape")
        print("     Use force=True to scrape anyway")
        return None
    
    print(f"\n📊 Scraping Top 25 Rankings (Week {week or 'Preseason'})")
    print("-" * 50)
    
    # Try scrapers in order
    scrapers = [
        ("d1baseball", scrape_d1baseball),
        ("ncaa", scrape_ncaa_rankings),
        ("espn", scrape_espn_rankings),
        ("baseball_america", scrape_baseball_america),
    ]
    
    raw_rankings = None
    source = None
    
    for name, scraper_func in scrapers:
        print(f"  Trying {name}...", end=" ")
        result = scraper_func()
        if result:
            print(f"✓ Got {len(result)} teams")
            raw_rankings = result
            source = name
            break
        else:
            print("✗")
    
    # Fall back to manual list
    if not raw_rankings:
        print("  Using fallback rankings (manual list)")
        raw_rankings = get_fallback_rankings()
        source = "fallback"
    
    # Get previous rankings for comparison
    previous = get_previous_rankings(poll)
    
    # Process rankings
    results = {
        "rankings": [],
        "source": source,
        "week": week,
        "date": today.strftime('%Y-%m-%d'),
        "new_teams": [],
        "dropped_teams": [],
        "movements": [],
    }
    
    current_team_ids = set()
    
    print(f"\n{'Rank':<5} {'Team':<25} {'Change':<10} {'Notes'}")
    print("-" * 60)
    
    for team_name, rank in raw_rankings:
        team_id = normalize_team_id(team_name)
        current_team_ids.add(team_id)
        
        # Check if team is new to database
        is_new_to_db = not team_exists(team_id)
        
        # Check if new to Top 25
        prev_rank = previous.get(team_id)
        is_new_to_top25 = prev_rank is None and len(previous) > 0
        
        # Calculate movement
        if prev_rank:
            change = prev_rank - rank
            if change > 0:
                change_str = f"↑{change}"
            elif change < 0:
                change_str = f"↓{abs(change)}"
            else:
                change_str = "—"
        elif is_new_to_top25:
            change_str = "NEW"
        else:
            change_str = "—"
        
        notes = []
        
        # Handle new team to database
        if is_new_to_db:
            add_team(team_id, team_name)
            elo = initialize_team_elo(team_id, rank)
            flag_for_roster_scrape(team_id)
            notes.append(f"Added (Elo: {elo:.0f})")
            print(f"\n🆕 NEW TEAM ENTERED TOP 25: {team_name} at #{rank}")
            print(f"   → Added to database")
            print(f"   → Initialized Elo: {elo:.0f}")
            print(f"   → Flagged for roster scrape")
            results["new_teams"].append({
                "team_id": team_id,
                "name": team_name,
                "rank": rank,
                "initial_elo": elo
            })
        elif is_new_to_top25:
            # Returning to Top 25 - update Elo if significantly different
            results["new_teams"].append({
                "team_id": team_id,
                "name": team_name,
                "rank": rank,
                "returning": True
            })
        
        # Ensure team exists with proper name (updates if exists)
        add_team(team_id, team_name)
        
        # Add ranking to database
        add_ranking(team_id, rank, poll, week, today.strftime('%Y-%m-%d'))
        
        # Track movements
        if prev_rank and abs(prev_rank - rank) >= 3:
            results["movements"].append({
                "team_id": team_id,
                "name": team_name,
                "prev_rank": prev_rank,
                "new_rank": rank,
                "change": prev_rank - rank
            })
        
        results["rankings"].append({
            "team_id": team_id,
            "name": team_name,
            "rank": rank,
            "prev_rank": prev_rank,
            "change": (prev_rank - rank) if prev_rank else None
        })
        
        notes_str = " | ".join(notes) if notes else ""
        print(f"{rank:<5} {team_name:<25} {change_str:<10} {notes_str}")
    
    # Find dropped teams and clear their current_rank
    previous_ids = set(previous.keys())
    dropped = previous_ids - current_team_ids
    
    if dropped:
        print("\n📉 Dropped from Top 25:")
        conn = get_connection()
        c = conn.cursor()
        for team_id in dropped:
            prev_rank = previous[team_id]
            print(f"   DROPPED: {team_id} fell from #{prev_rank}")
            results["dropped_teams"].append({
                "team_id": team_id,
                "prev_rank": prev_rank
            })
            # Clear the current_rank so they don't show in Top 25
            c.execute('''
                UPDATE teams SET current_rank = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (team_id,))
        conn.commit()
        conn.close()
    
    print("-" * 60)
    print(f"✓ Saved {len(results['rankings'])} rankings from {source}")
    
    # Save summary to file
    summary_file = BASE_DIR / "data" / "rankings" / f"rankings_{today.strftime('%Y-%m-%d')}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# MOVEMENT ANALYSIS
# =============================================================================

def analyze_movement(poll="d1baseball"):
    """
    Analyze rankings movement between weeks
    
    Returns summary of risers, fallers, new entries, and exits
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Get two most recent dates
    c.execute('''
        SELECT DISTINCT date FROM rankings_history
        WHERE poll = ?
        ORDER BY date DESC
        LIMIT 2
    ''', (poll,))
    
    dates = [row[0] for row in c.fetchall()]
    
    if len(dates) < 2:
        conn.close()
        return {
            "error": "Need at least 2 weeks of data for movement analysis",
            "weeks_available": len(dates)
        }
    
    current_date, previous_date = dates
    
    # Get rankings for both dates
    def get_rankings_by_date(date):
        c.execute('''
            SELECT rh.team_id, rh.rank, t.name
            FROM rankings_history rh
            LEFT JOIN teams t ON rh.team_id = t.id
            WHERE rh.poll = ? AND rh.date = ?
        ''', (poll, date))
        return {row[0]: {"rank": row[1], "name": row[2]} for row in c.fetchall()}
    
    current = get_rankings_by_date(current_date)
    previous = get_rankings_by_date(previous_date)
    
    conn.close()
    
    # Analyze
    risers = []
    fallers = []
    new_entries = []
    exits = []
    
    for team_id, data in current.items():
        if team_id in previous:
            change = previous[team_id]["rank"] - data["rank"]
            if change >= 3:
                risers.append({
                    "team": data["name"] or team_id,
                    "prev_rank": previous[team_id]["rank"],
                    "new_rank": data["rank"],
                    "change": change
                })
            elif change <= -3:
                fallers.append({
                    "team": data["name"] or team_id,
                    "prev_rank": previous[team_id]["rank"],
                    "new_rank": data["rank"],
                    "change": change
                })
        else:
            new_entries.append({
                "team": data["name"] or team_id,
                "rank": data["rank"]
            })
    
    for team_id in previous:
        if team_id not in current:
            exits.append({
                "team": previous[team_id]["name"] or team_id,
                "prev_rank": previous[team_id]["rank"]
            })
    
    # Sort
    risers.sort(key=lambda x: x["change"], reverse=True)
    fallers.sort(key=lambda x: x["change"])
    
    return {
        "current_date": current_date,
        "previous_date": previous_date,
        "risers": risers,
        "fallers": fallers,
        "new_entries": new_entries,
        "exits": exits
    }


# =============================================================================
# REPORT FUNCTIONS
# =============================================================================

def get_rankings_summary(poll="d1baseball"):
    """
    Get rankings summary suitable for PDF report
    
    Returns data structure for generate_report.py
    """
    current = get_current_top_25(poll)
    movement = analyze_movement(poll)
    
    summary = {
        "top_25": [],
        "biggest_risers": movement.get("risers", [])[:3],
        "biggest_fallers": movement.get("fallers", [])[:3],
        "new_entries": movement.get("new_entries", []),
        "exits": movement.get("exits", []),
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "week": get_current_week()
    }
    
    # Build Top 25 with previous rank
    previous = get_previous_rankings(poll)
    
    for team in current:
        team_id = team["id"]
        prev_rank = previous.get(team_id)
        
        summary["top_25"].append({
            "rank": team["current_rank"],
            "name": team["name"],
            "prev_rank": prev_rank,
            "conference": team.get("conference"),
            "change": (prev_rank - team["current_rank"]) if prev_rank else None
        })
    
    return summary


def print_rankings_report():
    """Print a formatted rankings report to console"""
    summary = get_rankings_summary()
    week = summary["week"]
    
    print(f"\n{'='*60}")
    print(f"📊 TOP 25 RANKINGS - Week {week or 'Preseason'}")
    print(f"{'='*60}")
    
    print(f"\n{'Rank':<5} {'Team':<25} {'Prev':<6} {'Chg':<5} {'Conf'}")
    print("-" * 55)
    
    for team in summary["top_25"]:
        prev = str(team["prev_rank"]) if team["prev_rank"] else "-"
        change = team["change"]
        if change:
            chg = f"+{change}" if change > 0 else str(change)
        else:
            chg = "-"
        conf = team["conference"] or ""
        print(f"{team['rank']:<5} {team['name']:<25} {prev:<6} {chg:<5} {conf}")
    
    # Movement summary
    if summary["biggest_risers"]:
        print(f"\n📈 BIGGEST RISERS:")
        for t in summary["biggest_risers"]:
            print(f"   {t['team']}: #{t['prev_rank']} → #{t['new_rank']} (+{t['change']})")
    
    if summary["biggest_fallers"]:
        print(f"\n📉 BIGGEST FALLERS:")
        for t in summary["biggest_fallers"]:
            print(f"   {t['team']}: #{t['prev_rank']} → #{t['new_rank']} ({t['change']})")
    
    if summary["new_entries"]:
        print(f"\n🆕 NEW ENTRIES:")
        for t in summary["new_entries"]:
            print(f"   {t['team']} at #{t['rank']}")
    
    if summary["exits"]:
        print(f"\n🚪 DROPPED OUT:")
        for t in summary["exits"]:
            print(f"   {t['team']} (was #{t['prev_rank']})")


# =============================================================================
# DAILY COLLECTION INTEGRATION
# =============================================================================

def run_weekly_rankings_update(force=False):
    """
    Called from daily_collection.py
    
    Only runs on Mondays unless force=True
    Returns results for logging
    """
    today = datetime.now()
    is_monday = today.weekday() == 0
    
    if not is_monday and not force:
        return {
            "skipped": True,
            "reason": f"Not Monday (today is {today.strftime('%A')})"
        }
    
    results = scrape_rankings(poll="d1baseball", force=True)
    
    if results:
        return {
            "success": True,
            "source": results["source"],
            "week": results["week"],
            "teams_ranked": len(results["rankings"]),
            "new_teams": len(results["new_teams"]),
            "dropped": len(results["dropped_teams"]),
            "movements": len(results["movements"])
        }
    
    return {
        "success": False,
        "error": "Failed to scrape rankings"
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Top 25 college baseball rankings")
    parser.add_argument("command", nargs="?", default="show",
                       choices=["scrape", "show", "report", "movement", "summary"],
                       help="Command to run")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force scrape even if not Monday")
    parser.add_argument("--poll", "-p", default="d1baseball",
                       help="Poll source (d1baseball, baseball_america, coaches)")
    
    args = parser.parse_args()
    
    if args.command == "scrape":
        results = scrape_rankings(poll=args.poll, force=args.force)
        if results:
            print(f"\n✓ Scraped {len(results['rankings'])} rankings")
            if results["new_teams"]:
                print(f"  {len(results['new_teams'])} new teams added")
            if results["dropped_teams"]:
                print(f"  {len(results['dropped_teams'])} teams dropped")
    
    elif args.command == "show":
        teams = get_current_top_25(args.poll)
        if teams:
            print(f"\n📊 Current Top 25 ({args.poll}):")
            print("-" * 40)
            for t in teams:
                conf = f" ({t['conference']})" if t.get('conference') else ""
                print(f"  {t['current_rank']:2}. {t['name']}{conf}")
        else:
            print("\n⚠️  No rankings in database")
            print("Run: python scrape_rankings.py scrape --force")
    
    elif args.command == "report":
        print_rankings_report()
    
    elif args.command == "movement":
        movement = analyze_movement(args.poll)
        if "error" in movement:
            print(f"\n⚠️  {movement['error']}")
        else:
            print(f"\n📊 Rankings Movement ({movement['previous_date']} → {movement['current_date']})")
            print("-" * 50)
            
            if movement["risers"]:
                print("\n📈 Risers (+3 or more):")
                for t in movement["risers"]:
                    print(f"   {t['team']}: #{t['prev_rank']} → #{t['new_rank']} (+{t['change']})")
            
            if movement["fallers"]:
                print("\n📉 Fallers (-3 or more):")
                for t in movement["fallers"]:
                    print(f"   {t['team']}: #{t['prev_rank']} → #{t['new_rank']} ({t['change']})")
            
            if movement["new_entries"]:
                print("\n🆕 New to Top 25:")
                for t in movement["new_entries"]:
                    print(f"   {t['team']} enters at #{t['rank']}")
            
            if movement["exits"]:
                print("\n🚪 Dropped out:")
                for t in movement["exits"]:
                    print(f"   {t['team']} (was #{t['prev_rank']})")
    
    elif args.command == "summary":
        summary = get_rankings_summary(args.poll)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
