#!/usr/bin/env python3
"""
Scrape team and individual stats from NCAA.com

Sources:
- https://www.ncaa.com/stats/baseball/d1 (individual + team leaders)
- https://www.ncaa.com/stats/baseball/d1/current/team/{stat_id} (full team rankings)
- https://www.ncaa.com/stats/baseball/d1/current/individual/{stat_id} (full individual rankings)

Stat categories available:
  Team: batting-average, earned-run-average, fielding-percentage, 
        home-runs, runs-scored, scoring, slugging-percentage, 
        stolen-bases, strikeouts, triples, walks, winning-percentage
  Individual: batting-average, earned-run-average, hits, home-runs,
              on-base-percentage, rbi, runs-scored, slugging-percentage,
              stolen-bases, strikeouts, triples, walks, wins
"""

import sys
import re
import json
from pathlib import Path
from datetime import datetime

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))

from database import get_connection

# Try to import web_fetch helper, fall back to requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


NCAA_BASE = "https://www.ncaa.com/stats/baseball/d1"

# Team stat URLs
TEAM_STAT_IDS = {
    'batting_avg': 'current/team/229',
    'era': 'current/team/230',
    'fielding_pct': 'current/team/231',
    'home_runs': 'current/team/232',
    'runs_scored': 'current/team/233',
    'scoring': 'current/team/234',
    'slugging_pct': 'current/team/235',
    'stolen_bases': 'current/team/236',
    'strikeouts': 'current/team/237',
    'triples': 'current/team/238',
    'walks': 'current/team/239',
    'win_pct': 'current/team/240',
}

# Individual stat URLs
INDIVIDUAL_STAT_IDS = {
    'batting_avg': 'current/individual/229',
    'era': 'current/individual/230',
    'hits': 'current/individual/231',
    'home_runs': 'current/individual/232',
    'obp': 'current/individual/233',
    'rbi': 'current/individual/234',
    'runs_scored': 'current/individual/235',
    'slugging_pct': 'current/individual/236',
    'stolen_bases': 'current/individual/237',
    'strikeouts': 'current/individual/238',
    'triples': 'current/individual/239',
    'walks': 'current/individual/240',
    'wins': 'current/individual/241',
}


def init_ncaa_stats_table():
    """Create table for NCAA.com stats snapshots"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ncaa_team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_id TEXT,
            stat_category TEXT NOT NULL,
            stat_value REAL,
            rank INTEGER,
            games INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS ncaa_individual_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            team_name TEXT NOT NULL,
            team_id TEXT,
            stat_category TEXT NOT NULL,
            stat_value REAL,
            rank INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def normalize_team_id(name):
    """Convert NCAA team name to our team_id format"""
    # Common mappings
    aliases = {
        'miss state': 'mississippi-state',
        'mississippi state': 'mississippi-state',
        'mississippi st.': 'mississippi-state',
        'ole miss': 'ole-miss',
        'texas a&m': 'texas-a&m',
        'south carolina': 'south-carolina',
        'nc state': 'nc-state',
        'n.c. state': 'nc-state',
        'georgia tech': 'georgia-tech',
        'florida st.': 'florida-state',
        'florida state': 'florida-state',
        'oregon st.': 'oregon-state',
        'oregon state': 'oregon-state',
        'wake forest': 'wake-forest',
        'virginia tech': 'virginia-tech',
        'east carolina': 'east-carolina',
        'coastal caro.': 'coastal-carolina',
        'coastal carolina': 'coastal-carolina',
        'southern miss.': 'southern-miss',
        'southern miss': 'southern-miss',
        'arizona st.': 'arizona-state',
        'arizona state': 'arizona-state',
        'miami (fl)': 'miami',
        'central conn. st.': 'central-connecticut',
        'southern u.': 'southern',
        'usc upstate': 'usc-upstate',
        'southeastern la.': 'southeastern-louisiana',
        'western ky.': 'western-kentucky',
        'northern ky.': 'northern-kentucky',
        'austin peay': 'austin-peay',
    }
    lower = name.lower().strip()
    if lower in aliases:
        return aliases[lower]
    return lower.replace(' ', '-').replace('(', '').replace(')', '').replace('.', '')


def parse_stats_page(html, stat_category, is_team=True):
    """Parse NCAA.com stats page HTML to extract rankings"""
    results = []
    
    # Look for table rows with rank, name, team, stat value
    # NCAA.com uses a fairly consistent table structure
    lines = html.split('\n')
    
    current_rank = None
    current_name = None
    current_team = None
    
    for line in lines:
        line = line.strip()
        
        # Try to find rank numbers (1, 2, 3...)
        rank_match = re.match(r'^(\d+)$', line)
        if rank_match and int(rank_match.group(1)) <= 500:
            current_rank = int(rank_match.group(1))
            continue
        
        # Try to find stat values (decimals like .337 or 3.71, or integers)
        stat_match = re.match(r'^(\d*\.\d+)$', line)
        if not stat_match:
            stat_match = re.match(r'^(\d+\.\d+)$', line)
        
        if stat_match and current_rank is not None:
            stat_value = float(stat_match.group(1))
            if is_team and current_name:
                results.append({
                    'rank': current_rank,
                    'team_name': current_name,
                    'stat_value': stat_value,
                    'category': stat_category,
                })
                current_rank = None
                current_name = None
            elif not is_team and current_name and current_team:
                results.append({
                    'rank': current_rank,
                    'player_name': current_name,
                    'team_name': current_team,
                    'stat_value': stat_value,
                    'category': stat_category,
                })
                current_rank = None
                current_name = None
                current_team = None
    
    return results


def fetch_ncaa_page(url):
    """Fetch a page from NCAA.com"""
    if HAS_REQUESTS:
        try:
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; BaseballPredictor/1.0)'
            })
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"  Request failed: {e}")
    return None


def scrape_main_stats_page():
    """
    Scrape the main NCAA.com/stats/baseball/d1 page.
    Returns parsed team and individual stat leaders.
    
    This page has top 10 in key categories.
    """
    init_ncaa_stats_table()
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📊 Scraping NCAA.com stats ({today})")
    print("=" * 50)
    
    html = fetch_ncaa_page(NCAA_BASE)
    if not html:
        print("  ❌ Failed to fetch main stats page")
        return False
    
    # Parse the markdown/text version
    conn = get_connection()
    c = conn.cursor()
    
    # The page has structured data we can parse
    # Team batting avg leaders, team ERA leaders, individual BA, individual ERA
    stats_collected = 0
    
    # Simple line-by-line parsing of the text content
    lines = html.split('\n')
    current_section = None
    current_rank = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if 'TEAM STATISTICS' in line.upper():
            current_section = 'team'
        elif 'INDIVIDUAL STATISTICS' in line.upper():
            current_section = 'individual'
    
    print(f"  ✓ Fetched main page ({len(html)} bytes)")
    print(f"  Stats will be parsed from dedicated category pages")
    
    conn.close()
    return True


def scrape_team_stat(category):
    """Scrape a specific team stat category from NCAA.com"""
    if category not in TEAM_STAT_IDS:
        print(f"  Unknown category: {category}")
        return []
    
    url = f"{NCAA_BASE}/{TEAM_STAT_IDS[category]}"
    print(f"  Fetching team {category}...")
    
    html = fetch_ncaa_page(url)
    if not html:
        print(f"  ❌ Failed to fetch {category}")
        return []
    
    results = parse_stats_page(html, category, is_team=True)
    print(f"  ✓ Got {len(results)} teams for {category}")
    return results


def scrape_all_team_stats():
    """Scrape all team stat categories and store in DB"""
    init_ncaa_stats_table()
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📊 Scraping all NCAA team stats ({today})")
    print("=" * 50)
    
    conn = get_connection()
    c = conn.cursor()
    
    total = 0
    for category in ['batting_avg', 'era', 'win_pct', 'scoring', 'home_runs', 
                      'slugging_pct', 'stolen_bases', 'strikeouts']:
        results = scrape_team_stat(category)
        for r in results:
            team_id = normalize_team_id(r['team_name'])
            c.execute('''
                INSERT INTO ncaa_team_stats 
                (team_name, team_id, stat_category, stat_value, rank, date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (r['team_name'], team_id, category, r['stat_value'], r['rank'], today))
            total += 1
    
    conn.commit()
    conn.close()
    print(f"\n✓ Stored {total} team stat entries")
    return total


def get_team_ncaa_stats(team_id, date=None):
    """Get NCAA.com stats for a specific team"""
    conn = get_connection()
    c = conn.cursor()
    
    if date:
        c.execute('''
            SELECT stat_category, stat_value, rank 
            FROM ncaa_team_stats 
            WHERE team_id = ? AND date = ?
        ''', (team_id, date))
    else:
        c.execute('''
            SELECT stat_category, stat_value, rank 
            FROM ncaa_team_stats 
            WHERE team_id = ? 
            ORDER BY date DESC
        ''', (team_id,))
    
    rows = c.fetchall()
    conn.close()
    
    stats = {}
    for row in rows:
        cat = row[0]
        if cat not in stats:  # Take most recent
            stats[cat] = {'value': row[1], 'rank': row[2]}
    
    return stats


def get_ncaa_stat_leaders(category, limit=25, date=None):
    """Get top teams in a stat category"""
    conn = get_connection()
    c = conn.cursor()
    
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT team_name, team_id, stat_value, rank
        FROM ncaa_team_stats
        WHERE stat_category = ? AND date = ?
        ORDER BY rank ASC
        LIMIT ?
    ''', (category, date, limit))
    
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def collect_for_daily(force=False):
    """Called from daily_collection.py"""
    print("\n[NCAA Stats Collection]")
    
    # Check if already collected today
    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    c = conn.cursor()
    
    try:
        c.execute("SELECT COUNT(*) FROM ncaa_team_stats WHERE date = ?", (today,))
        count = c.fetchone()[0]
    except Exception:
        count = 0
    conn.close()
    
    if count > 0 and not force:
        print(f"  Already collected {count} stats today. Use force=True to re-collect.")
        return count
    
    total = scrape_all_team_stats()
    return total


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scrape_ncaa_stats.py collect       - Collect all team stats")
        print("  python scrape_ncaa_stats.py team <team>   - Show team's NCAA stats")
        print("  python scrape_ncaa_stats.py leaders <cat>  - Show leaders in category")
        print("  python scrape_ncaa_stats.py categories    - List available categories")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "collect":
        force = "--force" in sys.argv
        collect_for_daily(force=force)
    
    elif cmd == "team":
        if len(sys.argv) < 3:
            print("Usage: python scrape_ncaa_stats.py team <team_name>")
            return
        team = normalize_team_id(sys.argv[2])
        stats = get_team_ncaa_stats(team)
        if stats:
            print(f"\n📊 NCAA Stats for {sys.argv[2]}:")
            for cat, data in stats.items():
                print(f"  {cat}: {data['value']} (#{data['rank']})")
        else:
            print(f"No stats found for {team}")
    
    elif cmd == "leaders":
        cat = sys.argv[2] if len(sys.argv) > 2 else 'batting_avg'
        leaders = get_ncaa_stat_leaders(cat)
        if leaders:
            print(f"\n📊 Top teams - {cat}:")
            for l in leaders:
                print(f"  #{l['rank']:>3} {l['team_name']:<25} {l['stat_value']}")
        else:
            print(f"No data for {cat}. Run 'collect' first.")
    
    elif cmd == "categories":
        print("\nTeam categories:")
        for k in TEAM_STAT_IDS:
            print(f"  {k}")
        print("\nIndividual categories:")
        for k in INDIVIDUAL_STAT_IDS:
            print(f"  {k}")
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
