#!/usr/bin/env python3
"""
Collect team and player stats from NCAA.com
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import re
import sys

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def fetch_ncaa_stats():
    """Fetch current stats from NCAA.com"""
    url = "https://www.ncaa.com/stats/baseball/d1"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching NCAA stats: {e}")
        return None
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    stats = {
        "fetched_at": datetime.now().isoformat(),
        "source": url,
        "individual_batting": [],
        "individual_pitching": [],
        "team_batting": [],
        "team_pitching": []
    }
    
    # Parse tables - NCAA.com structure varies, this is a starting point
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                # Extract rank, name, team, stat
                data = [c.get_text(strip=True) for c in cells]
                if data:
                    stats["individual_batting"].append(data)
    
    return stats

def fetch_team_stats(team_id):
    """Fetch stats for a specific team"""
    url = f"https://www.ncaa.com/schools/{team_id}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Parse team-specific stats
            return {"team": team_id, "fetched": datetime.now().isoformat()}
    except Exception as e:
        print(f"Error fetching {team_id}: {e}")
    
    return None

def save_stats(stats, filename):
    """Save stats to JSON file"""
    output_path = DATA_DIR / "snapshots" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved stats to {output_path}")

def main():
    print(f"Collecting NCAA stats at {datetime.now()}")
    
    stats = fetch_ncaa_stats()
    if stats:
        date_str = datetime.now().strftime("%Y-%m-%d")
        save_stats(stats, f"ncaa_stats_{date_str}.json")
        print(f"Collected {len(stats.get('individual_batting', []))} batting entries")
    else:
        print("Failed to collect stats")
        sys.exit(1)

if __name__ == "__main__":
    main()
