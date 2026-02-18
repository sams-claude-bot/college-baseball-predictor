#!/usr/bin/env python3
"""
Fill in athletics URLs for Sun Belt Conference teams.
"""

import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Sun Belt athletics URLs
SUN_BELT_URLS = [
    ('appalachian-state', 'https://appstatesports.com'),
    ('arkansas-state', 'https://www.astateredwolves.com'),
    ('georgia-southern', 'https://www.gseagles.com'),
    ('georgia-state', 'https://www.georgiastatesports.com'),
    ('james-madison', 'https://jmusports.com'),
    ('louisiana', 'https://ragincajuns.com'),  # Main site
    ('marshall', 'https://herdzone.com'),
    ('old-dominion', 'https://www.odusports.com'),
    ('south-alabama', 'https://www.usajaguars.com'),
    ('texas-state', 'https://www.txstatebobcats.com'),
    ('troy', 'https://troytrojans.com'),
    ('ul-monroe', 'https://www.ulmwarhawks.com'),
]

def update_athletics_urls():
    conn = sqlite3.connect(str(DB_PATH))
    
    for team_id, athletics_url in SUN_BELT_URLS:
        conn.execute("""
            UPDATE teams 
            SET athletics_url = ? 
            WHERE id = ? AND conference = 'Sun Belt'
        """, (athletics_url, team_id))
    
    conn.commit()
    conn.close()
    print(f"Updated {len(SUN_BELT_URLS)} Sun Belt team URLs")

if __name__ == '__main__':
    update_athletics_urls()