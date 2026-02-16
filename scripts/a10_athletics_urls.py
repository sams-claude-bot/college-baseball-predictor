#!/usr/bin/env python3
"""
Fill in athletics URLs for A-10 (Atlantic 10) teams.
"""

import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# A-10 athletics URLs  
A10_URLS = [
    ('davidson', 'https://www.davidsonwildcats.com'),
    ('dayton', 'https://www.daytonflyers.com'),
    ('fordham', 'https://www.fordhamsports.com'),
    ('george-mason', 'https://www.gomason.com'),
    ('george-washington', 'https://www.gwsports.com'),
    ('la-salle', 'https://www.goexplorers.com'),
    ('massachusetts', 'https://www.umassathletics.com'),  # Same as umass
    ('rhode-island', 'https://www.gorhody.com'),
    ('richmond', 'https://www.richmondspiders.com'),
    ('saint-josephs', 'https://www.sjuhawks.com'),
    ('saint-louis', 'https://www.slubillikens.com'),
    ('st-bonaventure', 'https://www.gobonnies.com'),
    ('vcu', 'https://www.vcuathletics.com'),
]

def update_athletics_urls():
    conn = sqlite3.connect(str(DB_PATH))
    
    for team_id, athletics_url in A10_URLS:
        conn.execute("""
            UPDATE teams 
            SET athletics_url = ? 
            WHERE id = ? AND conference = 'A-10'
        """, (athletics_url, team_id))
    
    conn.commit()
    conn.close()
    print(f"Updated {len(A10_URLS)} A-10 team URLs")

if __name__ == '__main__':
    update_athletics_urls()