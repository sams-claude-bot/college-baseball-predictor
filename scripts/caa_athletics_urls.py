#!/usr/bin/env python3
"""
Fill in athletics URLs for CAA (Colonial Athletic Association) teams.
"""

import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# CAA athletics URLs  
CAA_URLS = [
    ('charleston', 'https://www.charlestonuniversity.edu/athletics/'),
    ('delaware', 'https://www.bluehens.com'),
    ('drexel', 'https://www.drexeldragons.com'),
    ('elon', 'https://www.elonphoenix.com'),
    ('hofstra', 'https://www.hofstrapride.com'),
    ('monmouth', 'https://www.monmouthhawks.com'),
    ('north-carolina-at', 'https://www.ncataggies.com'),
    ('northeastern', 'https://www.nuhuskies.com'),
    ('stony-brook', 'https://www.stonybrookathletics.com'),
    ('towson', 'https://www.towsontigers.com'),
    ('unc-wilmington', 'https://www.uncwsports.com'),  # Same as uncw
    ('william-mary', 'https://www.tribeathletics.com'),
]

def update_athletics_urls():
    conn = sqlite3.connect(str(DB_PATH))
    
    for team_id, athletics_url in CAA_URLS:
        conn.execute("""
            UPDATE teams 
            SET athletics_url = ? 
            WHERE id = ? AND conference = 'CAA'
        """, (athletics_url, team_id))
    
    conn.commit()
    conn.close()
    print(f"Updated {len(CAA_URLS)} CAA team URLs")

if __name__ == '__main__':
    update_athletics_urls()