#!/usr/bin/env python3
"""
Fill in athletics URLs for AAC (American Athletic Conference) teams.
"""

import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# AAC athletics URLs
AAC_URLS = [
    ('charlotte', 'https://www.charlotte49ers.com'),
    ('east-carolina', 'https://www.ecupirates.com'),
    ('florida-atlantic', 'https://www.fausports.com'),
    ('memphis', 'https://www.gotigersgo.com'),
    ('north-texas', 'https://www.meangreensports.com'),
    ('rice', 'https://www.riceowls.com'),
    ('south-florida', 'https://www.gousfbulls.com'),
    ('temple', 'https://www.owlsports.com'),
    ('tulane', 'https://tulanegreenwave.com'),
    ('tulsa', 'https://www.tulsahurricane.com'),
    ('uab', 'https://www.uabsports.com'),
    ('utsa', 'https://www.goutsa.com'),
    ('wichita-state', 'https://www.goshockers.com'),
]

def update_athletics_urls():
    conn = sqlite3.connect(str(DB_PATH))
    
    for team_id, athletics_url in AAC_URLS:
        conn.execute("""
            UPDATE teams 
            SET athletics_url = ? 
            WHERE id = ? AND conference = 'AAC'
        """, (athletics_url, team_id))
    
    conn.commit()
    conn.close()
    print(f"Updated {len(AAC_URLS)} AAC team URLs")

if __name__ == '__main__':
    update_athletics_urls()