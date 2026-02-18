#!/usr/bin/env python3
"""
Update athletics URLs for Tier 2 conference teams.
This script uses known patterns and web lookups to find official athletics websites.
"""

import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Known athletics URLs for Tier 2 conference teams
# Format: team_id -> athletics_url
ATHLETICS_URLS = {
    # ASUN Conference
    "bellarmine": "https://gobulldogs.com",
    "central-arkansas": "https://ucasports.com",
    "eastern-kentucky": "https://ekusports.com",
    "florida-gulf-coast": "https://fgcuathletics.com",
    "jacksonville": "https://judolphins.com",
    "lipscomb": "https://lipscombsports.com",
    "north-alabama": "https://roarlions.com",
    "north-florida": "https://unfospreys.com",
    "queens-university": "https://queensathletics.com",
    "stetson": "https://gohatters.com",
    
    # Big East Conference
    "butler": "https://butlersports.com",
    "creighton": "https://gocreighton.com",
    "georgetown": "https://guhoyas.com",
    "providence": "https://friars.com",
    "seton-hall": "https://shupirates.com",
    "st-johns": "https://redstormsports.com",
    "uconn": "https://uconnhuskies.com",
    "villanova": "https://villanova.com/sports/baseball",
    "xavier": "https://goxavier.com",
    
    # C-USA Conference
    "florida-international": "https://fiusports.com",
    "jacksonville-state": "https://jsugamecocksports.com",
    "kennesaw-state": "https://ksuowls.com",
    "liberty": "https://libertyflames.com",
    "louisiana-tech": "https://latechsports.com",
    "middle-tennessee": "https://goblueraiders.com",
    "new-mexico-state": "https://nmstatesports.com",
    "sam-houston": "https://gobearkats.com",
    "western-kentucky": "https://wkusports.com",
    
    # MVC Conference (Missouri Valley)
    "belmont": "https://belmontbruins.com",
    "bradley": "https://bradleybraves.com",
    "dallas-baptist": "https://dbupatriots.com",
    "evansville": "https://gopurpleaces.com",
    "illinois-state": "https://goredbirds.com",
    "indiana-state": "https://gosycamores.com",
    "missouri-state": "https://missouristatebears.com",
    "southern-illinois": "https://siusalukis.com",
    "ualr": "https://lrtrojans.com",
    "valparaiso": "https://valpoathletics.com",
    
    # WCC Conference (West Coast)
    "gonzaga": "https://gozags.com",
    "loyola-marymount": "https://lmulions.com",
    "pacific": "https://pacifictigers.com",
    "pepperdine": "https://pepperdinewaves.com",
    "portland": "https://portlandpilots.com",
    "saint-marys": "https://smcgaels.com",
    "san-diego": "https://usdtoreros.com",
    "san-francisco": "https://usfdons.com",
    "santa-clara": "https://santaclarabroncos.com",
    
    # CAA Conference (already has 3 with URLs, add the rest)
    "campbell": "https://gocamels.com",
    "college-of-charleston": "https://cofcsports.com",
    "elon": "https://elonphoenix.com",
    "hofstra": "https://gohofstra.com",
    "monmouth": "https://monmouthhawks.com",
    "north-carolina-at": "https://ncataggies.com",
    "northeastern": "https://gonu.com",
    "stony-brook": "https://stonybrookathletics.com",
    "towson": "https://towsontigers.com",
    "uncw": "https://uncwsports.com",
    "william-mary": "https://tribeathletics.com",
}


def update_urls():
    """Update athletics URLs in the database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    updated = 0
    for team_id, url in ATHLETICS_URLS.items():
        # Check if team exists and needs update
        result = cursor.execute(
            "SELECT athletics_url FROM teams WHERE id = ?", (team_id,)
        ).fetchone()
        
        if result is None:
            print(f"⚠ Team not found: {team_id}")
            continue
        
        current_url = result[0]
        if current_url and current_url.strip():
            print(f"· {team_id}: already has URL")
            continue
        
        cursor.execute(
            "UPDATE teams SET athletics_url = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (url, team_id)
        )
        print(f"✓ {team_id}: {url}")
        updated += 1
    
    conn.commit()
    conn.close()
    
    print(f"\nUpdated {updated} teams with athletics URLs")
    return updated


def verify_coverage():
    """Check coverage of Tier 2 conferences."""
    conn = sqlite3.connect(str(DB_PATH))
    
    result = conn.execute("""
        SELECT conference, 
               COUNT(*) as teams,
               SUM(CASE WHEN athletics_url IS NOT NULL AND athletics_url != '' THEN 1 ELSE 0 END) as has_url
        FROM teams
        WHERE conference IN ('Sun Belt', 'AAC', 'A-10', 'CAA', 'WCC', 'MVC', 'Big East', 'C-USA', 'ASUN')
        GROUP BY conference
        ORDER BY conference
    """).fetchall()
    
    print("\nTier 2 Conference Coverage:")
    print("-" * 40)
    for conf, teams, has_url in result:
        pct = (has_url / teams * 100) if teams > 0 else 0
        status = "✓" if has_url == teams else "·"
        print(f"{status} {conf:12} {has_url:2}/{teams:2} ({pct:.0f}%)")
    
    conn.close()


if __name__ == '__main__':
    update_urls()
    verify_coverage()
