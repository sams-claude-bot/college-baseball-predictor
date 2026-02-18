#!/usr/bin/env python3
"""
Batch-set D1 baseball conference assignments from hardcoded mapping.
Much faster than querying ESPN one-by-one.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'

# team_id -> conference
# Only teams that play D1 baseball and are in our DB
CONFERENCE_MAP = {
    # SEC (16)
    'alabama': 'SEC', 'arkansas': 'SEC', 'auburn': 'SEC', 'florida': 'SEC',
    'georgia': 'SEC', 'kentucky': 'SEC', 'lsu': 'SEC', 'mississippi-state': 'SEC',
    'missouri': 'SEC', 'ole-miss': 'SEC', 'oklahoma': 'SEC',
    'south-carolina': 'SEC', 'tennessee': 'SEC', 'texas': 'SEC',
    'texas-a&m': 'SEC', 'vanderbilt': 'SEC',
    
    # Big Ten (18)
    'illinois': 'Big Ten', 'indiana': 'Big Ten', 'iowa': 'Big Ten',
    'maryland': 'Big Ten', 'michigan': 'Big Ten', 'michigan-state': 'Big Ten',
    'minnesota': 'Big Ten', 'nebraska': 'Big Ten', 'northwestern': 'Big Ten',
    'ohio-state': 'Big Ten', 'oregon': 'Big Ten', 'oregon-state': 'Big Ten',
    'penn-state': 'Big Ten', 'purdue': 'Big Ten', 'rutgers': 'Big Ten',
    'ucla': 'Big Ten', 'usc': 'Big Ten', 'washington': 'Big Ten',
    
    # ACC (17)
    'boston-college': 'ACC', 'clemson': 'ACC', 'duke': 'ACC',
    'florida-state': 'ACC', 'georgia-tech': 'ACC', 'louisville': 'ACC',
    'miami': 'ACC', 'miami-fl': 'ACC', 'north-carolina': 'ACC', 'unc': 'ACC',
    'nc-state': 'ACC', 'notre-dame': 'ACC', 'pittsburgh': 'ACC',
    'stanford': 'ACC', 'virginia': 'ACC', 'virginia-tech': 'ACC',
    'wake-forest': 'ACC', 'cal': 'ACC', 'california': 'ACC',
    
    # Big 12 (16 - minus 4 no-baseball = 12 baseball)
    'arizona': 'Big 12', 'arizona-state': 'Big 12', 'baylor': 'Big 12',
    'byu': 'Big 12', 'brigham-young': 'Big 12', 'cincinnati': 'Big 12',
    'houston': 'Big 12', 'kansas': 'Big 12', 'kansas-state': 'Big 12',
    'oklahoma-state': 'Big 12', 'tcu': 'Big 12', 'texas-tech': 'Big 12',
    'ucf': 'Big 12', 'west-virginia': 'Big 12', 'utah': 'Big 12',
    
    # AAC (14)
    'charlotte': 'AAC', 'east-carolina': 'AAC', 'fau': 'AAC',
    'florida-atlantic': 'AAC', 'memphis': 'AAC', 'north-texas': 'AAC',
    'rice': 'AAC', 'south-florida': 'AAC', 'usf': 'AAC',
    'temple': 'AAC', 'tulane': 'AAC', 'tulsa': 'AAC',
    'uab': 'AAC', 'utsa': 'AAC', 'wichita-state': 'AAC',
    
    # Sun Belt (14)
    'app-state': 'Sun Belt', 'appalachian-state': 'Sun Belt',
    'arkansas-state': 'Sun Belt', 'arkansas-state-red': 'Sun Belt',
    'coastal-carolina': 'Sun Belt', 'georgia-southern': 'Sun Belt',
    'georgia-state': 'Sun Belt', 'james-madison': 'Sun Belt',
    'louisiana': 'Sun Belt', 'louisiana-lafayette': 'Sun Belt',
    'marshall': 'Sun Belt', 'old-dominion': 'Sun Belt',
    'south-alabama': 'Sun Belt', 'southern-miss': 'Sun Belt',
    'texas-state': 'Sun Belt', 'troy': 'Sun Belt',
    'ul-monroe': 'Sun Belt',
    
    # Conference USA (8)
    'fiu': 'C-USA', 'florida-international': 'C-USA',
    'jacksonville-state': 'C-USA', 'kennesaw-state': 'C-USA',
    'liberty': 'C-USA', 'louisiana-tech': 'C-USA',
    'middle-tennessee': 'C-USA', 'new-mexico-state': 'C-USA',
    'sam-houston': 'C-USA', 'sam-houston-state': 'C-USA',
    'western-kentucky': 'C-USA',
    
    # MWC (7)
    'air-force': 'MWC', 'fresno-state': 'MWC', 'nevada': 'MWC',
    'new-mexico': 'MWC', 'san-diego-state': 'MWC', 'san-jose-state': 'MWC',
    'unlv': 'MWC',
    
    # WCC (8)
    'gonzaga': 'WCC', 'loyola-marymount': 'WCC', 'pepperdine': 'WCC',
    'portland': 'WCC', 'saint-marys': 'WCC', 'san-diego': 'WCC',
    'san-francisco': 'WCC', 'santa-clara': 'WCC',
    
    # Big East (8)
    'butler': 'Big East', 'creighton': 'Big East', 'georgetown': 'Big East',
    'providence': 'Big East', 'seton-hall': 'Big East',
    'st-johns': 'Big East', 'uconn': 'Big East', 'villanova': 'Big East',
    'xavier': 'Big East',
    
    # A-10 (13)
    'davidson': 'A-10', 'dayton': 'A-10', 'fordham': 'A-10',
    'george-mason': 'A-10', 'george-washington': 'A-10',
    'la-salle': 'A-10', 'massachusetts': 'A-10', 'rhode-island': 'A-10',
    'richmond': 'A-10', 'saint-josephs': 'A-10',
    'saint-louis': 'A-10', 'st-bonaventure': 'A-10', 'vcu': 'A-10',
    
    # CAA (11)
    'charleston': 'CAA', 'college-of-charleston': 'CAA',
    'delaware': 'CAA', 'drexel': 'CAA', 'elon': 'CAA',
    'hofstra': 'CAA', 'monmouth': 'CAA', 'northeastern': 'CAA',
    'north-carolina-at': 'CAA', 'stony-brook': 'CAA',
    'towson': 'CAA', 'william-mary': 'CAA', 'uncw': 'CAA',
    'unc-wilmington': 'CAA',
    
    # SoCon (10)
    'chattanooga': 'SoCon', 'the-citadel': 'SoCon', 'citadel': 'SoCon',
    'east-tennessee-state': 'SoCon', 'etsu': 'SoCon', 'furman': 'SoCon',
    'mercer': 'SoCon', 'samford': 'SoCon', 'unc-greensboro': 'SoCon',
    'vmi': 'SoCon', 'western-carolina': 'SoCon', 'wofford': 'SoCon',
    
    # ASUN (10)
    'bellarmine': 'ASUN', 'central-arkansas': 'ASUN',
    'eastern-kentucky': 'ASUN', 'florida-gulf-coast': 'ASUN', 'fgcu': 'ASUN',
    'jacksonville': 'ASUN', 'lipscomb': 'ASUN', 'north-alabama': 'ASUN',
    'north-florida': 'ASUN', 'queens-university': 'ASUN',
    'stetson': 'ASUN',
    
    # Southland (8)
    'houston-christian': 'Southland', 'incarnate-word': 'Southland',
    'lamar': 'Southland', 'mcneese': 'Southland',
    'nicholls': 'Southland', 'nicholls-state': 'Southland',
    'northwestern-state': 'Southland', 'southeastern-louisiana': 'Southland',
    'southeastern-la': 'Southland', 'texas-am-corpus-christi': 'Southland',
    
    # OVC (9)
    'austin-peay': 'OVC', 'eastern-illinois': 'OVC',
    'lindenwood': 'OVC', 'morehead-state': 'OVC',
    'murray-state': 'OVC', 'siu-edwardsville': 'OVC',
    'southeast-missouri': 'OVC', 'southern-indiana': 'OVC',
    'tennessee-tech': 'OVC', 'tennessee-state': 'OVC',
    'ut-martin': 'OVC',
    
    # MVC (10)
    'dallas-baptist': 'MVC', 'dbu': 'MVC',
    'evansville': 'MVC', 'illinois-state': 'MVC',
    'indiana-state': 'MVC', 'missouri-state': 'MVC',
    'southern-illinois': 'MVC', 'ualr': 'MVC',
    'valparaiso': 'MVC', 'bradley': 'MVC',
    
    # Big West (9)
    'cal-poly': 'Big West', 'cal-state-bakersfield': 'Big West',
    'cal-state-fullerton': 'Big West', 'cal-state-northridge': 'Big West',
    'hawaii': 'Big West', 'long-beach-state': 'Big West',
    'uc-davis': 'Big West', 'uc-irvine': 'Big West',
    'uc-riverside': 'Big West', 'uc-san-diego': 'Big West',
    'uc-santa-barbara': 'Big West', 'ucsb': 'Big West',
    
    # Patriot League
    'army': 'Patriot', 'army-black': 'Patriot',
    'bucknell': 'Patriot', 'holy-cross': 'Patriot',
    'lafayette': 'Patriot', 'lehigh': 'Patriot',
    'navy': 'Patriot',
    
    # Ivy League
    'columbia': 'Ivy', 'cornell': 'Ivy', 'dartmouth': 'Ivy',
    'harvard': 'Ivy', 'penn': 'Ivy', 'princeton': 'Ivy', 'yale': 'Ivy',
    'brown': 'Ivy',
    
    # MAAC (8)
    'canisius': 'MAAC', 'fairfield': 'MAAC', 'iona': 'MAAC',
    'manhattan': 'MAAC', 'marist': 'MAAC', 'niagara': 'MAAC',
    'quinnipiac': 'MAAC', 'rider': 'MAAC', 'siena': 'MAAC',
    'saint-peters': 'MAAC',
    
    # NEC (9)
    'central-connecticut': 'NEC', 'fairleigh-dickinson': 'NEC',
    'le-moyne': 'NEC', 'long-island': 'NEC',
    'mount-st-marys': 'NEC', 'sacred-heart': 'NEC',
    'stonehill': 'NEC', 'wagner': 'NEC',
    'merrimack': 'NEC',
    
    # Horizon League
    'cleveland-state': 'Horizon', 'illinois-chicago': 'Horizon',
    'milwaukee': 'Horizon', 'northern-kentucky': 'Horizon',
    'oakland': 'Horizon', 'purdue-fort-wayne': 'Horizon',
    'wright-state': 'Horizon', 'youngstown-state': 'Horizon',
    
    # MAC (12)
    'akron': 'MAC', 'ball-state': 'MAC', 'bowling-green': 'MAC',
    'central-michigan': 'MAC', 'eastern-michigan': 'MAC',
    'kent-state': 'MAC', 'miami-oh': 'MAC', 'miami-ohio': 'MAC',
    'northern-illinois': 'MAC', 'ohio': 'MAC', 'toledo': 'MAC',
    'western-michigan': 'MAC',
    
    # Summit League
    'north-dakota-state': 'Summit', 'omaha': 'Summit',
    'oral-roberts': 'Summit', 'south-dakota-state': 'Summit',
    'st-thomas-mn': 'Summit', 'western-illinois': 'Summit',
    'kansas-city': 'Summit',
    
    # America East (8)
    'albany': 'America East', 'binghamton': 'America East',
    'maine': 'America East', 'new-jersey-institute-of-technology': 'America East',
    'njit': 'America East', 'umass-lowell': 'America East',
    'umbc': 'America East', 'vermont': 'America East',
    
    # WAC (8)
    'abilene-christian': 'WAC', 'california-baptist': 'WAC',
    'grand-canyon': 'WAC', 'gcu': 'WAC',
    'seattle-u': 'WAC', 'seattle': 'WAC',
    'southern-utah': 'WAC', 'stephen-f-austin': 'WAC',
    'tarleton-state': 'WAC', 'utah-tech': 'WAC',
    'utah-valley': 'WAC',
    
    # Big South (8)
    'campbell': 'Big South', 'charleston-southern': 'Big South',
    'gardner-webb': 'Big South', 'high-point': 'Big South',
    'longwood': 'Big South', 'presbyterian': 'Big South',
    'radford': 'Big South', 'unc-asheville': 'Big South',
    'winthrop': 'Big South',
    
    # MEAC
    'alabama-state': 'MEAC', 'bethune-cookman': 'MEAC',
    'coppin-state': 'MEAC', 'delaware-state': 'MEAC',
    'howard': 'MEAC', 'maryland-eastern-shore': 'MEAC',
    'morgan-state': 'MEAC', 'norfolk-state': 'MEAC',
    'north-carolina-at': 'MEAC', 'south-carolina-state': 'MEAC',
    
    # SWAC
    'alabama-am': 'SWAC', 'alcorn-state': 'SWAC',
    'arkansas-pine-bluff': 'SWAC', 'grambling-state': 'SWAC',
    'jackson-state': 'SWAC', 'mississippi-valley-state': 'SWAC',
    'prairie-view': 'SWAC', 'southern': 'SWAC',
    'texas-southern': 'SWAC',
    
    # Independent or misc
    'washington-state': 'Independent',
}

def main():
    conn = sqlite3.connect(str(DB_PATH))
    
    updated = 0
    not_found = 0
    for team_id, conf in CONFERENCE_MAP.items():
        result = conn.execute("UPDATE teams SET conference = ? WHERE id = ? AND (conference IS NULL OR conference = '')", 
                            (conf, team_id))
        if result.rowcount > 0:
            updated += 1
        else:
            # Check if team exists at all
            exists = conn.execute("SELECT id FROM teams WHERE id = ?", (team_id,)).fetchone()
            if not exists:
                not_found += 1
    
    conn.commit()
    
    # Report
    total = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    with_conf = conn.execute("SELECT COUNT(*) FROM teams WHERE conference != '' AND conference IS NOT NULL").fetchone()[0]
    without = total - with_conf
    
    print(f"Updated: {updated} teams")
    print(f"Team IDs not in DB: {not_found}")
    print(f"Total teams: {total}")
    print(f"With conference: {with_conf}")
    print(f"Without conference: {without}")
    
    # Show conference breakdown
    rows = conn.execute("""
        SELECT conference, COUNT(*) as cnt FROM teams 
        WHERE conference != '' AND conference IS NOT NULL
        GROUP BY conference ORDER BY cnt DESC
    """).fetchall()
    print(f"\nConference breakdown:")
    for r in rows:
        print(f"  {r[0]:<15} {r[1]}")
    
    conn.close()


if __name__ == '__main__':
    main()
