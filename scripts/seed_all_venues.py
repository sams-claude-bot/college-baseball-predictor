#!/usr/bin/env python3
"""
Seed venue coordinates for all D1 baseball teams.

Uses Open-Meteo geocoding (free) to look up college locations by name,
then adds them to the venues table.

Usage:
    python3 scripts/seed_all_venues.py
"""

import json
import sqlite3
import time
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'

# Manual overrides for schools with ambiguous names or non-standard locations
MANUAL_COORDS = {
    # Already in P4 venues - skip
    'alabama': None, 'arkansas': None, 'auburn': None, 'florida': None,
    'georgia': None, 'kentucky': None, 'lsu': None, 'mississippi-state': None,
    'missouri': None, 'oklahoma': None, 'ole-miss': None, 'south-carolina': None,
    'tennessee': None, 'texas': None, 'texas-am': None, 'vanderbilt': None,
    'michigan': None, 'ohio-state': None, 'indiana': None, 'maryland': None,
    'michigan-state': None, 'minnesota': None, 'nebraska': None, 'penn-state': None,
    'purdue': None, 'rutgers': None, 'illinois': None, 'iowa': None,
    'northwestern': None, 'clemson': None, 'duke': None, 'florida-state': None,
    'georgia-tech': None, 'louisville': None, 'miami-fl': None, 'nc-state': None,
    'north-carolina': None, 'notre-dame': None, 'pittsburgh': None, 'virginia': None,
    'virginia-tech': None, 'wake-forest': None, 'boston-college': None, 'stanford': None,
    'california': None, 'tcu': None, 'texas-tech': None, 'baylor': None,
    'oklahoma-state': None, 'kansas': None, 'kansas-state': None, 'west-virginia': None,
    'arizona': None, 'arizona-state': None, 'byu': None, 'ucf': None,
    'cincinnati': None, 'houston': None,
    
    # Schools with tricky names - provide search hints
    'miami-oh': ('Miami University', 'Oxford', 'OH', 39.5097, -84.7351),
    'unc-wilmington': ('UNC Wilmington', 'Wilmington', 'NC', 34.2257, -77.8708),
    'unc-greensboro': ('UNC Greensboro', 'Greensboro', 'NC', 36.0687, -79.8102),
    'unc-asheville': ('UNC Asheville', 'Asheville', 'NC', 35.6175, -82.5653),
    'texas-state': ('Texas State University', 'San Marcos', 'TX', 29.8884, -97.9384),
    'texas-aandm-corpus-christi': ('Texas A&M Corpus Christi', 'Corpus Christi', 'TX', 27.7127, -97.3254),
    'louisiana': ('Louisiana Ragin Cajuns', 'Lafayette', 'LA', 30.2138, -92.0215),
    'ul-monroe': ('UL Monroe', 'Monroe', 'LA', 32.5285, -92.0740),
    'southern-miss': ('Southern Miss', 'Hattiesburg', 'MS', 31.3275, -89.3369),
    'middle-tennessee': ('Middle Tennessee', 'Murfreesboro', 'TN', 35.8491, -86.3670),
    'utsa': ('UTSA', 'San Antonio', 'TX', 29.5831, -98.6199),
    'uab': ('UAB', 'Birmingham', 'AL', 33.5021, -86.8062),
    'fiu': ('FIU', 'Miami', 'FL', 25.7563, -80.3756),
    'usf': ('USF', 'Tampa', 'FL', 28.0619, -82.4133),
    'fau': ('FAU', 'Boca Raton', 'FL', 26.3708, -80.1018),
    'unlv': ('UNLV', 'Las Vegas', 'NV', 36.1085, -115.1434),
    'utrgv': ('UTRGV', 'Edinburg', 'TX', 26.3058, -98.1752),
    'siu-edwardsville': ('SIUE', 'Edwardsville', 'IL', 38.7928, -89.9982),
    'ipfw': ('Purdue Fort Wayne', 'Fort Wayne', 'IN', 41.1152, -85.1100),
    'umbc': ('UMBC', 'Baltimore', 'MD', 39.2555, -76.7118),
    'uconn': ('UConn', 'Storrs', 'CT', 41.8084, -72.2495),
    'umass': ('UMass Amherst', 'Amherst', 'MA', 42.3868, -72.5301),
    'umass-lowell': ('UMass Lowell', 'Lowell', 'MA', 42.6559, -71.3243),
    'ualr': ('UALR', 'Little Rock', 'AR', 34.7245, -92.3417),
    'uic': ('UIC', 'Chicago', 'IL', 41.8708, -87.6505),
}

# Name mappings for geocoding
SCHOOL_NAMES = {
    'alabama-am': 'Alabama A&M University',
    'alabama-state': 'Alabama State University',
    'alcorn-state': 'Alcorn State University',
    'appalachian-state': 'Appalachian State University',
    'army': 'United States Military Academy',
    'ball-state': 'Ball State University',
    'bethune-cookman': 'Bethune-Cookman University',
    'bowling-green': 'Bowling Green State University',
    'bryant': 'Bryant University',
    'bucknell': 'Bucknell University',
    'cal-poly': 'Cal Poly San Luis Obispo',
    'cal-state-bakersfield': 'CSU Bakersfield',
    'cal-state-fullerton': 'CSU Fullerton',
    'cal-state-northridge': 'CSU Northridge',
    'campbell': 'Campbell University',
    'canisius': 'Canisius College',
    'central-arkansas': 'University of Central Arkansas',
    'central-connecticut': 'Central Connecticut State',
    'central-michigan': 'Central Michigan University',
    'charleston': 'College of Charleston',
    'charleston-southern': 'Charleston Southern University',
    'charlotte': 'UNC Charlotte',
    'coastal-carolina': 'Coastal Carolina University',
    'coppin-state': 'Coppin State University',
    'creighton': 'Creighton University',
    'dallas-baptist': 'Dallas Baptist University',
    'davidson': 'Davidson College',
    'dayton': 'University of Dayton',
    'delaware': 'University of Delaware',
    'delaware-state': 'Delaware State University',
    'detroit-mercy': 'University of Detroit Mercy',
    'east-carolina': 'East Carolina University',
    'east-tennessee-state': 'East Tennessee State University',
    'eastern-illinois': 'Eastern Illinois University',
    'eastern-kentucky': 'Eastern Kentucky University',
    'eastern-michigan': 'Eastern Michigan University',
    'elon': 'Elon University',
    'evansville': 'University of Evansville',
    'fairfield': 'Fairfield University',
    'fairleigh-dickinson': 'Fairleigh Dickinson University',
    'florida-atlantic': 'Florida Atlantic University',
    'florida-gulf-coast': 'Florida Gulf Coast University',
    'florida-international': 'Florida International University',
    'fordham': 'Fordham University',
    'fresno-state': 'Fresno State',
    'furman': 'Furman University',
    'gardner-webb': 'Gardner-Webb University',
    'george-mason': 'George Mason University',
    'george-washington': 'George Washington University',
    'georgetown': 'Georgetown University',
    'georgia-southern': 'Georgia Southern University',
    'georgia-state': 'Georgia State University',
    'gonzaga': 'Gonzaga University',
    'grambling-state': 'Grambling State University',
    'grand-canyon': 'Grand Canyon University',
    'hampton': 'Hampton University',
    'hartford': 'University of Hartford',
    'hawaii': 'University of Hawaii',
    'high-point': 'High Point University',
    'hofstra': 'Hofstra University',
    'holy-cross': 'College of the Holy Cross',
    'illinois-state': 'Illinois State University',
    'incarnate-word': 'University of the Incarnate Word',
    'indiana-state': 'Indiana State University',
    'iona': 'Iona University',
    'iowa-state': 'Iowa State University',
    'jackson-state': 'Jackson State University',
    'jacksonville': 'Jacksonville University',
    'jacksonville-state': 'Jacksonville State University',
    'james-madison': 'James Madison University',
    'kennesaw-state': 'Kennesaw State University',
    'kent-state': 'Kent State University',
    'la-salle': 'La Salle University',
    'lamar': 'Lamar University',
    'le-moyne': 'Le Moyne College',
    'lehigh': 'Lehigh University',
    'liberty': 'Liberty University',
    'lindenwood': 'Lindenwood University',
    'lipscomb': 'Lipscomb University',
    'little-rock': 'University of Arkansas Little Rock',
    'liu': 'Long Island University',
    'longwood': 'Longwood University',
    'louisiana-tech': 'Louisiana Tech University',
    'loyola-marymount': 'Loyola Marymount University',
    'maine': 'University of Maine',
    'manhattan': 'Manhattan College',
    'marist': 'Marist College',
    'marshall': 'Marshall University',
    'maryland-eastern-shore': 'Maryland Eastern Shore',
    'mcneese': 'McNeese State University',
    'memphis': 'University of Memphis',
    'mercer': 'Mercer University',
    'merrimack': 'Merrimack College',
    'mississippi-valley-state': 'Mississippi Valley State',
    'missouri-state': 'Missouri State University',
    'monmouth': 'Monmouth University',
    'morehead-state': 'Morehead State University',
    'mount-st-marys': 'Mount St. Mary\'s University',
    'murray-state': 'Murray State University',
    'navy': 'United States Naval Academy',
    'new-mexico': 'University of New Mexico',
    'new-mexico-state': 'New Mexico State University',
    'new-orleans': 'University of New Orleans',
    'niagara': 'Niagara University',
    'nicholls': 'Nicholls State University',
    'njit': 'New Jersey Institute of Technology',
    'norfolk-state': 'Norfolk State University',
    'north-alabama': 'University of North Alabama',
    'north-carolina-at': 'NC A&T State University',
    'north-dakota-state': 'North Dakota State University',
    'north-florida': 'University of North Florida',
    'northeastern': 'Northeastern University',
    'northern-colorado': 'University of Northern Colorado',
    'northern-illinois': 'Northern Illinois University',
    'northern-kentucky': 'Northern Kentucky University',
    'northwestern-state': 'Northwestern State University',
    'oakland': 'Oakland University',
    'ohio': 'Ohio University',
    'old-dominion': 'Old Dominion University',
    'omaha': 'University of Nebraska Omaha',
    'oral-roberts': 'Oral Roberts University',
    'oregon': 'University of Oregon',
    'oregon-state': 'Oregon State University',
    'pacific': 'University of the Pacific',
    'pepperdine': 'Pepperdine University',
    'portland': 'University of Portland',
    'prairie-view': 'Prairie View A&M University',
    'presbyterian': 'Presbyterian College',
    'princeton': 'Princeton University',
    'providence': 'Providence College',
    'quinnipiac': 'Quinnipiac University',
    'radford': 'Radford University',
    'rhode-island': 'University of Rhode Island',
    'rice': 'Rice University',
    'richmond': 'University of Richmond',
    'rider': 'Rider University',
    'sacramento-state': 'Sacramento State',
    'sacred-heart': 'Sacred Heart University',
    'saint-josephs': 'Saint Joseph\'s University',
    'saint-louis': 'Saint Louis University',
    'saint-peters': 'Saint Peter\'s University',
    'sam-houston': 'Sam Houston State University',
    'samford': 'Samford University',
    'san-diego': 'University of San Diego',
    'san-diego-state': 'San Diego State University',
    'san-francisco': 'University of San Francisco',
    'san-jose-state': 'San Jose State University',
    'santa-barbara': 'UC Santa Barbara',
    'santa-clara': 'Santa Clara University',
    'seattle': 'Seattle University',
    'seton-hall': 'Seton Hall University',
    'siena': 'Siena College',
    'south-alabama': 'University of South Alabama',
    'south-dakota-state': 'South Dakota State University',
    'south-florida': 'University of South Florida',
    'southeast-missouri': 'Southeast Missouri State',
    'southeastern-louisiana': 'Southeastern Louisiana',
    'southern': 'Southern University',
    'southern-illinois': 'Southern Illinois University',
    'st-bonaventure': 'St. Bonaventure University',
    'st-johns': 'St. John\'s University',
    'stephen-f-austin': 'Stephen F. Austin State',
    'stetson': 'Stetson University',
    'stonehill': 'Stonehill College',
    'stony-brook': 'Stony Brook University',
    'tarleton': 'Tarleton State University',
    'temple': 'Temple University',
    'texas-arlington': 'UT Arlington',
    'texas-southern': 'Texas Southern University',
    'the-citadel': 'The Citadel',
    'toledo': 'University of Toledo',
    'towson': 'Towson University',
    'troy': 'Troy University',
    'tulane': 'Tulane University',
    'tulsa': 'University of Tulsa',
    'usc': 'University of Southern California',
    'usc-upstate': 'USC Upstate',
    'utah': 'University of Utah',
    'utah-valley': 'Utah Valley University',
    'valparaiso': 'Valparaiso University',
    'villanova': 'Villanova University',
    'wagner': 'Wagner College',
    'washington': 'University of Washington',
    'washington-state': 'Washington State University',
    'western-carolina': 'Western Carolina University',
    'western-illinois': 'Western Illinois University',
    'western-kentucky': 'Western Kentucky University',
    'western-michigan': 'Western Michigan University',
    'wichita-state': 'Wichita State University',
    'william-mary': 'William & Mary',
    'winthrop': 'Winthrop University',
    'wofford': 'Wofford College',
    'wright-state': 'Wright State University',
    'xavier': 'Xavier University',
    'youngstown-state': 'Youngstown State University',
}


def geocode_school(name):
    """Use Open-Meteo geocoding to find school coordinates."""
    encoded = urllib.parse.quote(name)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            results = data.get('results', [])
            if results:
                r = results[0]
                return r.get('latitude'), r.get('longitude'), r.get('admin1', ''), r.get('country', '')
    except Exception as e:
        print(f"  Geocode error: {e}")
    return None, None, None, None


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def main():
    import urllib.parse
    
    # Load existing venues
    db = get_db()
    existing = set(row[0] for row in db.execute("SELECT team_id FROM venues").fetchall())
    print(f"Existing venues: {len(existing)}")
    
    # Load slug map for all D1 teams
    slug_map = json.loads(SLUGS_FILE.read_text()).get('team_id_to_d1bb_slug', {})
    print(f"D1 teams in slug map: {len(slug_map)}")
    
    added = 0
    failed = []
    
    for team_id in sorted(slug_map.keys()):
        if team_id in existing:
            continue
            
        # Check manual override first
        if team_id in MANUAL_COORDS:
            override = MANUAL_COORDS[team_id]
            if override is None:
                continue  # Already in venues
            stadium, city, state, lat, lon = override
            db.execute("""
                INSERT OR REPLACE INTO venues (team_id, stadium_name, city, state, latitude, longitude, is_dome)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (team_id, stadium, city, state, lat, lon))
            added += 1
            print(f"  {team_id}: {lat:.4f}, {lon:.4f} (manual)")
            continue
        
        # Try to geocode
        search_name = SCHOOL_NAMES.get(team_id, team_id.replace('-', ' ').title() + ' University')
        lat, lon, state, country = geocode_school(search_name)
        
        if lat and lon and country == 'United States':
            db.execute("""
                INSERT OR REPLACE INTO venues (team_id, city, state, latitude, longitude, is_dome)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (team_id, '', state, lat, lon))
            added += 1
            print(f"  {team_id}: {lat:.4f}, {lon:.4f} ({state})")
        else:
            failed.append(team_id)
            print(f"  {team_id}: FAILED (searched: {search_name})")
        
        time.sleep(0.2)  # Rate limit
    
    db.commit()
    db.close()
    
    print(f"\n{'='*50}")
    print(f"Added: {added} venues")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed teams: {', '.join(failed[:20])}")


if __name__ == '__main__':
    main()
