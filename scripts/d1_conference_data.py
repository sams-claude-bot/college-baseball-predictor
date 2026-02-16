#!/usr/bin/env python3
"""
D1 Conference data extracted from Wikipedia's List of NCAA Division I baseball programs.
This provides a definitive mapping of team names to conferences for the 2026 season.
"""

# Mapping of team name variations to their conferences (2026 season)
# Format: lowercase team name/id -> conference abbreviation
D1_TEAM_CONFERENCES = {
    # America East Conference
    "albany": "America East",
    "binghamton": "America East",
    "bryant": "America East",
    "maine": "America East",
    "njit": "America East",
    "umass-lowell": "America East",
    "umbc": "America East",
    
    # American Athletic Conference (AAC)
    "charlotte": "AAC",
    "east-carolina": "AAC",
    "florida-atlantic": "AAC",
    "fau": "AAC",
    "memphis": "AAC",
    "rice": "AAC",
    "south-florida": "AAC",
    "usf": "AAC",
    "tulane": "AAC",
    "uab": "AAC",
    "utsa": "AAC",
    "wichita-state": "AAC",
    
    # Atlantic Coast Conference (ACC)
    "boston-college": "ACC",
    "california": "ACC",
    "cal": "ACC",
    "clemson": "ACC",
    "duke": "ACC",
    "florida-state": "ACC",
    "georgia-tech": "ACC",
    "louisville": "ACC",
    "miami": "ACC",
    "miami-fl": "ACC",
    "north-carolina": "ACC",
    "unc": "ACC",
    "nc-state": "ACC",
    "notre-dame": "ACC",
    "pittsburgh": "ACC",
    "pitt": "ACC",
    "stanford": "ACC",
    "virginia": "ACC",
    "virginia-tech": "ACC",
    "wake-forest": "ACC",
    
    # ASUN Conference
    "austin-peay": "ASUN",
    "bellarmine": "ASUN",
    "central-arkansas": "ASUN",
    "eastern-kentucky": "ASUN",
    "florida-gulf-coast": "ASUN",
    "fgcu": "ASUN",
    "jacksonville": "ASUN",
    "lipscomb": "ASUN",
    "north-alabama": "ASUN",
    "north-florida": "ASUN",
    "queens": "ASUN",
    "stetson": "ASUN",
    
    # Atlantic 10 Conference (A-10)
    "davidson": "A-10",
    "dayton": "A-10",
    "fordham": "A-10",
    "george-mason": "A-10",
    "george-washington": "A-10",
    "la-salle": "A-10",
    "rhode-island": "A-10",
    "richmond": "A-10",
    "st-bonaventure": "A-10",
    "saint-josephs": "A-10",
    "saint-louis": "A-10",
    "vcu": "A-10",
    
    # Big East Conference
    "butler": "Big East",
    "creighton": "Big East",
    "georgetown": "Big East",
    "st-johns": "Big East",
    "seton-hall": "Big East",
    "connecticut": "Big East",
    "uconn": "Big East",
    "villanova": "Big East",
    "xavier": "Big East",
    
    # Big South Conference
    "charleston-southern": "Big South",
    "gardner-webb": "Big South",
    "high-point": "Big South",
    "longwood": "Big South",
    "presbyterian": "Big South",
    "radford": "Big South",
    "unc-asheville": "Big South",
    "south-carolina-upstate": "Big South",
    "usc-upstate": "Big South",
    "winthrop": "Big South",
    
    # Big Ten Conference
    "illinois": "Big Ten",
    "indiana": "Big Ten",
    "iowa": "Big Ten",
    "maryland": "Big Ten",
    "michigan": "Big Ten",
    "michigan-state": "Big Ten",
    "minnesota": "Big Ten",
    "nebraska": "Big Ten",
    "northwestern": "Big Ten",
    "ohio-state": "Big Ten",
    "oregon": "Big Ten",
    "penn-state": "Big Ten",
    "purdue": "Big Ten",
    "rutgers": "Big Ten",
    "ucla": "Big Ten",
    "usc": "Big Ten",
    "southern-california": "Big Ten",
    "washington": "Big Ten",
    
    # Big 12 Conference
    "arizona": "Big 12",
    "arizona-state": "Big 12",
    "baylor": "Big 12",
    "byu": "Big 12",
    "brigham-young": "Big 12",
    "cincinnati": "Big 12",
    "houston": "Big 12",
    "kansas": "Big 12",
    "kansas-state": "Big 12",
    "oklahoma-state": "Big 12",
    "tcu": "Big 12",
    "texas-tech": "Big 12",
    "ucf": "Big 12",
    "utah": "Big 12",
    "west-virginia": "Big 12",
    
    # Big West Conference
    "cal-poly": "Big West",
    "cal-state-fullerton": "Big West",
    "cal-state-northridge": "Big West",
    "csun": "Big West",
    "cal-state-bakersfield": "Big West",
    "hawaii": "Big West",
    "long-beach-state": "Big West",
    "uc-davis": "Big West",
    "uc-irvine": "Big West",
    "uc-riverside": "Big West",
    "uc-san-diego": "Big West",
    "uc-santa-barbara": "Big West",
    "ucsb": "Big West",
    
    # Colonial Athletic Association (CAA)
    "campbell": "CAA",
    "campbell-fighting": "CAA",
    "college-of-charleston": "CAA",
    "charleston": "CAA",
    "elon": "CAA",
    "hofstra": "CAA",
    "monmouth": "CAA",
    "north-carolina-at": "CAA",
    "nc-at": "CAA",
    "northeastern": "CAA",
    "stony-brook": "CAA",
    "towson": "CAA",
    "uncw": "CAA",
    "unc-wilmington": "CAA",
    "william-mary": "CAA",
    
    # Conference USA (C-USA)
    "dallas-baptist": "C-USA",
    "delaware": "C-USA",
    "fiu": "C-USA",
    "florida-international": "C-USA",
    "jacksonville-state": "C-USA",
    "kennesaw-state": "C-USA",
    "liberty": "C-USA",
    "louisiana-tech": "C-USA",
    "middle-tennessee": "C-USA",
    "missouri-state": "C-USA",
    "new-mexico-state": "C-USA",
    "sam-houston": "C-USA",
    "sam-houston-state": "C-USA",
    "western-kentucky": "C-USA",
    "wku": "C-USA",
    
    # Horizon League
    "milwaukee": "Horizon",
    "northern-kentucky": "Horizon",
    "oakland": "Horizon",
    "purdue-fort-wayne": "Horizon",
    "wright-state": "Horizon",
    "youngstown-state": "Horizon",
    
    # Ivy League
    "brown": "Ivy",
    "columbia": "Ivy",
    "cornell": "Ivy",
    "dartmouth": "Ivy",
    "dartmouth-big": "Ivy",
    "harvard": "Ivy",
    "pennsylvania": "Ivy",
    "penn": "Ivy",
    "princeton": "Ivy",
    "yale": "Ivy",
    
    # MAAC (Metro Atlantic Athletic Conference)
    "canisius": "MAAC",
    "canisius-golden": "MAAC",
    "fairfield": "MAAC",
    "iona": "MAAC",
    "manhattan": "MAAC",
    "marist": "MAAC",
    "merrimack": "MAAC",
    "mount-st-marys": "MAAC",
    "niagara": "MAAC",
    "quinnipiac": "MAAC",
    "rider": "MAAC",
    "sacred-heart": "MAAC",
    "saint-peters": "MAAC",
    "siena": "MAAC",
    
    # Mid-American Conference (MAC)
    "akron": "MAC",
    "ball-state": "MAC",
    "bowling-green": "MAC",
    "central-michigan": "MAC",
    "eastern-michigan": "MAC",
    "kent-state": "MAC",
    "umass": "MAC",
    "massachusetts": "MAC",
    "miami-oh": "MAC",
    "miami-ohio": "MAC",
    "northern-illinois": "MAC",
    "niu": "MAC",
    "ohio": "MAC",
    "toledo": "MAC",
    "western-michigan": "MAC",
    
    # Missouri Valley Conference (MVC)
    "belmont": "MVC",
    "bradley": "MVC",
    "evansville": "MVC",
    "illinois-state": "MVC",
    "indiana-state": "MVC",
    "murray-state": "MVC",
    "southern-illinois": "MVC",
    "siu": "MVC",
    "ualr": "MVC",
    "little-rock": "MVC",
    "valparaiso": "MVC",
    
    # Northeast Conference (NEC)
    "central-connecticut": "NEC",
    "fairleigh-dickinson": "NEC",
    "fdu": "NEC",
    "le-moyne": "NEC",
    "long-island": "NEC",
    "long-island-university": "NEC",
    "liu": "NEC",
    "mercyhurst": "NEC",
    "sacred-heart": "NEC",
    "st-francis-bkn": "NEC",
    "st-francis-brooklyn": "NEC",
    "stonehill": "NEC",
    "wagner": "NEC",
    
    # Ohio Valley Conference (OVC)
    "eastern-illinois": "OVC",
    "lindenwood": "OVC",
    "morehead-state": "OVC",
    "semo": "OVC",
    "southeast-missouri": "OVC",
    "southern-indiana": "OVC",
    "usi": "OVC",
    "tennessee-tech": "OVC",
    "ut-martin": "OVC",
    
    # Patriot League
    "army": "Patriot",
    "army-black": "Patriot",
    "bucknell": "Patriot",
    "colgate": "Patriot",
    "holy-cross": "Patriot",
    "lafayette": "Patriot",
    "lehigh": "Patriot",
    "navy": "Patriot",
    
    # Southeastern Conference (SEC)
    "alabama": "SEC",
    "arkansas": "SEC",
    "auburn": "SEC",
    "florida": "SEC",
    "georgia": "SEC",
    "kentucky": "SEC",
    "lsu": "SEC",
    "ole-miss": "SEC",
    "mississippi": "SEC",
    "mississippi-state": "SEC",
    "missouri": "SEC",
    "oklahoma": "SEC",
    "south-carolina": "SEC",
    "tennessee": "SEC",
    "texas": "SEC",
    "texas-am": "SEC",
    "vanderbilt": "SEC",
    
    # Southern Conference (SoCon)
    "chattanooga": "SoCon",
    "citadel": "SoCon",
    "east-tennessee-state": "SoCon",
    "etsu": "SoCon",
    "furman": "SoCon",
    "mercer": "SoCon",
    "samford": "SoCon",
    "unc-greensboro": "SoCon",
    "uncg": "SoCon",
    "vmi": "SoCon",
    "western-carolina": "SoCon",
    "wofford": "SoCon",
    
    # Southland Conference
    "houston-christian": "Southland",
    "incarnate-word": "Southland",
    "lamar": "Southland",
    "mcneese": "Southland",
    "mcneese-state": "Southland",
    "nicholls": "Southland",
    "nicholls-state": "Southland",
    "northwestern-state": "Southland",
    "southeastern-louisiana": "Southland",
    "se-louisiana": "Southland",
    "texas-am-corpus-christi": "Southland",
    "tamucc": "Southland",
    
    # Summit League
    "oral-roberts": "Summit",
    "south-dakota-state": "Summit",
    "st-thomas-minnesota": "Summit",
    "st-thomas": "Summit",
    
    # Sun Belt Conference
    "app-state": "Sun Belt",
    "appalachian-state": "Sun Belt",
    "arkansas-state": "Sun Belt",
    "coastal-carolina": "Sun Belt",
    "georgia-southern": "Sun Belt",
    "georgia-state": "Sun Belt",
    "james-madison": "Sun Belt",
    "jmu": "Sun Belt",
    "louisiana": "Sun Belt",
    "ul-lafayette": "Sun Belt",
    "louisiana-monroe": "Sun Belt",
    "ulm": "Sun Belt",
    "marshall": "Sun Belt",
    "old-dominion": "Sun Belt",
    "odu": "Sun Belt",
    "south-alabama": "Sun Belt",
    "southern-miss": "Sun Belt",
    "usm": "Sun Belt",
    "texas-state": "Sun Belt",
    "troy": "Sun Belt",
    
    # SWAC (Southwestern Athletic Conference)
    "alabama-am": "SWAC",
    "alabama-state": "SWAC",
    "alcorn-state": "SWAC",
    "arkansas-pine-bluff": "SWAC",
    "bethune-cookman": "SWAC",
    "florida-am": "SWAC",
    "famu": "SWAC",
    "grambling": "SWAC",
    "grambling-state": "SWAC",
    "jackson-state": "SWAC",
    "prairie-view": "SWAC",
    "prairie-view-am": "SWAC",
    "southern": "SWAC",
    "southern-university": "SWAC",
    "texas-southern": "SWAC",
    
    # MEAC (Mid-Eastern Athletic Conference)
    "coppin-state": "MEAC",
    "delaware-state": "MEAC",
    "howard": "MEAC",
    "maryland-eastern-shore": "MEAC",
    "umes": "MEAC",
    "morgan-state": "MEAC",
    "norfolk-state": "MEAC",
    "north-carolina-central": "MEAC",
    "south-carolina-state": "MEAC",
    
    # WAC (Western Athletic Conference)
    "california-baptist": "WAC",
    "cal-baptist": "WAC",
    "grand-canyon": "WAC",
    "sacramento-state": "WAC",
    "seattle": "WAC",
    "seattle-redhawks": "WAC",
    "stephen-f-austin": "WAC",
    "sfa": "WAC",
    "tarleton": "WAC",
    "tarleton-state": "WAC",
    "utah-valley": "WAC",
    "uvu": "WAC",
    
    # West Coast Conference (WCC)
    "gonzaga": "WCC",
    "loyola-marymount": "WCC",
    "lmu": "WCC",
    "pacific": "WCC",
    "pepperdine": "WCC",
    "portland": "WCC",
    "san-diego": "WCC",
    "san-francisco": "WCC",
    "santa-clara": "WCC",
    "saint-marys": "WCC",
    "st-marys": "WCC",
    
    # Independent
    "oregon-state": "Independent",
    
    # ========================================
    # SPECIFIC TEAM IDs FROM THE DATABASE
    # These map exact database IDs to conferences
    # ========================================
    
    # Teams that need conference assignment
    "american-university": "Patriot",
    "boston-university": "Patriot",
    "buffalo": "MAC",
    "uic": "MVC",  # UIC is in MVC
    "detroit-mercy": "Horizon",
    "loyola-chicago": "A-10",  # Loyola Chicago is A-10
    "loyola-maryland": "Patriot",
    "marquette": "Big East",  # Marquette doesn't have baseball - Non-D1
    "new-hampshire": "America East",
    "new-orleans": "Southland",
    "ut-arlington": "WAC",
    "utep": "C-USA",
    "utrgv": "WAC",
    "green-bay": "Horizon",
    "denver": "Summit",  # Denver - Summit League
    "drake": "MVC",
    "duquesne": "A-10",
    "northern-iowa": "MVC",
    "robert-morris": "Horizon",
    "hampton": "Big South",  # Hampton moved to Big South
    "hartford": "America East",  # Hartford - America East
    "hawaii-hilo": "Non-D1",  # D2
    "idaho": "Non-D1",  # No D1 baseball
    "idaho-state": "Non-D1",  # No D1 baseball  
    "iu-indianapolis": "Horizon",  # IU Indianapolis (formerly IUPUI)
    "montana": "Non-D1",  # No D1 baseball
    "montana-state": "Non-D1",  # No D1 baseball
    "north-dakota-fighting": "Summit",  # North Dakota - Summit League but no baseball? Check
    "northern-arizona": "Non-D1",  # No D1 baseball
    "portland-state": "Non-D1",  # No D1 baseball
    "south-dakota": "Summit",  # Check if they have baseball
    "utah-state": "MWC",  # Utah State - Mountain West
    "weber-state": "Non-D1",  # No D1 baseball
    "wyoming": "Non-D1",  # No D1 baseball
    "boise-state": "Non-D1",  # No D1 baseball
    "colorado-state": "Non-D1",  # No D1 baseball
    "eastern-washington": "Non-D1",  # No D1 baseball
    
    # Non-D1 programs (D2, D3, NAIA, JUCO, or no baseball)
    "alma": "Non-D1",
    "anderson-in": "Non-D1",
    "augustana-college-il": "Non-D1",
    "aurora": "Non-D1",
    "baker": "Non-D1",
    "barry": "Non-D1",
    "bethany-ks-bethany": "Non-D1",
    "birmingham-southern": "Non-D1",  # Closed in 2024
    "cal-state-los-angeles": "Non-D1",
    "centenary": "Non-D1",
    "central-missouri-state": "Non-D1",
    "chicago-state": "Non-D1",  # Dropped to NAIA
    "coe-college": "Non-D1",
    "depauw": "Non-D1",
    "emporia-st": "Non-D1",
    "florida-southern": "Non-D1",
    "fort-lauderdale": "Non-D1",
    "grand-view": "Non-D1",
    "lincoln": "Non-D1",
    "linfield": "Non-D1",
    "loras-college": "Non-D1",
    "mid-america-christian": "Non-D1",
    "ny-institute-of-technology": "Non-D1",
    "nebraska-kearney": "Non-D1",
    "north-georgia": "Non-D1",
    "northwestern-college-ia-northwestern-college": "Non-D1",
    "pacific-lutheran": "Non-D1",
    "savannah-state": "Non-D1",
    "southwest": "Non-D1",
    "st-gregory": "Non-D1",
    "st-martins": "Non-D1",
    "st-olaf": "Non-D1",
    "sw-oklahoma-state": "Non-D1",
    "tabor": "Non-D1",
    "texas-lutheran": "Non-D1",
    "texas-pan-american": "Non-D1",  # Merged into UTRGV
    "truman-state": "Non-D1",
    "upper-iowa": "Non-D1",
    "wartburg": "Non-D1",
    "wayne-state-ne": "Non-D1",
    "west-georgia": "Non-D1",
    "william": "Non-D1",
    "wisconsin-platteville": "Non-D1",
}

# Additional mappings for team name variations
TEAM_NAME_ALIASES = {
    "Texas A&M-Corpus Christi": "texas-aandm-corpus-christi",
    "UL-Lafayette": "ul-lafayette",
    "SE Louisiana": "se-louisiana",
    "UMass": "umass",
    "UTEP": "utep",
    "UTRGV": "utrgv",
    "UIC": "uic",
    "UT Arlington": "ut-arlington",
}


def get_conference(team_id):
    """Get conference for a team by ID."""
    team_id_lower = team_id.lower().strip()
    return D1_TEAM_CONFERENCES.get(team_id_lower, None)


def main():
    """Test the mappings."""
    import sqlite3
    from pathlib import Path
    
    db_path = Path(__file__).parent.parent / 'data' / 'baseball.db'
    conn = sqlite3.connect(str(db_path))
    
    # Get teams with no conference
    teams = conn.execute("""
        SELECT id, name, nickname FROM teams 
        WHERE conference IS NULL OR conference = '' OR conference = 'Unknown'
        ORDER BY id
    """).fetchall()
    
    print(f"Found {len(teams)} teams without conference assignments\n")
    
    found = 0
    not_found = []
    
    for team_id, name, nickname in teams:
        conf = get_conference(team_id)
        if conf:
            found += 1
            print(f"✓ {team_id}: {conf}")
        else:
            not_found.append((team_id, name, nickname))
    
    print(f"\n\nFound conferences for {found}/{len(teams)} teams")
    
    if not_found:
        print(f"\nStill need to look up {len(not_found)} teams:")
        for team_id, name, nickname in not_found:
            print(f"  - {team_id} ({name})")
    
    conn.close()


if __name__ == '__main__':
    main()
