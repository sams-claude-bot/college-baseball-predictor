#!/usr/bin/env python3
"""
Parse scraped roster data for Coastal Carolina and Southern Miss
Convert the messy HTML text into structured player data
"""

import re
import sys
from pathlib import Path

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))

try:
    from player_stats import add_player
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.player_stats import add_player

def parse_coastal_carolina_roster():
    """Parse CCU roster from scraped data"""
    
    # The scraped CCU roster data (cleaned up)
    roster_text = """
    INF 5'9" 165 lbs S/R So. Centralia, Ill. Centralia High School John A Logan College
    RHP 6'6" 210 lbs R/R Jr. Egg Harbor Township, N.J. Egg Harbor
    INF 5'9" 170 lbs L/R So. Knoxville, Tenn. Farragut High School Mississippi State
    INF 5'11" 185 lbs R/R Jr. Lexington, S.C. River Bluff
    INF 6'0" 180 lbs R/R Jr. Aledo, Texas Weatherford College
    INF 5'9" 190 lbs L/R Sr. Allentown, Pa. Parkland
    OF 5'9" 170 lbs R/R Sr. Chester Springs, Pa. Owen J. Roberts
    INF 6'0" 170 lbs R/R R-Sr. San Mateo, Calif. Aragon College of San Mateo
    1B/OF 6'1" 210 lbs L/R R-Jr. Lumberton, N.C. Brunswick CC
    OF 5'10" 195 lbs L/L So. Manteca, Calif. San Diego
    C 5'10" 195 lbs R/R So. Camp Hill, Pa. Camp Hill Senior HS
    LHP/1B 6'3" 235 lbs L/L Fr. Charleston, S.C. Bishop England
    LHP 6'5" 215 lbs L/L Jr. Myrtle Beach, S.C. Socastee
    LHP 6'5" 225 lbs L/L Fr. Jenison, Mich. Jenison
    RHP 6'2" 205 lbs R/R So. Edison, N.J. Edison HS
    OF 5'10" 180 lbs R/R Jr. Haymarket, Va. Longwood
    OF 6'1" 185 lbs R/R Sr. Jacksonville, Fla. Stetson
    C 5'10" 190 lbs R/R Jr. Ft. Mitchell, Ky. P27 Academy
    RHP 6'7" 210 lbs R/R Jr. Ringgold, Ga. Ringgold High School Georgia State
    LHP 5'10" 175 lbs L/L Fr. Mullins, S.C. Pee Dee Academy
    OF 6'0" 200 lbs L/R Sr. Sayre, Pa. Oklahoma
    RHP 6'0" 181 lbs S/R Fr. Hartsville, S.C. P27 Academy
    RHP 6'4" 220 lbs R/R So. Erdenheim, Pa. The William Penn Charter School
    RHP 6'1" 185 lbs R/R Sr. Wilmington, N.C. Laney UNCW
    LHP 6'1" 190 lbs L/L Fr. Summerville, S.C. Eastern Carolina Academy
    1B 6'3" 215 lbs R/R Gr. Cochran, Ga. North Oconee High School North Georgia
    C 6'0" 190 lbs R/R Fr. Lancaster, PA Hempfield
    RHP 6'4" 195 lbs R/R Fr. Sicklerville, N.J. Gloucester Catholic
    RHP 6'4" 200 lbs R/R R-Sr. Bluffton, S.C. May River
    LHP 5'11" 180 lbs L/L Jr. Rocky Point, N.Y. Rocky Point
    LHP 6'6" 215 lbs L/L Fr. Lincolnton, N.C. North Lincoln
    C 6'2" 210 lbs L/R So. Novato, Calif. Texas
    RHP 5'11" 190 lbs R/R So. Greenville, S.C. JL Mann HS
    RHP 6'4" 225 lbs R/R Gr. Snohomish, Wash. Monroe Bellevue College
    """
    
    # Extract player info using regex
    pattern = r'(\w+(?:/\w+)*)\s+([\d\'\"]+)\s+(\d+)\s+lbs\s+([LRS]/[LR])\s+(\w+\.?)\s+(.+)'
    players = []
    
    lines = [line.strip() for line in roster_text.strip().split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            position = match.group(1)
            height = match.group(2)
            weight = int(match.group(3))
            bats_throws = match.group(4)
            year = match.group(5)
            hometown_school = match.group(6)
            
            # Parse bats/throws
            bats, throws = bats_throws.split('/')
            
            # Generate a name since names aren't in the scraped data
            name = f"CCU Player #{i+1}"
            
            players.append({
                'name': name,
                'number': i + 1,
                'position': position,
                'year': year,
                'height': height,
                'weight': weight,
                'bats': bats,
                'throws': throws,
                'hometown': hometown_school.split()[0] if hometown_school else None
            })
    
    return players

def parse_southern_miss_roster():
    """Parse Southern Miss roster from scraped data"""
    
    # The scraped USM roster data (cleaned up and structured)
    roster_text = """
    Position OF Academic Year Sr. Height 6' 0'' Weight 198 lbs R/R Hometown Jupiter, Fla.
    Position OF Academic Year Sr. Height 6' 1'' Weight 202 lbs R/R Hometown Baton Rouge, La.
    Position INF Academic Year Jr. Height 6' 2'' Weight 194 lbs L/R
    Position OF Academic Year Fr. Height 5' 9'' Weight 185 lbs R/R
    Position INF/RHP Academic Year So. Height 6' 0'' Weight 200 lbs R/R
    Position RHP Academic Year Sr. Height 6' 0'' Weight 195 lbs R/R
    Position OF Academic Year R-Sr. Height 6' 2'' Weight 220 lbs L/R Hometown Paducah, Ky.
    Position INF Academic Year Fr. Height 5' 9'' Weight 200 lbs L/R
    Position RHP Academic Year Fr. Height 6' 0'' Weight 165 lbs R/R
    Position C Academic Year Jr. Height 5' 10'' Weight 190 lbs R/R
    Position OF Academic Year R-Jr. Height 6' 1'' Weight 225 lbs R/R
    Position LHP Academic Year Sr. Height 5' 11'' Weight 202 lbs L/L
    Position C Academic Year Fr. Height 5' 9'' Weight 175 lbs B/R
    Position RHP Academic Year So. Height 6' 2'' Weight 215 lbs R/R
    Position INF Academic Year So. Height 6' 3'' Weight 214 lbs R/R
    Position LHP Academic Year So. Height 6' 1'' Weight 210 lbs L/L
    Position INF Academic Year Sr. Height 6' 0'' Weight 212 lbs L/R
    Position RHP Academic Year Jr. Height 6' 4'' Weight 200 lbs R/R
    Position INF Academic Year Sr. Height 6' 2'' Weight 255 lbs L/R
    Position RHP Academic Year So. Height 6' 1'' Weight 200 lbs R/R
    Position RHP Academic Year Fr. Height 6' 2'' Weight 220 lbs R/R
    Position INF Academic Year Fr. Height 5' 8'' Weight 165 lbs L/R
    Position LHP Academic Year Fr. Height 5' 9'' Weight 180 lbs L/L
    Position LHP Academic Year R-So. Height 6' 0'' Weight 215 lbs L/L
    Position RHP Academic Year So. Height 6' 4'' Weight 215 lbs R/R Hometown San Antonio, Texas
    """
    
    players = []
    lines = [line.strip() for line in roster_text.strip().split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        # Extract using regex
        pos_match = re.search(r'Position (\w+(?:/\w+)*)', line)
        year_match = re.search(r'Academic Year ([\w-]+\.?)', line)
        height_match = re.search(r'Height ([\d\'\s\"]+)', line)
        weight_match = re.search(r'Weight (\d+)', line)
        bats_throws_match = re.search(r'(\w)/(\w)', line)
        hometown_match = re.search(r'Hometown (.+)', line)
        
        if pos_match and year_match and height_match and weight_match:
            position = pos_match.group(1)
            year = year_match.group(1)
            height = height_match.group(1).strip()
            weight = int(weight_match.group(1))
            
            bats = throws = None
            if bats_throws_match:
                bats = bats_throws_match.group(1)
                throws = bats_throws_match.group(2)
            
            hometown = hometown_match.group(1) if hometown_match else None
            
            # Generate name
            name = f"USM Player #{i+1}"
            
            players.append({
                'name': name,
                'number': i + 1,
                'position': position,
                'year': year,
                'height': height,
                'weight': weight,
                'bats': bats,
                'throws': throws,
                'hometown': hometown
            })
    
    return players

def load_parsed_rosters():
    """Load the parsed rosters into the database"""
    
    print("=== Loading Parsed Rosters ===")
    
    # Coastal Carolina
    print("\n🐓 Coastal Carolina:")
    ccu_players = parse_coastal_carolina_roster()
    ccu_added = 0
    
    for player in ccu_players:
        try:
            add_player('coastal-carolina', **player)
            ccu_added += 1
        except Exception as e:
            print(f"  ⚠️  Error adding CCU player: {e}")
    
    print(f"  ✓ Added {ccu_added} Coastal Carolina players")
    
    # Southern Miss
    print("\n🦅 Southern Miss:")
    usm_players = parse_southern_miss_roster()
    usm_added = 0
    
    for player in usm_players:
        try:
            add_player('southern-miss', **player)
            usm_added += 1
        except Exception as e:
            print(f"  ⚠️  Error adding USM player: {e}")
    
    print(f"  ✓ Added {usm_added} Southern Miss players")
    
    print(f"\n✅ Total added: {ccu_added + usm_added} players")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test parsing
        print("Testing Coastal Carolina parsing...")
        ccu = parse_coastal_carolina_roster()
        print(f"Found {len(ccu)} CCU players")
        for p in ccu[:3]:
            print(f"  {p}")
            
        print("\nTesting Southern Miss parsing...")
        usm = parse_southern_miss_roster()
        print(f"Found {len(usm)} USM players")
        for p in usm[:3]:
            print(f"  {p}")
    else:
        load_parsed_rosters()

if __name__ == "__main__":
    main()