#!/usr/bin/env python3
"""
Populate team colors in the database.
Adds primary_color and secondary_color columns to teams table with official brand colors.
"""

import sys
from pathlib import Path

# Add project root to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from database import get_connection

# Comprehensive team color mapping: team_id -> (primary_hex, secondary_hex)
# Colors are official brand colors from school style guides where available

TEAM_COLORS = {
    # ═══════════════════════════════════════════════════════════════
    # SEC (provided exact colors)
    # ═══════════════════════════════════════════════════════════════
    'alabama': ('#9E1B32', '#FFFFFF'),
    'arkansas': ('#9D2235', '#FFFFFF'),
    'auburn': ('#0C2340', '#E87722'),
    'florida': ('#0021A5', '#FA4616'),
    'georgia': ('#BA0C2F', '#000000'),
    'kentucky': ('#0033A0', '#FFFFFF'),
    'lsu': ('#461D7C', '#FDD023'),
    'mississippi-state': ('#660000', '#FFFFFF'),
    'missouri': ('#F1B82D', '#000000'),
    'oklahoma': ('#841617', '#FFFFFF'),
    'ole-miss': ('#CE1126', '#14213D'),
    'south-carolina': ('#73000A', '#000000'),
    'tennessee': ('#FF8200', '#FFFFFF'),
    'texas': ('#BF5700', '#FFFFFF'),
    'texas-am': ('#500000', '#FFFFFF'),
    'vanderbilt': ('#CFB53B', '#000000'),
    
    # ═══════════════════════════════════════════════════════════════
    # ACC (provided exact colors)
    # ═══════════════════════════════════════════════════════════════
    'boston-college': ('#98002E', '#BC9B6A'),
    'california': ('#003262', '#FDB515'),
    'clemson': ('#F56600', '#522D80'),
    'duke': ('#003087', '#FFFFFF'),
    'florida-state': ('#782F40', '#CEB888'),
    'georgia-tech': ('#B3A369', '#003057'),
    'louisville': ('#AD0000', '#000000'),
    'miami-fl': ('#F47321', '#005030'),
    'nc-state': ('#CC0000', '#000000'),
    'north-carolina': ('#7BAFD4', '#13294B'),
    'notre-dame': ('#0C2340', '#C99700'),
    'pittsburgh': ('#003594', '#FFB81C'),
    'stanford': ('#8C1515', '#007360'),
    'virginia': ('#232D4B', '#F84C1E'),
    'virginia-tech': ('#630031', '#CF4420'),
    'wake-forest': ('#9E7E38', '#000000'),
    
    # ═══════════════════════════════════════════════════════════════
    # Big 12 (provided exact colors)
    # ═══════════════════════════════════════════════════════════════
    'arizona': ('#CC0033', '#003366'),
    'arizona-state': ('#8C1D40', '#FFC627'),
    'baylor': ('#154734', '#FFB81C'),
    'byu': ('#002E5D', '#FFFFFF'),
    'cincinnati': ('#E00122', '#000000'),
    'houston': ('#C8102E', '#FFFFFF'),
    'kansas': ('#0051BA', '#E8000D'),
    'kansas-state': ('#512888', '#FFFFFF'),
    'oklahoma-state': ('#FF6600', '#000000'),
    'tcu': ('#4D1979', '#A3A9AC'),
    'texas-tech': ('#CC0000', '#000000'),
    'ucf': ('#BA9B37', '#000000'),
    'utah': ('#CC0000', '#FFFFFF'),
    'west-virginia': ('#002855', '#EAAA00'),
    
    # ═══════════════════════════════════════════════════════════════
    # Big Ten (provided exact colors)
    # ═══════════════════════════════════════════════════════════════
    'illinois': ('#E84A27', '#13294B'),
    'indiana': ('#990000', '#FFFFFF'),
    'iowa': ('#FFCD00', '#000000'),
    'maryland': ('#E03C31', '#FFD520'),
    'michigan': ('#00274C', '#FFCB05'),
    'michigan-state': ('#18453B', '#FFFFFF'),
    'minnesota': ('#7A0019', '#FFCC33'),
    'nebraska': ('#E41C38', '#FFFFFF'),
    'northwestern': ('#4E2A84', '#000000'),
    'ohio-state': ('#BB0000', '#666666'),
    'oregon': ('#154733', '#FEE11A'),
    'penn-state': ('#041E42', '#FFFFFF'),
    'purdue': ('#CEB888', '#000000'),
    'rutgers': ('#CC0033', '#FFFFFF'),
    'usc': ('#990000', '#FFC72C'),
    'washington': ('#4B2E83', '#E8D3A2'),
    
    # ═══════════════════════════════════════════════════════════════
    # Pac-12 / Other Power Programs (provided exact colors)
    # ═══════════════════════════════════════════════════════════════
    'ucla': ('#2D68C4', '#F2A900'),
    'oregon-state': ('#DC4405', '#000000'),
    'coastal-carolina': ('#006F71', '#A27752'),
    'washington-state': ('#981E32', '#5E6A71'),
    
    # ═══════════════════════════════════════════════════════════════
    # A-10 Conference
    # ═══════════════════════════════════════════════════════════════
    'davidson': ('#CC0000', '#000000'),
    'dayton': ('#CE1141', '#004B8D'),
    'fordham': ('#8C2633', '#FFFFFF'),
    'george-mason': ('#006633', '#FFCC33'),
    'george-washington': ('#004C97', '#FFCD00'),
    'la-salle': ('#0038A8', '#FFD200'),
    'massachusetts': ('#881C1C', '#000000'),
    'rhode-island': ('#75B2DD', '#041E42'),
    'richmond': ('#990000', '#000066'),
    'saint-josephs': ('#9E1B34', '#FFFFFF'),
    'saint-louis': ('#003DA5', '#FFFFFF'),
    'st-bonaventure': ('#6E2C00', '#FFFFFF'),
    'vcu': ('#FFCC00', '#000000'),
    
    # ═══════════════════════════════════════════════════════════════
    # AAC Conference
    # ═══════════════════════════════════════════════════════════════
    'charlotte': ('#005035', '#B19B6B'),
    'east-carolina': ('#592A8A', '#FFC425'),
    'florida-atlantic': ('#003366', '#CC0000'),
    'memphis': ('#003087', '#898D8D'),
    'rice': ('#003D7C', '#5E6A71'),
    'south-florida': ('#006747', '#CFC493'),
    'tulane': ('#006747', '#8CC63F'),
    'uab': ('#006341', '#FFD500'),
    'utsa': ('#0C2340', '#F47321'),
    'wichita-state': ('#FFCD00', '#000000'),
    
    # ═══════════════════════════════════════════════════════════════
    # ASUN Conference
    # ═══════════════════════════════════════════════════════════════
    'bellarmine': ('#9E1B34', '#FFFFFF'),
    'central-arkansas': ('#4F2D7F', '#808080'),
    'eastern-kentucky': ('#861F41', '#FFFFFF'),
    'florida-gulf-coast': ('#004C97', '#007A53'),
    'jacksonville': ('#006341', '#FFFFFF'),
    'lipscomb': ('#4F2683', '#FFC72C'),
    'north-alabama': ('#660066', '#FFCC00'),
    'north-florida': ('#00529B', '#5E6A71'),
    'queens-university': ('#003057', '#C5B783'),
    'stetson': ('#006747', '#FFFFFF'),
    
    # ═══════════════════════════════════════════════════════════════
    # America East Conference
    # ═══════════════════════════════════════════════════════════════
    'binghamton': ('#005A43', '#FFFFFF'),
    'bryant': ('#000000', '#FFCC00'),
    'maine': ('#003263', '#75B2DD'),
    'njit': ('#C41230', '#FFFFFF'),
    'albany': ('#461D7C', '#FFCC00'),  # UAlbany
    'umbc': ('#000000', '#FFCC00'),
    'umass-lowell': ('#0067B1', '#CC0000'),
    
    # ═══════════════════════════════════════════════════════════════
    # Big East Conference
    # ═══════════════════════════════════════════════════════════════
    'butler': ('#13294B', '#FFFFFF'),
    'creighton': ('#005CA9', '#FFFFFF'),
    'georgetown': ('#041E42', '#63666A'),
    'seton-hall': ('#0067B1', '#FFFFFF'),
    'st-johns': ('#CC0000', '#FFFFFF'),
    'uconn': ('#0E1A2C', '#FFFFFF'),
    'villanova': ('#00205B', '#FFFFFF'),
    'xavier': ('#002D72', '#9EA2A2'),
    
    # ═══════════════════════════════════════════════════════════════
    # Big Sky Conference
    # ═══════════════════════════════════════════════════════════════
    'sacramento-state': ('#043927', '#C4B581'),
    
    # ═══════════════════════════════════════════════════════════════
    # Big South Conference
    # ═══════════════════════════════════════════════════════════════
    'campbell': ('#F47920', '#000000'),
    'charleston-southern': ('#00529B', '#C8B273'),
    'gardner-webb': ('#C8102E', '#000000'),
    'high-point': ('#330072', '#FFFFFF'),
    'longwood': ('#003DA5', '#A50034'),
    'presbyterian': ('#00539F', '#CC0000'),
    'radford': ('#CC0000', '#003366'),
    'south-carolina-upstate': ('#00704A', '#000000'),
    'unc-asheville': ('#003DA5', '#FFFFFF'),
    'winthrop': ('#8B1F41', '#FFC72C'),
    
    # ═══════════════════════════════════════════════════════════════
    # Big West Conference
    # ═══════════════════════════════════════════════════════════════
    'cal-poly': ('#003D2D', '#C69214'),
    'cal-state-bakersfield': ('#003DA5', '#FFB81C'),
    'cal-state-fullerton': ('#00274C', '#FF6600'),
    'cal-state-northridge': ('#CC0000', '#000000'),
    'hawaii': ('#024731', '#FFFFFF'),
    'long-beach-state': ('#000000', '#FFC72C'),
    'uc-davis': ('#002855', '#C99700'),
    'uc-irvine': ('#0064A4', '#FFC72C'),
    'uc-riverside': ('#003DA5', '#FFB81C'),
    'uc-san-diego': ('#182B49', '#C69214'),
    'uc-santa-barbara': ('#003660', '#FEBC11'),
    
    # ═══════════════════════════════════════════════════════════════
    # C-USA Conference
    # ═══════════════════════════════════════════════════════════════
    'florida-international': ('#081E3F', '#B6862C'),
    'jacksonville-state': ('#CC0000', '#FFFFFF'),
    'kennesaw-state': ('#FDBB30', '#000000'),
    'liberty': ('#002D62', '#C41E3A'),
    'louisiana-tech': ('#002F8B', '#CC0000'),
    'middle-tennessee': ('#0066CC', '#FFFFFF'),
    'new-mexico-state': ('#8C0033', '#000000'),
    'sam-houston': ('#F47321', '#FFFFFF'),
    'western-kentucky': ('#CC0000', '#FFFFFF'),
    
    # ═══════════════════════════════════════════════════════════════
    # CAA Conference
    # ═══════════════════════════════════════════════════════════════
    'charleston': ('#8C1D40', '#C8B273'),
    'delaware': ('#00539F', '#FFD200'),
    'elon': ('#6B0F28', '#C9A227'),
    'hofstra': ('#00539B', '#FFD200'),
    'monmouth': ('#003087', '#5E6A71'),
    'north-carolina-at': ('#00539B', '#C89100'),
    'northeastern': ('#CC0000', '#000000'),
    'stony-brook': ('#990000', '#808080'),
    'towson': ('#FFC425', '#000000'),
    'unc-wilmington': ('#006666', '#FFC72C'),
    'william-mary': ('#115740', '#FFC72C'),
    
    # ═══════════════════════════════════════════════════════════════
    # Horizon League
    # ═══════════════════════════════════════════════════════════════
    'milwaukee': ('#000000', '#FFCC00'),
    'northern-kentucky': ('#000000', '#FFC72C'),
    'oakland': ('#000000', '#B89D5B'),
    'uic': ('#001E62', '#CC0033'),
    'wright-state': ('#007A33', '#FFCD00'),
    'youngstown-state': ('#CC0000', '#FFFFFF'),
    
    # ═══════════════════════════════════════════════════════════════
    # Ivy League
    # ═══════════════════════════════════════════════════════════════
    'brown': ('#4E3629', '#CC0000'),
    'columbia': ('#75AADB', '#002868'),
    'cornell': ('#B31B1B', '#FFFFFF'),
    'dartmouth': ('#00693E', '#FFFFFF'),
    'harvard': ('#A51C30', '#000000'),
    'pennsylvania': ('#011F5B', '#990000'),
    'princeton': ('#FF6600', '#000000'),
    'yale': ('#0F4D92', '#FFFFFF'),
    
    # ═══════════════════════════════════════════════════════════════
    # MAAC Conference
    # ═══════════════════════════════════════════════════════════════
    'canisius': ('#002D62', '#FFB81C'),
    'fairfield': ('#C8102E', '#FFFFFF'),
    'iona': ('#8B0000', '#FFD700'),
    'manhattan': ('#006633', '#FFFFFF'),
    'marist': ('#C8102E', '#FFFFFF'),
    'niagara': ('#582C83', '#FFFFFF'),
    'quinnipiac': ('#002D72', '#FFB81C'),
    'rider': ('#C8102E', '#000000'),
    'saint-peters': ('#00529B', '#FFFFFF'),
    'siena': ('#006633', '#FFCC33'),
    
    # ═══════════════════════════════════════════════════════════════
    # MAC Conference
    # ═══════════════════════════════════════════════════════════════
    'akron': ('#003366', '#C6933C'),
    'ball-state': ('#BA0C2F', '#FFFFFF'),
    'bowling-green': ('#F47321', '#5E3C16'),
    'central-michigan': ('#6A0032', '#FFC72C'),
    'eastern-michigan': ('#006633', '#FFFFFF'),
    'kent-state': ('#002664', '#EAAB00'),
    'miami-ohio': ('#B61E2E', '#FFFFFF'),
    'northern-illinois': ('#CC0000', '#000000'),
    'ohio': ('#00694E', '#FFFFFF'),
    'toledo': ('#15397F', '#FFC72C'),
    'western-michigan': ('#4F2C1D', '#FFC72C'),
    
    # ═══════════════════════════════════════════════════════════════
    # MEAC Conference
    # ═══════════════════════════════════════════════════════════════
    'alabama-state': ('#000000', '#FFC72C'),
    'bethune-cookman': ('#8C1D40', '#FFC72C'),
    'coppin-state': ('#002D62', '#C9A227'),
    'delaware-state': ('#D7182A', '#001489'),
    'maryland-eastern-shore': ('#8B0000', '#808080'),
    'norfolk-state': ('#007A33', '#C9A227'),
    
    # ═══════════════════════════════════════════════════════════════
    # MVC (Missouri Valley)
    # ═══════════════════════════════════════════════════════════════
    'belmont': ('#003087', '#CC0000'),
    'bradley': ('#CC0000', '#FFFFFF'),
    'dallas-baptist': ('#002D72', '#CC0000'),
    'evansville': ('#663399', '#F47321'),
    'illinois-state': ('#CE1126', '#FFFFFF'),
    'indiana-state': ('#00539B', '#FFFFFF'),
    'missouri-state': ('#5F0F1A', '#FFFFFF'),
    'southern-illinois': ('#660000', '#FFFFFF'),
    'valparaiso': ('#613318', '#FFC72C'),
    
    # ═══════════════════════════════════════════════════════════════
    # Mountain West Conference
    # ═══════════════════════════════════════════════════════════════
    'air-force': ('#003087', '#8A8D8F'),
    'fresno-state': ('#CC0033', '#002E6D'),
    'nevada': ('#003366', '#A5ACAF'),
    'new-mexico': ('#CC0033', '#8A8D8F'),
    'san-diego-state': ('#CC0033', '#000000'),
    'san-jose-state': ('#0055A2', '#E5A823'),
    'unlv': ('#CC0033', '#8F8E8C'),
    
    # ═══════════════════════════════════════════════════════════════
    # NEC (Northeast Conference)
    # ═══════════════════════════════════════════════════════════════
    'central-connecticut': ('#00529B', '#FFFFFF'),
    'fairleigh-dickinson': ('#002D72', '#C8102E'),
    'le-moyne': ('#006747', '#FFD100'),
    'long-island-university': ('#000000', '#FFCD00'),
    'mercyhurst': ('#006747', '#003087'),
    'merrimack': ('#002D72', '#FFC72C'),
    'mount-st-marys': ('#002D72', '#FFFFFF'),
    'sacred-heart': ('#CC0033', '#8A8D8F'),
    'stonehill': ('#582C83', '#FFFFFF'),
    'wagner': ('#006633', '#FFFFFF'),
    
    # ═══════════════════════════════════════════════════════════════
    # Non-D1 / Independent (still in database)
    # ═══════════════════════════════════════════════════════════════
    'colgate': ('#821019', '#FFFFFF'),
    'furman': ('#582C83', '#FFFFFF'),
    'hawaii-hilo': ('#CC0000', '#000000'),
    'new-haven': ('#003087', '#FFC72C'),
    'syracuse': ('#F76900', '#002D72'),
    'tulsa': ('#002D72', '#C8102E'),
    'west-georgia': ('#002D72', '#CC0000'),
    
    # ═══════════════════════════════════════════════════════════════
    # OVC (Ohio Valley Conference)
    # ═══════════════════════════════════════════════════════════════
    'austin-peay': ('#CC0033', '#FFFFFF'),
    'eastern-illinois': ('#004B8D', '#8A8D8F'),
    'lindenwood': ('#000000', '#FFD100'),
    'little-rock': ('#8C1D40', '#808080'),
    'morehead-state': ('#003087', '#FFC72C'),
    'murray-state': ('#002D62', '#FFC72C'),
    'siu-edwardsville': ('#CC0000', '#000000'),
    'southeast-missouri': ('#CC0033', '#000000'),
    'southern-indiana': ('#CC0033', '#003366'),
    'tennessee-tech': ('#4F2D7F', '#FFC72C'),
    'ut-martin': ('#002D62', '#F77F00'),
    
    # ═══════════════════════════════════════════════════════════════
    # Patriot League
    # ═══════════════════════════════════════════════════════════════
    'army': ('#000000', '#C5B783'),
    'bucknell': ('#002D62', '#F77F00'),
    'holy-cross': ('#602D89', '#FFFFFF'),
    'lafayette': ('#6C0633', '#FFFFFF'),
    'lehigh': ('#653600', '#FFFFFF'),
    'navy': ('#00205B', '#C5B783'),
    
    # ═══════════════════════════════════════════════════════════════
    # SWAC Conference
    # ═══════════════════════════════════════════════════════════════
    'alabama-am': ('#660000', '#FFFFFF'),
    'alcorn-state': ('#582C83', '#FFD100'),
    'arkansas-pine-bluff': ('#000000', '#FFC72C'),
    'florida-am': ('#FF6600', '#006338'),
    'grambling-state': ('#000000', '#FFC72C'),
    'jackson-state': ('#002D72', '#CC0000'),
    'mississippi-valley-state': ('#006633', '#FFFFFF'),
    'prairie-view': ('#582C83', '#FFC72C'),
    'southern': ('#0033A0', '#FFD100'),
    'texas-southern': ('#660000', '#808080'),
    
    # ═══════════════════════════════════════════════════════════════
    # SoCon (Southern Conference)
    # ═══════════════════════════════════════════════════════════════
    'east-tennessee-state': ('#002D62', '#FFC72C'),
    'mercer': ('#F47321', '#000000'),
    'samford': ('#002D72', '#CC0033'),
    'the-citadel': ('#003087', '#FFFFFF'),
    'unc-greensboro': ('#002D62', '#FFC72C'),
    'vmi': ('#CC0000', '#FFCC00'),
    'western-carolina': ('#582C83', '#FFC72C'),
    'wofford': ('#8B7500', '#000000'),
    
    # ═══════════════════════════════════════════════════════════════
    # Southland Conference
    # ═══════════════════════════════════════════════════════════════
    'houston-christian': ('#002D72', '#FF6600'),
    'incarnate-word': ('#CC0000', '#000000'),
    'lamar': ('#CC0000', '#FFFFFF'),
    'mcneese': ('#002D72', '#FFD100'),
    'new-orleans': ('#002D62', '#8A8D8F'),
    'nicholls': ('#CC0033', '#808080'),
    'northwestern-state': ('#582C83', '#F77F00'),
    'southeastern-louisiana': ('#006338', '#FFC72C'),
    'texas-aandm-corpus-christi': ('#003366', '#007A53'),
    
    # ═══════════════════════════════════════════════════════════════
    # Summit League
    # ═══════════════════════════════════════════════════════════════
    'north-dakota-state': ('#006633', '#FFC72C'),
    'northern-colorado': ('#002D62', '#FFC72C'),
    'omaha': ('#000000', '#CC0033'),
    'oral-roberts': ('#002D62', '#C5B783'),
    'south-dakota-state': ('#002D62', '#FFC72C'),
    'st-thomas-minnesota': ('#582C83', '#808080'),
    'western-illinois': ('#582C83', '#FFC72C'),
    
    # ═══════════════════════════════════════════════════════════════
    # Sun Belt Conference
    # ═══════════════════════════════════════════════════════════════
    'appalachian-state': ('#000000', '#FFCC00'),
    'arkansas-state': ('#CC092F', '#000000'),
    'georgia-southern': ('#002D62', '#87714D'),
    'georgia-state': ('#0039A6', '#CC0033'),
    'james-madison': ('#450084', '#C5A47E'),
    'louisiana': ('#CE181E', '#FFFFFF'),
    'marshall': ('#00B140', '#FFFFFF'),
    'old-dominion': ('#003057', '#7C878E'),
    'south-alabama': ('#00205B', '#CC0000'),
    'southern-miss': ('#000000', '#FFCC00'),
    'texas-state': ('#501214', '#C5A47E'),
    'troy': ('#8B0000', '#808080'),
    'ul-monroe': ('#660000', '#FFC72C'),
    
    # ═══════════════════════════════════════════════════════════════
    # WAC Conference
    # ═══════════════════════════════════════════════════════════════
    'abilene-christian': ('#582C83', '#FFFFFF'),
    'california-baptist': ('#002D72', '#FFB81C'),
    'grand-canyon': ('#522D80', '#FFFFFF'),
    'seattle': ('#CC0033', '#FFFFFF'),
    'stephen-f-austin': ('#4F2D7F', '#FFFFFF'),
    'tarleton-state': ('#582C83', '#FFFFFF'),
    'ut-arlington': ('#0064B1', '#F77F00'),
    'utrgv': ('#002D62', '#F77F00'),
    'utah-tech': ('#CC0033', '#002D62'),
    'utah-valley': ('#006341', '#FFFFFF'),
    
    # ═══════════════════════════════════════════════════════════════
    # WCC Conference
    # ═══════════════════════════════════════════════════════════════
    'gonzaga': ('#002D62', '#CC0033'),
    'loyola-marymount': ('#8B1F41', '#00529B'),
    'pacific': ('#F77F00', '#000000'),
    'pepperdine': ('#002D62', '#FF6600'),
    'portland': ('#582C83', '#FFFFFF'),
    'saint-marys': ('#CC0033', '#002D62'),
    'san-diego': ('#002D72', '#75B2DD'),
    'san-francisco': ('#006338', '#FFCC00'),
    'santa-clara': ('#8C1515', '#FFFFFF'),
}

def add_color_columns(conn):
    """Add color columns to teams table if they don't exist."""
    cursor = conn.cursor()
    
    # Check if columns exist
    cursor.execute("PRAGMA table_info(teams)")
    columns = {row['name'] for row in cursor.fetchall()}
    
    if 'primary_color' not in columns:
        print("Adding primary_color column...")
        cursor.execute("ALTER TABLE teams ADD COLUMN primary_color TEXT")
    
    if 'secondary_color' not in columns:
        print("Adding secondary_color column...")
        cursor.execute("ALTER TABLE teams ADD COLUMN secondary_color TEXT")
    
    conn.commit()
    print("✓ Color columns ready")


def populate_colors(conn):
    """Update all teams with their colors."""
    cursor = conn.cursor()
    
    # Get all teams
    cursor.execute("SELECT id, name FROM teams")
    teams = cursor.fetchall()
    
    updated = 0
    missing = []
    
    for team in teams:
        team_id = team['id']
        if team_id in TEAM_COLORS:
            primary, secondary = TEAM_COLORS[team_id]
            cursor.execute('''
                UPDATE teams 
                SET primary_color = ?, secondary_color = ?
                WHERE id = ?
            ''', (primary, secondary, team_id))
            updated += 1
        else:
            missing.append((team_id, team['name']))
    
    conn.commit()
    
    print(f"✓ Updated {updated} teams with colors")
    
    if missing:
        print(f"\n⚠ {len(missing)} teams without color mapping:")
        for team_id, name in sorted(missing):
            print(f"  - {team_id}: {name}")
            # Set a default gray color for unmapped teams
            cursor.execute('''
                UPDATE teams 
                SET primary_color = ?, secondary_color = ?
                WHERE id = ? AND primary_color IS NULL
            ''', ('#5E6A71', '#FFFFFF', team_id))
        conn.commit()
        print(f"  (set default gray for unmapped teams)")


def verify_colors(conn):
    """Verify colors were populated correctly."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN primary_color IS NOT NULL THEN 1 ELSE 0 END) as with_colors
        FROM teams
    ''')
    result = cursor.fetchone()
    print(f"\n📊 Verification: {result['with_colors']}/{result['total']} teams have colors")
    
    # Show a few examples
    cursor.execute('''
        SELECT id, name, primary_color, secondary_color 
        FROM teams 
        WHERE primary_color IS NOT NULL 
        ORDER BY name 
        LIMIT 10
    ''')
    print("\nSample colors:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['primary_color']} / {row['secondary_color']}")


def main():
    print("🎨 Populating NCAA Team Colors")
    print("=" * 50)
    
    conn = get_connection()
    
    try:
        add_color_columns(conn)
        populate_colors(conn)
        verify_colors(conn)
        print("\n✅ Team colors populated successfully!")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
