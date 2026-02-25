#!/usr/bin/env python3
"""
Download ESPN team logos for all college baseball teams.
Maps our team slugs to ESPN team IDs and downloads 500px logos.
"""

import os
import sys
import json
import sqlite3
import urllib.request
from pathlib import Path

# Manual overrides for tricky mappings (our_slug -> espn_display_name)
MANUAL_OVERRIDES = {
    'miami-fl': 'Miami Hurricanes',
    'miami-oh': 'Miami (OH) RedHawks',
    'usc': 'USC Trojans',
    'lsu': 'LSU Tigers',
    'ucf': 'UCF Knights',
    'uconn': 'UConn Huskies',
    'unlv': 'UNLV Rebels',
    'utsa': 'UTSA Roadrunners',
    'utep': 'UTEP Miners',
    'fiu': 'FIU Panthers',
    'vcu': 'VCU Rams',
    'ole-miss': 'Ole Miss Rebels',
    'smu': 'SMU Mustangs',
    'tcu': 'TCU Horned Frogs',
    'uab': 'UAB Blazers',
    'unc': 'North Carolina Tar Heels',
    'unc-wilmington': 'UNC Wilmington Seahawks',
    'unc-greensboro': 'UNC Greensboro Spartans',
    'unc-asheville': 'UNC Asheville Bulldogs',
    'north-carolina-at': 'North Carolina A&T Aggies',
    'florida-gulf-coast': 'Florida Gulf Coast Eagles',
    'texas-am': 'Texas A&M Aggies',
    'texas-am-corpus-christi': 'Texas A&M-CC Islanders',
    'louisiana': 'Louisiana Ragin\' Cajuns',
    'louisiana-monroe': 'Louisiana-Monroe Warhawks',
    'louisiana-tech': 'Louisiana Tech Bulldogs',
    'central-florida': 'UCF Knights',
    'south-florida': 'South Florida Bulls',
    'tennessee-martin': 'UT Martin Skyhawks',
    'middle-tennessee': 'Middle Tennessee Blue Raiders',
    'east-tennessee-state': 'East Tennessee State Buccaneers',
    'south-carolina-upstate': 'USC Upstate Spartans',
    'texas-southern': 'Texas Southern Tigers',
    'texas-rio-grande-valley': 'UTRGV Vaqueros',
    'prairie-view': 'Prairie View A&M Panthers',
    'alabama-am': 'Alabama A&M Bulldogs',
    'bethune-cookman': 'Bethune-Cookman Wildcats',
    'florida-am': 'Florida A&M Rattlers',
    'north-carolina-central': 'NC Central Eagles',
    'south-carolina-state': 'South Carolina State Bulldogs',
    'arkansas-pine-bluff': 'Arkansas-Pine Bluff Golden Lions',
    'maryland-eastern-shore': 'Maryland-Eastern Shore Hawks',
    'cal-state-fullerton': 'Cal State Fullerton Titans',
    'cal-state-northridge': 'Cal State Northridge Matadors',
    'cal-state-bakersfield': 'Cal State Bakersfield Roadrunners',
    'long-beach-state': 'Long Beach State Beach',
    'san-jose-state': 'San José State Spartans',
    'san-diego-state': 'San Diego State Aztecs',
    'fresno-state': 'Fresno State Bulldogs',
    'boise-state': 'Boise State Broncos',
    'colorado-state': 'Colorado State Rams',
    'utah-state': 'Utah State Aggies',
    'new-mexico-state': 'New Mexico State Aggies',
    'penn-state': 'Penn State Nittany Lions',
    'ohio-state': 'Ohio State Buckeyes',
    'michigan-state': 'Michigan State Spartans',
    'iowa-state': 'Iowa State Cyclones',
    'kansas-state': 'Kansas State Wildcats',
    'oklahoma-state': 'Oklahoma State Cowboys',
    'arizona-state': 'Arizona State Sun Devils',
    'washington-state': 'Washington State Cougars',
    'oregon-state': 'Oregon State Beavers',
    'florida-state': 'Florida State Seminoles',
    'georgia-state': 'Georgia State Panthers',
    'mississippi-state': 'Mississippi State Bulldogs',
    'ball-state': 'Ball State Cardinals',
    'kent-state': 'Kent State Golden Flashes',
    'wright-state': 'Wright State Raiders',
    'youngstown-state': 'Youngstown State Penguins',
    'cleveland-state': 'Cleveland State Vikings',
    'wichita-state': 'Wichita State Shockers',
    'illinois-state': 'Illinois State Redbirds',
    'indiana-state': 'Indiana State Sycamores',
    'missouri-state': 'Missouri State Bears',
    'murray-state': 'Murray State Racers',
    'morehead-state': 'Morehead State Eagles',
    'appalachian-state': 'Appalachian State Mountaineers',
    'coastal-carolina': 'Coastal Carolina Chanticleers',
    'western-kentucky': 'Western Kentucky Hilltoppers',
    'northwestern-state': 'Northwestern State Demons',
    'mcneese-state': 'McNeese Cowboys',
    'nicholls-state': 'Nicholls Colonels',
    'southeastern-louisiana': 'Southeastern Louisiana Lions',
    'stephen-f-austin': 'Stephen F. Austin Lumberjacks',
    'sam-houston-state': 'Sam Houston Bearkats',
    'houston-christian': 'Houston Christian Huskies',
    'tarleton-state': 'Tarleton State Texans',
    'western-illinois': 'Western Illinois Leathernecks',
    'southern-illinois': 'Southern Illinois Salukis',
    'northern-illinois': 'Northern Illinois Huskies',
    'byu': 'BYU Cougars',
    'nc-state': 'NC State Wolfpack',
    'uic': 'UIC Flames',
    'umass': 'UMass Minutemen',
    'umbc': 'UMBC Retrievers',
    'njit': 'NJIT Highlanders',
    'ipfw': 'Purdue Fort Wayne Mastodons',
    'iupui': 'IUPUI Jaguars',
    'siu-edwardsville': 'SIU Edwardsville Cougars',
    'ut-arlington': 'UT Arlington Mavericks',
    'ut-san-antonio': 'UTSA Roadrunners',
    'ut-rio-grande-valley': 'UTRGV Vaqueros',
    'little-rock': 'Little Rock Trojans',
    'southeastern-missouri-state': 'Southeast Missouri State Redhawks',
    'austin-peay': 'Austin Peay Governors',
    'tennessee-tech': 'Tennessee Tech Golden Eagles',
    'jacksonville-state': 'Jacksonville State Gamecocks',
    'kennesaw-state': 'Kennesaw State Owls',
    'central-arkansas': 'Central Arkansas Bears',
    'sacramento-state': 'Sacramento State Hornets',
    'weber-state': 'Weber State Wildcats',
    'southern-utah': 'Southern Utah Thunderbirds',
    'northern-colorado': 'Northern Colorado Bears',
    'grand-canyon': 'Grand Canyon Antelopes',
    'seattle': 'Seattle Redhawks',
    'abilene-christian': 'Abilene Christian Wildcats',
    'california-baptist': 'California Baptist Lancers',
    'dixie-state': 'Utah Tech Trailblazers',
    'alcorn-state': 'Alcorn State Braves',
    'jackson-state': 'Jackson State Tigers',
    'grambling-state': 'Grambling Tigers',
    'southern': 'Southern Jaguars',
    'alabama-state': 'Alabama State Hornets',
    'coppin-state': 'Coppin State Eagles',
    'delaware-state': 'Delaware State Hornets',
    'norfolk-state': 'Norfolk State Spartans',
    'morgan-state': 'Morgan State Bears',
    'howard': 'Howard Bison',
    # Additional mappings for remaining teams
    'utrgv': 'UT Rio Grande Valley Vaqueros',
    'western-carolina': 'Western Carolina Catamounts',
    'eastern-kentucky': 'Eastern Kentucky Colonels',
    'northern-kentucky': 'Northern Kentucky Norse',
    'eastern-michigan': 'Eastern Michigan Eagles',
    'southern-indiana': 'Southern Indiana Screaming Eagles',
    'charleston-southern': 'Charleston Southern Buccaneers',
    'hawaii': 'Hawai\'i Rainbow Warriors',
    'pittsburgh': 'Pittsburgh Panthers',
    'eastern-illinois': 'Eastern Illinois Panthers',
    'seattle': 'Seattle U Redhawks',
    'southeastern-louisiana': 'SE Louisiana Lions',
    'central-michigan': 'Central Michigan Chippewas',
    'uc-santa-barbara': 'UC Santa Barbara Gauchos',
    'florida-atlantic': 'Florida Atlantic Owls',
    'western-michigan': 'Western Michigan Broncos',
    'georgia-southern': 'Georgia Southern Eagles',
    'appalachian-state': 'App State Mountaineers',
    'miami-ohio': 'Miami (OH) RedHawks',
    'southeast-missouri': 'Southeast Missouri State Redhawks',
    'loyola-marymount': 'Loyola Marymount Lions',
    'central-connecticut': 'Central Connecticut Blue Devils',
    'fairleigh-dickinson': 'Fairleigh Dickinson Knights',
    'florida-international': 'Florida International Panthers',
    'george-washington': 'George Washington Revolutionaries',
    'hawaii-hilo': 'Hawai\'i Hilo Vulcans',
    'maryland-eastern-shore': 'Maryland Eastern Shore Hawks',
    'massachusetts': 'Massachusetts Minutemen',
    'mississippi-valley-state': 'Mississippi Valley State Delta Devils',
    'pennsylvania': 'Pennsylvania Quakers',
    'south-carolina-upstate': 'South Carolina Upstate Spartans',
    'st-thomas-minnesota': 'St. Thomas-Minnesota Tommies',
    'texas-aandm-corpus-christi': 'Texas A&M-Corpus Christi Islanders',
    'albany': 'UAlbany Great Danes',
    'new-haven': 'New Haven Chargers',
}


def normalize_name(name):
    """Normalize team name for matching."""
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [' university', ' college', ' state', ' a&m']:
        name = name.replace(suffix, '')
    # Remove punctuation
    for ch in "'-().":
        name = name.replace(ch, '')
    return name.strip()


def main():
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'baseball.db'
    logos_dir = project_root / 'web' / 'static' / 'logos'
    
    # Create logos directory
    logos_dir.mkdir(parents=True, exist_ok=True)
    
    # Load our teams
    conn = sqlite3.connect(db_path)
    our_teams = conn.execute('SELECT id, name FROM teams').fetchall()
    conn.close()
    print(f"Our teams: {len(our_teams)}")
    
    # Fetch ESPN teams
    print("Fetching ESPN team data...")
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams?limit=500"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    
    espn_teams = {}
    for t in data['sports'][0]['leagues'][0]['teams']:
        team = t['team']
        logo_url = team['logos'][0]['href'] if team.get('logos') else None
        espn_teams[team['id']] = {
            'name': team.get('displayName', ''),
            'short': team.get('shortDisplayName', ''),
            'abbrev': team.get('abbreviation', ''),
            'logo': logo_url,
        }
    print(f"ESPN teams: {len(espn_teams)}")
    
    # Build lookup by normalized name
    espn_by_name = {}
    espn_by_short = {}
    for eid, e in espn_teams.items():
        espn_by_name[normalize_name(e['name'])] = (eid, e)
        espn_by_short[normalize_name(e['short'])] = (eid, e)
    
    # Also index by display name directly
    espn_by_display = {e['name'].lower(): (eid, e) for eid, e in espn_teams.items()}
    
    matched = 0
    unmatched = []
    
    for our_id, our_name in our_teams:
        logo_path = logos_dir / f"{our_id}.png"
        
        # Skip if already downloaded
        if logo_path.exists():
            matched += 1
            continue
        
        espn_info = None
        
        # 1. Check manual overrides
        if our_id in MANUAL_OVERRIDES:
            override_name = MANUAL_OVERRIDES[our_id].lower()
            if override_name in espn_by_display:
                espn_info = espn_by_display[override_name]
        
        # 2. Exact display name match
        if not espn_info and our_name.lower() in espn_by_display:
            espn_info = espn_by_display[our_name.lower()]
        
        # 3. Normalized name match
        if not espn_info:
            norm = normalize_name(our_name)
            if norm in espn_by_name:
                espn_info = espn_by_name[norm]
            elif norm in espn_by_short:
                espn_info = espn_by_short[norm]
        
        # 4. Try variations
        if not espn_info:
            variations = [
                our_name + ' State',
                our_name.replace(' State', ''),
                our_name + ' University',
            ]
            for var in variations:
                norm_var = normalize_name(var)
                if norm_var in espn_by_name:
                    espn_info = espn_by_name[norm_var]
                    break
        
        if espn_info:
            eid, e = espn_info
            logo_url = e['logo']
            if logo_url:
                try:
                    print(f"Downloading: {our_id} <- {e['name']}")
                    urllib.request.urlretrieve(logo_url, logo_path)
                    matched += 1
                except Exception as ex:
                    print(f"  Error downloading {our_id}: {ex}")
                    unmatched.append((our_id, our_name))
            else:
                unmatched.append((our_id, our_name))
        else:
            unmatched.append((our_id, our_name))
    
    print(f"\nMatched: {matched}/{len(our_teams)}")
    print(f"Unmatched: {len(unmatched)}")
    for u in unmatched:
        print(f"  {u[0]:35} {u[1]}")


if __name__ == '__main__':
    main()
