#!/usr/bin/env python3
"""
Fill missing game times from D1Baseball scrape data.
Converts Unix timestamps to Central time and matches to DB games.
"""
import json
import sqlite3
import sys
from datetime import datetime
import pytz

DB_PATH = 'data/baseball.db'
TARGET_DATE = '2026-02-27'

# D1Baseball data scraped from their scores page
D1B_GAMES = [
    {"away": "arizonast", "home": "missst", "awayName": "Arizona State", "homeName": "Mississippi State", "ts": 1772193600},
    {"away": "citadel", "home": "floridast", "awayName": "The Citadel", "homeName": "Florida State", "ts": 1772193600},
    {"away": "sacheart", "home": "ncstate", "awayName": "Sacred Heart", "homeName": "NC State", "ts": 1772204400},
    {"away": "loyolamary", "home": "wake", "awayName": "Loyola Marymount", "homeName": "Wake Forest", "ts": 1772204400},
    {"away": "oregonst", "home": "houston", "awayName": "Oregon State", "homeName": "Houston", "ts": 1772204400},
    {"away": "oakland", "home": "georgia", "awayName": "Oakland", "homeName": "Georgia", "ts": 1772204400},
    {"away": "ucla", "home": "tennessee", "awayName": "UCLA", "homeName": "Tennessee", "ts": 1772208000},
    {"away": "utarl", "home": "arkansas", "awayName": "UT Arlington", "homeName": "Arkansas", "ts": 1772208000},
    {"away": "nwestern", "home": "gatech", "awayName": "Northwestern", "homeName": "Georgia Tech", "ts": 1772208000},
    {"away": "wvirginia", "home": "kennesawst", "awayName": "West Virginia", "homeName": "Kennesaw State", "ts": 1772208000},
    {"away": "le-moyne", "home": "unc", "awayName": "Le Moyne", "homeName": "North Carolina", "ts": 1772208000},
    {"away": "stjohns", "home": "kentucky", "awayName": "St Johns", "homeName": "Kentucky", "ts": 1772208000},
    {"away": "baylor", "home": "olemiss", "awayName": "Baylor", "homeName": "Ole Miss", "ts": 1772208300},
    {"away": "gonzaga", "home": "oklahoma", "awayName": "Gonzaga", "homeName": "Oklahoma", "ts": 1772211600},
    {"away": "davidson", "home": "wake", "awayName": "Davidson", "homeName": "Wake Forest", "ts": 1772217000},
    {"away": "clemson", "home": "scarolina", "awayName": "Clemson", "homeName": "South Carolina", "ts": 1772218801},
    {"away": "florida", "home": "miamifl", "awayName": "Florida", "homeName": "Miami", "ts": 1772218801},
    {"away": "nebraska", "home": "auburn", "awayName": "Nebraska", "homeName": "Auburn", "ts": 1772218801},
    {"away": "smiss", "home": "latech", "awayName": "Southern Miss", "homeName": "Louisiana Tech", "ts": 1772218801},
    {"away": "new-haven", "home": "tcu", "awayName": "New Haven", "homeName": "TCU", "ts": 1772218801},
    {"away": "dartmouth", "home": "lsu", "awayName": "Dartmouth", "homeName": "LSU", "ts": 1772220600},
    {"away": "coastcar", "home": "texas", "awayName": "Coastal Carolina", "homeName": "Texas", "ts": 1772222400},
    {"away": "vatech", "home": "texasam", "awayName": "Virginia Tech", "homeName": "Texas AM", "ts": 1772222400},
    {"away": "cornell", "home": "richmond", "awayName": "Cornell", "homeName": "Richmond", "ts": 1772197200},
    {"away": "fordham", "home": "stetson", "awayName": "Fordham", "homeName": "Stetson", "ts": 1772200800},
    {"away": "stpeters", "home": "georgewash", "awayName": "Saint Peters", "homeName": "George Washington", "ts": 1772202600},
    {"away": "bellarmine", "home": "stlouis", "awayName": "Bellarmine", "homeName": "Saint Louis", "ts": 1772204400},
    {"away": "la-salle", "home": "coppinst", "awayName": "La Salle", "homeName": "Coppin State", "ts": 1772204400},
    {"away": "uri", "home": "kentst", "awayName": "Rhode Island", "homeName": "Kent State", "ts": 1772204400},
    {"away": "vcu", "home": "virginia", "awayName": "VCU", "homeName": "Virginia", "ts": 1772204400},
    {"away": "stbonny", "home": "navy", "awayName": "St Bonaventure", "homeName": "Navy", "ts": 1772204400},
    {"away": "dayton", "home": "ohio", "awayName": "Dayton", "homeName": "Ohio", "ts": 1772204400},
    {"away": "stjosephs", "home": "wofford", "awayName": "Saint Josephs", "homeName": "Wofford", "ts": 1772213400},
    {"away": "lehigh", "home": "georgemas", "awayName": "Lehigh", "homeName": "George Mason", "ts": 1772218800},
    {"away": "notredame", "home": "alabamaam", "awayName": "Notre Dame", "homeName": "Alabama AM", "ts": 1772193600},
    {"away": "uncgreen", "home": "pittsburgh", "awayName": "UNC Greensboro", "homeName": "Pittsburgh", "ts": 1772204400},
    {"away": "cmichigan", "home": "louisville", "awayName": "Central Michigan", "homeName": "Louisville", "ts": 1772204400},
    {"away": "princeton", "home": "duke", "awayName": "Princeton", "homeName": "Duke", "ts": 1772208000},
    {"away": "fresnost", "home": "stanford", "awayName": "Fresno State", "homeName": "Stanford", "ts": 1772211900},
    {"away": "bostoncoll", "home": "flgulfcst", "awayName": "Boston College", "homeName": "Florida Gulf Coast", "ts": 1772217000},
    {"away": "sacstate", "home": "california", "awayName": "Sacramento State", "homeName": "California", "ts": 1772226000},
    {"away": "maine", "home": "winthrop", "awayName": "Maine", "homeName": "Winthrop", "ts": 1772204400},
    {"away": "bryant", "home": "radford", "awayName": "Bryant", "homeName": "Radford", "ts": 1772204400},
    {"away": "sunybing", "home": "highpoint", "awayName": "Binghamton", "homeName": "High Point", "ts": 1772208000},
    {"away": "utsa", "home": "ohiost", "awayName": "UTSA", "homeName": "Ohio State", "ts": 1772193600},
    {"away": "charlotte", "home": "olddom", "awayName": "Charlotte", "homeName": "Old Dominion", "ts": 1772200800},
    {"away": "jacksonvil", "home": "uab", "awayName": "Jacksonville", "homeName": "UAB", "ts": 1772204400},
    {"away": "nebomaha", "home": "wichitast", "awayName": "Omaha", "homeName": "Wichita State", "ts": 1772208300},
    {"away": "rutgers", "home": "ecarolina", "awayName": "Rutgers", "homeName": "East Carolina", "ts": 1772209800},
    {"away": "jksonvilst", "home": "memphis", "awayName": "Jacksonville State", "homeName": "Memphis", "ts": 1772211900},
    {"away": "sflorida", "home": "ucf", "awayName": "South Florida", "homeName": "UCF", "ts": 1772215200},
    {"away": "villanova", "home": "flatlantic", "awayName": "Villanova", "homeName": "Florida Atlantic", "ts": 1772217000},
    {"away": "ekentucky", "home": "tulane", "awayName": "Eastern Kentucky", "homeName": "Tulane", "ts": 1772220600},
    {"away": "harvard", "home": "rice", "awayName": "Harvard", "homeName": "Rice", "ts": 1772220900},
    {"away": "austinpeay", "home": "cincy", "awayName": "Austin Peay", "homeName": "Cincinnati", "ts": 1772204400},
    {"away": "sdakotast", "home": "carkansas", "awayName": "South Dakota State", "homeName": "Central Arkansas", "ts": 1772208000},
    {"away": "toledo", "home": "lipscomb", "awayName": "Toledo", "homeName": "Lipscomb", "ts": 1772208000},
    {"away": "southern-indiana", "home": "northalabama", "awayName": "Southern Indiana", "homeName": "North Alabama", "ts": 1772208000},
    {"away": "army", "home": "queens-nc", "awayName": "Army", "homeName": "Queens NC", "ts": 1772215200},
    {"away": "presbytrn", "home": "nflorida", "awayName": "Presbyterian", "homeName": "North Florida", "ts": 1772215200},
    {"away": "west-georgia", "home": "salabama", "awayName": "West Georgia", "homeName": "South Alabama", "ts": 1772220600},
    {"away": "calstbaker", "home": "txtech", "awayName": "CSU Bakersfield", "homeName": "Texas Tech", "ts": 1772204400},
    {"away": "columbia", "home": "kansasst", "awayName": "Columbia", "homeName": "Kansas State", "ts": 1772204400},
    {"away": "byu", "home": "washst", "awayName": "BYU", "homeName": "Washington State", "ts": 1772208300},
    {"away": "samhouston", "home": "okstate", "awayName": "Sam Houston", "homeName": "Oklahoma State", "ts": 1772211600},
    {"away": "kansas", "home": "minnesota", "awayName": "Kansas", "homeName": "Minnesota", "ts": 1772218801},
    {"away": "utah", "home": "ucsb", "awayName": "Utah", "homeName": "UC Santa Barbara", "ts": 1772219100},
    {"away": "oregon", "home": "arizona", "awayName": "Oregon", "homeName": "Arizona", "ts": 1772226000},
    {"away": "tarletonst", "home": "creighton", "awayName": "Tarleton", "homeName": "Creighton", "ts": 1772204400},
    {"away": "fairldick", "home": "georgetown", "awayName": "Fairleigh Dickinson", "homeName": "Georgetown", "ts": 1772204400},
    {"away": "butler", "home": "marshall", "awayName": "Butler", "homeName": "Marshall", "ts": 1772204400},
    {"away": "uconn", "home": "uncwilm", "awayName": "Connecticut", "homeName": "UNC Wilmington", "ts": 1772208000},
    {"away": "setonhall", "home": "sillinois", "awayName": "Seton Hall", "homeName": "Southern Illinois", "ts": 1772218800},
    {"away": "xavier", "home": "calstfull", "awayName": "Xavier", "homeName": "Cal State Fullerton", "ts": 1772227800},
    {"away": "youngst", "home": "longwood", "awayName": "Youngstown State", "homeName": "Longwood", "ts": 1772204400},
    {"away": "uncashe", "home": "missvalley", "awayName": "UNC Asheville", "homeName": "Mississippi Valley State", "ts": 1772204400},
    {"away": "brown", "home": "charlsouth", "awayName": "Brown", "homeName": "Charleston Southern", "ts": 1772211600},
    {"away": "appalst", "home": "gardwebb", "awayName": "Appalachian State", "homeName": "Gardner-Webb", "ts": 1772215200},
    {"away": "uscupstate", "home": "georgiasou", "awayName": "USC Upstate", "homeName": "Georgia Southern", "ts": 1772217000},
    {"away": "illinois", "home": "michiganst", "awayName": "Illinois", "homeName": "Michigan State", "ts": 1772200800},
    {"away": "indiana", "home": "wkentucky", "awayName": "Indiana", "homeName": "Western Kentucky", "ts": 1772208000},
    {"away": "purdue", "home": "marist", "awayName": "Purdue", "homeName": "Marist", "ts": 1772208000},
    {"away": "wagner", "home": "maryland", "awayName": "Wagner", "homeName": "Maryland", "ts": 1772208000},
    {"away": "pennst", "home": "indianast", "awayName": "Penn State", "homeName": "Indiana State", "ts": 1772215200},
    {"away": "iowa", "home": "alabama", "awayName": "Iowa", "homeName": "Alabama", "ts": 1772218801},
    {"away": "sandiegost", "home": "washington", "awayName": "San Diego State", "homeName": "Washington", "ts": 1772222700},
    {"away": "michigan", "home": "sandiego", "awayName": "Michigan", "homeName": "San Diego", "ts": 1772226000},
    {"away": "usc", "home": "calpoly", "awayName": "Southern California", "homeName": "Cal Poly", "ts": 1772226300},
    {"away": "vandy", "home": "ucirvine", "awayName": "Vanderbilt", "homeName": "UC Irvine", "ts": 1772211600},
    {"away": "ucriver", "home": "utvalley", "awayName": "UC Riverside", "homeName": "Utah Valley", "ts": 1772215500},
    {"away": "ballst", "home": "hawaii", "awayName": "Ball State", "homeName": "Hawaii", "ts": 1772217000},
    {"away": "ucsandiego", "home": "ulala", "awayName": "UC San Diego", "homeName": "Louisiana", "ts": 1772218801},
    {"away": "wmichigan", "home": "calstnorth", "awayName": "Western Michigan", "homeName": "Cal State Northridge", "ts": 1772222400},
    {"away": "nevada", "home": "ucdavis", "awayName": "Nevada", "homeName": "UC Davis", "ts": 1772226000},
    {"away": "washst", "home": "longbeach", "awayName": "Washington State", "homeName": "Long Beach State", "ts": 1772226300},
    {"away": "arkansaslr", "home": "missourist", "awayName": "Little Rock", "homeName": "Missouri State", "ts": 1772208000},
    {"away": "rider", "home": "delaware", "awayName": "Rider", "homeName": "Delaware", "ts": 1772208000},
    {"away": "hofstra", "home": "liberty", "awayName": "Hofstra", "homeName": "Liberty", "ts": 1772208000},
    {"away": "illinoisst", "home": "mtennst", "awayName": "Illinois State", "homeName": "Middle Tennessee", "ts": 1772208000},
    {"away": "stonybrook", "home": "flinternat", "awayName": "Stony Brook", "homeName": "Florida International", "ts": 1772215200},
    {"away": "airforce", "home": "dallasbapt", "awayName": "Air Force", "homeName": "Dallas Baptist", "ts": 1772220600},
    {"away": "prairview", "home": "nmstate", "awayName": "Prairie View", "homeName": "New Mexico State", "ts": 1772222400},
    {"away": "grambling", "home": "neastern", "awayName": "Grambling", "homeName": "Northeastern", "ts": 1772200800},
    {"away": "akron", "home": "ncat", "awayName": "Akron", "homeName": "North Carolina AT", "ts": 1772208000},
    {"away": "fairfield", "home": "elon", "awayName": "Fairfield", "homeName": "Elon", "ts": 1772208000},
    {"away": "wrightst", "home": "campbell", "awayName": "Wright State", "homeName": "Campbell", "ts": 1772211600},
    {"away": "canisius", "home": "towson", "awayName": "Canisius", "homeName": "Towson", "ts": 1772211600},
    {"away": "lafayette", "home": "willmary", "awayName": "Lafayette", "homeName": "William  Mary", "ts": 1772215200},
    {"away": "jamesmad", "home": "charleston", "awayName": "James Madison", "homeName": "College of Charleston", "ts": 1772215200},
    {"away": "eillinois", "home": "nkentucky", "awayName": "Eastern Illinois", "homeName": "Northern Kentucky", "ts": 1772200800},
    {"away": "wiscmilw", "home": "evansville", "awayName": "Milwaukee", "homeName": "Evansville", "ts": 1772208000},
    {"away": "yale", "home": "pepperdine", "awayName": "Yale", "homeName": "Pepperdine", "ts": 1772209800},
    {"away": "upenn", "home": "mercer", "awayName": "Pennsylvania", "homeName": "Mercer", "ts": 1772215200},
    {"away": "merrimack", "home": "bucknell", "awayName": "Merrimack", "homeName": "Bucknell", "ts": 1772200800},
    {"away": "siena", "home": "umass", "awayName": "Siena", "homeName": "Massachusetts", "ts": 1772200800},
    {"away": "quinnipiac", "home": "etennst", "awayName": "Quinnipiac", "homeName": "East Tennessee State", "ts": 1772204400},
    {"away": "mtstmarys", "home": "norfolkst", "awayName": "Mount St Marys", "homeName": "Norfolk State", "ts": 1772204400},
    {"away": "niagara", "home": "portland", "awayName": "Niagara", "homeName": "Portland", "ts": 1772226000},
    {"away": "nillinois", "home": "lindenwood", "awayName": "Northern Illinois", "homeName": "Lindenwood", "ts": 1772200800},
    {"away": "miamioh", "home": "semost", "awayName": "Miami OH", "homeName": "Southeast Missouri State", "ts": 1772208000},
    {"away": "emichigan", "home": "silledward", "awayName": "Eastern Michigan", "homeName": "SIU Edwardsville", "ts": 1772208000},
    {"away": "bowlgreen", "home": "samford", "awayName": "Bowling Green", "homeName": "Samford", "ts": 1772211600},
    {"away": "mercyhurst", "home": "murrayst", "awayName": "Mercyhurst", "homeName": "Murray State", "ts": 1772204400},
    {"away": "bradley", "home": "tnmartin", "awayName": "Bradley", "homeName": "Tennessee-Martin", "ts": 1772208000},
    {"away": "georgiast", "home": "belmont", "awayName": "Georgia State", "homeName": "Belmont", "ts": 1772211600},
    {"away": "illchicago", "home": "tntech", "awayName": "Illinois-Chicago", "homeName": "Tennessee Tech", "ts": 1772211600},
    {"away": "valpo", "home": "alabamast", "awayName": "Valparaiso", "homeName": "Alabama State", "ts": 1772218801},
    {"away": "oralrob", "home": "airforce", "awayName": "Oral Roberts", "homeName": "Air Force", "ts": 1772204400},
    {"away": "stthomas", "home": "nmexico", "awayName": "St Thomas", "homeName": "New Mexico", "ts": 1772208000},
    {"away": "sanjosest", "home": "stmarysca", "awayName": "San Jose State", "homeName": "Saint Marys", "ts": 1772211600},
    {"away": "pacific", "home": "gcanyon", "awayName": "Pacific", "homeName": "Grand Canyon", "ts": 1772222400},
    {"away": "unlv", "home": "santaclara", "awayName": "UNLV", "homeName": "Santa Clara", "ts": 1772226000},
    {"away": "stonehill", "home": "vmi", "awayName": "Stonehill", "homeName": "VMI", "ts": 1772204400},
    {"away": "cconnst", "home": "floridaam", "awayName": "Central Connecticut", "homeName": "Florida AM", "ts": 1772208000},
    {"away": "delawarest", "home": "bethcook", "awayName": "Delaware State", "homeName": "Bethune-Cookman", "ts": 1772209800},
    {"away": "willinois", "home": "ulamo", "awayName": "Western Illinois", "homeName": "UL Monroe", "ts": 1772200800},
    {"away": "morehead", "home": "southernu", "awayName": "Morehead State", "homeName": "Southern", "ts": 1772218801},
    {"away": "holycross", "home": "seattleu", "awayName": "Holy Cross", "homeName": "Seattle", "ts": 1772215200},
    {"away": "ndakotast", "home": "missouri", "awayName": "North Dakota State", "homeName": "Missouri", "ts": 1772218800},
    {"away": "wcarolina", "home": "troy", "awayName": "Western Carolina", "homeName": "Troy", "ts": 1772193600},
    {"away": "mcneese", "home": "nicholls", "awayName": "McNeese", "homeName": "Nicholls", "ts": 1772218801},
    {"away": "utrio", "home": "lamar", "awayName": "UT Rio Grande Valley", "homeName": "Lamar", "ts": 1772219100},
    {"away": "sela", "home": "nwstate", "awayName": "Southeastern Louisiana", "homeName": "Northwestern State", "ts": 1772220600},
    {"away": "tamucc", "home": "txsouth", "awayName": "Texas AM-Corpus Christi", "homeName": "Texas Southern", "ts": 1772220600},
    {"away": "incarnword", "home": "houstnbapt", "awayName": "Incarnate Word", "homeName": "Houston Christian", "ts": 1772220600},
    {"away": "sfaustin", "home": "norleans", "awayName": "Stephen F Austin", "homeName": "New Orleans", "ts": 1772220600},
    {"away": "ncolorado", "home": "utah-tech", "awayName": "Northern Colorado", "homeName": "Utah Tech", "ts": 1772215500},
    {"away": "arkansaspb", "home": "arkansasst", "awayName": "Arkansas-Pine Bluff", "homeName": "Arkansas State", "ts": 1772208000},
    {"away": "abilchrist", "home": "txstate", "awayName": "Abilene Christian", "homeName": "Texas State", "ts": 1772218801},
    {"away": "alcornst", "home": "jacksonst", "awayName": "Alcorn State", "homeName": "Jackson State", "ts": 1772215200},
    {"away": "californiabaptist", "home": "sanfran", "awayName": "California Baptist", "homeName": "San Francisco", "ts": 1772211600},
]

# D1Baseball slug -> our team_id mapping
# Built by examining d1baseball URL slugs vs our teams table
D1B_TO_TEAM_ID = {
    'arizonast': 'arizona-state',
    'missst': 'mississippi-state',
    'citadel': 'the-citadel',
    'floridast': 'florida-state',
    'sacheart': 'sacred-heart',
    'ncstate': 'nc-state',
    'loyolamary': 'loyola-marymount',
    'wake': 'wake-forest',
    'oregonst': 'oregon-state',
    'houston': 'houston',
    'oakland': 'oakland',
    'georgia': 'georgia',
    'ucla': 'ucla',
    'tennessee': 'tennessee',
    'utarl': 'ut-arlington',
    'arkansas': 'arkansas',
    'nwestern': 'northwestern',
    'gatech': 'georgia-tech',
    'wvirginia': 'west-virginia',
    'kennesawst': 'kennesaw-state',
    'le-moyne': 'le-moyne',
    'unc': 'north-carolina',
    'stjohns': 'st-johns',
    'kentucky': 'kentucky',
    'baylor': 'baylor',
    'olemiss': 'ole-miss',
    'gonzaga': 'gonzaga',
    'oklahoma': 'oklahoma',
    'davidson': 'davidson',
    'clemson': 'clemson',
    'scarolina': 'south-carolina',
    'florida': 'florida',
    'miamifl': 'miami-fl',
    'nebraska': 'nebraska',
    'auburn': 'auburn',
    'smiss': 'southern-miss',
    'latech': 'louisiana-tech',
    'new-haven': 'new-haven',
    'tcu': 'tcu',
    'dartmouth': 'dartmouth',
    'lsu': 'lsu',
    'coastcar': 'coastal-carolina',
    'texas': 'texas',
    'vatech': 'virginia-tech',
    'texasam': 'texas-am',
    'cornell': 'cornell',
    'richmond': 'richmond',
    'fordham': 'fordham',
    'stetson': 'stetson',
    'stpeters': 'saint-peters',
    'georgewash': 'george-washington',
    'bellarmine': 'bellarmine',
    'stlouis': 'saint-louis',
    'la-salle': 'la-salle',
    'coppinst': 'coppin-state',
    'uri': 'rhode-island',
    'kentst': 'kent-state',
    'vcu': 'vcu',
    'virginia': 'virginia',
    'stbonny': 'st-bonaventure',
    'navy': 'navy',
    'dayton': 'dayton',
    'ohio': 'ohio',
    'stjosephs': 'saint-josephs',
    'wofford': 'wofford',
    'lehigh': 'lehigh',
    'georgemas': 'george-mason',
    'notredame': 'notre-dame',
    'alabamaam': 'alabama-am',
    'uncgreen': 'unc-greensboro',
    'pittsburgh': 'pittsburgh',
    'cmichigan': 'central-michigan',
    'louisville': 'louisville',
    'princeton': 'princeton',
    'duke': 'duke',
    'fresnost': 'fresno-state',
    'stanford': 'stanford',
    'bostoncoll': 'boston-college',
    'flgulfcst': 'florida-gulf-coast',
    'sacstate': 'sacramento-state',
    'california': 'california',
    'maine': 'maine',
    'winthrop': 'winthrop',
    'bryant': 'bryant',
    'radford': 'radford',
    'sunybing': 'binghamton',
    'highpoint': 'high-point',
    'utsa': 'utsa',
    'ohiost': 'ohio-state',
    'charlotte': 'charlotte',
    'olddom': 'old-dominion',
    'jacksonvil': 'jacksonville',
    'uab': 'uab',
    'nebomaha': 'omaha',
    'wichitast': 'wichita-state',
    'rutgers': 'rutgers',
    'ecarolina': 'east-carolina',
    'jksonvilst': 'jacksonville-state',
    'memphis': 'memphis',
    'sflorida': 'south-florida',
    'ucf': 'ucf',
    'villanova': 'villanova',
    'flatlantic': 'florida-atlantic',
    'ekentucky': 'eastern-kentucky',
    'tulane': 'tulane',
    'harvard': 'harvard',
    'rice': 'rice',
    'austinpeay': 'austin-peay',
    'cincy': 'cincinnati',
    'sdakotast': 'south-dakota-state',
    'carkansas': 'central-arkansas',
    'toledo': 'toledo',
    'lipscomb': 'lipscomb',
    'southern-indiana': 'southern-indiana',
    'northalabama': 'north-alabama',
    'army': 'army',
    'queens-nc': 'queens',
    'presbytrn': 'presbyterian',
    'nflorida': 'north-florida',
    'west-georgia': 'west-georgia',
    'salabama': 'south-alabama',
    'calstbaker': 'cal-state-bakersfield',
    'txtech': 'texas-tech',
    'columbia': 'columbia',
    'kansasst': 'kansas-state',
    'byu': 'byu',
    'washst': 'washington-state',
    'samhouston': 'sam-houston',
    'okstate': 'oklahoma-state',
    'kansas': 'kansas',
    'minnesota': 'minnesota',
    'utah': 'utah',
    'ucsb': 'uc-santa-barbara',
    'oregon': 'oregon',
    'arizona': 'arizona',
    'tarletonst': 'tarleton-state',
    'creighton': 'creighton',
    'fairldick': 'fairleigh-dickinson',
    'georgetown': 'georgetown',
    'butler': 'butler',
    'marshall': 'marshall',
    'uconn': 'uconn',
    'uncwilm': 'unc-wilmington',
    'setonhall': 'seton-hall',
    'sillinois': 'southern-illinois',
    'xavier': 'xavier',
    'calstfull': 'cal-state-fullerton',
    'youngst': 'youngstown-state',
    'longwood': 'longwood',
    'uncashe': 'unc-asheville',
    'missvalley': 'mississippi-valley-state',
    'brown': 'brown',
    'charlsouth': 'charleston-southern',
    'appalst': 'appalachian-state',
    'gardwebb': 'gardner-webb',
    'uscupstate': 'usc-upstate',
    'georgiasou': 'georgia-southern',
    'illinois': 'illinois',
    'michiganst': 'michigan-state',
    'indiana': 'indiana',
    'wkentucky': 'western-kentucky',
    'purdue': 'purdue',
    'marist': 'marist',
    'wagner': 'wagner',
    'maryland': 'maryland',
    'pennst': 'penn-state',
    'indianast': 'indiana-state',
    'iowa': 'iowa',
    'alabama': 'alabama',
    'sandiegost': 'san-diego-state',
    'washington': 'washington',
    'michigan': 'michigan',
    'sandiego': 'san-diego',
    'usc': 'usc',
    'calpoly': 'cal-poly',
    'vandy': 'vanderbilt',
    'ucirvine': 'uc-irvine',
    'ucriver': 'uc-riverside',
    'utvalley': 'utah-valley',
    'ballst': 'ball-state',
    'hawaii': 'hawaii',
    'ucsandiego': 'uc-san-diego',
    'ulala': 'louisiana',
    'wmichigan': 'western-michigan',
    'calstnorth': 'cal-state-northridge',
    'nevada': 'nevada',
    'ucdavis': 'uc-davis',
    'longbeach': 'long-beach-state',
    'arkansaslr': 'little-rock',
    'missourist': 'missouri-state',
    'rider': 'rider',
    'delaware': 'delaware',
    'hofstra': 'hofstra',
    'liberty': 'liberty',
    'illinoisst': 'illinois-state',
    'mtennst': 'middle-tennessee',
    'stonybrook': 'stony-brook',
    'flinternat': 'fiu',
    'airforce': 'air-force',
    'dallasbapt': 'dallas-baptist',
    'prairview': 'prairie-view',
    'nmstate': 'new-mexico-state',
    'grambling': 'grambling',
    'neastern': 'northeastern',
    'akron': 'akron',
    'ncat': 'north-carolina-at',
    'fairfield': 'fairfield',
    'elon': 'elon',
    'wrightst': 'wright-state',
    'campbell': 'campbell',
    'canisius': 'canisius',
    'towson': 'towson',
    'lafayette': 'lafayette',
    'willmary': 'william-mary',
    'jamesmad': 'james-madison',
    'charleston': 'college-of-charleston',
    'eillinois': 'eastern-illinois',
    'nkentucky': 'northern-kentucky',
    'wiscmilw': 'milwaukee',
    'evansville': 'evansville',
    'yale': 'yale',
    'pepperdine': 'pepperdine',
    'upenn': 'penn',
    'mercer': 'mercer',
    'merrimack': 'merrimack',
    'bucknell': 'bucknell',
    'siena': 'siena',
    'umass': 'umass',
    'quinnipiac': 'quinnipiac',
    'etennst': 'east-tennessee-state',
    'mtstmarys': 'mount-st-marys',
    'norfolkst': 'norfolk-state',
    'niagara': 'niagara',
    'portland': 'portland',
    'nillinois': 'northern-illinois',
    'lindenwood': 'lindenwood',
    'miamioh': 'miami-ohio',
    'semost': 'southeast-missouri',
    'emichigan': 'eastern-michigan',
    'silledward': 'siu-edwardsville',
    'bowlgreen': 'bowling-green',
    'samford': 'samford',
    'mercyhurst': 'mercyhurst',
    'murrayst': 'murray-state',
    'bradley': 'bradley',
    'tnmartin': 'ut-martin',
    'georgiast': 'georgia-state',
    'belmont': 'belmont',
    'illchicago': 'uic',
    'tntech': 'tennessee-tech',
    'valpo': 'valparaiso',
    'alabamast': 'alabama-state',
    'oralrob': 'oral-roberts',
    'stthomas': 'st-thomas-minnesota',
    'nmexico': 'new-mexico',
    'sanjosest': 'san-jose-state',
    'stmarysca': 'saint-marys',
    'pacific': 'pacific',
    'gcanyon': 'grand-canyon',
    'unlv': 'unlv',
    'santaclara': 'santa-clara',
    'stonehill': 'stonehill',
    'vmi': 'vmi',
    'cconnst': 'central-connecticut',
    'floridaam': 'florida-am',
    'delawarest': 'delaware-state',
    'bethcook': 'bethune-cookman',
    'willinois': 'western-illinois',
    'ulamo': 'ul-monroe',
    'morehead': 'morehead-state',
    'southernu': 'southern',
    'holycross': 'holy-cross',
    'seattleu': 'seattle',
    'ndakotast': 'north-dakota-state',
    'missouri': 'missouri',
    'wcarolina': 'western-carolina',
    'troy': 'troy',
    'mcneese': 'mcneese',
    'nicholls': 'nicholls',
    'utrio': 'utrgv',
    'lamar': 'lamar',
    'sela': 'southeastern-louisiana',
    'nwstate': 'northwestern-state',
    'tamucc': 'texas-aandm-corpus-christi',
    'txsouth': 'texas-southern',
    'incarnword': 'incarnate-word',
    'houstnbapt': 'houston-christian',
    'sfaustin': 'stephen-f-austin',
    'norleans': 'new-orleans',
    'ncolorado': 'northern-colorado',
    'utah-tech': 'utah-tech',
    'arkansaspb': 'arkansas-pine-bluff',
    'arkansasst': 'arkansas-state',
    'abilchrist': 'abilene-christian',
    'txstate': 'texas-state',
    'alcornst': 'alcorn-state',
    'jacksonst': 'jackson-state',
    'californiabaptist': 'california-baptist',
    'sanfran': 'san-francisco',
}


def ts_to_central(ts):
    """Convert D1Baseball pseudo-UTC timestamp to Central time string.
    
    D1B encodes Eastern time as fake-UTC (the UTC hour IS the Eastern hour).
    To get Central: read UTC hour, subtract 1 (EST = CST + 1 in standard time).
    Feb 2026 is standard time (DST starts March).
    """
    # Base = midnight Feb 27 UTC = 1772150400
    base = 1772150400
    seconds_from_midnight = ts - base
    et_hours = seconds_from_midnight // 3600
    et_minutes = (seconds_from_midnight % 3600) // 60
    
    # Eastern -> Central: subtract 1 hour
    ct_hours = et_hours - 1
    if ct_hours < 0:
        ct_hours += 24
    
    ampm = 'AM' if ct_hours < 12 else 'PM'
    display_hour = ct_hours
    if display_hour == 0:
        display_hour = 12
    elif display_hour > 12:
        display_hour -= 12
    return f"{display_hour}:{et_minutes:02d} {ampm}"


def main():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get all team IDs in our DB
    cur.execute("SELECT id FROM teams")
    valid_ids = {r['id'] for r in cur.fetchall()}
    
    # Also get team_aliases for fallback matching
    cur.execute("SELECT alias, team_id FROM team_aliases")
    alias_map = {}
    for r in cur.fetchall():
        alias_map[r['alias'].lower()] = r['team_id']
    
    # Get games missing times for target date
    cur.execute("""
        SELECT id, away_team_id, home_team_id, time 
        FROM games 
        WHERE date = ? AND (time IS NULL OR time = '' OR time = 'TBA' OR time = 'TBD')
    """, (TARGET_DATE,))
    missing_games = cur.fetchall()
    
    print(f"Games missing times for {TARGET_DATE}: {len(missing_games)}")
    
    # Build lookup of (away_id, home_id) -> game row
    missing_lookup = {}
    for g in missing_games:
        missing_lookup[(g['away_team_id'], g['home_team_id'])] = g
    
    # Also build swapped lookup
    missing_lookup_swapped = {}
    for g in missing_games:
        missing_lookup_swapped[(g['home_team_id'], g['away_team_id'])] = g
    
    updated = 0
    not_found_team = []
    not_in_db = []
    already_has_time = 0
    
    for d1g in D1B_GAMES:
        away_id = D1B_TO_TEAM_ID.get(d1g['away'])
        home_id = D1B_TO_TEAM_ID.get(d1g['home'])
        
        if not away_id or not home_id:
            not_found_team.append(f"{d1g['awayName']} @ {d1g['homeName']} (slug: {d1g['away']}@{d1g['home']})")
            continue
        
        # Verify team IDs exist in DB
        if away_id not in valid_ids:
            # Try alias
            away_id = alias_map.get(away_id, away_id)
        if home_id not in valid_ids:
            home_id = alias_map.get(home_id, home_id)
            
        if away_id not in valid_ids or home_id not in valid_ids:
            not_found_team.append(f"{d1g['awayName']}({away_id}) @ {d1g['homeName']}({home_id}) - not in teams table")
            continue
        
        time_str = ts_to_central(d1g['ts'])
        
        # Try normal order
        game = missing_lookup.get((away_id, home_id))
        swapped = False
        if not game:
            # Try swapped
            game = missing_lookup.get((home_id, away_id))
            if game:
                swapped = True
        if not game:
            # Check if game exists but already has time
            cur.execute("""
                SELECT time FROM games 
                WHERE date = ? AND (
                    (away_team_id = ? AND home_team_id = ?) OR
                    (away_team_id = ? AND home_team_id = ?)
                )
            """, (TARGET_DATE, away_id, home_id, home_id, away_id))
            existing = cur.fetchone()
            if existing:
                already_has_time += 1
            else:
                not_in_db.append(f"{d1g['awayName']}({away_id}) @ {d1g['homeName']}({home_id})")
            continue
        
        # Update the time
        cur.execute("UPDATE games SET time = ? WHERE id = ?", (time_str, game['id']))
        updated += 1
        swap_note = " (swapped)" if swapped else ""
        print(f"  Updated: {d1g['awayName']} @ {d1g['homeName']} -> {time_str}{swap_note}")
    
    conn.commit()
    
    # Check how many still missing
    cur.execute("""
        SELECT COUNT(*) as cnt FROM games 
        WHERE date = ? AND (time IS NULL OR time = '' OR time = 'TBA' OR time = 'TBD')
    """, (TARGET_DATE,))
    still_missing = cur.fetchone()['cnt']
    
    print(f"\n--- Summary ---")
    print(f"Updated: {updated}")
    print(f"Already had time: {already_has_time}")
    print(f"Still missing times: {still_missing}")
    
    if not_found_team:
        print(f"\nTeam ID not resolved ({len(not_found_team)}):")
        for t in not_found_team:
            print(f"  {t}")
    
    if not_in_db:
        print(f"\nNot in our DB for {TARGET_DATE} ({len(not_in_db)}):")
        for t in not_in_db:
            print(f"  {t}")
    
    conn.close()


if __name__ == '__main__':
    main()
