#!/usr/bin/env python3
"""
Weather integration for college baseball predictions.

Uses Open-Meteo (free, no API key) to fetch weather forecasts for game locations.
Wind direction + speed, temperature, and humidity significantly affect run scoring.

Usage:
    python3 scripts/weather.py seed-venues          # Populate SEC/P4 stadium coords
    python3 scripts/weather.py fetch --date 2026-02-20   # Fetch weather for date's games
    python3 scripts/weather.py fetch --upcoming     # Fetch for next 3 days of games
"""

import argparse
import json
import sqlite3
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# SEC stadium data (lat, lon, is_dome, city, state, stadium_name)
SEC_VENUES = {
    'alabama': (33.2098, -87.5504, 0, 'Tuscaloosa', 'AL', 'Sewell-Thomas Stadium'),
    'arkansas': (36.0686, -94.1780, 0, 'Fayetteville', 'AR', 'Baum-Walker Stadium'),
    'auburn': (32.6030, -85.4900, 0, 'Auburn', 'AL', 'Plainsman Park'),
    'florida': (29.6465, -82.3487, 0, 'Gainesville', 'FL', 'Florida Ballpark'),
    'georgia': (33.9480, -83.3696, 0, 'Athens', 'GA', 'Foley Field'),
    'kentucky': (38.0285, -84.5050, 0, 'Lexington', 'KY', 'Kentucky Proud Park'),
    'lsu': (30.4060, -91.1835, 0, 'Baton Rouge', 'LA', 'Alex Box Stadium'),
    'mississippi-state': (33.4558, -88.7959, 0, 'Starkville', 'MS', 'Dudy Noble Field'),
    'missouri': (38.9400, -92.3280, 0, 'Columbia', 'MO', 'Taylor Stadium'),
    'oklahoma': (35.2059, -97.4457, 0, 'Norman', 'OK', 'L. Dale Mitchell Park'),
    'ole-miss': (34.3620, -89.5370, 0, 'Oxford', 'MS', 'Swayze Field'),
    'south-carolina': (33.9727, -81.0248, 0, 'Columbia', 'SC', 'Founders Park'),
    'tennessee': (35.9550, -83.9295, 0, 'Knoxville', 'TN', 'Lindsey Nelson Stadium'),
    'texas': (30.2830, -97.7325, 1, 'Austin', 'TX', 'UFCU Disch-Falk Field'),  # Not dome but noting
    'texas-am': (30.6100, -96.3400, 0, 'College Station', 'TX', 'Olsen Field'),
    'vanderbilt': (36.1447, -86.8095, 0, 'Nashville', 'TN', 'Hawkins Field'),
}

# Additional P4 venues (Big Ten, ACC, Big 12 - top programs)
P4_VENUES = {
    # Big Ten
    'michigan': (42.2650, -83.7490, 0, 'Ann Arbor', 'MI', 'Ray Fisher Stadium'),
    'ohio-state': (40.0075, -83.0275, 0, 'Columbus', 'OH', 'Bill Davis Stadium'),
    'indiana': (39.1780, -86.5180, 0, 'Bloomington', 'IN', 'Bart Kaufman Field'),
    'maryland': (38.9897, -76.9378, 0, 'College Park', 'MD', 'Bob "Turtle" Smith Stadium'),
    'michigan-state': (42.7251, -84.4807, 0, 'East Lansing', 'MI', 'McLane Stadium'),
    'minnesota': (44.9740, -93.2277, 0, 'Minneapolis', 'MN', 'Siebert Field'),
    'nebraska': (40.8202, -96.7005, 0, 'Lincoln', 'NE', 'Hawks Field'),
    'penn-state': (40.8120, -77.8560, 0, 'University Park', 'PA', 'Medlar Field'),
    'purdue': (40.4380, -86.9250, 0, 'West Lafayette', 'IN', 'Alexander Field'),
    'rutgers': (40.5230, -74.4370, 0, 'Piscataway', 'NJ', 'Bainton Field'),
    'illinois': (40.1020, -88.2350, 0, 'Champaign', 'IL', 'Illinois Field'),
    'iowa': (41.6680, -91.5530, 0, 'Iowa City', 'IA', 'Duane Banks Field'),
    'northwestern': (42.0650, -87.6950, 0, 'Evanston', 'IL', 'Rocky Miller Park'),
    # ACC
    'clemson': (34.6770, -82.8420, 0, 'Clemson', 'SC', 'Doug Kingsmore Stadium'),
    'duke': (36.0025, -78.9440, 0, 'Durham', 'NC', 'Durham Bulls Athletic Park'),
    'florida-state': (30.4380, -84.3010, 0, 'Tallahassee', 'FL', 'Dick Howser Stadium'),
    'georgia-tech': (33.7720, -84.3920, 0, 'Atlanta', 'GA', 'Russ Chandler Stadium'),
    'louisville': (38.2195, -85.7590, 0, 'Louisville', 'KY', 'Jim Patterson Stadium'),
    'miami-fl': (25.7210, -80.2840, 0, 'Coral Gables', 'FL', 'Mark Light Field'),
    'nc-state': (35.7870, -78.6720, 0, 'Raleigh', 'NC', 'Doak Field'),
    'north-carolina': (35.9700, -79.0480, 0, 'Chapel Hill', 'NC', 'Boshamer Stadium'),
    'notre-dame': (41.6980, -86.2340, 0, 'South Bend', 'IN', 'Frank Eck Stadium'),
    'pittsburgh': (40.4440, -79.9580, 0, 'Pittsburgh', 'PA', 'Charles L. Cost Field'),
    'virginia': (38.0460, -78.5130, 0, 'Charlottesville', 'VA', 'Disharoon Park'),
    'virginia-tech': (37.2270, -80.4220, 0, 'Blacksburg', 'VA', 'English Field'),
    'wake-forest': (36.1330, -80.2770, 0, 'Winston-Salem', 'NC', 'David F. Couch Ballpark'),
    'boston-college': (42.3350, -71.1680, 0, 'Chestnut Hill', 'MA', 'Eddie Pellagrini Diamond'),
    'stanford': (37.4346, -122.1609, 0, 'Stanford', 'CA', 'Klein Field at Sunken Diamond'),
    'california': (37.8708, -122.2503, 0, 'Berkeley', 'CA', 'Evans Diamond'),
    # Big 12
    'tcu': (32.7100, -97.3680, 0, 'Fort Worth', 'TX', 'Lupton Stadium'),
    'texas-tech': (33.5910, -101.8740, 0, 'Lubbock', 'TX', 'Dan Law Field'),
    'baylor': (31.5540, -97.1190, 0, 'Waco', 'TX', 'Baylor Ballpark'),
    'oklahoma-state': (36.1270, -97.0710, 0, 'Stillwater', 'OK', "O'Brate Stadium"),
    'kansas': (38.9620, -95.2520, 0, 'Lawrence', 'KS', 'Hoglund Ballpark'),
    'kansas-state': (39.2010, -96.5840, 0, 'Manhattan', 'KS', 'Tointon Family Stadium'),
    'west-virginia': (39.6480, -79.9540, 0, 'Morgantown', 'WV', 'Monongalia County Ballpark'),
    'arizona': (32.2285, -110.9486, 0, 'Tucson', 'AZ', 'Hi Corbett Field'),
    'arizona-state': (33.4260, -111.9320, 0, 'Tempe', 'AZ', 'Phoenix Municipal Stadium'),
    'byu': (40.2570, -111.6490, 0, 'Provo', 'UT', 'Larry H. Miller Field'),
    'ucf': (28.6024, -81.1918, 0, 'Orlando', 'FL', 'John Euliano Park'),
    'cincinnati': (39.1320, -84.5160, 0, 'Cincinnati', 'OH', 'UC Baseball Stadium'),
    'houston': (29.7210, -95.3440, 1, 'Houston', 'TX', 'Darryl & Lori Schroeder Park'),
}

ALL_VENUES = {**SEC_VENUES, **P4_VENUES}


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def seed_venues():
    """Populate venues table with stadium coordinates."""
    db = get_db()
    inserted = 0
    
    for team_id, data in ALL_VENUES.items():
        lat, lon, is_dome, city, state, stadium = data
        try:
            db.execute("""
                INSERT OR REPLACE INTO venues 
                (team_id, latitude, longitude, is_dome, city, state, stadium_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (team_id, lat, lon, is_dome, city, state, stadium))
            inserted += 1
        except Exception as e:
            print(f"  Error for {team_id}: {e}")
    
    db.commit()
    db.close()
    print(f"Seeded {inserted} venues")
    return inserted


def fetch_weather(lat: float, lon: float, game_datetime: datetime) -> dict:
    """Fetch weather forecast from Open-Meteo for a specific location and time."""
    
    # Open-Meteo API (free, no key needed)
    date_str = game_datetime.strftime('%Y-%m-%d')
    hour = game_datetime.hour
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,"
        f"precipitation_probability,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        f"&timezone=America%2FChicago"
        f"&start_date={date_str}&end_date={date_str}"
    )
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  Weather API error: {e}")
        return None
    
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    
    # Find the hour closest to game time
    target_idx = min(hour, len(times) - 1) if times else 0
    
    return {
        'temp_f': hourly.get('temperature_2m', [None])[target_idx],
        'feels_like_f': hourly.get('apparent_temperature', [None])[target_idx],
        'humidity_pct': hourly.get('relative_humidity_2m', [None])[target_idx],
        'wind_speed_mph': hourly.get('wind_speed_10m', [None])[target_idx],
        'wind_direction_deg': hourly.get('wind_direction_10m', [None])[target_idx],
        'wind_gust_mph': hourly.get('wind_gusts_10m', [None])[target_idx],
        'precip_prob_pct': hourly.get('precipitation_probability', [None])[target_idx],
        'cloud_cover_pct': hourly.get('cloud_cover', [None])[target_idx],
    }


def fetch_games_weather(date_str: str = None, upcoming: bool = False):
    """Fetch weather for games on a specific date or upcoming games."""
    db = get_db()
    
    if upcoming:
        # Next 3 days
        today = datetime.now().strftime('%Y-%m-%d')
        future = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
        games = db.execute("""
            SELECT g.id, g.date, g.home_team_id, g.time as start_time,
                   v.latitude, v.longitude, v.is_dome
            FROM games g
            LEFT JOIN venues v ON g.home_team_id = v.team_id
            LEFT JOIN game_weather gw ON g.id = gw.game_id
            WHERE g.date BETWEEN ? AND ?
              AND v.latitude IS NOT NULL
              AND gw.game_id IS NULL
            ORDER BY g.date, g.time
        """, (today, future)).fetchall()
    else:
        games = db.execute("""
            SELECT g.id, g.date, g.home_team_id, g.time as start_time,
                   v.latitude, v.longitude, v.is_dome
            FROM games g
            LEFT JOIN venues v ON g.home_team_id = v.team_id
            LEFT JOIN game_weather gw ON g.id = gw.game_id
            WHERE g.date = ?
              AND v.latitude IS NOT NULL
              AND gw.game_id IS NULL
        """, (date_str,)).fetchall()
    
    print(f"Found {len(games)} games needing weather data")
    
    fetched = 0
    for game in games:
        game_id = game['id']
        lat, lon = game['latitude'], game['longitude']
        is_dome = game['is_dome']
        
        # Parse game datetime (assume 6 PM if no start_time)
        date_part = game['date']
        time_part = game['start_time'] or '18:00'
        try:
            game_dt = datetime.strptime(f"{date_part} {time_part}", '%Y-%m-%d %H:%M')
        except:
            game_dt = datetime.strptime(f"{date_part} 18:00", '%Y-%m-%d %H:%M')
        
        if is_dome:
            # Dome games get neutral weather
            weather = {
                'temp_f': 72, 'feels_like_f': 72, 'humidity_pct': 50,
                'wind_speed_mph': 0, 'wind_direction_deg': 0, 'wind_gust_mph': 0,
                'precip_prob_pct': 0, 'cloud_cover_pct': 0
            }
            print(f"  {game_id}: dome (neutral weather)")
        else:
            weather = fetch_weather(lat, lon, game_dt)
            if not weather:
                continue
            print(f"  {game_id}: {weather['temp_f']:.0f}°F, wind {weather['wind_speed_mph']:.0f}mph @ {weather['wind_direction_deg']}°")
        
        db.execute("""
            INSERT OR REPLACE INTO game_weather 
            (game_id, temp_f, feels_like_f, humidity_pct, wind_speed_mph, 
             wind_direction_deg, wind_gust_mph, precip_prob_pct, cloud_cover_pct, is_dome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, weather['temp_f'], weather['feels_like_f'], weather['humidity_pct'],
            weather['wind_speed_mph'], weather['wind_direction_deg'], weather['wind_gust_mph'],
            weather['precip_prob_pct'], weather['cloud_cover_pct'], is_dome
        ))
        fetched += 1
    
    db.commit()
    db.close()
    print(f"Fetched weather for {fetched} games")
    return fetched


def show_weather(date_str: str):
    """Display weather for games on a date."""
    db = get_db()
    games = db.execute("""
        SELECT g.id, t1.name as away, t2.name as home, 
               gw.temp_f, gw.wind_speed_mph, gw.wind_direction_deg, gw.precip_prob_pct, gw.is_dome
        FROM games g
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        LEFT JOIN game_weather gw ON g.id = gw.game_id
        WHERE g.date = ?
        ORDER BY g.time
    """, (date_str,)).fetchall()
    
    print(f"\nWeather for {date_str}:")
    print("-" * 70)
    for g in games:
        if g['is_dome']:
            wx = "🏟️ DOME"
        elif g['temp_f']:
            wind_dir = ['N','NE','E','SE','S','SW','W','NW'][int((g['wind_direction_deg'] + 22.5) // 45) % 8]
            wx = f"{g['temp_f']:.0f}°F, {g['wind_speed_mph']:.0f}mph {wind_dir}"
            if g['precip_prob_pct'] and g['precip_prob_pct'] > 30:
                wx += f", {g['precip_prob_pct']:.0f}% rain"
        else:
            wx = "No data"
        print(f"  {g['away']:20} @ {g['home']:20} | {wx}")
    
    db.close()


def main():
    parser = argparse.ArgumentParser(description='Weather integration for baseball predictions')
    parser.add_argument('command', choices=['seed-venues', 'fetch', 'show'])
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')
    parser.add_argument('--upcoming', action='store_true', help='Fetch for next 3 days')
    args = parser.parse_args()
    
    if args.command == 'seed-venues':
        seed_venues()
    elif args.command == 'fetch':
        if args.upcoming:
            fetch_games_weather(upcoming=True)
        elif args.date:
            fetch_games_weather(date_str=args.date)
        else:
            print("Specify --date or --upcoming")
    elif args.command == 'show':
        if args.date:
            show_weather(args.date)
        else:
            show_weather(datetime.now().strftime('%Y-%m-%d'))


if __name__ == '__main__':
    main()
