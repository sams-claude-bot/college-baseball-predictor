#!/usr/bin/env python3
"""
Backfill weather for historical games (2024, 2025) using Open-Meteo Archive API.
Maps historical team names (with mascots) to our teams table to get venue coordinates.
"""

import sqlite3
import requests
import time
import re
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"

# Common mascot suffixes to strip
MASCOTS = [
    'Demon Deacons', 'Horned Frogs', 'Red Raiders', 'Yellow Jackets', 'Blue Devils',
    'Crimson Tide', 'Fighting Irish', 'Tar Heels', 'Golden Eagles', 'Golden Gophers',
    'Sun Devils', 'Cardinal', 'Ducks', 'Beavers', 'Bruins', 'Trojans', 'Wildcats',
    'Bulldogs', 'Tigers', 'Gators', 'Razorbacks', 'Commodores', 'Aggies', 'Longhorns',
    'Sooners', 'Cowboys', 'Bears', 'Volunteers', 'Gamecocks', 'Cavaliers', 'Hokies',
    'Wolfpack', 'Seminoles', 'Hurricanes', 'Pirates', 'Hoosiers', 'Buckeyes',
    'Spartans', 'Wolverines', 'Hawkeyes', 'Boilermakers', 'Huskers', 'Badgers',
    'Fighting Illini', 'Nittany Lions', 'Terrapins', 'Scarlet Knights', 'Mountaineers',
    'Chanticleers', 'Owls', 'Knights', 'Cougars', 'Panthers', 'Cardinals', 'Dolphins',
    'Hilltoppers', 'Zips', 'Lopes', 'Thundering Herd', 'Jaguars', 'Flames',
    'Huskies', 'Golden Bears', 'Broncos', 'Aztecs', 'Rebels', 'Colonels',
    'Bearcats', 'Anteaters', 'Mustangs', 'Miners', 'Roadrunners', 'Bobcats',
    'Wave', 'Green Wave', 'Blue Jays', 'Mean Green', 'Ragin Cajuns', "Ragin' Cajuns",
    'Red Wolves', 'Shockers', 'Phoenix', 'Runnin Rebels', 'Lobos', 'Toreros',
]


def strip_mascot(full_name: str) -> str:
    """Remove mascot from team name to get school name."""
    name = full_name.strip()
    
    # Sort by length descending so longer mascots match first
    for mascot in sorted(MASCOTS, key=len, reverse=True):
        if name.endswith(' ' + mascot):
            return name[:-len(mascot)-1].strip()
    
    # If no mascot found, try taking first 1-3 words
    words = name.split()
    if len(words) > 2:
        # Try 2 words first
        return ' '.join(words[:2])
    return name


def build_team_mapping(conn) -> dict:
    """Build mapping from historical team names to team_id."""
    c = conn.cursor()
    
    # Get all our teams
    c.execute("SELECT id, name FROM teams")
    teams = {row[1].lower(): row[0] for row in c.fetchall()}
    teams.update({row[0].replace('-', ' '): row[0] for row in c.execute("SELECT id FROM teams")})
    
    # Get unique historical team names
    c.execute("SELECT DISTINCT home_team FROM historical_games")
    historical_names = [row[0] for row in c.fetchall()]
    
    mapping = {}
    unmatched = []
    
    for full_name in historical_names:
        school_name = strip_mascot(full_name)
        
        # Try exact match
        school_lower = school_name.lower()
        if school_lower in teams:
            mapping[full_name] = teams[school_lower]
            continue
        
        # Try with common variations
        variations = [
            school_lower,
            school_lower.replace(' ', '-'),
            school_lower.replace('state', '').strip() + '-state' if 'state' in school_lower else None,
            school_lower.replace('&', 'and'),
        ]
        
        matched = False
        for var in variations:
            if var and var in teams:
                mapping[full_name] = teams[var]
                matched = True
                break
        
        if not matched:
            # Try fuzzy matching - check if any team name is contained
            for team_name, team_id in teams.items():
                if school_lower in team_name or team_name in school_lower:
                    mapping[full_name] = team_id
                    matched = True
                    break
        
        if not matched:
            unmatched.append(full_name)
    
    print(f"Mapped {len(mapping)}/{len(historical_names)} teams", flush=True)
    if unmatched[:5]:
        print(f"Sample unmatched: {unmatched[:5]}", flush=True)
    
    return mapping


def fetch_historical_weather(lat: float, lon: float, date: str, hour: int = 14) -> dict:
    """Fetch historical weather from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago"
    }
    
    try:
        resp = requests.get(HISTORICAL_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        target_idx = min(hour, len(times) - 1) if times else 0
        
        return {
            "temp_f": hourly.get("temperature_2m", [None])[target_idx],
            "humidity_pct": hourly.get("relative_humidity_2m", [None])[target_idx],
            "precip_prob_pct": hourly.get("precipitation_probability", [None])[target_idx],
            "wind_speed_mph": hourly.get("wind_speed_10m", [None])[target_idx],
            "wind_direction_deg": hourly.get("wind_direction_10m", [None])[target_idx],
        }
    except Exception as e:
        return None


def backfill_historical_weather():
    """Backfill weather for historical_games table."""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Create historical_game_weather table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS historical_game_weather (
            game_id INTEGER PRIMARY KEY,
            temp_f REAL,
            humidity_pct REAL,
            wind_speed_mph REAL,
            wind_direction_deg REAL,
            precip_prob_pct REAL,
            is_dome INTEGER DEFAULT 0,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES historical_games(id)
        )
    """)
    conn.commit()
    
    # Build team mapping
    team_mapping = build_team_mapping(conn)
    
    # Get historical games that need weather
    c.execute("""
        SELECT h.id, h.date, h.home_team
        FROM historical_games h
        LEFT JOIN historical_game_weather hgw ON h.id = hgw.game_id
        WHERE hgw.game_id IS NULL
        ORDER BY h.date DESC
    """)
    
    games = c.fetchall()
    print(f"Found {len(games)} historical games to process", flush=True)
    
    success = 0
    skipped_no_venue = 0
    errors = 0
    
    for i, game in enumerate(games):
        game_id = game["id"]
        date = game["date"]
        home_team = game["home_team"]
        
        # Get team_id from mapping
        team_id = team_mapping.get(home_team)
        if not team_id:
            skipped_no_venue += 1
            continue
        
        # Get venue coords
        c.execute("SELECT latitude, longitude, is_dome FROM venues WHERE team_id = ?", (team_id,))
        venue = c.fetchone()
        
        if not venue or not venue["latitude"]:
            skipped_no_venue += 1
            continue
        
        lat, lon, is_dome = venue["latitude"], venue["longitude"], venue["is_dome"] or 0
        
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(games)}] success={success}, skipped={skipped_no_venue}", flush=True)
        
        weather = fetch_historical_weather(lat, lon, date)
        
        if weather:
            c.execute("""
                INSERT OR REPLACE INTO historical_game_weather 
                (game_id, temp_f, humidity_pct, wind_speed_mph, wind_direction_deg, 
                 precip_prob_pct, is_dome, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                game_id,
                weather.get("temp_f"),
                weather.get("humidity_pct"),
                weather.get("wind_speed_mph"),
                weather.get("wind_direction_deg"),
                weather.get("precip_prob_pct"),
                is_dome
            ))
            success += 1
        else:
            errors += 1
        
        # Rate limiting
        time.sleep(0.15)
        
        # Commit every 200 games
        if (i + 1) % 200 == 0:
            conn.commit()
            print(f"  -- Committed {i+1} games (success: {success}) --")
    
    conn.commit()
    conn.close()
    
    print(f"\n=== Historical Weather Backfill Complete ===")
    print(f"Success: {success}")
    print(f"Skipped (no venue): {skipped_no_venue}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    backfill_historical_weather()
