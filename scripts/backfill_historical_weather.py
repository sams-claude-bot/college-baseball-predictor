#!/usr/bin/env python3
"""
Backfill historical weather data for completed games using Open-Meteo Archive API.

Open-Meteo Historical API: https://open-meteo.com/en/docs/historical-weather-api
- Free, no API key required
- Data available from 1940 to 5 days ago
- Rate limit: ~10 requests/second (be conservative)
"""

import sqlite3
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

# Open-Meteo Historical Weather API
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"


def parse_game_time(time_str: str) -> int:
    """Convert time string like '2:00 PM' to hour (0-23)."""
    if not time_str:
        return 14  # Default to 2 PM if no time
    
    time_str = time_str.strip().upper()
    try:
        # Handle formats like "2:00 PM", "6:30 PM", "12:00 PM"
        if ":" in time_str:
            parts = time_str.replace("AM", "").replace("PM", "").strip().split(":")
            hour = int(parts[0])
            is_pm = "PM" in time_str
            
            if is_pm and hour != 12:
                hour += 12
            elif not is_pm and hour == 12:
                hour = 0
            
            return hour
    except:
        pass
    
    return 14  # Default


def fetch_historical_weather(lat: float, lon: float, date: str, hour: int) -> dict:
    """
    Fetch historical weather from Open-Meteo Archive API.
    
    Returns dict with weather data or None on error.
    """
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
        
        # Find the closest hour
        target_idx = min(hour, len(times) - 1) if times else 0
        
        return {
            "temp_f": hourly.get("temperature_2m", [None])[target_idx],
            "humidity_pct": hourly.get("relative_humidity_2m", [None])[target_idx],
            "precip_prob_pct": hourly.get("precipitation_probability", [None])[target_idx],
            "wind_speed_mph": hourly.get("wind_speed_10m", [None])[target_idx],
            "wind_direction_deg": hourly.get("wind_direction_10m", [None])[target_idx],
        }
        
    except Exception as e:
        print(f"  Error fetching weather: {e}")
        return None


def backfill_weather():
    """Backfill weather for all completed games missing weather data."""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Find games that need weather backfill
    c.execute("""
        SELECT g.id, g.date, g.time, v.latitude, v.longitude, v.is_dome, t.name as home_team
        FROM games g
        JOIN venues v ON g.home_team_id = v.team_id
        JOIN teams t ON g.home_team_id = t.id
        LEFT JOIN game_weather gw ON g.id = gw.game_id
        WHERE g.status = 'final' 
          AND v.latitude IS NOT NULL
          AND gw.game_id IS NULL
        ORDER BY g.date
    """)
    
    games = c.fetchall()
    print(f"Found {len(games)} games needing weather backfill")
    
    if not games:
        print("All games already have weather data!")
        return
    
    # Check if dates are within historical API range (up to 2 days ago for safety)
    cutoff_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    success = 0
    skipped = 0
    errors = 0
    
    for i, game in enumerate(games):
        game_id = game["id"]
        date = game["date"]
        time_str = game["time"]
        lat = game["latitude"]
        lon = game["longitude"]
        is_dome = game["is_dome"]
        home_team = game["home_team"]
        
        # Skip games too recent for historical API
        if date > cutoff_date:
            print(f"  [{i+1}/{len(games)}] {game_id}: Too recent for historical API (need 5+ days)")
            skipped += 1
            continue
        
        hour = parse_game_time(time_str)
        
        print(f"  [{i+1}/{len(games)}] {game_id} @ {home_team}: {date} {hour}:00...")
        
        weather = fetch_historical_weather(lat, lon, date, hour)
        
        if weather:
            c.execute("""
                INSERT OR REPLACE INTO game_weather 
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
                is_dome or 0
            ))
            success += 1
            print(f"    ✓ {weather.get('temp_f')}°F, {weather.get('wind_speed_mph')} mph wind")
        else:
            errors += 1
        
        # Rate limiting - be conservative
        time.sleep(0.2)
        
        # Commit every 50 games
        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  -- Committed {i+1} games --")
    
    conn.commit()
    conn.close()
    
    print(f"\n=== Backfill Complete ===")
    print(f"Success: {success}")
    print(f"Skipped (too recent): {skipped}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    backfill_weather()
