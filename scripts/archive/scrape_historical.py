#!/usr/bin/env python3
"""Scrape NCAA D1 college baseball game results from ESPN API for 2024 and 2025 seasons."""

import json
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import sys
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'baseball.db')
PROGRESS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_scrape_progress.json')

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={date}&limit=200"

# Postseason date ranges
POSTSEASON_2024 = {
    'regionals': ('2024-05-31', '2024-06-03'),
    'super_regionals': ('2024-06-07', '2024-06-10'),
    'cws': ('2024-06-14', '2024-06-24'),
}
POSTSEASON_2025 = {
    'regionals': ('2025-05-30', '2025-06-02'),
    'super_regionals': ('2025-06-06', '2025-06-09'),
    'cws': ('2025-06-13', '2025-06-23'),
}

SEASONS = {
    2024: {'start': '2024-02-16', 'end': '2024-06-25', 'postseason': POSTSEASON_2024},
    2025: {'start': '2025-02-14', 'end': '2025-06-23', 'postseason': POSTSEASON_2025},
}


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            date TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_score INTEGER NOT NULL,
            home_score INTEGER NOT NULL,
            neutral_site INTEGER DEFAULT 0,
            postseason INTEGER DEFAULT 0,
            source TEXT,
            UNIQUE(date, away_team, home_team)
        )
    """)
    conn.commit()


def is_postseason(date_str, season):
    ps = SEASONS[season]['postseason']
    for _, (start, end) in ps.items():
        if start <= date_str <= end:
            return 1
    return 0


def fetch_date(date_str):
    """Fetch games for a single date from ESPN API."""
    url = ESPN_URL.format(date=date_str.replace('-', ''))
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"  ERROR fetching {date_str}: {e}")
        return None

    games = []
    for event in data.get('events', []):
        for comp in event.get('competitions', []):
            status = comp.get('status', {}).get('type', {})
            if not status.get('completed', False):
                continue

            competitors = comp.get('competitors', [])
            if len(competitors) != 2:
                continue

            home = away = None
            for c in competitors:
                if c.get('homeAway') == 'home':
                    home = c
                elif c.get('homeAway') == 'away':
                    away = c

            if not home or not away:
                continue

            try:
                home_score = int(home['score'])
                away_score = int(away['score'])
            except (ValueError, KeyError):
                continue

            home_team = home.get('team', {}).get('displayName', '')
            away_team = away.get('team', {}).get('displayName', '')
            neutral = 1 if comp.get('neutralSite', False) else 0

            if home_team and away_team:
                games.append({
                    'date': date_str,
                    'away_team': away_team,
                    'home_team': home_team,
                    'away_score': away_score,
                    'home_score': home_score,
                    'neutral_site': neutral,
                })

    return games


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f)


def scrape_season(conn, season):
    info = SEASONS[season]
    start = datetime.strptime(info['start'], '%Y-%m-%d')
    end = datetime.strptime(info['end'], '%Y-%m-%d')

    progress = load_progress()
    season_key = str(season)
    completed_dates = set(progress.get(season_key, []))

    total_games = 0
    current = start
    total_days = (end - start).days + 1
    day_num = 0

    while current <= end:
        day_num += 1
        date_str = current.strftime('%Y-%m-%d')

        if date_str in completed_dates:
            current += timedelta(days=1)
            continue

        games = fetch_date(date_str)
        if games is None:
            # Retry once after delay
            time.sleep(3)
            games = fetch_date(date_str)
            if games is None:
                print(f"  SKIPPING {date_str} after retry failure")
                current += timedelta(days=1)
                continue

        postseason = is_postseason(date_str, season)

        inserted = 0
        for g in games:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO historical_games 
                    (season, date, away_team, home_team, away_score, home_score, neutral_site, postseason, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (season, g['date'], g['away_team'], g['home_team'],
                      g['away_score'], g['home_score'], g['neutral_site'],
                      postseason, 'espn'))
                if conn.total_changes:
                    inserted += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        total_games += len(games)

        if len(games) > 0:
            print(f"  {date_str}: {len(games)} games ({day_num}/{total_days})")

        # Track progress
        completed_dates.add(date_str)
        progress[season_key] = list(completed_dates)
        if day_num % 7 == 0:
            save_progress(progress)

        # Rate limit: be nice to ESPN
        time.sleep(0.5)
        current += timedelta(days=1)

    save_progress(progress)
    return total_games


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    for season in [2024, 2025]:
        print(f"\n=== Scraping {season} season ===")
        count = scrape_season(conn, season)
        print(f"  Total games found: {count}")

    # Summary
    for season in [2024, 2025]:
        row = conn.execute("SELECT count(*), min(date), max(date) FROM historical_games WHERE season=?", (season,)).fetchone()
        print(f"\n{season}: {row[0]} games, {row[1]} to {row[2]}")

    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
