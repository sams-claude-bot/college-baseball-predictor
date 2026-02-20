#!/usr/bin/env python3
"""
Backfill historical games from ESPN API for additional seasons.

Extends the existing scrape_historical.py to cover 2021-2023.
Uses the same ESPN scoreboard API and table schema.

Usage:
    python3 scripts/backfill_historical.py                    # All missing seasons
    python3 scripts/backfill_historical.py --season 2023      # Single season
    python3 scripts/backfill_historical.py --dry-run           # Count only
"""

import argparse
import json
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
PROGRESS_PATH = PROJECT_DIR / 'data' / 'historical_backfill_progress.json'

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={date}&limit=200"

SEASONS = {
    2021: {'start': '2021-02-19', 'end': '2021-06-30'},
    2022: {'start': '2022-02-18', 'end': '2022-06-27'},
    2023: {'start': '2023-02-17', 'end': '2023-06-26'},
}

# Postseason roughly starts late May
POSTSEASON_STARTS = {2021: '2021-06-04', 2022: '2022-06-03', 2023: '2023-06-02'}


def fetch_date(date_str):
    """Fetch completed games for a date from ESPN."""
    url = ESPN_URL.format(date=date_str.replace('-', ''))
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR {date_str}: {e}")
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
                else:
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
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress))


def scrape_season(conn, season, dry_run=False):
    info = SEASONS[season]
    start = datetime.strptime(info['start'], '%Y-%m-%d')
    end = datetime.strptime(info['end'], '%Y-%m-%d')
    ps_start = POSTSEASON_STARTS.get(season, '9999-99-99')

    progress = load_progress()
    season_key = str(season)
    completed = set(progress.get(season_key, []))

    total_games = 0
    total_inserted = 0
    current = start
    total_days = (end - start).days + 1

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        day_num = (current - start).days + 1

        if date_str in completed:
            current += timedelta(days=1)
            continue

        if dry_run:
            current += timedelta(days=1)
            continue

        games = fetch_date(date_str)
        if games is None:
            time.sleep(3)
            games = fetch_date(date_str)
            if games is None:
                print(f"  SKIP {date_str}")
                current += timedelta(days=1)
                continue

        postseason = 1 if date_str >= ps_start else 0
        inserted = 0

        for g in games:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO historical_games
                    (season, date, away_team, home_team, away_score, home_score, 
                     neutral_site, postseason, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (season, g['date'], g['away_team'], g['home_team'],
                      g['away_score'], g['home_score'], g['neutral_site'],
                      postseason, 'espn'))
                inserted += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        total_games += len(games)
        total_inserted += inserted

        if len(games) > 0 and day_num % 7 == 0:
            print(f"  {date_str}: {day_num}/{total_days} days, {total_games} games so far")

        completed.add(date_str)
        progress[season_key] = list(completed)
        if day_num % 14 == 0:
            save_progress(progress)

        time.sleep(0.3)  # Rate limit
        current += timedelta(days=1)

    save_progress(progress)
    return total_games, total_inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, help='Single season to backfill')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row

    # Check existing data
    existing = {}
    for row in conn.execute("SELECT season, COUNT(*) as cnt FROM historical_games GROUP BY season"):
        existing[row['season']] = row['cnt']
    print("Existing data:")
    for s in sorted(existing):
        print(f"  {s}: {existing[s]} games")

    seasons = [args.season] if args.season else sorted(SEASONS.keys())
    seasons = [s for s in seasons if s in SEASONS]

    for season in seasons:
        ex = existing.get(season, 0)
        if ex > 100:
            print(f"\n⏭️  {season}: already has {ex} games, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Scraping {season} season...")
        print(f"{'='*50}")

        if args.dry_run:
            info = SEASONS[season]
            days = (datetime.strptime(info['end'], '%Y-%m-%d') - 
                    datetime.strptime(info['start'], '%Y-%m-%d')).days
            print(f"  Would scrape {days} days ({info['start']} to {info['end']})")
            continue

        total, inserted = scrape_season(conn, season, dry_run=args.dry_run)
        print(f"  Found: {total} games, inserted: {inserted}")

    # Final summary
    print(f"\n{'='*50}")
    print("FINAL COUNTS:")
    for row in conn.execute("SELECT season, COUNT(*) as cnt FROM historical_games GROUP BY season ORDER BY season"):
        print(f"  {row['season']}: {row['cnt']} games")
    total = conn.execute("SELECT COUNT(*) FROM historical_games").fetchone()[0]
    print(f"  TOTAL: {total}")

    conn.close()


if __name__ == '__main__':
    main()
