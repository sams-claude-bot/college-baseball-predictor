#!/usr/bin/env python3
"""
Fetch PEAR Ratings data and store in database.

Usage:
    python3 scripts/fetch_pear_ratings.py           # Fetch current season
    python3 scripts/fetch_pear_ratings.py 2025      # Fetch specific season
"""

import sys
import requests
import difflib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection, init_pear_ratings_table

API_URL = "https://pearatings.com/api/cbase/stats"


def _build_name_lookup(conn):
    """Build lookup dicts from teams table and team_aliases table."""
    c = conn.cursor()

    # teams.name -> team_id (case-insensitive)
    c.execute("SELECT id, name FROM teams")
    name_to_id = {}
    id_to_name = {}
    for row in c.fetchall():
        name_to_id[row["name"].lower()] = row["id"]
        id_to_name[row["id"]] = row["name"]

    # team_aliases.alias -> team_id
    c.execute("SELECT alias, team_id FROM team_aliases")
    alias_to_id = {}
    for row in c.fetchall():
        alias_to_id[row["alias"].lower()] = row["team_id"]

    return name_to_id, alias_to_id, id_to_name


def _resolve_team(pear_name, name_to_id, alias_to_id, all_names):
    """Resolve a PEAR team name to our internal team_id.

    Strategy:
      1. Exact match on teams.name (case-insensitive)
      2. Exact match on team_aliases.alias (case-insensitive)
      3. Fuzzy match against all known names/aliases (threshold 0.8)
    """
    lower = pear_name.lower().strip()

    # 1. Exact match on team name
    if lower in name_to_id:
        return name_to_id[lower]

    # 2. Exact match on alias
    if lower in alias_to_id:
        return alias_to_id[lower]

    # 3. Fuzzy match
    matches = difflib.get_close_matches(lower, all_names, n=1, cutoff=0.8)
    if matches:
        best = matches[0]
        return name_to_id.get(best) or alias_to_id.get(best)

    return None


def fetch_pear_ratings(season=None):
    """Fetch PEAR stats API and upsert into pear_ratings table."""
    if season is None:
        season = datetime.now().year

    print(f"Fetching PEAR ratings for {season}...")
    resp = requests.get(API_URL, params={"season": season}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    stats = data.get("stats", [])
    if not stats:
        print("No stats returned from PEAR API")
        return

    print(f"Got {len(stats)} teams from PEAR API")

    conn = get_connection()

    # Ensure table exists
    init_pear_ratings_table()

    name_to_id, alias_to_id, id_to_name = _build_name_lookup(conn)
    all_names = list(name_to_id.keys()) + list(alias_to_id.keys())

    c = conn.cursor()
    now = datetime.now().isoformat()

    matched = 0
    unmatched = []

    for team in stats:
        pear_name = team.get("Team", "")
        team_id = _resolve_team(pear_name, name_to_id, alias_to_id, all_names)

        if not team_id:
            unmatched.append(pear_name)
            continue

        matched += 1

        c.execute('''
            INSERT INTO pear_ratings (
                team_name, team_id, season, rating, net_score, net_rank,
                rqi, rqi_rank, sos_rank, sor_rank, elo, elo_rank, rpi_rank,
                resume_quality, avg_expected_wins, fwar, owar_z, pwar_z,
                wpoe_pct, pythag, killshots, conceded, kshot_ratio,
                era, whip, kp9, rpg, ba, obp, slg, ops, woba, iso, pct,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_id, season) DO UPDATE SET
                team_name = excluded.team_name,
                rating = excluded.rating,
                net_score = excluded.net_score,
                net_rank = excluded.net_rank,
                rqi = excluded.rqi,
                rqi_rank = excluded.rqi_rank,
                sos_rank = excluded.sos_rank,
                sor_rank = excluded.sor_rank,
                elo = excluded.elo,
                elo_rank = excluded.elo_rank,
                rpi_rank = excluded.rpi_rank,
                resume_quality = excluded.resume_quality,
                avg_expected_wins = excluded.avg_expected_wins,
                fwar = excluded.fwar,
                owar_z = excluded.owar_z,
                pwar_z = excluded.pwar_z,
                wpoe_pct = excluded.wpoe_pct,
                pythag = excluded.pythag,
                killshots = excluded.killshots,
                conceded = excluded.conceded,
                kshot_ratio = excluded.kshot_ratio,
                era = excluded.era,
                whip = excluded.whip,
                kp9 = excluded.kp9,
                rpg = excluded.rpg,
                ba = excluded.ba,
                obp = excluded.obp,
                slg = excluded.slg,
                ops = excluded.ops,
                woba = excluded.woba,
                iso = excluded.iso,
                pct = excluded.pct,
                fetched_at = excluded.fetched_at
        ''', (
            pear_name,
            team_id,
            season,
            team.get("Rating"),
            team.get("NET_Score"),
            team.get("NET"),
            team.get("RQI"),
            team.get("PRR"),          # PRR = RQI rank
            team.get("SOS"),
            team.get("SOR"),
            team.get("ELO"),
            _safe_int(team.get("ELO_Rank")),
            team.get("RPI"),
            team.get("resume_quality"),
            team.get("avg_expected_wins"),
            team.get("fWAR"),
            team.get("oWAR_z"),
            team.get("pWAR_z"),
            team.get("wpoe_pct"),
            team.get("PYTHAG"),
            team.get("Killshots"),
            team.get("Conceded"),
            team.get("KSHOT_Ratio"),
            team.get("ERA"),
            team.get("WHIP"),
            team.get("KP9"),
            team.get("RPG"),
            team.get("BA"),
            team.get("OBP"),
            team.get("SLG"),
            team.get("OPS"),
            team.get("wOBA"),
            team.get("ISO"),
            team.get("PCT"),
            now,
        ))

    conn.commit()
    conn.close()

    print(f"\nMatched: {matched}/{len(stats)} teams")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for name in sorted(unmatched):
            print(f"  - {name}")
        print("\nTip: Add missing teams to team_aliases table for future runs.")


def _safe_int(val):
    """Convert to int if possible, else None."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else None
    fetch_pear_ratings(season)
