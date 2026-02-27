#!/usr/bin/env python3
"""
D1Baseball Pre-Game Discovery — replaces StatBroadcast pregame scan.

Scrapes d1baseball.com/scores/ for the full game slate for a target date.
One browser load (~10s) replaces hundreds of SB endpoint probes.

Phases:
  1. Scrape D1B scores page → full game list with times, status, scores
  2. Fill missing game times in DB
  3. Detect status changes (scheduled → in-progress/final)
  4. Report DB + SB coverage stats

Usage:
    python3 scripts/d1b_pregame_discovery.py              # Today
    python3 scripts/d1b_pregame_discovery.py --date 2026-02-28
    python3 scripts/d1b_pregame_discovery.py -v
"""

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

DB_PATH = PROJECT / "data" / "baseball.db"
MAPPING = PROJECT / "data" / "d1b_slug_mapping.json"
OPENCLAW_BIN = "/home/sam/.npm-global/bin/openclaw"
PROFILE = "openclaw"

CT = pytz.timezone("America/Chicago")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

def run_browser_cmd(args, timeout=30):
    """Run an openclaw browser CLI command and return parsed JSON."""
    cmd = [OPENCLAW_BIN, "browser"] + args + ["--browser-profile", PROFILE, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError("Browser command failed: {}".format(result.stderr[:300]))
    return json.loads(result.stdout)


# JS to extract full game data from D1B score tiles (including SB event IDs)
EXTRACT_JS = """() => {
    const tiles = document.querySelectorAll('.d1-score-tile');
    const seen = new Set();
    const games = [];
    tiles.forEach(t => {
        const key = t.dataset.key;
        if (seen.has(key)) return;
        seen.add(key);

        const away = (t.querySelector('.team-1 a[href*="/team/"]') || {}).href || '';
        const home = (t.querySelector('.team-2 a[href*="/team/"]') || {}).href || '';
        const awaySlug = away.match(/\\/team\\/([^/]+)/)?.[1] || '';
        const homeSlug = home.match(/\\/team\\/([^/]+)/)?.[1] || '';

        const inProgress = t.dataset.inProgress === 'true' || t.dataset.inProgress === '1';
        const isOver = t.dataset.isOver === 'true' || t.dataset.isOver === '1';
        const ts = parseInt(t.dataset.matchupTime) || 0;

        const awayScoreEl = t.querySelector('.team-1 .score, .team-1 .d1-score');
        const homeScoreEl = t.querySelector('.team-2 .score, .team-2 .d1-score');
        const awayScore = awayScoreEl ? parseInt(awayScoreEl.textContent.trim()) : null;
        const homeScore = homeScoreEl ? parseInt(homeScoreEl.textContent.trim()) : null;

        // Extract StatBroadcast event ID from "Live Stats" links
        const sbLinks = t.querySelectorAll('a[href*="statbroadcast"], a[href*="statb.us"]');
        let sbId = null;
        sbLinks.forEach(a => {
            if (sbId) return;
            const m = a.href.match(/id=(\\d+)/) || a.href.match(/\\/b\\/(\\d+)/);
            if (m) sbId = parseInt(m[1]);
        });

        if (awaySlug && homeSlug) {
            games.push({
                a: awaySlug, h: homeSlug,
                ip: inProgress, over: isOver,
                as: isNaN(awayScore) ? null : awayScore,
                hs: isNaN(homeScore) ? null : homeScore,
                t: ts,
                sb: sbId
            });
        }
    });
    return games;
}"""


def scrape_d1b_games(date_str):
    """Scrape D1B scores page for all games on a date.

    Returns list of dicts with keys: a(way slug), h(ome slug),
    ip (in-progress), over, as (away score), hs (home score), t (timestamp).
    """
    d1b_date = date_str.replace("-", "")
    url = "https://d1baseball.com/scores/?date={}".format(d1b_date)

    logger.info("Navigating to %s", url)
    run_browser_cmd(["navigate", url], timeout=20)

    # Wait for JS rendering
    time.sleep(5)

    # Verify tiles loaded
    count_js = '() => document.querySelectorAll(".d1-score-tile").length'
    tile_count = run_browser_cmd(["evaluate", "--fn", count_js]).get("result", 0)

    if tile_count == 0:
        logger.warning("No tiles on first try, retrying after 5 s …")
        time.sleep(5)
        tile_count = run_browser_cmd(["evaluate", "--fn", count_js]).get("result", 0)

    if tile_count == 0:
        logger.error("No score tiles found after retry")
        return []

    logger.info("Found %d score tiles", tile_count)

    resp = run_browser_cmd(["evaluate", "--fn", EXTRACT_JS])
    return resp.get("result", [])


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def ts_to_central(ts, date_str):
    """Convert D1B pseudo-UTC timestamp to Central time string.

    D1B encodes Eastern time as seconds-since-midnight-UTC for the date.
    We subtract 1 h to get Central (both zones share DST transitions).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    base_ts = int((dt - datetime(1970, 1, 1)).total_seconds())
    seconds_from_midnight = ts - base_ts
    total_minutes = seconds_from_midnight // 60
    et_hours = total_minutes // 60
    et_minutes = total_minutes % 60

    ct_hours = et_hours - 1
    if ct_hours < 0:
        ct_hours += 24

    ampm = "AM" if ct_hours < 12 else "PM"
    display_hour = ct_hours
    if display_hour == 0:
        display_hour = 12
    elif display_hour > 12:
        display_hour -= 12
    return "{}:{:02d} {}".format(display_hour, et_minutes, ampm)


# ---------------------------------------------------------------------------
# DB updates
# ---------------------------------------------------------------------------

def register_sb_events(d1b_games, date_str, slug_map, conn):
    """Register SB event IDs found on D1B into statbroadcast_events.

    Uses the SB client to fetch event info (xml_file, group_id) for each
    new event, then upserts into the events table.

    Returns (registered_count, skipped_count).
    """
    from statbroadcast_discovery import ensure_table, _upsert_sb_event, match_game
    from statbroadcast_client import StatBroadcastClient
    from html import unescape

    ensure_table(conn)
    client = StatBroadcastClient()

    # Get existing SB events for this date
    existing = set(
        r[0] for r in conn.execute(
            "SELECT sb_event_id FROM statbroadcast_events WHERE game_date = ?",
            (date_str,),
        ).fetchall()
    )

    # Build game lookup for matching
    game_lookup = {}
    for r in conn.execute(
        "SELECT id, away_team_id, home_team_id FROM games WHERE date = ?",
        (date_str,),
    ).fetchall():
        d = dict(r)
        game_lookup[(d["away_team_id"], d["home_team_id"])] = d["id"]

    registered = 0
    skipped = 0

    for dg in d1b_games:
        sb_id = dg.get("sb")
        if not sb_id or sb_id in existing:
            continue

        away_id = slug_map.get(dg["a"])
        home_id = slug_map.get(dg["h"])
        if not away_id or not home_id:
            continue

        # Match to our game
        game_id = game_lookup.get((away_id, home_id)) or game_lookup.get((home_id, away_id))

        # Fetch event info from SB to get xml_file and group_id
        try:
            info = client.get_event_info(sb_id)
        except Exception as e:
            logger.debug("Could not fetch SB event info for %s: %s", sb_id, e)
            skipped += 1
            continue

        if not info or info.get("sport") != "bsgame":
            skipped += 1
            continue

        _upsert_sb_event(conn, {
            "sb_event_id": sb_id,
            "game_id": game_id,
            "home_team": unescape(info.get("home", "")),
            "visitor_team": unescape(info.get("visitor", "")),
            "home_team_id": home_id,
            "visitor_team_id": away_id,
            "game_date": date_str,
            "group_id": info.get("group_id", ""),
            "xml_file": info.get("xml_file", ""),
            "completed": 1 if info.get("completed") else 0,
        })
        registered += 1
        existing.add(sb_id)
        logger.info("  Registered SB event %s for %s @ %s (game %s)",
                     sb_id, dg["a"], dg["h"], game_id or "unmatched")

        time.sleep(0.15)  # gentle rate limit

    conn.commit()
    return registered, skipped


def process_games(d1b_games, date_str, slug_map, conn, verbose=False):
    """Match D1B games to DB rows and apply updates.

    Returns a stats dict.
    """
    # Fetch all DB games for this date
    db_rows = conn.execute(
        "SELECT id, away_team_id, home_team_id, status, time, away_score, home_score "
        "FROM games WHERE date = ?",
        (date_str,),
    ).fetchall()

    # Build lookup by (away, home) — try both orderings during matching
    game_lookup = {}
    for r in db_rows:
        d = dict(r)
        game_lookup[(d["away_team_id"], d["home_team_id"])] = d

    stats = {
        "d1b_total": len(d1b_games),
        "db_total": len(db_rows),
        "matched": 0,
        "times_filled": 0,
        "status_updated": 0,
        "scores_updated": 0,
        "unmapped_slugs": [],
        "not_in_db": [],
    }

    for dg in d1b_games:
        away_id = slug_map.get(dg["a"])
        home_id = slug_map.get(dg["h"])

        if not away_id or not home_id:
            stats["unmapped_slugs"].append("{}@{}".format(dg["a"], dg["h"]))
            continue

        # Try canonical order first, then swapped
        game = game_lookup.get((away_id, home_id)) or game_lookup.get((home_id, away_id))
        if not game:
            stats["not_in_db"].append("{}@{}".format(away_id, home_id))
            continue

        stats["matched"] += 1
        game_id = game["id"]

        # --- Fill missing time ---
        cur_time = game["time"] or ""
        if cur_time in ("", "TBD", "TBA") and dg["t"] > 0:
            time_str = ts_to_central(dg["t"], date_str)
            conn.execute("UPDATE games SET time = ? WHERE id = ?", (time_str, game_id))
            stats["times_filled"] += 1
            logger.debug("  Time filled: %s → %s", game_id, time_str)

        # --- Status promotion ---
        if game["status"] == "scheduled" and (dg["ip"] or dg["over"]):
            new_status = "final" if dg["over"] else "in-progress"
            conn.execute("UPDATE games SET status = ? WHERE id = ?", (new_status, game_id))
            stats["status_updated"] += 1
            logger.info("  Status: %s → %s", game_id, new_status)

        # --- Score updates (only for live/final games) ---
        if (dg["ip"] or dg["over"]) and dg["as"] is not None and dg["hs"] is not None:
            if game["away_score"] != dg["as"] or game["home_score"] != dg["hs"]:
                conn.execute(
                    "UPDATE games SET away_score = ?, home_score = ? WHERE id = ?",
                    (dg["as"], dg["hs"], game_id),
                )
                stats["scores_updated"] += 1

    conn.commit()
    return stats


def sb_coverage(conn, date_str):
    """Return (total_games, games_with_active_sb_event) for the date."""
    row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN se.sb_event_id IS NOT NULL THEN 1 ELSE 0 END) as with_sb
        FROM games g
        LEFT JOIN statbroadcast_events se
            ON g.id = se.game_id AND se.completed = 0
        WHERE g.date = ? AND g.status != 'cancelled'
    """, (date_str,)).fetchone()
    return row["total"], row["with_sb"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(date_str, verbose=False):
    """Full discovery pipeline for one date."""
    # Load slug mapping
    if not MAPPING.exists():
        logger.error("Missing slug mapping: %s", MAPPING)
        return None

    with open(str(MAPPING)) as f:
        slug_map = json.load(f)

    # Scrape D1B
    d1b_games = scrape_d1b_games(date_str)
    if not d1b_games:
        print("No games found on D1B for {}".format(date_str))
        return None

    # Process
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row

    stats = process_games(d1b_games, date_str, slug_map, conn, verbose)

    # Register new SB events found on D1B
    sb_registered, sb_skipped = register_sb_events(d1b_games, date_str, slug_map, conn)
    stats["sb_registered"] = sb_registered
    stats["sb_skipped"] = sb_skipped

    total_games, sb_count = sb_coverage(conn, date_str)
    conn.close()

    stats["sb_total"] = total_games
    stats["sb_covered"] = sb_count
    return stats


def main():
    parser = argparse.ArgumentParser(description="D1Baseball pre-game discovery")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    date_str = args.date or datetime.now(CT).strftime("%Y-%m-%d")

    stats = run(date_str, args.verbose)
    if stats is None:
        sys.exit(1)

    # Pretty summary
    print("\n=== D1B Pre-Game Discovery: {} ===".format(date_str))
    print("D1B games scraped:   {}".format(stats["d1b_total"]))
    print("DB games today:      {}".format(stats["db_total"]))
    print("Matched to DB:       {}".format(stats["matched"]))
    print("Times filled:        {}".format(stats["times_filled"]))
    print("Status updates:      {}".format(stats["status_updated"]))
    print("Score updates:       {}".format(stats["scores_updated"]))
    print("SB events registered: {} new".format(stats["sb_registered"]))
    print("SB coverage:         {}/{} games have active SB events".format(
        stats["sb_covered"], stats["sb_total"]))

    unmapped = set(stats["unmapped_slugs"])
    if unmapped:
        print("Unmapped D1B slugs ({}): {}".format(
            len(unmapped), ", ".join(sorted(unmapped)[:15])))

    not_in_db = set(stats["not_in_db"])
    if not_in_db and args.verbose:
        print("D1B games not in DB ({}): {}".format(
            len(not_in_db), ", ".join(sorted(not_in_db)[:15])))


if __name__ == "__main__":
    main()
