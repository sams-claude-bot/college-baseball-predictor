#!/usr/bin/env python3
"""
Finalize Games — Clean up non-final games from a given date.

Phase 0: Scrape D1Baseball scores page for final results (fast, reliable)
Phase 1: Sync remaining from D1Baseball team pages (handles doubleheaders)
Phase 2: Mark remaining as postponed/canceled if date has passed

Usage:
    python3 scripts/finalize_games.py                    # Yesterday
    python3 scripts/finalize_games.py --date 2026-02-21  # Specific date
    python3 scripts/finalize_games.py --dry-run           # Preview only
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from run_utils import ScriptRunner
from verify_team_schedule import fetch_d1bb_schedule, load_d1bb_slugs, load_reverse_slug_map
from d1b_team_sync import sync_team
from schedule_gateway import ScheduleGateway

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
D1B_SLUG_MAPPING = PROJECT_DIR / 'data' / 'd1b_slug_mapping.json'
OPENCLAW_BIN = "/home/sam/.npm-global/bin/openclaw"
BROWSER_PROFILE = "openclaw"


def run_browser_cmd(args: list, timeout: int = 30) -> dict:
    """Run an openclaw browser CLI command and return parsed JSON."""
    cmd = [OPENCLAW_BIN, "browser"] + args + ["--browser-profile", BROWSER_PROFILE, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Browser command failed: {result.stderr[:200]}")
    return json.loads(result.stdout)


def scrape_d1b_scores_page(date_str: str) -> list:
    """Scrape D1B scores page for final/canceled games.
    
    Returns list of dicts with:
        away_slug, home_slug, status ('final'|'canceled'|'postponed'),
        away_score (int|None), home_score (int|None)
    """
    d1b_date = date_str.replace("-", "")
    url = f"https://d1baseball.com/scores/?date={d1b_date}"
    
    run_browser_cmd(["navigate", url])
    time.sleep(5)  # Wait for JS rendering
    
    js = """() => {
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
            
            const isOver = t.dataset.isOver === 'true' || t.dataset.isOver === '1';
            const isCanceled = t.className.includes('status-canceled');
            const isPostponed = t.className.includes('status-postponed');
            
            // Extract scores from h5 elements (format: "R{runs}H{hits}E{errors}")
            const h5s = t.querySelectorAll('h5');
            let awayRHE = '', homeRHE = '';
            let statusText = '';
            let scoreIdx = 0;
            for (const h of h5s) {
                const txt = h.textContent.trim();
                if (/^R\\d/.test(txt)) {
                    if (scoreIdx === 0) awayRHE = txt;
                    else homeRHE = txt;
                    scoreIdx++;
                }
                if (/FINAL|Canceled|Postponed/i.test(txt)) {
                    statusText = txt.toUpperCase();
                }
            }
            
            if (awaySlug && homeSlug) {
                games.push({
                    a: awaySlug, h: homeSlug,
                    over: isOver,
                    canceled: isCanceled,
                    postponed: isPostponed,
                    status: statusText,
                    ar: awayRHE,
                    hr: homeRHE
                });
            }
        });
        return games;
    }"""
    
    resp = run_browser_cmd(["evaluate", "--fn", js])
    return resp.get("result", [])


def parse_rhe_score(rhe_str: str):
    """Parse runs from D1B RHE string like 'R13H11E4' -> 13."""
    m = re.match(r'R(\d+)', rhe_str)
    return int(m.group(1)) if m else None


def finalize_from_d1b_scores(db, date_str, slug_map, dry_run=False, verbose=False):
    """Phase 0: Scrape D1B scores page and finalize/cancel games.
    
    Returns dict with counts: finalized, canceled, errors.
    """
    stats = {'finalized': 0, 'canceled': 0, 'errors': 0}
    
    try:
        d1b_games = scrape_d1b_scores_page(date_str)
    except Exception as e:
        print(f"  ⚠️  D1B scores page scrape failed: {e}", file=sys.stderr)
        stats['errors'] = 1
        return stats
    
    print(f"  D1B scores page returned {len(d1b_games)} unique games")
    
    # Get non-final games for this date
    nonfinal = db.execute("""
        SELECT id, home_team_id, away_team_id, status
        FROM games WHERE date = ? AND status NOT IN ('final', 'canceled', 'cancelled')
    """, (date_str,)).fetchall()
    
    if not nonfinal:
        print("  No non-final games to process")
        return stats
    
    # Build lookup: (away_team_id, home_team_id) -> game row
    game_lookup = {}
    for g in nonfinal:
        game_lookup[(g['away_team_id'], g['home_team_id'])] = dict(g)
    
    gw = ScheduleGateway(db)
    
    for dg in d1b_games:
        away_id = slug_map.get(dg['a'])
        home_id = slug_map.get(dg['h'])
        if not away_id or not home_id:
            continue
        
        game = game_lookup.get((away_id, home_id))
        if not game:
            continue
        
        game_id = game['id']
        
        # Handle canceled games
        if dg.get('canceled') or 'CANCELED' in dg.get('status', ''):
            if not dry_run:
                gw.mark_cancelled(game_id, reason='Canceled per D1Baseball')
            if verbose:
                print(f"  🚫 Canceled: {game_id}")
            stats['canceled'] += 1
            continue
        
        # Handle postponed games
        if dg.get('postponed') or 'POSTPONED' in dg.get('status', ''):
            if not dry_run:
                gw.mark_postponed(game_id, reason='Postponed per D1Baseball')
            if verbose:
                print(f"  ⏸️  Postponed (D1B): {game_id}")
            continue
        
        # Handle final games
        if dg.get('over') or 'FINAL' in dg.get('status', ''):
            away_score = parse_rhe_score(dg.get('ar', ''))
            home_score = parse_rhe_score(dg.get('hr', ''))
            
            if away_score is not None and home_score is not None:
                if not dry_run:
                    gw.finalize_game(game_id, home_score, away_score)
                if verbose:
                    print(f"  ✅ Finalized: {game_id} ({away_score}-{home_score})")
                stats['finalized'] += 1
            else:
                if verbose:
                    print(f"  ⚠️  Final but no scores: {game_id} (ar={dg.get('ar')}, hr={dg.get('hr')})")
    
    if not dry_run:
        db.commit()
    
    return stats


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_nonfinal_teams(db, date_str):
    """Return set of team IDs that have non-final games on date."""
    rows = db.execute("""
        SELECT home_team_id, away_team_id FROM games
        WHERE date = ? AND status != 'final'
    """, (date_str,)).fetchall()
    teams = set()
    for r in rows:
        teams.add(r['home_team_id'])
        teams.add(r['away_team_id'])
    return teams


def mark_postponed(db, date_str, dry_run=False, verbose=False):
    """Mark remaining non-final games from a past date as postponed.
    
    Only marks games where D1BB also has no result (confirming it wasn't played).
    Games stuck as in-progress with partial scores get marked too if D1BB has no result.
    """
    rows = db.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.status, g.home_score, g.away_score
        FROM games g
        WHERE g.date = ? AND g.status != 'final'
    """, (date_str,)).fetchall()
    
    gw = ScheduleGateway(db)
    marked = 0
    for r in rows:
        if not dry_run:
            gw.mark_postponed(r['id'], reason='No result on D1Baseball')
        if verbose:
            print(f"  ⏸️  Postponed: {r['id']} (was {r['status']})")
        marked += 1
    
    return marked


def main():
    parser = argparse.ArgumentParser(description='Finalize games from a given date')
    parser.add_argument('--date', '-d', help='Date to finalize (YYYY-MM-DD, default: yesterday)')
    parser.add_argument('--dry-run', action='store_true', help='Preview only')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-postpone', action='store_true', help='Skip marking games as postponed')
    args = parser.parse_args()

    runner = ScriptRunner("finalize_games")
    
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    runner.info(f"Finalizing games for {target_date}")
    
    db = get_db()
    slugs = load_d1bb_slugs()
    reverse = load_reverse_slug_map()
    
    # Count non-final before
    before = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status != 'final'",
        (target_date,)
    ).fetchone()['c']
    
    total_games = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ?",
        (target_date,)
    ).fetchone()['c']
    
    runner.info(f"Total games: {total_games} | Non-final: {before}")
    
    if before == 0:
        runner.info("All games already final — nothing to do")
        runner.finish()
        return
    
    # Phase 0: Scrape D1B scores page (fast, catches games team pages miss)
    runner.info(f"Phase 0: Scraping D1B scores page for {target_date}...")
    try:
        with open(D1B_SLUG_MAPPING) as f:
            d1b_slug_map = json.load(f)
        p0_stats = finalize_from_d1b_scores(db, target_date, d1b_slug_map,
                                            dry_run=args.dry_run, verbose=args.verbose)
        runner.info(f"Phase 0 results: {p0_stats['finalized']} finalized, "
                    f"{p0_stats['canceled']} canceled")
    except Exception as e:
        runner.warn(f"Phase 0 failed (non-fatal): {e}")
    
    # Re-check how many are still non-final
    before = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status NOT IN ('final', 'canceled', 'cancelled', 'postponed')",
        (target_date,)
    ).fetchone()['c']
    
    if before == 0:
        runner.info("All games resolved after Phase 0 — skipping team page sync")
    else:
        runner.info(f"Still {before} non-final after Phase 0, proceeding to team pages...")
    
    # Phase 1: Sync scores from D1BB team pages (only if Phase 0 left unresolved games)
    total_scored = 0
    total_created = 0
    errors = []
    
    if before > 0:
        teams = get_nonfinal_teams(db, target_date)
        teams_with_slugs = {t for t in teams if t in slugs}
        runner.info(f"Phase 1: Syncing {len(teams_with_slugs)} teams from D1Baseball...")
        
        for i, tid in enumerate(sorted(teams_with_slugs)):
            try:
                stats = sync_team(db, tid, slugs[tid], reverse,
                                dry_run=args.dry_run, verbose=args.verbose)
                total_scored += stats['scored']
                total_created += stats['created']
                if stats['errors']:
                    errors.append(tid)
            except Exception as e:
                runner.warn(f"Error syncing {tid}: {e}")
                errors.append(tid)
            
            if not args.dry_run and i % 10 == 9:
                db.commit()
            time.sleep(0.3)
        
        if not args.dry_run:
            db.commit()
        
        runner.info(f"Phase 1 results: {total_scored} scored, {total_created} created, {len(errors)} errors")
    else:
        runner.info("Phase 1: Skipped — all games resolved by Phase 0")
    
    # Phase 2: Mark remaining as postponed (if date is in the past)
    remaining = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status NOT IN ('final', 'canceled', 'cancelled', 'postponed')",
        (target_date,)
    ).fetchone()['c']
    
    if remaining > 0 and not args.no_postpone:
        today = datetime.now().strftime('%Y-%m-%d')
        if target_date < today:
            runner.info(f"Phase 2: Marking {remaining} remaining games as postponed...")
            marked = mark_postponed(db, target_date, dry_run=args.dry_run, verbose=args.verbose)
            if not args.dry_run:
                db.commit()
            runner.info(f"Marked {marked} games as postponed")
        else:
            runner.info(f"Phase 2: Skipped — {target_date} is today or future ({remaining} still pending)")
    
    # Final count
    final_nonfinal = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status NOT IN ('final', 'postponed', 'canceled')",
        (target_date,)
    ).fetchone()['c']
    
    final_count = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status = 'final'",
        (target_date,)
    ).fetchone()['c']
    
    postponed_count = db.execute(
        "SELECT COUNT(*) as c FROM games WHERE date = ? AND status = 'postponed'",
        (target_date,)
    ).fetchone()['c']
    
    db.close()
    
    runner.info(f"Final: {final_count} final, {postponed_count} postponed, {final_nonfinal} unresolved")
    runner.add_stat("date", target_date)
    runner.add_stat("scored", total_scored)
    runner.add_stat("postponed", postponed_count)
    runner.add_stat("unresolved", final_nonfinal)
    runner.finish()


if __name__ == '__main__':
    main()
