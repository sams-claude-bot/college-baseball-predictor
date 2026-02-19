#!/usr/bin/env python3
"""
Compute Team Batting Quality Metrics

Computes lineup-level batting quality from player_stats:
- Core lineup quality (top 9 hitters by AB, AB-weighted)
- Power metrics (ISO, HR rate, SLG)
- Plate discipline (K%, BB%, OBP)
- Bench depth (hitters 10+)
- Contact quality (BABIP, wOBA, wRC+)
- Lineup concentration (how reliant on a few bats)

Usage:
    python3 scripts/compute_batting_quality.py
    python3 scripts/compute_batting_quality.py --team lsu
    python3 scripts/compute_batting_quality.py --show-top 25
"""

import argparse
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / "data" / "baseball.db"

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def get_db():
    db = sqlite3.connect(str(DB_PATH), timeout=30)
    db.row_factory = sqlite3.Row
    return db


def ensure_table(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS team_batting_quality (
            team_id TEXT PRIMARY KEY,

            -- Core lineup (top 9 by AB, AB-weighted)
            lineup_avg REAL,
            lineup_obp REAL,
            lineup_slg REAL,
            lineup_ops REAL,
            lineup_woba REAL,
            lineup_wrc_plus REAL,
            lineup_iso REAL,
            lineup_babip REAL,
            lineup_k_pct REAL,
            lineup_bb_pct REAL,

            -- Power
            hr_per_game REAL,
            extra_base_hit_pct REAL,    -- (2B+3B+HR) / hits

            -- Bench (hitters 10+)
            bench_avg REAL,
            bench_ops REAL,
            bench_depth INTEGER,         -- number of bench hitters with AB

            -- Lineup concentration
            top3_ab_pct REAL,           -- top 3 hitters AB / total AB
            lineup_hhi REAL,            -- Herfindahl of AB distribution

            -- Run production
            runs_per_game REAL,
            total_abs INTEGER,
            total_hits INTEGER,
            total_hrs INTEGER,

            -- Quality counts
            elite_bats INTEGER,          -- hitters with OPS >= .900 and AB >= 10
            solid_bats INTEGER,          -- hitters with OPS >= .750 and AB >= 10
            weak_bats INTEGER,           -- hitters with OPS < .550 and AB >= 10

            last_updated TEXT
        )
    """)
    db.commit()


def compute_team_batting(db, team_id):
    """Compute batting quality metrics for one team."""
    hitters = db.execute("""
        SELECT at_bats, hits, doubles, triples, home_runs, walks, strikeouts,
               batting_avg, obp, slg, ops, woba, wrc_plus, iso, babip,
               k_pct, bb_pct, runs, name, games
        FROM player_stats
        WHERE team_id = ? AND at_bats > 0
        ORDER BY at_bats DESC
    """, (team_id,)).fetchall()

    if not hitters:
        return None

    total_ab = sum(h['at_bats'] for h in hitters)
    if total_ab == 0:
        return None

    # Core lineup = top 9 by AB
    lineup = hitters[:9]
    bench = hitters[9:]

    lineup_ab = sum(h['at_bats'] for h in lineup)

    def ab_weighted(group, field, default):
        """AB-weighted average of a stat."""
        group_ab = sum(h['at_bats'] for h in group)
        if group_ab == 0:
            return default
        return sum((h[field] or default) * h['at_bats'] for h in group) / group_ab

    # Lineup stats (AB-weighted)
    l_avg = ab_weighted(lineup, 'batting_avg', 0.250)
    l_obp = ab_weighted(lineup, 'obp', 0.320)
    l_slg = ab_weighted(lineup, 'slg', 0.380)
    l_ops = ab_weighted(lineup, 'ops', 0.700)
    l_woba = ab_weighted(lineup, 'woba', 0.310)
    l_wrc = ab_weighted(lineup, 'wrc_plus', 100.0)
    l_iso = ab_weighted(lineup, 'iso', 0.130)
    l_babip = ab_weighted(lineup, 'babip', 0.300)
    l_k_pct = ab_weighted(lineup, 'k_pct', 20.0)
    l_bb_pct = ab_weighted(lineup, 'bb_pct', 8.0)

    # Power
    total_hrs = sum(h['home_runs'] or 0 for h in hitters)
    total_games = max(h['games'] or 1 for h in hitters)  # approx team games
    hr_per_game = total_hrs / max(total_games, 1)

    total_hits = sum(h['hits'] or 0 for h in hitters)
    total_2b = sum(h['doubles'] or 0 for h in hitters)
    total_3b = sum(h['triples'] or 0 for h in hitters)
    xbh = total_2b + total_3b + total_hrs
    xbh_pct = xbh / max(total_hits, 1)

    # Bench
    bench_ab = sum(h['at_bats'] for h in bench)
    b_avg = ab_weighted(bench, 'batting_avg', 0.230) if bench else 0.230
    b_ops = ab_weighted(bench, 'ops', 0.600) if bench else 0.600

    # Concentration
    top3 = hitters[:3]
    top3_ab = sum(h['at_bats'] for h in top3)
    top3_ab_pct = top3_ab / total_ab

    ab_shares = [(h['at_bats'] / total_ab) for h in hitters]
    hhi = sum(s ** 2 for s in ab_shares)

    # Run production
    total_runs = sum(h['runs'] or 0 for h in hitters)
    rpg = total_runs / max(total_games, 1)

    # Quality counts (minimum 10 AB)
    min_ab = 10
    elite = sum(1 for h in hitters if (h['ops'] or 0) >= 0.900 and h['at_bats'] >= min_ab)
    solid = sum(1 for h in hitters if (h['ops'] or 0) >= 0.750 and h['at_bats'] >= min_ab)
    weak = sum(1 for h in hitters if (h['ops'] or 0) < 0.550 and h['at_bats'] >= min_ab)

    return {
        'team_id': team_id,
        'lineup_avg': l_avg,
        'lineup_obp': l_obp,
        'lineup_slg': l_slg,
        'lineup_ops': l_ops,
        'lineup_woba': l_woba,
        'lineup_wrc_plus': l_wrc,
        'lineup_iso': l_iso,
        'lineup_babip': l_babip,
        'lineup_k_pct': l_k_pct,
        'lineup_bb_pct': l_bb_pct,
        'hr_per_game': hr_per_game,
        'extra_base_hit_pct': xbh_pct,
        'bench_avg': b_avg,
        'bench_ops': b_ops,
        'bench_depth': len(bench),
        'top3_ab_pct': top3_ab_pct,
        'lineup_hhi': hhi,
        'runs_per_game': rpg,
        'total_abs': total_ab,
        'total_hits': total_hits,
        'total_hrs': total_hrs,
        'elite_bats': elite,
        'solid_bats': solid,
        'weak_bats': weak,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def compute_all(db, team_id=None):
    """Compute batting quality for all teams (or one)."""
    ensure_table(db)

    if team_id:
        teams = [(team_id,)]
    else:
        teams = db.execute("SELECT DISTINCT team_id FROM player_stats WHERE at_bats > 0").fetchall()

    updated = 0
    for (tid,) in teams:
        result = compute_team_batting(db, tid)
        if result:
            cols = list(result.keys())
            placeholders = ', '.join(['?'] * len(cols))
            col_str = ', '.join(cols)
            db.execute(
                f"INSERT OR REPLACE INTO team_batting_quality ({col_str}) VALUES ({placeholders})",
                [result[c] for c in cols]
            )
            updated += 1

    db.commit()
    log.info(f"Updated batting quality for {updated} teams")
    return updated


def show_top(db, n=25):
    """Show top teams by lineup OPS."""
    rows = db.execute("""
        SELECT team_id, lineup_ops, lineup_woba, lineup_avg, lineup_k_pct, lineup_bb_pct,
               hr_per_game, runs_per_game, elite_bats, solid_bats, lineup_iso
        FROM team_batting_quality
        WHERE total_abs >= 100
        ORDER BY lineup_ops DESC LIMIT ?
    """, (n,)).fetchall()

    print(f"\n{'Team':<25} {'OPS':>6} {'wOBA':>6} {'AVG':>6} {'K%':>5} {'BB%':>5} {'HR/G':>5} {'R/G':>5} {'Elite':>6} {'Solid':>6}")
    print("-" * 95)
    for r in rows:
        print(f"{r['team_id']:<25} {r['lineup_ops']:>6.3f} {r['lineup_woba']:>6.3f} {r['lineup_avg']:>6.3f} {r['lineup_k_pct']:>5.1f} {r['lineup_bb_pct']:>5.1f} {r['hr_per_game']:>5.2f} {r['runs_per_game']:>5.1f} {r['elite_bats']:>6} {r['solid_bats']:>6}")


def main():
    parser = argparse.ArgumentParser(description='Compute team batting quality metrics')
    parser.add_argument('--team', help='Single team ID')
    parser.add_argument('--show-top', type=int, help='Show top N teams')
    args = parser.parse_args()

    db = get_db()
    compute_all(db, team_id=args.team)

    if args.show_top:
        show_top(db, args.show_top)


if __name__ == '__main__':
    main()
