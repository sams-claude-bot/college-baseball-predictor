#!/usr/bin/env python3
"""
Sam's RPI (Rating Percentage Index) Calculator

Computes RPI for all D1 teams using the standard formula with home/away weighting.

Formula:
    RPI = 0.25 * WP + 0.50 * OWP + 0.25 * OOWP

Where:
    WP   = Team's weighted winning percentage
           (road wins × 1.3, home wins × 0.7, neutral × 1.0)
    OWP  = Average winning percentage of opponents (excluding games vs this team)
    OOWP = Average of opponents' OWP values

This is "Sam's RPI" — our own calculation from game data.
NCAA's official RPI will be pulled separately for comparison.

Usage:
    python3 scripts/compute_rpi.py                # Compute and save
    python3 scripts/compute_rpi.py --show 25      # Show top 25
    python3 scripts/compute_rpi.py --team auburn   # Single team detail
    python3 scripts/compute_rpi.py --compare       # Compare Sam's RPI vs NCAA RPI (when available)
"""

import argparse
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Home/away weighting per NCAA RPI reform (2013)
HOME_WIN_WEIGHT = 0.7
ROAD_WIN_WEIGHT = 1.3
NEUTRAL_WIN_WEIGHT = 1.0
HOME_LOSS_WEIGHT = 1.3
ROAD_LOSS_WEIGHT = 0.7
NEUTRAL_LOSS_WEIGHT = 1.0


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_all_results(db):
    """Load all final games with home/away info."""
    rows = db.execute("""
        SELECT id, home_team_id, away_team_id, home_score, away_score, winner_id
        FROM games
        WHERE status = 'final'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND home_team_id != away_team_id
    """).fetchall()
    return [dict(r) for r in rows]


def compute_weighted_wp(team_id, games):
    """
    Compute weighted winning percentage for a team.
    Road wins worth more, home wins worth less.
    """
    weighted_wins = 0.0
    weighted_losses = 0.0

    for g in games:
        if g['home_team_id'] == team_id:
            is_home = True
            won = g['home_score'] > g['away_score']
        elif g['away_team_id'] == team_id:
            is_home = False
            won = g['away_score'] > g['home_score']
        else:
            continue

        # TODO: detect neutral site games (for now all games treated as home/away)
        if won:
            weight = HOME_WIN_WEIGHT if is_home else ROAD_WIN_WEIGHT
            weighted_wins += weight
        else:
            weight = HOME_LOSS_WEIGHT if is_home else ROAD_LOSS_WEIGHT
            weighted_losses += weight

    total = weighted_wins + weighted_losses
    if total == 0:
        return None, 0
    return weighted_wins / total, int(weighted_wins + weighted_losses)


def compute_owp(team_id, games, all_teams):
    """
    Compute Opponents' Winning Percentage.
    For each opponent, compute their win% EXCLUDING games against this team.
    Then average across all opponents faced.
    """
    # Find all opponents
    opponents = []
    for g in games:
        if g['home_team_id'] == team_id:
            opponents.append(g['away_team_id'])
        elif g['away_team_id'] == team_id:
            opponents.append(g['home_team_id'])

    if not opponents:
        return None

    # For each opponent, compute their record excluding games vs team_id
    opp_wps = []
    for opp_id in opponents:
        wins = 0
        losses = 0
        for g in games:
            # Skip games between opponent and team_id
            if (g['home_team_id'] == team_id and g['away_team_id'] == opp_id) or \
               (g['away_team_id'] == team_id and g['home_team_id'] == opp_id):
                continue

            if g['home_team_id'] == opp_id:
                if g['home_score'] > g['away_score']:
                    wins += 1
                else:
                    losses += 1
            elif g['away_team_id'] == opp_id:
                if g['away_score'] > g['home_score']:
                    wins += 1
                else:
                    losses += 1

        total = wins + losses
        if total > 0:
            opp_wps.append(wins / total)

    if not opp_wps:
        return None
    return sum(opp_wps) / len(opp_wps)


def compute_oowp(team_id, games, owp_cache):
    """
    Compute Opponents' Opponents' Winning Percentage.
    Average of each opponent's OWP value.
    """
    opponents = set()
    for g in games:
        if g['home_team_id'] == team_id:
            opponents.add(g['away_team_id'])
        elif g['away_team_id'] == team_id:
            opponents.add(g['home_team_id'])

    if not opponents:
        return None

    oowp_values = []
    for opp_id in opponents:
        if opp_id in owp_cache and owp_cache[opp_id] is not None:
            oowp_values.append(owp_cache[opp_id])

    if not oowp_values:
        return None
    return sum(oowp_values) / len(oowp_values)


def compute_all_rpi(db):
    """Compute Sam's RPI for all teams."""
    games = get_all_results(db)

    # Get all teams that have played
    all_teams = set()
    for g in games:
        all_teams.add(g['home_team_id'])
        all_teams.add(g['away_team_id'])

    print(f"Computing RPI for {len(all_teams)} teams from {len(games)} games...")

    # Step 1: Compute WP for all teams
    wp_cache = {}
    for team_id in all_teams:
        wp, count = compute_weighted_wp(team_id, games)
        wp_cache[team_id] = (wp, count)

    # Step 2: Compute OWP for all teams
    owp_cache = {}
    for team_id in all_teams:
        owp_cache[team_id] = compute_owp(team_id, games, all_teams)

    # Step 3: Compute OOWP for all teams
    oowp_cache = {}
    for team_id in all_teams:
        oowp_cache[team_id] = compute_oowp(team_id, games, owp_cache)

    # Step 4: Compute final RPI
    results = []
    for team_id in all_teams:
        wp, game_count = wp_cache[team_id]
        owp = owp_cache[team_id]
        oowp = oowp_cache[team_id]

        if wp is None or owp is None or oowp is None:
            continue

        rpi = 0.25 * wp + 0.50 * owp + 0.25 * oowp

        results.append({
            'team_id': team_id,
            'rpi': round(rpi, 6),
            'wp': round(wp, 6),
            'owp': round(owp, 6),
            'oowp': round(oowp, 6),
            'games_played': game_count,
        })

    # Rank by RPI
    results.sort(key=lambda x: x['rpi'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    return results


def save_rpi(db, results):
    """Save RPI to database."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS team_rpi (
            team_id TEXT PRIMARY KEY,
            sams_rpi REAL,
            sams_rank INTEGER,
            wp REAL,
            owp REAL,
            oowp REAL,
            games_played INTEGER,
            ncaa_rpi REAL,
            ncaa_rank INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for r in results:
        db.execute("""
            INSERT OR REPLACE INTO team_rpi
            (team_id, sams_rpi, sams_rank, wp, owp, oowp, games_played, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (r['team_id'], r['rpi'], r['rank'], r['wp'], r['owp'], r['oowp'],
              r['games_played']))

    db.commit()
    print(f"Saved RPI for {len(results)} teams")


def show_rpi(db, n=25):
    """Show top N teams by Sam's RPI."""
    rows = db.execute("""
        SELECT r.sams_rank, r.team_id, t.name, t.conference,
               r.sams_rpi, r.wp, r.owp, r.oowp, r.games_played,
               r.ncaa_rpi, r.ncaa_rank,
               e.rating as elo
        FROM team_rpi r
        JOIN teams t ON r.team_id = t.id
        LEFT JOIN elo_ratings e ON r.team_id = e.team_id
        ORDER BY r.sams_rank
        LIMIT ?
    """, (n,)).fetchall()

    print(f"\n{'Rank':<5} {'Team':<25} {'Conf':<8} {'RPI':>7} {'WP':>7} {'OWP':>7} {'OOWP':>7} {'GP':>4} {'Elo':>6}", end='')

    # Show NCAA RPI if available
    has_ncaa = any(r['ncaa_rpi'] is not None for r in rows)
    if has_ncaa:
        print(f"  {'NCAA':>7} {'Diff':>6}", end='')
    print()
    print("-" * (95 if has_ncaa else 82))

    for r in rows:
        line = f"{r['sams_rank']:<5} {(r['name'] or r['team_id']):<25} {(r['conference'] or '?'):<8} "
        line += f"{r['sams_rpi']:>7.4f} {r['wp']:>7.4f} {r['owp']:>7.4f} {r['oowp']:>7.4f} "
        line += f"{r['games_played']:>4} {r['elo'] or 0:>6.0f}"
        if has_ncaa:
            ncaa = r['ncaa_rpi']
            if ncaa:
                diff = r['sams_rpi'] - ncaa
                line += f"  {ncaa:>7.4f} {diff:>+6.4f}"
            else:
                line += f"  {'N/A':>7} {'':>6}"
        print(line)


def show_team(db, team_id):
    """Show detailed RPI breakdown for a single team."""
    r = db.execute("""
        SELECT r.*, t.name, t.conference, e.rating as elo
        FROM team_rpi r
        JOIN teams t ON r.team_id = t.id
        LEFT JOIN elo_ratings e ON r.team_id = e.team_id
        WHERE r.team_id = ?
    """, (team_id,)).fetchone()

    if not r:
        print(f"No RPI data for {team_id}")
        return

    print(f"\n=== {r['name']} ({r['conference']}) ===")
    print(f"Sam's RPI:  {r['sams_rpi']:.4f} (Rank #{r['sams_rank']})")
    if r['ncaa_rpi']:
        print(f"NCAA RPI:   {r['ncaa_rpi']:.4f} (Rank #{r['ncaa_rank']})")
        print(f"Difference: {r['sams_rpi'] - r['ncaa_rpi']:+.4f}")
    print(f"Elo:        {r['elo']:.0f}" if r['elo'] else "Elo:        N/A")
    print(f"\nComponents:")
    print(f"  WP (weighted):  {r['wp']:.4f}  (25% of RPI)")
    print(f"  OWP:            {r['owp']:.4f}  (50% of RPI)")
    print(f"  OOWP:           {r['oowp']:.4f}  (25% of RPI)")
    print(f"  Games played:   {r['games_played']}")

    # Show opponents
    games = db.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id, g.home_score, g.away_score,
               CASE WHEN g.home_team_id = ? THEN at.name ELSE ht.name END as opponent,
               CASE WHEN g.home_team_id = ? THEN 'vs' ELSE '@' END as loc,
               opp_rpi.sams_rpi as opp_rpi, opp_rpi.sams_rank as opp_rank
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN team_rpi opp_rpi ON opp_rpi.team_id = CASE WHEN g.home_team_id = ? THEN g.away_team_id ELSE g.home_team_id END
        WHERE g.status = 'final' AND (g.home_team_id = ? OR g.away_team_id = ?)
        ORDER BY g.date
    """, (team_id, team_id, team_id, team_id, team_id)).fetchall()

    if games:
        print(f"\nResults:")
        for g in games:
            is_home = g['home_team_id'] == team_id
            our_score = g['home_score'] if is_home else g['away_score']
            their_score = g['away_score'] if is_home else g['home_score']
            result = 'W' if our_score > their_score else 'L'
            opp_info = f"(RPI #{g['opp_rank']})" if g['opp_rank'] else ""
            print(f"  {g['date']} {g['loc']} {g['opponent']:<20} {result} {our_score}-{their_score}  {opp_info}")


def main():
    parser = argparse.ArgumentParser(description="Sam's RPI Calculator")
    parser.add_argument('--show', type=int, nargs='?', const=25, help='Show top N teams (default 25)')
    parser.add_argument('--team', type=str, help='Show detail for a team')
    parser.add_argument('--compare', action='store_true', help='Compare Sam\'s RPI vs NCAA RPI')
    parser.add_argument('--dry-run', action='store_true', help='Compute but don\'t save')
    args = parser.parse_args()

    db = get_db()

    if args.team:
        show_team(db, args.team)
        db.close()
        return

    if args.show:
        show_rpi(db, args.show)
        db.close()
        return

    if args.compare:
        show_rpi(db, 50)
        db.close()
        return

    # Compute and save
    results = compute_all_rpi(db)

    if not args.dry_run:
        save_rpi(db, results)

    # Show top 25
    if results:
        print(f"\nTop 25 by Sam's RPI:")
        print(f"{'Rank':<5} {'Team':<25} {'RPI':>7} {'WP':>7} {'OWP':>7} {'OOWP':>7} {'GP':>4}")
        print("-" * 65)
        for r in results[:25]:
            db_team = db.execute("SELECT name FROM teams WHERE id = ?", (r['team_id'],)).fetchone()
            name = db_team['name'] if db_team else r['team_id']
            print(f"{r['rank']:<5} {name:<25} {r['rpi']:>7.4f} {r['wp']:>7.4f} {r['owp']:>7.4f} {r['oowp']:>7.4f} {r['games_played']:>4}")

    db.close()


if __name__ == '__main__':
    main()
