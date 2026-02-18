#!/usr/bin/env python3
"""
Compute Strength of Schedule (SOS) for all teams.

SOS = average Elo rating of opponents faced, weighted by recency.
Also computes future SOS from remaining scheduled games.

Usage:
    python3 scripts/compute_sos.py              # Compute and save
    python3 scripts/compute_sos.py --show 25    # Show top 25 hardest schedules
"""

import argparse
import sqlite3
import math
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def compute_all_sos(db):
    """Compute SOS for all teams with completed games."""
    # Load Elo ratings
    elo = {}
    for row in db.execute("SELECT team_id, rating FROM elo_ratings"):
        elo[row['team_id']] = row['rating']
    
    # Get all teams with games
    teams = db.execute("""
        SELECT DISTINCT team_id FROM (
            SELECT home_team_id as team_id FROM games WHERE status = 'final'
            UNION
            SELECT away_team_id FROM games WHERE status = 'final'
        )
    """).fetchall()
    
    results = []
    
    for (team_id,) in teams:
        # Past SOS: opponents faced in completed games
        opponents = db.execute("""
            SELECT CASE WHEN home_team_id = ? THEN away_team_id ELSE home_team_id END as opp_id,
                   date
            FROM games
            WHERE status = 'final' AND (home_team_id = ? OR away_team_id = ?)
            ORDER BY date ASC
        """, (team_id, team_id, team_id)).fetchall()
        
        if not opponents:
            continue
        
        # Recency-weighted average of opponent Elo
        total_weight = 0
        weighted_elo = 0
        opp_elos = []
        for i, opp in enumerate(opponents):
            opp_elo = elo.get(opp['opp_id'], 1450)
            weight = 1.0 + i * 0.1  # More recent opponents weighted slightly more
            weighted_elo += opp_elo * weight
            total_weight += weight
            opp_elos.append(opp_elo)
        
        past_sos = weighted_elo / total_weight if total_weight > 0 else 1450
        
        # Future SOS: scheduled opponents
        future_opps = db.execute("""
            SELECT CASE WHEN home_team_id = ? THEN away_team_id ELSE home_team_id END as opp_id
            FROM games
            WHERE status = 'scheduled' AND (home_team_id = ? OR away_team_id = ?)
        """, (team_id, team_id, team_id)).fetchall()
        
        future_elos = [elo.get(o['opp_id'], 1450) for o in future_opps]
        future_sos = sum(future_elos) / len(future_elos) if future_elos else None
        
        # Overall SOS (past + future)
        all_elos = opp_elos + future_elos
        overall_sos = sum(all_elos) / len(all_elos) if all_elos else 1450
        
        # Hardest opponent faced
        max_opp_elo = max(opp_elos) if opp_elos else 1450
        
        results.append({
            'team_id': team_id,
            'past_sos': round(past_sos, 1),
            'future_sos': round(future_sos, 1) if future_sos else None,
            'overall_sos': round(overall_sos, 1),
            'games_played': len(opponents),
            'games_remaining': len(future_opps),
            'hardest_opp_elo': round(max_opp_elo, 1),
        })
    
    return results


def save_sos(db, results):
    """Save SOS to database."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS team_sos (
            team_id TEXT PRIMARY KEY,
            past_sos REAL,
            future_sos REAL,
            overall_sos REAL,
            games_played INTEGER,
            games_remaining INTEGER,
            hardest_opp_elo REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    for r in results:
        db.execute("""
            INSERT OR REPLACE INTO team_sos 
            (team_id, past_sos, future_sos, overall_sos, games_played, games_remaining, hardest_opp_elo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (r['team_id'], r['past_sos'], r['future_sos'], r['overall_sos'],
              r['games_played'], r['games_remaining'], r['hardest_opp_elo']))
    
    db.commit()


def show_sos(db, n=25):
    """Show teams ranked by past SOS (hardest schedules)."""
    rows = db.execute("""
        SELECT s.team_id, t.name, t.conference, s.past_sos, s.future_sos, 
               s.overall_sos, s.games_played, s.hardest_opp_elo,
               e.rating as elo,
               a.win_pct
        FROM team_sos s
        JOIN teams t ON s.team_id = t.id
        LEFT JOIN elo_ratings e ON s.team_id = e.team_id
        LEFT JOIN team_aggregate_stats a ON s.team_id = a.team_id
        WHERE s.games_played >= 3
        ORDER BY s.past_sos DESC
        LIMIT ?
    """, (n,)).fetchall()
    
    print(f"\n{'#':>3} {'Team':<22} {'Conf':<8} {'Elo':>5} {'W%':>5} {'SOS':>5} {'fSOS':>5} {'GP':>3} {'Best Opp':>8}")
    print("-" * 75)
    for i, r in enumerate(rows, 1):
        fsos = f"{r['future_sos']:.0f}" if r['future_sos'] else "N/A"
        wpct = f"{r['win_pct']:.3f}" if r['win_pct'] else "N/A"
        print(f"{i:>3} {r['name']:<22} {r['conference'] or ''::<8} {r['elo'] or 0:>5.0f} {wpct:>5} {r['past_sos']:>5.0f} {fsos:>5} {r['games_played']:>3} {r['hardest_opp_elo']:>8.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', type=int, nargs='?', const=25, help='Show top N hardest schedules')
    args = parser.parse_args()
    
    db = get_db()
    
    results = compute_all_sos(db)
    save_sos(db, results)
    print(f"✅ Computed SOS for {len(results)} teams")
    
    if args.show is not None:
        show_sos(db, args.show)
    
    db.close()


if __name__ == '__main__':
    main()
