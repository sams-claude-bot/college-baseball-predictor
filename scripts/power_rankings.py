#!/usr/bin/env python3
"""
Power Rankings - Round-Robin Simulation

Runs every team against every other team using the ensemble model,
then ranks by average win probability. Much richer than Elo alone.

Stores results in `power_rankings` table with week-over-week movement tracking.

Usage:
    python3 scripts/power_rankings.py [--top N] [--conference SEC] [--min-games 5]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

import warnings
import logging

# Suppress noisy model warnings during bulk simulation
logging.getLogger().setLevel(logging.ERROR)

from database import get_connection
from models.compare_models import MODELS


def get_eligible_teams(min_games=3):
    """Get teams with enough completed games for meaningful predictions."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT t.id, t.name, t.conference, COUNT(g.id) as games_played
        FROM teams t
        JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id)
            AND g.status = 'final'
        GROUP BY t.id
        HAVING games_played >= ?
        ORDER BY t.name
    ''', (min_games,))
    teams = [dict(r) for r in c.fetchall()]
    conn.close()
    return teams


def run_round_robin(teams, verbose=False):
    """
    Simulate every team vs every other team on a neutral site.
    Returns dict of team_id -> {total_win_prob, games_simulated, avg_win_prob, avg_projected_runs, ...}
    """
    ensemble = MODELS.get('ensemble')
    if not ensemble:
        print("ERROR: Ensemble model not found")
        sys.exit(1)

    team_ids = [t['id'] for t in teams]
    n = len(team_ids)
    
    # Initialize accumulators
    results = {}
    for t in teams:
        results[t['id']] = {
            'team_id': t['id'],
            'team_name': t['name'],
            'conference': t['conference'],
            'total_win_prob': 0.0,
            'total_projected_runs': 0.0,
            'total_projected_runs_against': 0.0,
            'matchups': 0,
            'dominant_wins': 0,  # >70% win prob
            'close_matchups': 0,  # 45-55% either way
        }

    total_matchups = n * (n - 1) // 2
    done = 0
    errors = 0

    for i in range(n):
        for j in range(i + 1, n):
            home_id = team_ids[i]
            away_id = team_ids[j]
            
            try:
                pred = ensemble.predict_game(home_id, away_id, neutral_site=True)
                home_prob = pred.get('home_win_probability', 0.5)
                away_prob = pred.get('away_win_probability', 0.5)
                home_runs = pred.get('projected_home_runs', 0)
                away_runs = pred.get('projected_away_runs', 0)

                # Accumulate for home team
                results[home_id]['total_win_prob'] += home_prob
                results[home_id]['total_projected_runs'] += home_runs
                results[home_id]['total_projected_runs_against'] += away_runs
                results[home_id]['matchups'] += 1
                if home_prob > 0.70:
                    results[home_id]['dominant_wins'] += 1
                if 0.45 <= home_prob <= 0.55:
                    results[home_id]['close_matchups'] += 1

                # Accumulate for away team
                results[away_id]['total_win_prob'] += away_prob
                results[away_id]['total_projected_runs'] += away_runs
                results[away_id]['total_projected_runs_against'] += home_runs
                results[away_id]['matchups'] += 1
                if away_prob > 0.70:
                    results[away_id]['dominant_wins'] += 1
                if 0.45 <= home_prob <= 0.55:
                    results[away_id]['close_matchups'] += 1

            except Exception as e:
                errors += 1
                # Give both teams neutral 0.5 on error
                results[home_id]['total_win_prob'] += 0.5
                results[home_id]['matchups'] += 1
                results[away_id]['total_win_prob'] += 0.5
                results[away_id]['matchups'] += 1

            done += 1
            if verbose and done % 500 == 0:
                print(f"  {done}/{total_matchups} matchups ({done*100//total_matchups}%)...")

    if verbose and errors:
        print(f"  ⚠️  {errors} prediction errors (used 0.5 fallback)")

    # Calculate averages
    for tid, r in results.items():
        m = max(r['matchups'], 1)
        r['avg_win_prob'] = r['total_win_prob'] / m
        r['avg_projected_runs'] = r['total_projected_runs'] / m
        r['avg_projected_runs_against'] = r['total_projected_runs_against'] / m
        r['avg_run_diff'] = r['avg_projected_runs'] - r['avg_projected_runs_against']
        r['dominance_pct'] = r['dominant_wins'] / m  # % of matchups with >70% win prob

    return results


def get_previous_rankings():
    """Get the most recent power rankings for movement tracking."""
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT team_id, rank FROM power_rankings
            WHERE date = (SELECT MAX(date) FROM power_rankings)
            ORDER BY rank
        ''')
        prev = {row['team_id']: row['rank'] for row in c.fetchall()}
    except Exception:
        prev = {}
    conn.close()
    return prev


def store_rankings(ranked_teams, date_str=None):
    """Store rankings in the database."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    prev = get_previous_rankings()
    
    conn = get_connection()
    c = conn.cursor()
    
    # Create table if needed
    c.execute('''
        CREATE TABLE IF NOT EXISTS power_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            power_score REAL NOT NULL,
            avg_win_prob REAL,
            avg_run_diff REAL,
            avg_projected_runs REAL,
            dominance_pct REAL,
            prev_rank INTEGER,
            rank_change INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_power_rankings_date ON power_rankings(date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_power_rankings_team ON power_rankings(team_id)')
    
    # Don't duplicate — remove existing for this date
    c.execute('DELETE FROM power_rankings WHERE date = ?', (date_str,))
    
    for i, team in enumerate(ranked_teams):
        rank = i + 1
        tid = team['team_id']
        prev_rank = prev.get(tid)
        rank_change = (prev_rank - rank) if prev_rank else None
        
        c.execute('''
            INSERT INTO power_rankings 
            (team_id, rank, power_score, avg_win_prob, avg_run_diff, 
             avg_projected_runs, dominance_pct, prev_rank, rank_change, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tid, rank,
            team['avg_win_prob'],
            team['avg_win_prob'],
            team['avg_run_diff'],
            team['avg_projected_runs'],
            team['dominance_pct'],
            prev_rank, rank_change,
            date_str
        ))
    
    conn.commit()
    conn.close()
    print(f"✅ Stored {len(ranked_teams)} power rankings for {date_str}")


def print_rankings(ranked_teams, top_n=25, conference=None):
    """Pretty-print the rankings."""
    filtered = ranked_teams
    if conference:
        filtered = [t for t in ranked_teams if t.get('conference', '').upper() == conference.upper()]
    
    display = filtered[:top_n]
    
    header = f"\n{'Rank':<6}{'Team':<30}{'Conf':<8}{'Score':<8}{'Avg W%':<8}{'Run Diff':<10}{'Dom%':<7}"
    print(header)
    print("=" * len(header))
    
    prev = get_previous_rankings()
    
    for i, team in enumerate(display):
        # Find overall rank (not filtered rank)
        overall_rank = ranked_teams.index(team) + 1
        tid = team['team_id']
        prev_rank = prev.get(tid)
        
        if prev_rank:
            change = prev_rank - overall_rank
            if change > 0:
                arrow = f" ↑{change}"
            elif change < 0:
                arrow = f" ↓{abs(change)}"
            else:
                arrow = " –"
        else:
            arrow = " NEW"
        
        print(f"#{overall_rank:<5}{team['team_name']:<30}{team.get('conference',''):<8}"
              f"{team['avg_win_prob']:.3f}   {team['avg_win_prob']*100:.1f}%   "
              f"{team['avg_run_diff']:+.2f}     {team['dominance_pct']*100:.0f}%{arrow}")


def main():
    parser = argparse.ArgumentParser(description='Generate Model Power Rankings')
    parser.add_argument('--top', type=int, default=25, help='Show top N teams')
    parser.add_argument('--conference', type=str, help='Filter by conference')
    parser.add_argument('--min-games', type=int, default=3, help='Minimum games played')
    parser.add_argument('--store', action='store_true', help='Store results in database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show progress')
    parser.add_argument('--all', action='store_true', help='Show all teams (not just top N)')
    args = parser.parse_args()

    print(f"🏟️  Generating Model Power Rankings...")
    print(f"   Min games: {args.min_games}")
    
    teams = get_eligible_teams(args.min_games)
    print(f"   Eligible teams: {len(teams)}")
    n = len(teams)
    total = n * (n - 1) // 2
    print(f"   Total matchups to simulate: {total:,}")
    print()

    results = run_round_robin(teams, verbose=args.verbose)
    
    # Sort by average win probability (descending)
    ranked = sorted(results.values(), key=lambda x: x['avg_win_prob'], reverse=True)
    
    top_n = len(ranked) if args.all else args.top
    print_rankings(ranked, top_n=top_n, conference=args.conference)
    
    if args.store:
        store_rankings(ranked)
    
    return ranked


if __name__ == '__main__':
    main()
