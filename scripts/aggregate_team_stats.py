#!/usr/bin/env python3
"""
Aggregate Team Stats from Game Results

Computes team-level batting and pitching stats from completed games in the database.
Works for ALL D1 teams — no scraping needed.

Usage:
    python3 scripts/aggregate_team_stats.py              # All teams
    python3 scripts/aggregate_team_stats.py --team lsu   # Single team
    python3 scripts/aggregate_team_stats.py --show-top 25  # Show top 25 by win%
"""

import argparse
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('aggregate')


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_team_stats_table(db):
    """Create or update team_aggregate_stats table."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS team_aggregate_stats (
            team_id TEXT PRIMARY KEY,
            games INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            win_pct REAL DEFAULT 0,
            runs_scored INTEGER DEFAULT 0,
            runs_allowed INTEGER DEFAULT 0,
            runs_per_game REAL DEFAULT 0,
            runs_allowed_per_game REAL DEFAULT 0,
            run_differential INTEGER DEFAULT 0,
            run_diff_per_game REAL DEFAULT 0,
            pythagorean_pct REAL DEFAULT 0,
            home_wins INTEGER DEFAULT 0,
            home_losses INTEGER DEFAULT 0,
            away_wins INTEGER DEFAULT 0,
            away_losses INTEGER DEFAULT 0,
            last_10_wins INTEGER DEFAULT 0,
            last_10_losses INTEGER DEFAULT 0,
            streak TEXT DEFAULT '',
            avg_margin REAL DEFAULT 0,
            close_games_wins INTEGER DEFAULT 0,
            close_games_losses INTEGER DEFAULT 0,
            blowout_wins INTEGER DEFAULT 0,
            blowout_losses INTEGER DEFAULT 0,
            conference TEXT DEFAULT '',
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        )
    """)
    db.commit()


def compute_team_stats(db, team_id=None):
    """Compute aggregate stats from completed games."""
    where = "AND (g.home_team_id = ? OR g.away_team_id = ?)" if team_id else ""
    params = (team_id, team_id) if team_id else ()
    
    # Get all teams with completed games
    if team_id:
        teams = [(team_id,)]
    else:
        teams = db.execute("""
            SELECT DISTINCT team_id FROM (
                SELECT home_team_id as team_id FROM games WHERE status = 'final'
                UNION
                SELECT away_team_id as team_id FROM games WHERE status = 'final'
            ) ORDER BY team_id
        """).fetchall()
    
    updated = 0
    for (tid,) in teams:
        # Get all completed games for this team
        games = db.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score, date
            FROM games
            WHERE status = 'final'
            AND (home_team_id = ? OR away_team_id = ?)
            AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date ASC
        """, (tid, tid)).fetchall()
        
        if not games:
            continue
        
        stats = {
            'games': 0, 'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'home_wins': 0, 'home_losses': 0,
            'away_wins': 0, 'away_losses': 0,
            'close_wins': 0, 'close_losses': 0,
            'blowout_wins': 0, 'blowout_losses': 0,
            'margins': [],
            'recent_results': [],
        }
        
        streak_type = None
        streak_count = 0
        
        for g in games:
            is_home = g['home_team_id'] == tid
            rs = g['home_score'] if is_home else g['away_score']
            ra = g['away_score'] if is_home else g['home_score']
            won = rs > ra
            margin = rs - ra
            
            stats['games'] += 1
            stats['runs_scored'] += rs
            stats['runs_allowed'] += ra
            stats['margins'].append(margin)
            stats['recent_results'].append(won)
            
            if won:
                stats['wins'] += 1
                if is_home: stats['home_wins'] += 1
                else: stats['away_wins'] += 1
                if margin <= 2: stats['close_wins'] += 1
                if margin >= 7: stats['blowout_wins'] += 1
            else:
                stats['losses'] += 1
                if is_home: stats['home_losses'] += 1
                else: stats['away_losses'] += 1
                if abs(margin) <= 2: stats['close_losses'] += 1
                if abs(margin) >= 7: stats['blowout_losses'] += 1
            
            if streak_type == won:
                streak_count += 1
            else:
                streak_type = won
                streak_count = 1
        
        gp = stats['games']
        rs = stats['runs_scored']
        ra = stats['runs_allowed']
        
        win_pct = stats['wins'] / gp if gp else 0
        rpg = rs / gp if gp else 0
        rapg = ra / gp if gp else 0
        rd = rs - ra
        rdpg = rd / gp if gp else 0
        
        # Pythagorean
        rs2 = rs ** 2
        ra2 = ra ** 2
        pyth = rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5
        
        # Last 10
        last10 = stats['recent_results'][-10:]
        l10w = sum(1 for x in last10 if x)
        l10l = len(last10) - l10w
        
        # Streak
        streak = f"{'W' if streak_type else 'L'}{streak_count}"
        
        # Average margin
        avg_margin = sum(stats['margins']) / len(stats['margins']) if stats['margins'] else 0
        
        # Get conference
        conf_row = db.execute("SELECT conference FROM teams WHERE id = ?", (tid,)).fetchone()
        conf = conf_row['conference'] if conf_row and conf_row['conference'] else ''
        
        db.execute("""
            INSERT OR REPLACE INTO team_aggregate_stats (
                team_id, games, wins, losses, win_pct,
                runs_scored, runs_allowed, runs_per_game, runs_allowed_per_game,
                run_differential, run_diff_per_game, pythagorean_pct,
                home_wins, home_losses, away_wins, away_losses,
                last_10_wins, last_10_losses, streak, avg_margin,
                close_games_wins, close_games_losses,
                blowout_wins, blowout_losses,
                conference, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            tid, gp, stats['wins'], stats['losses'], round(win_pct, 3),
            rs, ra, round(rpg, 2), round(rapg, 2),
            rd, round(rdpg, 2), round(pyth, 3),
            stats['home_wins'], stats['home_losses'],
            stats['away_wins'], stats['away_losses'],
            l10w, l10l, streak, round(avg_margin, 2),
            stats['close_wins'], stats['close_losses'],
            stats['blowout_wins'], stats['blowout_losses'],
            conf
        ))
        updated += 1
    
    db.commit()
    return updated


def show_top(db, n=25):
    """Show top N teams by win%."""
    rows = db.execute("""
        SELECT ts.team_id, t.name, ts.conference, ts.games, ts.wins, ts.losses,
               ts.win_pct, ts.runs_per_game, ts.runs_allowed_per_game,
               ts.run_diff_per_game, ts.pythagorean_pct, ts.streak,
               ts.last_10_wins, ts.last_10_losses
        FROM team_aggregate_stats ts
        JOIN teams t ON ts.team_id = t.id
        WHERE ts.games >= 3
        ORDER BY ts.win_pct DESC, ts.run_diff_per_game DESC
        LIMIT ?
    """, (n,)).fetchall()
    
    print(f"\n{'#':>3} {'Team':<25} {'Conf':<8} {'W-L':>7} {'Win%':>6} {'RPG':>5} {'RAPG':>5} {'RD/G':>6} {'Pyth':>6} {'L10':>5} {'Strk':>5}")
    print("-" * 100)
    for i, r in enumerate(rows, 1):
        print(f"{i:>3} {r['name']:<25} {r['conference']:<8} {r['wins']}-{r['losses']:>2} {r['win_pct']:>6.3f} {r['runs_per_game']:>5.1f} {r['runs_allowed_per_game']:>5.1f} {r['run_diff_per_game']:>+6.1f} {r['pythagorean_pct']:>6.3f} {r['last_10_wins']}-{r['last_10_losses']} {r['streak']:>5}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate team stats from game results')
    parser.add_argument('--team', help='Single team ID')
    parser.add_argument('--show-top', type=int, help='Show top N teams')
    args = parser.parse_args()
    
    db = get_db()
    ensure_team_stats_table(db)
    
    if args.show_top:
        show_top(db, args.show_top)
        db.close()
        return
    
    log.info('Computing aggregate team stats...')
    updated = compute_team_stats(db, args.team)
    log.info(f'✅ Updated stats for {updated} teams')
    
    if not args.team:
        show_top(db, 25)
    
    db.close()


if __name__ == '__main__':
    main()
