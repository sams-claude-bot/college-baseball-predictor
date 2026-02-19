#!/usr/bin/env python3
"""
Compute Staff-Level Pitching Quality Metrics

Instead of trying to predict who starts, we compute team pitching staff
quality metrics that capture:
- Ace quality (best starter's stats)
- Rotation depth (top 3 starters combined)
- Bullpen quality (non-starters combined)
- Staff concentration (how reliant on one arm)
- Overall staff efficiency

These get stored in team_pitching_quality table and used as features.

Usage:
    python3 scripts/compute_pitching_quality.py              # All teams
    python3 scripts/compute_pitching_quality.py --team lsu   # Single team
    python3 scripts/compute_pitching_quality.py --show-top 25
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / "data" / "baseball.db"
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from run_utils import ScriptRunner


def get_db():
    db = sqlite3.connect(str(DB_PATH), timeout=30)
    db.row_factory = sqlite3.Row
    return db


def ensure_table(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS team_pitching_quality (
            team_id TEXT PRIMARY KEY,
            
            -- Ace quality (top starter by IP)
            ace_era REAL,
            ace_whip REAL,
            ace_k_per_9 REAL,
            ace_bb_per_9 REAL,
            ace_fip REAL,
            ace_innings REAL,
            
            -- Rotation quality (top 3 starters by IP, IP-weighted)
            rotation_era REAL,
            rotation_whip REAL,
            rotation_k_per_9 REAL,
            rotation_bb_per_9 REAL,
            rotation_fip REAL,
            rotation_innings REAL,
            
            -- Bullpen quality (everyone else)
            bullpen_era REAL,
            bullpen_whip REAL,
            bullpen_k_per_9 REAL,
            bullpen_bb_per_9 REAL,
            bullpen_fip REAL,
            bullpen_innings REAL,
            
            -- Staff depth & concentration
            staff_size INTEGER,           -- pitchers with > 0 IP
            starter_count INTEGER,        -- pitchers with games_started > 0
            ace_ip_pct REAL,              -- ace IP / total staff IP (concentration)
            top3_ip_pct REAL,             -- top 3 IP / total staff IP
            innings_hhi REAL,             -- Herfindahl index of IP distribution (0=spread, 1=one guy)
            
            -- Overall staff
            staff_era REAL,
            staff_whip REAL,
            staff_k_per_9 REAL,
            staff_bb_per_9 REAL,
            staff_fip REAL,
            staff_total_ip REAL,
            
            -- Quality counts
            quality_arms INTEGER,         -- pitchers with ERA < 3.50 and IP >= 3
            shutdown_arms INTEGER,        -- pitchers with ERA < 2.00 and IP >= 3
            liability_arms INTEGER,       -- pitchers with ERA > 7.00 and IP >= 3
            
            last_updated TEXT
        )
    """)
    db.commit()


def compute_team_pitching(db, team_id):
    """Compute pitching quality metrics for one team."""
    
    # Get all pitchers with innings
    pitchers = db.execute("""
        SELECT innings_pitched, era, whip, k_per_9, bb_per_9, fip, xfip,
               games_started, games_pitched, earned_runs, strikeouts_pitched,
               walks_allowed, hits_allowed, saves,
               name
        FROM player_stats 
        WHERE team_id = ? AND innings_pitched > 0
        ORDER BY innings_pitched DESC
    """, (team_id,)).fetchall()
    
    if not pitchers:
        return None
    
    total_ip = sum(p['innings_pitched'] for p in pitchers)
    if total_ip == 0:
        return None
    
    # Identify starters vs bullpen
    # Use games_started > 0 OR top 3 by IP as proxy
    starters_by_gs = [p for p in pitchers if (p['games_started'] or 0) > 0]
    
    # Top 3 by IP (our "rotation")
    top3 = pitchers[:3]
    rest = pitchers[3:]
    
    # Ace = #1 by IP
    ace = pitchers[0]
    
    def ip_weighted_stats(group):
        """Compute IP-weighted average stats for a group of pitchers."""
        group_ip = sum(p['innings_pitched'] for p in group)
        if group_ip == 0:
            return {'era': 4.50, 'whip': 1.35, 'k_per_9': 7.5, 'bb_per_9': 3.5, 'fip': 4.50, 'ip': 0}
        
        w_era = sum((p['era'] or 4.50) * p['innings_pitched'] for p in group) / group_ip
        w_whip = sum((p['whip'] or 1.35) * p['innings_pitched'] for p in group) / group_ip
        w_k9 = sum((p['k_per_9'] or 7.5) * p['innings_pitched'] for p in group) / group_ip
        w_bb9 = sum((p['bb_per_9'] or 3.5) * p['innings_pitched'] for p in group) / group_ip
        w_fip = sum((p['fip'] or (p['era'] or 4.50)) * p['innings_pitched'] for p in group) / group_ip
        
        return {'era': w_era, 'whip': w_whip, 'k_per_9': w_k9, 'bb_per_9': w_bb9, 'fip': w_fip, 'ip': group_ip}
    
    rotation = ip_weighted_stats(top3)
    bullpen = ip_weighted_stats(rest) if rest else ip_weighted_stats(top3)  # fallback
    staff = ip_weighted_stats(pitchers)
    
    # Innings concentration (Herfindahl-Hirschman Index)
    ip_shares = [(p['innings_pitched'] / total_ip) for p in pitchers]
    hhi = sum(s ** 2 for s in ip_shares)
    
    # Quality counts (need minimum IP to count)
    min_ip = 3.0
    quality_arms = sum(1 for p in pitchers if (p['era'] or 99) < 3.50 and p['innings_pitched'] >= min_ip)
    shutdown_arms = sum(1 for p in pitchers if (p['era'] or 99) < 2.00 and p['innings_pitched'] >= min_ip)
    liability_arms = sum(1 for p in pitchers if (p['era'] or 0) > 7.00 and p['innings_pitched'] >= min_ip)
    
    return {
        'team_id': team_id,
        'ace_era': ace['era'] or 4.50,
        'ace_whip': ace['whip'] or 1.35,
        'ace_k_per_9': ace['k_per_9'] or 7.5,
        'ace_bb_per_9': ace['bb_per_9'] or 3.5,
        'ace_fip': ace['fip'] or (ace['era'] or 4.50),
        'ace_innings': ace['innings_pitched'],
        'rotation_era': rotation['era'],
        'rotation_whip': rotation['whip'],
        'rotation_k_per_9': rotation['k_per_9'],
        'rotation_bb_per_9': rotation['bb_per_9'],
        'rotation_fip': rotation['fip'],
        'rotation_innings': rotation['ip'],
        'bullpen_era': bullpen['era'],
        'bullpen_whip': bullpen['whip'],
        'bullpen_k_per_9': bullpen['k_per_9'],
        'bullpen_bb_per_9': bullpen['bb_per_9'],
        'bullpen_fip': bullpen['fip'],
        'bullpen_innings': bullpen['ip'],
        'staff_size': len(pitchers),
        'starter_count': len(starters_by_gs),
        'ace_ip_pct': ace['innings_pitched'] / total_ip,
        'top3_ip_pct': sum(p['innings_pitched'] for p in top3) / total_ip,
        'innings_hhi': hhi,
        'staff_era': staff['era'],
        'staff_whip': staff['whip'],
        'staff_k_per_9': staff['k_per_9'],
        'staff_bb_per_9': staff['bb_per_9'],
        'staff_fip': staff['fip'],
        'staff_total_ip': total_ip,
        'quality_arms': quality_arms,
        'shutdown_arms': shutdown_arms,
        'liability_arms': liability_arms,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def compute_all(db, team_id=None, runner=None):
    """Compute pitching quality for all teams (or one)."""
    ensure_table(db)
    
    if team_id:
        teams = [(team_id,)]
    else:
        teams = db.execute("SELECT DISTINCT team_id FROM player_stats WHERE innings_pitched > 0").fetchall()
    
    updated = 0
    for (tid,) in teams:
        result = compute_team_pitching(db, tid)
        if result:
            cols = list(result.keys())
            placeholders = ', '.join(['?'] * len(cols))
            col_str = ', '.join(cols)
            db.execute(
                f"INSERT OR REPLACE INTO team_pitching_quality ({col_str}) VALUES ({placeholders})",
                [result[c] for c in cols]
            )
            updated += 1
    
    db.commit()
    if runner:
        runner.info(f"Updated pitching quality for {updated} teams")
    return updated


def show_top(db, n=25):
    """Show top teams by rotation ERA."""
    rows = db.execute("""
        SELECT team_id, rotation_era, rotation_whip, rotation_k_per_9, 
               bullpen_era, ace_era, ace_innings, staff_size, quality_arms,
               innings_hhi, staff_total_ip
        FROM team_pitching_quality 
        WHERE staff_total_ip >= 15
        ORDER BY rotation_era ASC LIMIT ?
    """, (n,)).fetchall()
    
    print(f"\n{'Team':<25} {'Rot ERA':>8} {'Rot WHIP':>9} {'Rot K/9':>8} {'BP ERA':>7} {'Ace ERA':>8} {'Ace IP':>7} {'Staff':>6} {'QArms':>6} {'HHI':>5}")
    print("-" * 105)
    for r in rows:
        print(f"{r['team_id']:<25} {r['rotation_era']:>8.2f} {r['rotation_whip']:>9.2f} {r['rotation_k_per_9']:>8.1f} {r['bullpen_era']:>7.2f} {r['ace_era']:>8.2f} {r['ace_innings']:>7.1f} {r['staff_size']:>6} {r['quality_arms']:>6} {r['innings_hhi']:>5.3f}")


def main():
    parser = argparse.ArgumentParser(description='Compute team pitching quality metrics')
    parser.add_argument('--team', help='Single team ID')
    parser.add_argument('--show-top', type=int, help='Show top N teams')
    args = parser.parse_args()
    
    runner = ScriptRunner("compute_pitching_quality")
    
    db = get_db()
    updated = compute_all(db, team_id=args.team, runner=runner)
    
    runner.add_stat("teams_updated", updated)
    
    if args.show_top:
        show_top(db, args.show_top)
    
    runner.finish()


if __name__ == '__main__':
    main()
