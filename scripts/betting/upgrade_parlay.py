#!/usr/bin/env python3
"""
Upgrade today's parlay with totals bets when new odds arrive.

Run after totals/FD odds are scraped (e.g., 9:30 AM pre-game scheduler).
Compares totals candidates to the weakest ML leg and swaps if the totals
bet has a higher calibrated edge.

Rules:
  - Only upgrades if no game in the parlay has started yet
  - Only swaps the weakest leg if the totals edge beats it
  - Never downgrades — only improves
  - No same-game parlays (totals game must differ from other legs)
  - Logs every change to upgrade_log column

Usage:
    python3 -m scripts.betting.upgrade_parlay [--date YYYY-MM-DD] [--dry-run]
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytz

BASE = Path(__file__).parent.parent.parent
DB_PATH = str(BASE / 'data' / 'baseball.db')

# Minimum edge for a totals bet to be considered for parlay upgrade
MIN_TOTALS_EDGE = 8.0
# Minimum edge improvement over the weakest leg to justify a swap
MIN_IMPROVEMENT = 2.0


def get_today(date_override=None):
    if date_override:
        return date_override
    ct = pytz.timezone('America/Chicago')
    return datetime.now(ct).strftime('%Y-%m-%d')


def is_any_leg_live(conn, legs):
    """Check if any parlay leg's game has started."""
    for leg in legs:
        gid = leg.get('game_id', '')
        if not gid:
            continue
        row = conn.execute(
            "SELECT status FROM games WHERE id = ?", (gid,)
        ).fetchone()
        if row and row[0] in ('in-progress', 'final'):
            return True
    return False


def get_totals_candidates(conn, date_str, exclude_game_ids):
    """Find totals bets with strong edges from today's betting lines."""
    rows = conn.execute("""
        SELECT bl.game_id, bl.over_under,
               g.home_team_id, g.away_team_id, g.status
        FROM betting_lines bl
        JOIN games g ON g.id = bl.game_id
        WHERE bl.date = ?
          AND bl.over_under IS NOT NULL
          AND g.status = 'scheduled'
    """, (date_str,)).fetchall()

    candidates = []
    for r in rows:
        gid = r[0]
        ou_line = r[1]
        if gid in exclude_game_ids:
            continue

        # Get model projection for this game's total
        proj_row = conn.execute("""
            SELECT predicted_total
            FROM totals_predictions
            WHERE game_id = ? ORDER BY predicted_at DESC LIMIT 1
        """, (gid,)).fetchone()

        if not proj_row or proj_row[0] is None:
            continue

        model_total = proj_row[0]
        diff = model_total - ou_line
        if abs(diff) < 1.0:
            continue

        lean = 'OVER' if diff > 0 else 'UNDER'
        edge = abs(diff) / ou_line * 100  # edge as % of line
        est_prob = min(0.5 + abs(diff) * 0.06, 0.85)

        if edge < MIN_TOTALS_EDGE:
            continue

        away_name = r[3].replace('-', ' ').title() if r[3] else ''
        home_name = r[2].replace('-', ' ').title() if r[2] else ''

        candidates.append({
            'type': 'Total',
            'game_id': gid,
            'date': date_str,
            'pick': '{} {}'.format(lean, ou_line),
            'matchup': '{} @ {}'.format(away_name, home_name),
            'odds': -110,
            'prob': est_prob,
            'edge': edge,
            'total_diff': diff,
        })

    candidates.sort(key=lambda x: x['edge'], reverse=True)
    return candidates


def upgrade_parlay(date_str=None, dry_run=False):
    """Attempt to upgrade today's parlay with better totals bets.

    Returns dict with 'upgraded' bool and 'log' string.
    """
    date_str = get_today(date_str)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    # Load stored parlay
    row = conn.execute(
        "SELECT id, legs_json, american_odds, decimal_odds, model_prob "
        "FROM tracked_parlays WHERE date = ?", (date_str,)
    ).fetchone()

    if not row:
        conn.close()
        return {'upgraded': False, 'log': 'No parlay found for {}'.format(date_str)}

    parlay_id = row['id']
    legs = json.loads(row['legs_json'])

    # Check if any game has started — if so, parlay is locked
    if is_any_leg_live(conn, legs):
        conn.close()
        return {'upgraded': False, 'log': 'Parlay locked — at least one game has started'}

    # Find the weakest leg by edge
    weakest_idx = None
    weakest_edge = float('inf')
    for i, leg in enumerate(legs):
        leg_edge = leg.get('edge', 0)
        if leg_edge < weakest_edge:
            weakest_edge = leg_edge
            weakest_idx = i

    if weakest_idx is None:
        conn.close()
        return {'upgraded': False, 'log': 'Could not identify weakest leg'}

    weakest = legs[weakest_idx]

    # Get totals candidates (exclude games already in parlay)
    used_game_ids = set(leg.get('game_id', '') for leg in legs)
    candidates = get_totals_candidates(conn, date_str, used_game_ids)

    if not candidates:
        conn.close()
        return {'upgraded': False, 'log': 'No totals candidates with {}%+ edge'.format(MIN_TOTALS_EDGE)}

    best_total = candidates[0]

    # Only swap if improvement exceeds threshold
    improvement = best_total['edge'] - weakest_edge
    if improvement < MIN_IMPROVEMENT:
        conn.close()
        return {
            'upgraded': False,
            'log': 'Best total ({:.1f}% edge) doesn\'t beat weakest leg ({:.1f}% edge) by {}%+'.format(
                best_total['edge'], weakest_edge, MIN_IMPROVEMENT
            ),
        }

    # Perform the swap
    old_leg = legs[weakest_idx].copy()
    legs[weakest_idx] = best_total

    # Recalculate parlay odds
    def ml_to_decimal(ml):
        return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)

    decimal_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        decimal_odds *= ml_to_decimal(leg['odds'])
        combined_prob *= leg['prob']

    american = round((decimal_odds - 1) * 100) if decimal_odds > 2 else round(-100 / (decimal_odds - 1))
    payout = round(25 * decimal_odds, 2)

    log_entry = (
        'Upgraded leg {} at {}: {} ({} {:.1f}% edge) -> {} ({} {:.1f}% edge) '
        '[+{:.1f}% improvement]'
    ).format(
        weakest_idx + 1,
        datetime.now(pytz.timezone('America/Chicago')).strftime('%H:%M'),
        old_leg.get('pick', '?'), old_leg.get('type', '?'), old_leg.get('edge', 0),
        best_total['pick'], best_total['type'], best_total['edge'],
        improvement,
    )

    if dry_run:
        conn.close()
        return {'upgraded': True, 'log': '[DRY RUN] ' + log_entry, 'new_legs': legs}

    # Update DB
    conn.execute("""
        UPDATE tracked_parlays
        SET legs_json = ?, american_odds = ?, decimal_odds = ?,
            model_prob = ?, payout = ?,
            updated_at = datetime('now'),
            upgrade_log = COALESCE(upgrade_log || char(10), '') || ?
        WHERE id = ?
    """, (
        json.dumps(legs), american, round(decimal_odds, 4),
        round(combined_prob, 4), payout,
        log_entry, parlay_id,
    ))
    conn.commit()
    conn.close()

    return {'upgraded': True, 'log': log_entry}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Upgrade parlay with totals')
    parser.add_argument('--date', help='Date override (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change')
    args = parser.parse_args()

    result = upgrade_parlay(date_str=args.date, dry_run=args.dry_run)
    print(result['log'])
    if result.get('new_legs'):
        for i, leg in enumerate(result['new_legs']):
            print('  Leg {}: {} {} ({:.1f}% edge)'.format(
                i + 1, leg.get('type', '?'), leg.get('pick', '?'), leg.get('edge', 0)))
