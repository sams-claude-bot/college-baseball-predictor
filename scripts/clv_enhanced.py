#!/usr/bin/env python3
"""Enhanced CLV analysis: true closing lines, retroactive updates, correlation reports.

Usage:
    python3 scripts/clv_enhanced.py --update          # Retroactive CLV recalculation
    python3 scripts/clv_enhanced.py --report           # Opening vs closing spread report
    python3 scripts/clv_enhanced.py --correlation      # CLV vs win-rate correlation
    python3 scripts/clv_enhanced.py --all              # All of the above
"""
import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from capture_closing_lines import compute_clv, get_best_closing_line, _parse_game_time
from database import get_connection


# ────────────────────────────────────────────────
# Core CLV helpers
# ────────────────────────────────────────────────

def compute_enhanced_clv(game_id, db=None):
    """Compare bet opening ML vs best available closing line for a game.

    Returns list of dicts: [{table, row_id, opening_ml, closing_ml, clv_implied, clv_cents, source}]
    """
    own_conn = db is None
    if own_conn:
        db = get_connection()

    results = []
    try:
        # Get game info for datetime parsing
        game = db.execute(
            "SELECT date, time, home_team_id FROM games WHERE id = ?", (game_id,)
        ).fetchone()
        if not game:
            return results

        game_dt = _parse_game_time(game['date'], game['time']) if game['time'] else None
        best = get_best_closing_line(db, game_id, game_dt)
        if not best:
            return results

        for table in ('tracked_bets', 'tracked_confident_bets'):
            rows = db.execute(
                f"SELECT id, pick_team_id, moneyline, is_home FROM {table} WHERE game_id = ?",
                (game_id,)
            ).fetchall()
            for row in rows:
                row = dict(row)
                picked_home = row['pick_team_id'] == game['home_team_id'] or row['is_home'] == 1
                closing_ml = best['home_ml'] if picked_home else best['away_ml']
                if closing_ml is None:
                    continue
                clv_implied, clv_cents = compute_clv(row['moneyline'], closing_ml)
                results.append({
                    'table': table,
                    'row_id': row['id'],
                    'opening_ml': row['moneyline'],
                    'closing_ml': closing_ml,
                    'clv_implied': clv_implied,
                    'clv_cents': clv_cents,
                    'source': best['source'],
                })
    finally:
        if own_conn:
            db.close()
    return results


def retroactive_clv_update(db=None):
    """Recalculate CLV for all graded bets using best-available closing data.

    Only updates bets that have been graded (won IS NOT NULL).
    Returns count of bets updated.
    """
    own_conn = db is None
    if own_conn:
        db = get_connection()

    updated = 0
    try:
        for table in ('tracked_bets', 'tracked_confident_bets'):
            rows = db.execute(
                f"SELECT id, game_id, pick_team_id, moneyline, is_home FROM {table} WHERE won IS NOT NULL"
            ).fetchall()

            for row in rows:
                row = dict(row)
                game = db.execute(
                    "SELECT date, time, home_team_id FROM games WHERE id = ?",
                    (row['game_id'],)
                ).fetchone()
                if not game:
                    continue

                game_dt = _parse_game_time(game['date'], game['time']) if game['time'] else None
                best = get_best_closing_line(db, row['game_id'], game_dt)
                if not best:
                    continue

                picked_home = row['pick_team_id'] == game['home_team_id'] or row['is_home'] == 1
                closing_ml = best['home_ml'] if picked_home else best['away_ml']
                if closing_ml is None:
                    continue

                clv_implied, clv_cents = compute_clv(row['moneyline'], closing_ml)
                db.execute(
                    f"UPDATE {table} SET closing_ml = ?, clv_implied = ?, clv_cents = ? WHERE id = ?",
                    (closing_ml, clv_implied, clv_cents, row['id'])
                )
                updated += 1

        db.commit()
    finally:
        if own_conn:
            db.close()
    return updated


# ────────────────────────────────────────────────
# Analytical reports
# ────────────────────────────────────────────────

def clv_vs_result_correlation(start_date=None, end_date=None, db=None):
    """Analyze whether +CLV bets win more often than -CLV bets.

    Returns dict with positive_clv and negative_clv groups, each containing
    count, wins, win_rate, avg_clv, avg_profit.
    """
    own_conn = db is None
    if own_conn:
        db = get_connection()

    try:
        where = "WHERE clv_implied IS NOT NULL AND won IS NOT NULL"
        params = []
        if start_date:
            where += " AND date >= ?"
            params.append(start_date)
        if end_date:
            where += " AND date <= ?"
            params.append(end_date)

        rows = []
        for table in ('tracked_bets', 'tracked_confident_bets'):
            rs = db.execute(
                f"SELECT clv_implied, clv_cents, won, profit FROM {table} {where}", params
            ).fetchall()
            rows.extend([dict(r) for r in rs])

        if not rows:
            return {'positive_clv': None, 'negative_clv': None, 'total': 0}

        pos = [r for r in rows if r['clv_implied'] > 0]
        neg = [r for r in rows if r['clv_implied'] <= 0]

        def _group_stats(group):
            if not group:
                return {'count': 0, 'wins': 0, 'win_rate': None, 'avg_clv': None, 'avg_profit': None}
            wins = sum(1 for r in group if r['won'] == 1)
            profits = [r['profit'] for r in group if r['profit'] is not None]
            return {
                'count': len(group),
                'wins': wins,
                'win_rate': round(wins / len(group), 4),
                'avg_clv': round(sum(r['clv_cents'] for r in group) / len(group), 2),
                'avg_profit': round(sum(profits) / len(profits), 2) if profits else None,
            }

        return {
            'positive_clv': _group_stats(pos),
            'negative_clv': _group_stats(neg),
            'total': len(rows),
        }
    finally:
        if own_conn:
            db.close()


def opening_vs_closing_spread(start_date=None, end_date=None, db=None):
    """Average line movement from opening to best closing line.

    Returns dict with avg_home_move, avg_away_move, game_count, and per-game details.
    """
    own_conn = db is None
    if own_conn:
        db = get_connection()

    def american_to_prob(ml):
        if ml is None:
            return None
        ml = float(ml)
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)

    try:
        where = "WHERE snapshot_type = 'opening'"
        params = []
        if start_date:
            where += " AND date >= ?"
            params.append(start_date)
        if end_date:
            where += " AND date <= ?"
            params.append(end_date)

        openings = db.execute(
            f"SELECT game_id, date, home_ml, away_ml FROM betting_line_history {where}", params
        ).fetchall()

        if not openings:
            return {'avg_home_move': None, 'avg_away_move': None, 'game_count': 0}

        moves_home = []
        moves_away = []

        for op in openings:
            op = dict(op)
            game_id = op['game_id']
            if op['home_ml'] is None or op['away_ml'] is None:
                continue

            # Get game datetime for best closing line lookup
            game = db.execute(
                "SELECT time FROM games WHERE id = ?", (game_id,)
            ).fetchone()
            game_dt = _parse_game_time(op['date'], game['time']) if game and game['time'] else None

            best = get_best_closing_line(db, game_id, game_dt)
            if not best or best['home_ml'] is None or best['away_ml'] is None:
                continue

            op_home_p = american_to_prob(op['home_ml'])
            cl_home_p = american_to_prob(best['home_ml'])
            op_away_p = american_to_prob(op['away_ml'])
            cl_away_p = american_to_prob(best['away_ml'])

            if all(v is not None for v in [op_home_p, cl_home_p, op_away_p, cl_away_p]):
                moves_home.append(cl_home_p - op_home_p)
                moves_away.append(cl_away_p - op_away_p)

        if not moves_home:
            return {'avg_home_move': None, 'avg_away_move': None, 'game_count': 0}

        return {
            'avg_home_move': round(sum(moves_home) / len(moves_home) * 100, 3),
            'avg_away_move': round(sum(moves_away) / len(moves_away) * 100, 3),
            'game_count': len(moves_home),
        }
    finally:
        if own_conn:
            db.close()


# ────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Enhanced CLV analysis')
    parser.add_argument('--update', action='store_true', help='Retroactive CLV recalculation')
    parser.add_argument('--report', action='store_true', help='Opening vs closing spread report')
    parser.add_argument('--correlation', action='store_true', help='CLV vs result correlation')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    if not any([args.update, args.report, args.correlation, args.all]):
        parser.print_help()
        return

    if args.update or args.all:
        print("=== Retroactive CLV Update ===")
        count = retroactive_clv_update()
        print(f"Updated {count} bets with best-available closing lines.\n")

    if args.correlation or args.all:
        print("=== CLV vs Result Correlation ===")
        corr = clv_vs_result_correlation(args.start, args.end)
        print(f"Total bets with CLV: {corr['total']}")
        for label, key in [('Positive CLV', 'positive_clv'), ('Negative CLV', 'negative_clv')]:
            g = corr[key]
            if g and g['count'] > 0:
                print(f"  {label}: {g['count']} bets, {g['wins']} wins "
                      f"({g['win_rate']:.1%}), avg CLV {g['avg_clv']:+.1f}c"
                      f", avg profit ${g['avg_profit']:.2f}" if g['avg_profit'] is not None else "")
            else:
                print(f"  {label}: no data")
        print()

    if args.report or args.all:
        print("=== Opening vs Closing Spread ===")
        spread = opening_vs_closing_spread(args.start, args.end)
        if spread['game_count'] > 0:
            print(f"Games analyzed: {spread['game_count']}")
            print(f"Avg home implied prob movement: {spread['avg_home_move']:+.3f}%")
            print(f"Avg away implied prob movement: {spread['avg_away_move']:+.3f}%")
        else:
            print("No opening/closing pairs found.")
        print()


if __name__ == '__main__':
    main()
