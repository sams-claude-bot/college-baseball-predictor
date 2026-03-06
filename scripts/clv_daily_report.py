#!/usr/bin/env python3
"""
Daily CLV (Closing Line Value) Report.

Generates a markdown summary of yesterday's graded bets with CLV data.
Outputs to artifacts/clv_daily_YYYY-MM-DD.md

Usage:
    python3 scripts/clv_daily_report.py              # yesterday's report
    python3 scripts/clv_daily_report.py --date 2026-03-05
    python3 scripts/clv_daily_report.py --json        # JSON output to stdout
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from database import get_connection


def _fetch_clv_bets(conn, date_str):
    """Fetch all graded bets with CLV data for a given date."""
    bets = []

    ml_rows = conn.execute("""
        SELECT game_id, pick_team_name, opponent_name, moneyline,
               model_prob, dk_implied, edge, won, profit,
               closing_ml, clv_implied, clv_cents
        FROM tracked_bets
        WHERE date = ? AND won IS NOT NULL AND clv_implied IS NOT NULL
    """, (date_str,)).fetchall()
    for r in ml_rows:
        bets.append({**dict(r), 'type': 'ML'})

    conf_rows = conn.execute("""
        SELECT game_id, pick_team_name, opponent_name, moneyline,
               models_agree, models_total, avg_prob, confidence, won, profit,
               closing_ml, clv_implied, clv_cents
        FROM tracked_confident_bets
        WHERE date = ? AND won IS NOT NULL AND clv_implied IS NOT NULL
    """, (date_str,)).fetchall()
    for r in conf_rows:
        bets.append({**dict(r), 'type': 'CONSENSUS'})

    return bets


def _fetch_alltime_clv(conn):
    """Fetch all-time CLV averages across both tables."""
    all_clv = []

    for table, bet_type in [('tracked_bets', 'ML'), ('tracked_confident_bets', 'CONSENSUS')]:
        rows = conn.execute(f"""
            SELECT clv_implied, clv_cents, won
            FROM {table}
            WHERE clv_implied IS NOT NULL AND won IS NOT NULL
        """).fetchall()
        for r in rows:
            all_clv.append({'clv_implied': r['clv_implied'], 'clv_cents': r['clv_cents'],
                            'won': r['won'], 'type': bet_type})

    return all_clv


def generate_daily_report(date_str, conn=None):
    """Generate daily CLV report data.

    Returns dict with report data, or None if no data.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    bets = _fetch_clv_bets(conn, date_str)
    alltime = _fetch_alltime_clv(conn)

    if own_conn:
        conn.close()

    if not bets:
        return None

    # Daily stats
    total = len(bets)
    avg_clv_implied = sum(b['clv_implied'] for b in bets) / total
    avg_clv_cents = sum(b['clv_cents'] for b in bets) / total
    positive_clv = [b for b in bets if b['clv_cents'] > 0]
    negative_clv = [b for b in bets if b['clv_cents'] <= 0]

    pos_wins = sum(1 for b in positive_clv if b['won'])
    pos_total = len(positive_clv)
    neg_wins = sum(1 for b in negative_clv if b['won'])
    neg_total = len(negative_clv)

    total_profit = sum(b['profit'] for b in bets if b['profit'] is not None)

    # By type breakdown
    by_type = {}
    for bet_type in ('ML', 'CONSENSUS'):
        typed = [b for b in bets if b['type'] == bet_type]
        if typed:
            by_type[bet_type] = {
                'count': len(typed),
                'avg_clv_implied': round(sum(b['clv_implied'] for b in typed) / len(typed), 4),
                'avg_clv_cents': round(sum(b['clv_cents'] for b in typed) / len(typed), 2),
                'wins': sum(1 for b in typed if b['won']),
                'profit': round(sum(b['profit'] for b in typed if b['profit'] is not None), 2),
            }

    # All-time running averages
    alltime_stats = None
    if alltime:
        at_total = len(alltime)
        at_pos = [a for a in alltime if a['clv_cents'] > 0]
        at_neg = [a for a in alltime if a['clv_cents'] <= 0]
        alltime_stats = {
            'total_bets': at_total,
            'avg_clv_implied': round(sum(a['clv_implied'] for a in alltime) / at_total, 4),
            'avg_clv_cents': round(sum(a['clv_cents'] for a in alltime) / at_total, 2),
            'pos_clv_win_rate': round(sum(1 for a in at_pos if a['won']) / len(at_pos) * 100, 1) if at_pos else 0,
            'neg_clv_win_rate': round(sum(1 for a in at_neg if a['won']) / len(at_neg) * 100, 1) if at_neg else 0,
        }

    # Notable bets: top 3 by absolute CLV cents
    notable = sorted(bets, key=lambda b: abs(b['clv_cents']), reverse=True)[:3]

    return {
        'date': date_str,
        'total_bets': total,
        'avg_clv_implied': round(avg_clv_implied, 4),
        'avg_clv_cents': round(avg_clv_cents, 2),
        'positive_clv_count': pos_total,
        'negative_clv_count': neg_total,
        'pos_clv_win_rate': round(pos_wins / pos_total * 100, 1) if pos_total else 0,
        'neg_clv_win_rate': round(neg_wins / neg_total * 100, 1) if neg_total else 0,
        'total_profit': round(total_profit, 2),
        'by_type': by_type,
        'alltime': alltime_stats,
        'notable': [
            {
                'team': b['pick_team_name'],
                'opponent': b['opponent_name'],
                'ml': b['moneyline'],
                'closing_ml': b['closing_ml'],
                'clv_cents': round(b['clv_cents'], 2),
                'won': bool(b['won']),
                'type': b['type'],
            }
            for b in notable
        ],
    }


def render_markdown(data):
    """Render report data as markdown."""
    lines = []
    lines.append(f"# CLV Daily Report — {data['date']}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total bets with CLV | {data['total_bets']} |")
    lines.append(f"| Avg CLV (implied) | {data['avg_clv_implied']:+.4f} |")
    lines.append(f"| Avg CLV (cents) | {data['avg_clv_cents']:+.2f} |")
    lines.append(f"| +CLV bets | {data['positive_clv_count']} |")
    lines.append(f"| -CLV bets | {data['negative_clv_count']} |")
    lines.append(f"| Day profit | ${data['total_profit']:+.2f} |")
    lines.append("")

    # Win rate by CLV
    lines.append("## Win Rate: +CLV vs -CLV")
    lines.append("")
    lines.append(f"| Group | Win Rate |")
    lines.append(f"|-------|----------|")
    lines.append(f"| +CLV bets | {data['pos_clv_win_rate']:.1f}% ({data['positive_clv_count']} bets) |")
    lines.append(f"| -CLV bets | {data['neg_clv_win_rate']:.1f}% ({data['negative_clv_count']} bets) |")
    lines.append("")

    # By type
    if data['by_type']:
        lines.append("## Breakdown by Type")
        lines.append("")
        lines.append(f"| Type | Count | Avg CLV (cents) | Wins | Profit |")
        lines.append(f"|------|-------|-----------------|------|--------|")
        for bet_type, stats in data['by_type'].items():
            lines.append(
                f"| {bet_type} | {stats['count']} | {stats['avg_clv_cents']:+.2f} "
                f"| {stats['wins']}/{stats['count']} | ${stats['profit']:+.2f} |"
            )
        lines.append("")

    # Notable bets
    if data['notable']:
        lines.append("## Notable Bets")
        lines.append("")
        lines.append(f"| Team | Opponent | Open ML | Close ML | CLV (cents) | Won | Type |")
        lines.append(f"|------|----------|---------|----------|-------------|-----|------|")
        for b in data['notable']:
            won_str = "W" if b['won'] else "L"
            lines.append(
                f"| {b['team']} | {b['opponent']} | {b['ml']} | {b['closing_ml']:.0f} "
                f"| {b['clv_cents']:+.2f} | {won_str} | {b['type']} |"
            )
        lines.append("")

    # All-time running averages
    if data['alltime']:
        at = data['alltime']
        lines.append("## All-Time Running Averages")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total bets with CLV | {at['total_bets']} |")
        lines.append(f"| Avg CLV (implied) | {at['avg_clv_implied']:+.4f} |")
        lines.append(f"| Avg CLV (cents) | {at['avg_clv_cents']:+.2f} |")
        lines.append(f"| +CLV win rate | {at['pos_clv_win_rate']:.1f}% |")
        lines.append(f"| -CLV win rate | {at['neg_clv_win_rate']:.1f}% |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Daily CLV report")
    parser.add_argument('--date', type=str, default=None,
                        help="Report date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument('--json', action='store_true', dest='json_output',
                        help="Output structured JSON to stdout")
    args = parser.parse_args()

    date_str = args.date or (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    data = generate_daily_report(date_str)

    if data is None:
        print(f"No CLV data for {date_str}")
        sys.exit(0)

    if args.json_output:
        print(json.dumps(data, indent=2))
    else:
        md = render_markdown(data)
        artifacts_dir = PROJECT_ROOT / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        out_path = artifacts_dir / f"clv_daily_{date_str}.md"
        out_path.write_text(md)
        print(f"CLV daily report written to {out_path}")
        print(f"  {data['total_bets']} bets | Avg CLV: {data['avg_clv_cents']:+.2f} cents | Profit: ${data['total_profit']:+.2f}")


if __name__ == '__main__':
    main()
