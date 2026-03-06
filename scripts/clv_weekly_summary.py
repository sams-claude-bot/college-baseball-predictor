#!/usr/bin/env python3
"""
Weekly CLV (Closing Line Value) Summary.

Generates a trailing 7-day CLV aggregation report.
Outputs to artifacts/clv_weekly_YYYY-MM-DD.md

Usage:
    python3 scripts/clv_weekly_summary.py              # trailing 7 days from yesterday
    python3 scripts/clv_weekly_summary.py --end-date 2026-03-05
    python3 scripts/clv_weekly_summary.py --json
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


def _fetch_week_bets(conn, start_date, end_date):
    """Fetch all graded bets with CLV data for a date range."""
    bets = []

    ml_rows = conn.execute("""
        SELECT date, game_id, pick_team_name, opponent_name, moneyline,
               won, profit, closing_ml, clv_implied, clv_cents
        FROM tracked_bets
        WHERE date BETWEEN ? AND ? AND won IS NOT NULL AND clv_implied IS NOT NULL
        ORDER BY date
    """, (start_date, end_date)).fetchall()
    for r in ml_rows:
        bets.append({**dict(r), 'type': 'ML'})

    conf_rows = conn.execute("""
        SELECT date, game_id, pick_team_name, opponent_name, moneyline,
               won, profit, closing_ml, clv_implied, clv_cents
        FROM tracked_confident_bets
        WHERE date BETWEEN ? AND ? AND won IS NOT NULL AND clv_implied IS NOT NULL
        ORDER BY date
    """, (start_date, end_date)).fetchall()
    for r in conf_rows:
        bets.append({**dict(r), 'type': 'CONSENSUS'})

    return bets


def _compute_trend_direction(daily_data):
    """Compute trend direction from daily CLV averages.

    Returns 'improving', 'declining', or 'flat'.
    """
    if len(daily_data) < 2:
        return 'flat'

    dates = sorted(daily_data.keys())
    values = [daily_data[d]['avg_clv_cents'] for d in dates]

    # Simple: compare first half avg to second half avg
    mid = len(values) // 2
    first_half = sum(values[:mid]) / mid if mid > 0 else 0
    second_half = sum(values[mid:]) / (len(values) - mid) if (len(values) - mid) > 0 else 0

    diff = second_half - first_half
    if diff > 0.5:
        return 'improving'
    elif diff < -0.5:
        return 'declining'
    return 'flat'


def _compute_correlation(bets):
    """Compute simple CLV vs profit correlation.

    Returns Pearson correlation coefficient, or None if insufficient data.
    """
    valid = [(b['clv_cents'], b['profit']) for b in bets if b['profit'] is not None]
    if len(valid) < 3:
        return None

    n = len(valid)
    x = [v[0] for v in valid]
    y = [v[1] for v in valid]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

    if std_x == 0 or std_y == 0:
        return None

    return round(cov / (std_x * std_y), 3)


def generate_weekly_summary(end_date_str, conn=None):
    """Generate weekly CLV summary data.

    Returns dict with report data, or None if no data.
    """
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=6)
    start_date_str = start_dt.strftime('%Y-%m-%d')

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    bets = _fetch_week_bets(conn, start_date_str, end_date_str)

    if own_conn:
        conn.close()

    if not bets:
        return None

    total = len(bets)
    avg_clv_implied = sum(b['clv_implied'] for b in bets) / total
    avg_clv_cents = sum(b['clv_cents'] for b in bets) / total
    total_profit = sum(b['profit'] for b in bets if b['profit'] is not None)

    positive_clv = [b for b in bets if b['clv_cents'] > 0]
    negative_clv = [b for b in bets if b['clv_cents'] <= 0]
    pos_wins = sum(1 for b in positive_clv if b['won'])
    neg_wins = sum(1 for b in negative_clv if b['won'])

    # Daily breakdown
    daily_data = {}
    for b in bets:
        d = b['date']
        if d not in daily_data:
            daily_data[d] = {'bets': 0, 'clv_sum': 0.0, 'profit': 0.0, 'wins': 0}
        daily_data[d]['bets'] += 1
        daily_data[d]['clv_sum'] += b['clv_cents']
        daily_data[d]['profit'] += b['profit'] if b['profit'] is not None else 0
        daily_data[d]['wins'] += 1 if b['won'] else 0
    for d in daily_data:
        daily_data[d]['avg_clv_cents'] = round(daily_data[d]['clv_sum'] / daily_data[d]['bets'], 2)

    # By type
    by_type = {}
    for bet_type in ('ML', 'CONSENSUS'):
        typed = [b for b in bets if b['type'] == bet_type]
        if typed:
            by_type[bet_type] = {
                'count': len(typed),
                'avg_clv_cents': round(sum(b['clv_cents'] for b in typed) / len(typed), 2),
                'wins': sum(1 for b in typed if b['won']),
                'profit': round(sum(b['profit'] for b in typed if b['profit'] is not None), 2),
            }

    trend_direction = _compute_trend_direction(daily_data)
    correlation = _compute_correlation(bets)

    return {
        'start_date': start_date_str,
        'end_date': end_date_str,
        'total_bets': total,
        'avg_clv_implied': round(avg_clv_implied, 4),
        'avg_clv_cents': round(avg_clv_cents, 2),
        'positive_clv_count': len(positive_clv),
        'negative_clv_count': len(negative_clv),
        'pos_clv_win_rate': round(pos_wins / len(positive_clv) * 100, 1) if positive_clv else 0,
        'neg_clv_win_rate': round(neg_wins / len(negative_clv) * 100, 1) if negative_clv else 0,
        'total_profit': round(total_profit, 2),
        'by_type': by_type,
        'daily': {d: {
            'bets': daily_data[d]['bets'],
            'avg_clv_cents': daily_data[d]['avg_clv_cents'],
            'profit': round(daily_data[d]['profit'], 2),
            'wins': daily_data[d]['wins'],
        } for d in sorted(daily_data.keys())},
        'trend_direction': trend_direction,
        'clv_profit_correlation': correlation,
    }


def render_markdown(data):
    """Render weekly summary data as markdown."""
    lines = []
    lines.append(f"# CLV Weekly Summary — {data['start_date']} to {data['end_date']}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total bets | {data['total_bets']} |")
    lines.append(f"| Avg CLV (implied) | {data['avg_clv_implied']:+.4f} |")
    lines.append(f"| Avg CLV (cents) | {data['avg_clv_cents']:+.2f} |")
    lines.append(f"| +CLV / -CLV | {data['positive_clv_count']} / {data['negative_clv_count']} |")
    lines.append(f"| Week profit | ${data['total_profit']:+.2f} |")
    lines.append(f"| Trend | {data['trend_direction']} |")
    if data['clv_profit_correlation'] is not None:
        lines.append(f"| CLV-Profit correlation | {data['clv_profit_correlation']:+.3f} |")
    lines.append("")

    # Win rate comparison
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

    # Daily breakdown
    if data['daily']:
        lines.append("## Daily Breakdown")
        lines.append("")
        lines.append(f"| Date | Bets | Avg CLV (cents) | Wins | Profit |")
        lines.append(f"|------|------|-----------------|------|--------|")
        for date, stats in data['daily'].items():
            lines.append(
                f"| {date} | {stats['bets']} | {stats['avg_clv_cents']:+.2f} "
                f"| {stats['wins']}/{stats['bets']} | ${stats['profit']:+.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Weekly CLV summary (trailing 7 days)")
    parser.add_argument('--end-date', type=str, default=None,
                        help="End date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument('--json', action='store_true', dest='json_output',
                        help="Output structured JSON to stdout")
    args = parser.parse_args()

    end_date = args.end_date or (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    data = generate_weekly_summary(end_date)

    if data is None:
        print(f"No CLV data for week ending {end_date}")
        sys.exit(0)

    if args.json_output:
        print(json.dumps(data, indent=2))
    else:
        md = render_markdown(data)
        artifacts_dir = PROJECT_ROOT / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        out_path = artifacts_dir / f"clv_weekly_{end_date}.md"
        out_path.write_text(md)
        print(f"CLV weekly summary written to {out_path}")
        print(f"  {data['total_bets']} bets | Avg CLV: {data['avg_clv_cents']:+.2f} cents | Trend: {data['trend_direction']}")


if __name__ == '__main__':
    main()
