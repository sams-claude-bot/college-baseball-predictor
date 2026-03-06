#!/usr/bin/env python3
"""
Strategy-Level P&L Report

Breaks down betting performance by strategy:
  - EV Moneyline (tracked_bets)
  - Consensus (tracked_confident_bets)
  - Spreads (tracked_bets_spreads WHERE bet_type='spread')
  - Totals (tracked_bets_spreads WHERE bet_type='total')
  - Parlays (tracked_parlays)

Outputs a markdown report to artifacts/ and optionally JSON to stdout.

Usage:
    python3 scripts/strategy_pl_report.py
    python3 scripts/strategy_pl_report.py --json
    python3 scripts/strategy_pl_report.py --start-date 2026-02-18 --end-date 2026-03-06
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from database import get_connection

DEFAULT_START = "2026-02-18"


def _safe_query(cursor, sql, params):
    """Execute a query, returning [] if the table doesn't exist."""
    try:
        cursor.execute(sql, params)
        return [dict(r) for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []


def query_strategy_data(conn, start_date, end_date):
    """Query all betting tables and return raw rows per strategy."""
    c = conn.cursor()
    params = (start_date, end_date)

    return {
        "EV Moneyline": _safe_query(
            c, "SELECT * FROM tracked_bets WHERE won IS NOT NULL AND date BETWEEN ? AND ?", params
        ),
        "Consensus": _safe_query(
            c, "SELECT * FROM tracked_confident_bets WHERE won IS NOT NULL AND date BETWEEN ? AND ?", params
        ),
        "Spreads": _safe_query(
            c, "SELECT * FROM tracked_bets_spreads WHERE won IS NOT NULL AND bet_type='spread' AND date BETWEEN ? AND ?", params
        ),
        "Totals": _safe_query(
            c, "SELECT * FROM tracked_bets_spreads WHERE won IS NOT NULL AND bet_type='total' AND date BETWEEN ? AND ?", params
        ),
        "Parlays": _safe_query(
            c, "SELECT * FROM tracked_parlays WHERE won IS NOT NULL AND date BETWEEN ? AND ?", params
        ),
    }


def compute_strategy_stats(rows, bet_amount_default=100):
    """Compute W-L, profit, avg edge, ROI for a list of bet rows."""
    if not rows:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_pct": 0.0,
            "profit": 0.0,
            "wagered": 0.0,
            "avg_edge": 0.0,
            "roi": 0.0,
        }

    wins = sum(1 for r in rows if r["won"])
    losses = len(rows) - wins
    profit = sum((r.get("profit") or 0) for r in rows)
    wagered = sum((r.get("bet_amount") or bet_amount_default) for r in rows)

    edges = [r["edge"] for r in rows if r.get("edge") is not None]
    avg_edge = sum(edges) / len(edges) if edges else 0.0

    roi = (profit / wagered * 100) if wagered else 0.0

    return {
        "bets": len(rows),
        "wins": wins,
        "losses": losses,
        "win_pct": round(wins / len(rows) * 100, 1),
        "profit": round(profit, 2),
        "wagered": round(wagered, 2),
        "avg_edge": round(avg_edge, 2),
        "roi": round(roi, 1),
    }


def build_daily_pl(all_strategies):
    """Build a combined daily P&L timeline across all strategies."""
    daily = defaultdict(float)
    for _name, rows in all_strategies.items():
        for r in rows:
            daily[r["date"]] += r.get("profit") or 0

    timeline = []
    cumulative = 0
    for d in sorted(daily.keys()):
        cumulative += daily[d]
        timeline.append({
            "date": d,
            "daily_pl": round(daily[d], 2),
            "cumulative_pl": round(cumulative, 2),
        })
    return timeline


def build_report(strategy_stats, daily_timeline, start_date, end_date):
    """Build markdown report string."""
    lines = []
    lines.append(f"# Strategy P&L Report")
    lines.append(f"**Period:** {start_date} to {end_date}")
    lines.append("")

    # Per-strategy breakdown
    lines.append("## Strategy Breakdown")
    lines.append("")
    lines.append("| Strategy | Bets | W-L | Win% | Profit | Avg Edge | ROI% |")
    lines.append("|----------|------|-----|------|--------|----------|------|")
    total_bets = 0
    total_wins = 0
    total_losses = 0
    total_profit = 0.0
    total_wagered = 0.0
    for name, stats in strategy_stats.items():
        wl = f"{stats['wins']}-{stats['losses']}"
        profit_str = f"${stats['profit']:+.2f}"
        edge_str = f"{stats['avg_edge']:.2f}%" if stats['avg_edge'] else "—"
        roi_str = f"{stats['roi']:+.1f}%"
        lines.append(
            f"| {name} | {stats['bets']} | {wl} | {stats['win_pct']:.1f}% | {profit_str} | {edge_str} | {roi_str} |"
        )
        total_bets += stats["bets"]
        total_wins += stats["wins"]
        total_losses += stats["losses"]
        total_profit += stats["profit"]
        total_wagered += stats["wagered"]

    # Overall row
    overall_win_pct = round(total_wins / total_bets * 100, 1) if total_bets else 0
    overall_roi = round(total_profit / total_wagered * 100, 1) if total_wagered else 0
    lines.append(
        f"| **TOTAL** | **{total_bets}** | **{total_wins}-{total_losses}** | **{overall_win_pct:.1f}%** | **${total_profit:+.2f}** | — | **{overall_roi:+.1f}%** |"
    )
    lines.append("")

    # Daily timeline
    if daily_timeline:
        lines.append("## Daily P&L Timeline")
        lines.append("")
        lines.append("| Date | Daily P&L | Cumulative |")
        lines.append("|------|-----------|------------|")
        for entry in daily_timeline:
            lines.append(
                f"| {entry['date']} | ${entry['daily_pl']:+.2f} | ${entry['cumulative_pl']:+.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_report(conn=None, start_date=None, end_date=None):
    """Main entry point: query data, compute stats, return structured results.

    Returns (strategy_stats dict, daily_timeline list, report_md string).
    """
    if start_date is None:
        start_date = DEFAULT_START
    if end_date is None:
        end_date = date.today().isoformat()

    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        raw = query_strategy_data(conn, start_date, end_date)
        strategy_stats = {name: compute_strategy_stats(rows) for name, rows in raw.items()}
        daily_timeline = build_daily_pl(raw)
        report_md = build_report(strategy_stats, daily_timeline, start_date, end_date)
        return strategy_stats, daily_timeline, report_md
    finally:
        if close_conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Strategy-Level P&L Report")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--start-date", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    strategy_stats, daily_timeline, report_md = generate_report(
        start_date=args.start_date, end_date=args.end_date
    )

    if args.json:
        output = {
            "period": {"start": args.start_date, "end": args.end_date},
            "strategies": strategy_stats,
            "daily_timeline": daily_timeline,
        }
        print(json.dumps(output, indent=2))
    else:
        # Write markdown report
        artifacts_dir = PROJECT_DIR / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifacts_dir / f"strategy_pl_{args.end_date}.md"
        report_path.write_text(report_md)
        print(report_md)
        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
