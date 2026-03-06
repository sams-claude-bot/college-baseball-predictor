#!/usr/bin/env python3
"""Bet Timing & Edge Analysis.

Analyzes how model edge evolves across line snapshots, identifies optimal
bet timing, and checks whether post-lock line movement confirms model picks.

Usage:
    python3 scripts/bet_timing_analysis.py --game <game_id>
    python3 scripts/bet_timing_analysis.py --report --start-date 2026-02-01 --end-date 2026-03-01
    python3 scripts/bet_timing_analysis.py --confirmation --start-date 2026-02-01 --end-date 2026-03-01
    python3 scripts/bet_timing_analysis.py --report --start-date 2026-02-01 --end-date 2026-03-01 --markdown
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.database import get_connection


def american_to_prob(ml):
    """Convert American odds to implied probability."""
    if ml is None:
        return None
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


# ──────────────────────────────────────────────
# a) Edge timeline for a single game
# ──────────────────────────────────────────────

def edge_timeline(game_id, conn=None):
    """Return edge at each periodic snapshot for a game.

    Uses meta_ensemble predicted_home_prob and computes
    model_prob − market_implied at each snapshot.

    Returns list of dicts:
        [{timestamp, market_implied, model_prob, edge_pp, ml}, ...]
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # Get meta_ensemble prediction
        row = conn.execute(
            "SELECT predicted_home_prob FROM model_predictions "
            "WHERE game_id = ? AND model_name = 'meta_ensemble'",
            (game_id,)
        ).fetchone()
        if not row:
            return []

        model_prob = row['predicted_home_prob']

        # Get periodic snapshots ordered by time
        snaps = conn.execute(
            "SELECT home_ml, captured_at FROM betting_line_history "
            "WHERE game_id = ? AND snapshot_type = 'periodic' "
            "ORDER BY captured_at ASC",
            (game_id,)
        ).fetchall()

        timeline = []
        for snap in snaps:
            snap = dict(snap)
            ml = snap['home_ml']
            market_implied = american_to_prob(ml)
            if market_implied is None:
                continue
            edge_pp = round((model_prob - market_implied) * 100, 2)
            timeline.append({
                'timestamp': snap['captured_at'],
                'market_implied': round(market_implied, 4),
                'model_prob': round(model_prob, 4),
                'edge_pp': edge_pp,
                'ml': ml,
            })

        return timeline
    finally:
        if close_conn:
            conn.close()


# ──────────────────────────────────────────────
# b) Optimal timing report
# ──────────────────────────────────────────────

def _find_snapshot_near_time(snapshots, target_hour, target_minute):
    """Find the snapshot closest to target_hour:target_minute on its date."""
    from datetime import datetime

    best = None
    best_diff = None
    for snap in snapshots:
        captured = snap['captured_at']
        try:
            dt = datetime.fromisoformat(captured)
        except (ValueError, TypeError):
            continue
        snap_minutes = dt.hour * 60 + dt.minute
        target_minutes = target_hour * 60 + target_minute
        diff = abs(snap_minutes - target_minutes)
        if best_diff is None or diff < best_diff:
            best = snap
            best_diff = diff
    return best


def optimal_timing_report(start_date, end_date, conn=None):
    """Compute edge at open / lock / close for tracked bets with line history.

    Returns dict with:
        - game_details: per-game edge at open, lock, close
        - avg_edge_at_open, avg_edge_at_lock, avg_edge_at_close
        - avg_edge_erosion_lock_to_close
        - won_avg_edge_at_close, lost_avg_edge_at_close
        - count
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        # Get tracked bets in date range
        bets = conn.execute(
            "SELECT game_id, model_prob, is_home, won FROM tracked_bets "
            "WHERE date >= ? AND date <= ? AND won IS NOT NULL",
            (start_date, end_date)
        ).fetchall()

        game_details = []

        for bet in bets:
            bet = dict(bet)
            game_id = bet['game_id']
            model_prob = bet['model_prob']
            is_home = bet['is_home']

            # Get all snapshots for this game
            all_snaps = conn.execute(
                "SELECT home_ml, away_ml, snapshot_type, captured_at "
                "FROM betting_line_history WHERE game_id = ? "
                "ORDER BY captured_at ASC",
                (game_id,)
            ).fetchall()
            all_snaps = [dict(s) for s in all_snaps]

            if not all_snaps:
                continue

            ml_col = 'home_ml' if is_home else 'away_ml'

            # Opening: first snapshot
            open_ml = all_snaps[0][ml_col]
            open_implied = american_to_prob(open_ml)
            if open_implied is None:
                continue

            # Lock time: closest to 9:15 AM
            periodic = [s for s in all_snaps if s['snapshot_type'] == 'periodic']
            lock_snap = _find_snapshot_near_time(periodic, 9, 15) if periodic else None
            lock_ml = lock_snap[ml_col] if lock_snap else open_ml
            lock_implied = american_to_prob(lock_ml)

            # Closing: last snapshot with snapshot_type='closing', or last overall
            closing_snaps = [s for s in all_snaps if s['snapshot_type'] == 'closing']
            close_snap = closing_snaps[-1] if closing_snaps else all_snaps[-1]
            close_ml = close_snap[ml_col]
            close_implied = american_to_prob(close_ml)
            if lock_implied is None or close_implied is None:
                continue

            edge_open = round((model_prob - open_implied) * 100, 2)
            edge_lock = round((model_prob - lock_implied) * 100, 2)
            edge_close = round((model_prob - close_implied) * 100, 2)
            erosion = round(edge_lock - edge_close, 2)

            game_details.append({
                'game_id': game_id,
                'won': bet['won'],
                'edge_at_open': edge_open,
                'edge_at_lock': edge_lock,
                'edge_at_close': edge_close,
                'edge_erosion': erosion,
            })

        if not game_details:
            return {
                'count': 0, 'game_details': [],
                'avg_edge_at_open': None, 'avg_edge_at_lock': None,
                'avg_edge_at_close': None, 'avg_edge_erosion_lock_to_close': None,
                'won_avg_edge_at_close': None, 'lost_avg_edge_at_close': None,
            }

        n = len(game_details)
        avg_open = round(sum(g['edge_at_open'] for g in game_details) / n, 2)
        avg_lock = round(sum(g['edge_at_lock'] for g in game_details) / n, 2)
        avg_close = round(sum(g['edge_at_close'] for g in game_details) / n, 2)
        avg_erosion = round(sum(g['edge_erosion'] for g in game_details) / n, 2)

        won = [g for g in game_details if g['won'] == 1]
        lost = [g for g in game_details if g['won'] == 0]
        won_avg = round(sum(g['edge_at_close'] for g in won) / len(won), 2) if won else None
        lost_avg = round(sum(g['edge_at_close'] for g in lost) / len(lost), 2) if lost else None

        return {
            'count': n,
            'game_details': game_details,
            'avg_edge_at_open': avg_open,
            'avg_edge_at_lock': avg_lock,
            'avg_edge_at_close': avg_close,
            'avg_edge_erosion_lock_to_close': avg_erosion,
            'won_avg_edge_at_close': won_avg,
            'lost_avg_edge_at_close': lost_avg,
        }
    finally:
        if close_conn:
            conn.close()


# ──────────────────────────────────────────────
# c) Line confirms model
# ──────────────────────────────────────────────

def line_confirms_model(start_date, end_date, conn=None):
    """Check if lines moved toward our pick after lock.

    A line "confirms" our model when the closing implied probability for
    the picked side is higher than the opening implied probability.

    Returns dict:
        pct_confirmed, avg_confirmation_size,
        win_rate_confirmed, win_rate_not_confirmed, count
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        bets = conn.execute(
            "SELECT game_id, is_home, won FROM tracked_bets "
            "WHERE date >= ? AND date <= ? AND won IS NOT NULL",
            (start_date, end_date)
        ).fetchall()

        confirmed = []
        not_confirmed = []

        for bet in bets:
            bet = dict(bet)
            game_id = bet['game_id']
            is_home = bet['is_home']

            snaps = conn.execute(
                "SELECT home_ml, away_ml, snapshot_type, captured_at "
                "FROM betting_line_history WHERE game_id = ? "
                "ORDER BY captured_at ASC",
                (game_id,)
            ).fetchall()
            snaps = [dict(s) for s in snaps]

            if len(snaps) < 2:
                continue

            ml_col = 'home_ml' if is_home else 'away_ml'

            open_ml = snaps[0][ml_col]
            open_implied = american_to_prob(open_ml)

            # Closing: prefer closing snapshot, else last
            closing = [s for s in snaps if s['snapshot_type'] == 'closing']
            close_snap = closing[-1] if closing else snaps[-1]
            close_ml = close_snap[ml_col]
            close_implied = american_to_prob(close_ml)

            if open_implied is None or close_implied is None:
                continue

            move = close_implied - open_implied  # positive = line moved toward pick
            entry = {'won': bet['won'], 'move': round(move * 100, 2)}

            if move > 0:
                confirmed.append(entry)
            else:
                not_confirmed.append(entry)

        total = len(confirmed) + len(not_confirmed)
        if total == 0:
            return {
                'count': 0, 'pct_confirmed': None,
                'avg_confirmation_size': None,
                'win_rate_confirmed': None,
                'win_rate_not_confirmed': None,
            }

        pct = round(len(confirmed) / total * 100, 1)
        avg_size = round(sum(c['move'] for c in confirmed) / len(confirmed), 2) if confirmed else 0.0

        conf_wins = sum(1 for c in confirmed if c['won'] == 1)
        wr_conf = round(conf_wins / len(confirmed) * 100, 1) if confirmed else None

        not_wins = sum(1 for c in not_confirmed if c['won'] == 1)
        wr_not = round(not_wins / len(not_confirmed) * 100, 1) if not_confirmed else None

        return {
            'count': total,
            'pct_confirmed': pct,
            'avg_confirmation_size': avg_size,
            'win_rate_confirmed': wr_conf,
            'win_rate_not_confirmed': wr_not,
        }
    finally:
        if close_conn:
            conn.close()


# ──────────────────────────────────────────────
# CLI output helpers
# ──────────────────────────────────────────────

def print_edge_timeline(game_id, conn=None):
    """Print formatted edge timeline for a game."""
    tl = edge_timeline(game_id, conn=conn)
    if not tl:
        print(f"No edge timeline data for game {game_id}")
        return tl

    print(f"\n=== Edge Timeline: {game_id} ===\n")
    print(f"  {'Timestamp':<22} {'ML':>6} {'Mkt Impl':>9} {'Model':>7} {'Edge':>7}")
    print(f"  {'-'*22} {'-'*6} {'-'*9} {'-'*7} {'-'*7}")
    for pt in tl:
        print(f"  {pt['timestamp']:<22} {pt['ml']:>6.0f} {pt['market_implied']:>8.1%}"
              f" {pt['model_prob']:>6.1%} {pt['edge_pp']:>+6.1f}pp")
    print()
    return tl


def print_timing_report(report):
    """Print formatted optimal timing report."""
    if report['count'] == 0:
        print("No games with both tracked bets and line history found.")
        return

    print(f"\n=== Optimal Timing Report ({report['count']} games) ===\n")
    print(f"  Avg Edge at Open:   {report['avg_edge_at_open']:>+6.2f}pp")
    print(f"  Avg Edge at Lock:   {report['avg_edge_at_lock']:>+6.2f}pp")
    print(f"  Avg Edge at Close:  {report['avg_edge_at_close']:>+6.2f}pp")
    print(f"  Avg Erosion (Lock→Close): {report['avg_edge_erosion_lock_to_close']:>+6.2f}pp")
    print()
    if report['won_avg_edge_at_close'] is not None:
        print(f"  Won  — Avg Edge at Close: {report['won_avg_edge_at_close']:>+6.2f}pp")
    if report['lost_avg_edge_at_close'] is not None:
        print(f"  Lost — Avg Edge at Close: {report['lost_avg_edge_at_close']:>+6.2f}pp")
    print()


def print_confirmation_report(report):
    """Print formatted line confirmation report."""
    if report['count'] == 0:
        print("No games with line movement data found.")
        return

    print(f"\n=== Line Confirmation Report ({report['count']} games) ===\n")
    print(f"  Lines confirmed model:      {report['pct_confirmed']:.1f}%")
    print(f"  Avg confirmation size:      {report['avg_confirmation_size']:>+.2f}pp")
    print()
    if report['win_rate_confirmed'] is not None:
        print(f"  Win rate (confirmed):       {report['win_rate_confirmed']:.1f}%")
    if report['win_rate_not_confirmed'] is not None:
        print(f"  Win rate (not confirmed):   {report['win_rate_not_confirmed']:.1f}%")
    print()


def render_timing_markdown(report):
    """Render optimal timing report as markdown."""
    if report['count'] == 0:
        return "# Bet Timing Report\n\nNo data available.\n"

    lines = [
        "# Bet Timing Report\n",
        f"**Games analyzed:** {report['count']}\n",
        "## Edge at Key Moments\n",
        "| Moment | Avg Edge |",
        "|--------|----------|",
        f"| Open | {report['avg_edge_at_open']:+.2f}pp |",
        f"| Lock (~9:15 AM) | {report['avg_edge_at_lock']:+.2f}pp |",
        f"| Close | {report['avg_edge_at_close']:+.2f}pp |",
        f"| Erosion (Lock→Close) | {report['avg_edge_erosion_lock_to_close']:+.2f}pp |",
        "",
        "## Edge at Close by Outcome\n",
    ]
    if report['won_avg_edge_at_close'] is not None:
        lines.append(f"- **Won:** {report['won_avg_edge_at_close']:+.2f}pp")
    if report['lost_avg_edge_at_close'] is not None:
        lines.append(f"- **Lost:** {report['lost_avg_edge_at_close']:+.2f}pp")
    lines.append("")
    return "\n".join(lines)


def render_confirmation_markdown(report):
    """Render line confirmation report as markdown."""
    if report['count'] == 0:
        return "# Line Confirmation Report\n\nNo data available.\n"

    lines = [
        "# Line Confirmation Report\n",
        f"**Games analyzed:** {report['count']}\n",
        "## Results\n",
        f"- Lines confirmed model: **{report['pct_confirmed']:.1f}%**",
        f"- Avg confirmation size: **{report['avg_confirmation_size']:+.2f}pp**",
        "",
    ]
    if report['win_rate_confirmed'] is not None:
        lines.append(f"- Win rate (confirmed): **{report['win_rate_confirmed']:.1f}%**")
    if report['win_rate_not_confirmed'] is not None:
        lines.append(f"- Win rate (not confirmed): **{report['win_rate_not_confirmed']:.1f}%**")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bet Timing & Edge Analysis")
    parser.add_argument("--game", help="Game ID for edge timeline")
    parser.add_argument("--report", action="store_true", help="Optimal timing report")
    parser.add_argument("--confirmation", action="store_true", help="Line confirmation report")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--markdown", action="store_true", help="Save markdown to artifacts/")
    args = parser.parse_args()

    artifacts_dir = PROJECT_DIR / 'artifacts'

    if args.game:
        print_edge_timeline(args.game)

    elif args.report:
        if not args.start_date or not args.end_date:
            print("Error: --report requires --start-date and --end-date")
            sys.exit(1)
        report = optimal_timing_report(args.start_date, args.end_date)
        print_timing_report(report)
        if args.markdown:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            md = render_timing_markdown(report)
            out = artifacts_dir / 'bet_timing_report.md'
            out.write_text(md)
            print(f"Saved to {out}")

    elif args.confirmation:
        if not args.start_date or not args.end_date:
            print("Error: --confirmation requires --start-date and --end-date")
            sys.exit(1)
        report = line_confirms_model(args.start_date, args.end_date)
        print_confirmation_report(report)
        if args.markdown:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            md = render_confirmation_markdown(report)
            out = artifacts_dir / 'line_confirmation_report.md'
            out.write_text(md)
            print(f"Saved to {out}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
