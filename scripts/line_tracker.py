#!/usr/bin/env python3
"""Track betting line movement throughout the day.

Periodically snapshots ALL active betting lines into betting_line_history,
enabling line movement analysis, steam move detection, and better CLV accuracy.

Works alongside capture_closing_lines.py which handles the closing-specific logic.

Usage:
    python3 scripts/line_tracker.py                  # Snapshot all active lines
    python3 scripts/line_tracker.py --summary        # Print line movement summary for today
    python3 scripts/line_tracker.py --movers         # Show biggest line movers today
    python3 scripts/line_tracker.py --movers --top 10

Cron (every 5 min during game hours):
    */5 10-23 * * * cd /home/sam/college-baseball-predictor && python3 scripts/line_tracker.py >> logs/cron/$(date +%%Y-%%m-%%d)_line_tracker.log 2>&1
"""
import argparse
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Minimum minutes between periodic snapshots for the same game
# Prevents excessive writes while still catching movement
MIN_SNAPSHOT_INTERVAL_MINUTES = 4


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def get_ct_now():
    """Get current Central Time as a naive datetime."""
    utc_now = datetime.now(timezone.utc)
    month = utc_now.month
    ct_offset = timedelta(hours=-5) if 3 <= month <= 10 else timedelta(hours=-6)
    return utc_now + ct_offset


def american_to_prob(ml):
    """Convert American odds to implied probability."""
    if ml is None:
        return None
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def snapshot_all_lines(db=None):
    """Snapshot all active betting lines into history.

    Only writes a new snapshot if:
    - No periodic snapshot exists for this game yet today, OR
    - The line has moved since the last snapshot, OR
    - It's been >= MIN_SNAPSHOT_INTERVAL_MINUTES since last snapshot
    """
    close_db = False
    if db is None:
        db = get_db()
        close_db = True

    ct_now = get_ct_now()
    today = ct_now.strftime('%Y-%m-%d')
    now_str = ct_now.strftime('%Y-%m-%dT%H:%M:%S')

    # Get all betting lines for today's games
    lines = db.execute("""
        SELECT bl.game_id, bl.date, bl.home_team_id, bl.away_team_id, bl.book,
               bl.home_ml, bl.away_ml, bl.over_under, bl.over_odds, bl.under_odds
        FROM betting_lines bl
        JOIN games g ON bl.game_id = g.id
        WHERE bl.date = ? AND g.status IN ('scheduled', 'in-progress', 'in_progress')
    """, (today,)).fetchall()

    captured = 0
    skipped_unchanged = 0
    skipped_recent = 0

    for bl in lines:
        bl = dict(bl)
        game_id = bl['game_id']

        # Skip if no ML data
        if bl['home_ml'] is None and bl['away_ml'] is None:
            continue

        # Get the most recent periodic snapshot for this game
        last_snap = db.execute("""
            SELECT home_ml, away_ml, over_under, over_odds, under_odds, captured_at
            FROM betting_line_history
            WHERE game_id = ? AND snapshot_type = 'periodic'
            ORDER BY captured_at DESC LIMIT 1
        """, (game_id,)).fetchone()

        if last_snap:
            last_snap = dict(last_snap)

            # Check if line has moved
            line_moved = (
                last_snap['home_ml'] != bl['home_ml'] or
                last_snap['away_ml'] != bl['away_ml'] or
                last_snap['over_under'] != bl['over_under']
            )

            if not line_moved:
                # Check time since last snapshot
                try:
                    last_time = datetime.fromisoformat(last_snap['captured_at'])
                    minutes_since = (ct_now.replace(tzinfo=None) - last_time.replace(tzinfo=None)).total_seconds() / 60
                    if minutes_since < MIN_SNAPSHOT_INTERVAL_MINUTES:
                        skipped_recent += 1
                        continue
                except (ValueError, TypeError):
                    pass

                skipped_unchanged += 1
                continue

        # Insert periodic snapshot
        db.execute("""
            INSERT INTO betting_line_history
                (game_id, date, home_team_id, away_team_id, book,
                 home_ml, away_ml, over_under, over_odds, under_odds,
                 snapshot_type, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'periodic', ?)
        """, (game_id, today, bl['home_team_id'], bl['away_team_id'],
              bl['book'], bl['home_ml'], bl['away_ml'], bl['over_under'],
              bl['over_odds'], bl['under_odds'], now_str))
        captured += 1

    db.commit()
    if close_db:
        db.close()

    print(f"[{now_str}] Line tracker: {captured} snapshots, "
          f"{skipped_unchanged} unchanged, {skipped_recent} too recent")
    return captured, skipped_unchanged, skipped_recent


def get_line_movement(db=None, date_str=None):
    """Get line movement data for a date. Returns list of dicts with movement info."""
    close_db = False
    if db is None:
        db = get_db()
        close_db = True

    if date_str is None:
        date_str = get_ct_now().strftime('%Y-%m-%d')

    movements = []

    # Get all games with line history for the date
    games = db.execute("""
        SELECT DISTINCT blh.game_id,
               COALESCE(h.name, g.home_team_id) as home_name,
               COALESCE(a.name, g.away_team_id) as away_name,
               g.time, g.status
        FROM betting_line_history blh
        JOIN games g ON blh.game_id = g.id
        LEFT JOIN teams h ON g.home_team_id = h.id
        LEFT JOIN teams a ON g.away_team_id = a.id
        WHERE blh.date = ?
        ORDER BY g.time
    """, (date_str,)).fetchall()

    for game in games:
        game = dict(game)
        game_id = game['game_id']

        # Get all snapshots for this game, chronologically
        snaps = db.execute("""
            SELECT home_ml, away_ml, over_under, snapshot_type, captured_at
            FROM betting_line_history
            WHERE game_id = ? AND date = ?
            ORDER BY captured_at ASC
        """, (game_id, date_str)).fetchall()

        if len(snaps) < 1:
            continue

        first = dict(snaps[0])
        last = dict(snaps[-1])

        # Calculate movement
        home_ml_open = first['home_ml']
        home_ml_latest = last['home_ml']
        away_ml_open = first['away_ml']
        away_ml_latest = last['away_ml']

        home_prob_open = american_to_prob(home_ml_open)
        home_prob_latest = american_to_prob(home_ml_latest)

        prob_move = None
        if home_prob_open is not None and home_prob_latest is not None:
            prob_move = round((home_prob_latest - home_prob_open) * 100, 2)

        ou_open = first['over_under']
        ou_latest = last['over_under']
        ou_move = None
        if ou_open is not None and ou_latest is not None:
            ou_move = round(ou_latest - ou_open, 1)

        movements.append({
            'game_id': game_id,
            'home_name': game['home_name'],
            'away_name': game['away_name'],
            'time': game['time'],
            'status': game['status'],
            'snapshots': len(snaps),
            'home_ml_open': home_ml_open,
            'home_ml_latest': home_ml_latest,
            'away_ml_open': away_ml_open,
            'away_ml_latest': away_ml_latest,
            'home_prob_move': prob_move,
            'ou_open': ou_open,
            'ou_latest': ou_latest,
            'ou_move': ou_move,
        })

    if close_db:
        db.close()

    return movements


def get_game_line_history(db, game_id):
    """Get chronological line history snapshots for a single game.

    Returns list of dicts: {captured_at, home_ml, away_ml, home_prob, over_under, snapshot_type}
    """
    rows = db.execute("""
        SELECT home_ml, away_ml, over_under, snapshot_type, captured_at
        FROM betting_line_history
        WHERE game_id = ?
        ORDER BY captured_at ASC
    """, (game_id,)).fetchall()

    snapshots = []
    for r in rows:
        r = dict(r)
        snapshots.append({
            'captured_at': r['captured_at'],
            'home_ml': r['home_ml'],
            'away_ml': r['away_ml'],
            'home_prob': round(american_to_prob(r['home_ml']) * 100, 2) if american_to_prob(r['home_ml']) is not None else None,
            'over_under': r['over_under'],
            'snapshot_type': r['snapshot_type'],
        })
    return snapshots


def detect_steam_moves(db=None, date_str=None, threshold_pp=3.0, window_minutes=30):
    """Detect steam moves: large probability shifts within a short window.

    A steam move is a rapid line movement (>= threshold_pp percentage points)
    occurring within window_minutes, indicating sharp action.

    Returns list of {game_id, home_name, away_name, move_pp, window_min, direction}
    """
    close_db = False
    if db is None:
        db = get_db()
        close_db = True

    if date_str is None:
        date_str = get_ct_now().strftime('%Y-%m-%d')

    steam_moves = []

    games = db.execute("""
        SELECT DISTINCT blh.game_id,
               COALESCE(h.name, g.home_team_id) as home_name,
               COALESCE(a.name, g.away_team_id) as away_name
        FROM betting_line_history blh
        JOIN games g ON blh.game_id = g.id
        LEFT JOIN teams h ON g.home_team_id = h.id
        LEFT JOIN teams a ON g.away_team_id = a.id
        WHERE blh.date = ?
    """, (date_str,)).fetchall()

    for game in games:
        game = dict(game)
        game_id = game['game_id']

        snaps = db.execute("""
            SELECT home_ml, captured_at
            FROM betting_line_history
            WHERE game_id = ? AND date = ?
            ORDER BY captured_at ASC
        """, (game_id, date_str)).fetchall()

        if len(snaps) < 2:
            continue

        # Sliding window: check every pair (i, j) where j > i
        for i in range(len(snaps)):
            snap_i = dict(snaps[i])
            prob_i = american_to_prob(snap_i['home_ml'])
            if prob_i is None:
                continue
            try:
                time_i = datetime.fromisoformat(snap_i['captured_at'])
            except (ValueError, TypeError):
                continue

            for j in range(i + 1, len(snaps)):
                snap_j = dict(snaps[j])
                prob_j = american_to_prob(snap_j['home_ml'])
                if prob_j is None:
                    continue
                try:
                    time_j = datetime.fromisoformat(snap_j['captured_at'])
                except (ValueError, TypeError):
                    continue

                elapsed_min = (time_j - time_i).total_seconds() / 60
                if elapsed_min > window_minutes:
                    break  # No need to check further for this i

                move_pp = round((prob_j - prob_i) * 100, 2)
                if abs(move_pp) >= threshold_pp:
                    steam_moves.append({
                        'game_id': game_id,
                        'home_name': game['home_name'],
                        'away_name': game['away_name'],
                        'move_pp': move_pp,
                        'window_min': round(elapsed_min, 1),
                        'direction': 'home' if move_pp > 0 else 'away',
                    })
                    # Only report the first steam move per game
                    break
            else:
                continue
            break

    if close_db:
        db.close()

    return steam_moves


def print_summary(date_str=None):
    """Print line movement summary for a date."""
    db = get_db()
    if date_str is None:
        date_str = get_ct_now().strftime('%Y-%m-%d')

    movements = get_line_movement(db, date_str)

    total_snaps = db.execute(
        "SELECT COUNT(*) FROM betting_line_history WHERE date = ?",
        (date_str,)
    ).fetchone()[0]

    by_type = db.execute(
        "SELECT snapshot_type, COUNT(*) as cnt FROM betting_line_history WHERE date = ? GROUP BY snapshot_type",
        (date_str,)
    ).fetchall()

    db.close()

    print(f"\n=== Line Movement Summary — {date_str} ===")
    print(f"Total snapshots: {total_snaps}")
    for row in by_type:
        print(f"  {row['snapshot_type']}: {row['cnt']}")
    print(f"Games tracked: {len(movements)}")

    if movements:
        moved = [m for m in movements if m['home_prob_move'] and abs(m['home_prob_move']) > 0.5]
        print(f"Games with ML movement (>0.5pp): {len(moved)}")

        ou_moved = [m for m in movements if m['ou_move'] and abs(m['ou_move']) > 0]
        print(f"Games with O/U movement: {len(ou_moved)}")
    print()


def print_movers(date_str=None, top_n=5):
    """Print biggest line movers for a date."""
    movements = get_line_movement(date_str=date_str)

    if not movements:
        print("No line movement data available.")
        return

    # Sort by absolute probability movement
    with_moves = [m for m in movements if m['home_prob_move'] is not None]
    with_moves.sort(key=lambda m: abs(m['home_prob_move']), reverse=True)

    print(f"\n=== Top {top_n} Line Movers ===\n")
    for m in with_moves[:top_n]:
        direction = "→ HOME" if m['home_prob_move'] > 0 else "→ AWAY"
        print(f"  {m['away_name']} @ {m['home_name']} ({m['time'] or 'TBD'})")
        print(f"    Home ML: {m['home_ml_open']} → {m['home_ml_latest']} "
              f"({m['home_prob_move']:+.1f}pp {direction})")
        if m['ou_move'] and m['ou_move'] != 0:
            print(f"    O/U: {m['ou_open']} → {m['ou_latest']} ({m['ou_move']:+.1f})")
        print(f"    Snapshots: {m['snapshots']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Track betting line movement")
    parser.add_argument("--summary", action="store_true", help="Print movement summary")
    parser.add_argument("--movers", action="store_true", help="Show biggest movers")
    parser.add_argument("--top", type=int, default=5, help="Number of top movers to show")
    parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.date)
    elif args.movers:
        print_movers(args.date, args.top)
    else:
        snapshot_all_lines()


if __name__ == '__main__':
    main()
