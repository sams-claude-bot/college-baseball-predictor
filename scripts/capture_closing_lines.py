#!/usr/bin/env python3
"""Capture closing lines for games about to start.

Runs via cron every 15 minutes. For each game starting in the next 30-60 minutes,
saves the current betting_lines snapshot as the 'closing' line in betting_line_history
and updates tracked bets with closing_ml for CLV calculation.

Usage:
    python3 scripts/capture_closing_lines.py
"""
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'


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


def compute_clv(opening_ml, closing_ml):
    """Compute CLV in implied probability points.

    Positive CLV = you got a better price than the closing line.
    """
    if opening_ml is None or closing_ml is None:
        return None, None

    def american_to_prob(ml):
        ml = float(ml)
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)

    opening_prob = american_to_prob(opening_ml)
    closing_prob = american_to_prob(closing_ml)
    clv_implied = closing_prob - opening_prob
    # CLV in "cents" — same as implied but expressed in percentage points
    clv_cents = round(clv_implied * 100, 2)
    return round(clv_implied, 6), clv_cents


def capture_closing_lines():
    """Find games starting soon and capture their closing lines."""
    db = get_db()
    ct_now = get_ct_now()
    today = ct_now.strftime('%Y-%m-%d')

    # Find games starting in the next 30-60 minutes
    # Games table stores time as HH:MM or HH:MM AM/PM format
    games = db.execute("""
        SELECT g.id, g.date, g.time, g.home_team_id, g.away_team_id, g.status
        FROM games g
        WHERE g.date = ? AND g.status = 'scheduled' AND g.time IS NOT NULL
    """, (today,)).fetchall()

    captured = 0
    skipped = 0
    clv_updated = 0

    for game in games:
        game = dict(game)
        game_time_str = game['time']
        if not game_time_str:
            skipped += 1
            continue

        # Parse game time (formats: "7:00 PM", "19:00", "7:00p", etc.)
        game_dt = _parse_game_time(today, game_time_str)
        if game_dt is None:
            skipped += 1
            continue

        # Only capture for games starting in 0-60 minutes
        minutes_until = (game_dt - ct_now.replace(tzinfo=None)).total_seconds() / 60
        if minutes_until < 0 or minutes_until > 60:
            continue

        game_id = game['id']

        # Check if we already captured closing line for this game
        existing = db.execute(
            "SELECT id FROM betting_line_history WHERE game_id = ? AND snapshot_type = 'closing'",
            (game_id,)
        ).fetchone()
        if existing:
            continue

        # Get current betting_lines (most recent snapshot)
        bl = db.execute("""
            SELECT home_ml, away_ml, over_under, over_odds, under_odds
            FROM betting_lines
            WHERE game_id = ? AND book = 'draftkings'
            ORDER BY captured_at DESC LIMIT 1
        """, (game_id,)).fetchone()

        if not bl or (bl['home_ml'] is None and bl['away_ml'] is None):
            skipped += 1
            continue

        # Insert closing snapshot
        db.execute("""
            INSERT INTO betting_line_history
                (game_id, date, home_team_id, away_team_id, book,
                 home_ml, away_ml, over_under, over_odds, under_odds,
                 snapshot_type)
            VALUES (?, ?, ?, ?, 'draftkings', ?, ?, ?, ?, ?, 'closing')
        """, (game_id, today, game['home_team_id'], game['away_team_id'],
              bl['home_ml'], bl['away_ml'], bl['over_under'],
              bl['over_odds'], bl['under_odds']))
        captured += 1

        # Update tracked_bets with closing_ml and CLV
        clv_updated += _update_bet_clv(db, game_id, game['home_team_id'],
                                        bl['home_ml'], bl['away_ml'])

    db.commit()
    db.close()
    print(f"Closing lines: {captured} captured, {skipped} skipped, {clv_updated} bets updated with CLV")
    return captured, skipped, clv_updated


def _parse_game_time(date_str, time_str):
    """Parse a game time string into a datetime. Returns None on failure."""
    if not time_str:
        return None
    time_str = time_str.strip()

    # Try common formats
    for fmt in ('%I:%M %p', '%H:%M', '%I:%M%p', '%I %p'):
        try:
            t = datetime.strptime(time_str, fmt)
            d = datetime.strptime(date_str, '%Y-%m-%d')
            return d.replace(hour=t.hour, minute=t.minute)
        except ValueError:
            continue

    return None


def _update_bet_clv(db, game_id, home_team_id, closing_home_ml, closing_away_ml):
    """Update CLV columns for tracked bets on this game. Returns count updated."""
    updated = 0

    for table in ('tracked_bets', 'tracked_confident_bets'):
        rows = db.execute(
            f"SELECT id, pick_team_id, moneyline, is_home FROM {table} WHERE game_id = ? AND closing_ml IS NULL",
            (game_id,)
        ).fetchall()

        for row in rows:
            row = dict(row)
            # Determine closing ML for the side we picked
            picked_home = row['pick_team_id'] == home_team_id or row['is_home'] == 1
            closing_ml = closing_home_ml if picked_home else closing_away_ml

            if closing_ml is None:
                continue

            opening_ml = row['moneyline']
            clv_implied, clv_cents = compute_clv(opening_ml, closing_ml)

            db.execute(
                f"UPDATE {table} SET closing_ml = ?, clv_implied = ?, clv_cents = ? WHERE id = ?",
                (closing_ml, clv_implied, clv_cents, row['id'])
            )
            updated += 1

    return updated


if __name__ == '__main__':
    capture_closing_lines()
