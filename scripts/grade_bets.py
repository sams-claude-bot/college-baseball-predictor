#!/usr/bin/env python3
"""
Grade all ungraded bets (EV, consensus, parlays, totals) against final game results.

Run daily after games finish, or on demand.
Usage: python3 scripts/grade_bets.py [--dry-run]
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def find_game(conn, game_id):
    """Find a game by ID, with fallback for reversed team order in game_id."""
    row = conn.execute(
        "SELECT id, status, winner_id, home_team_id, away_team_id, "
        "home_score, away_score FROM games WHERE id = ?",
        (game_id,)
    ).fetchone()
    if row:
        return dict(row)

    # Try reversed team order: swap the two team slugs in the game_id
    # Format: YYYY-MM-DD_away_home → try YYYY-MM-DD_home_away
    parts = game_id.split('_', 1)
    if len(parts) == 2:
        date_part = parts[0]
        rest = parts[1]
        # Find the teams by looking for matching date + teams
        teams = rest.split('_')
        if len(teams) >= 2:
            # Try all permutations (handles multi-word team slugs)
            reversed_id = f"{date_part}_{'_'.join(reversed(teams))}"
            row = conn.execute(
                "SELECT id, status, winner_id, home_team_id, away_team_id, "
                "home_score, away_score FROM games WHERE id = ?",
                (reversed_id,)
            ).fetchone()
            if row:
                return dict(row)

    # Last resort: fuzzy match on date + teams
    date_prefix = game_id[:10]
    candidates = conn.execute(
        "SELECT id, status, winner_id, home_team_id, away_team_id, "
        "home_score, away_score FROM games WHERE date = ?",
        (date_prefix,)
    ).fetchall()
    
    # Extract team slugs from the game_id
    slug_part = game_id[11:]  # everything after YYYY-MM-DD_
    for c in candidates:
        c = dict(c)
        if c['home_team_id'] in slug_part and c['away_team_id'] in slug_part:
            return c

    return None


def ml_profit(bet_amount, moneyline, won):
    """Calculate profit for a moneyline bet."""
    if won:
        if moneyline > 0:
            return bet_amount * (moneyline / 100)
        else:
            return bet_amount * (100 / abs(moneyline))
    else:
        return -bet_amount


def grade_ev_bets(conn, dry_run=False):
    """Grade tracked_bets (EV bets)."""
    rows = conn.execute(
        "SELECT id, game_id, pick_team_id, moneyline, bet_amount "
        "FROM tracked_bets WHERE won IS NULL"
    ).fetchall()

    graded = 0
    for r in rows:
        game = find_game(conn, r['game_id'])
        if not game or game['status'] != 'final' or not game['winner_id']:
            continue

        won = 1 if game['winner_id'] == r['pick_team_id'] else 0
        profit = ml_profit(r['bet_amount'], r['moneyline'], won)

        print(f"  EV #{r['id']}: {r['pick_team_id']} → {'WON' if won else 'LOST'} (${profit:+.2f})")

        if not dry_run:
            conn.execute(
                "UPDATE tracked_bets SET won = ?, profit = ? WHERE id = ?",
                (won, round(profit, 2), r['id'])
            )
        graded += 1

    return graded


def grade_consensus_bets(conn, dry_run=False):
    """Grade tracked_confident_bets (consensus bets)."""
    rows = conn.execute(
        "SELECT id, game_id, pick_team_id, moneyline, bet_amount "
        "FROM tracked_confident_bets WHERE won IS NULL"
    ).fetchall()

    graded = 0
    for r in rows:
        game = find_game(conn, r['game_id'])
        if not game or game['status'] != 'final' or not game['winner_id']:
            continue

        won = 1 if game['winner_id'] == r['pick_team_id'] else 0
        profit = ml_profit(r['bet_amount'], r['moneyline'], won)

        print(f"  Consensus #{r['id']}: {r['pick_team_id']} → {'WON' if won else 'LOST'} (${profit:+.2f})")

        if not dry_run:
            conn.execute(
                "UPDATE tracked_confident_bets SET won = ?, profit = ? WHERE id = ?",
                (won, round(profit, 2), r['id'])
            )
        graded += 1

    return graded


def resolve_pick_team_id(conn, pick_name, game):
    """Resolve a team name to a team_id using the game's home/away teams."""
    pick_lower = pick_name.lower().strip()

    # Check home team
    home_row = conn.execute("SELECT id, name FROM teams WHERE id = ?", (game['home_team_id'],)).fetchone()
    if home_row and pick_lower in home_row['name'].lower():
        return game['home_team_id']

    # Check away team
    away_row = conn.execute("SELECT id, name FROM teams WHERE id = ?", (game['away_team_id'],)).fetchone()
    if away_row and pick_lower in away_row['name'].lower():
        return game['away_team_id']

    # Try matching the slug directly
    if pick_lower.replace(' ', '-') == game['home_team_id']:
        return game['home_team_id']
    if pick_lower.replace(' ', '-') == game['away_team_id']:
        return game['away_team_id']

    # Fuzzy: pick name appears in team ID
    if pick_lower.replace(' ', '-') in game['home_team_id']:
        return game['home_team_id']
    if pick_lower.replace(' ', '-') in game['away_team_id']:
        return game['away_team_id']

    return None


def grade_parlays(conn, dry_run=False):
    """Grade tracked_parlays."""
    rows = conn.execute(
        "SELECT id, date, legs_json, american_odds, bet_amount, payout "
        "FROM tracked_parlays WHERE won IS NULL"
    ).fetchall()

    graded = 0
    for r in rows:
        legs = json.loads(r['legs_json'])
        all_final = True
        legs_won = 0
        legs_lost = 0
        leg_results = []

        for leg in legs:
            game = find_game(conn, leg.get('game_id', ''))
            if not game or game['status'] != 'final':
                all_final = False
                leg_results.append('pending')
                continue

            leg_type = leg.get('type', 'ML')
            pick_name = leg.get('pick', leg.get('pick_label', ''))

            if leg_type in ('ML', 'CONSENSUS'):
                # Resolve pick to team_id
                pick_team_id = leg.get('pick_team') or resolve_pick_team_id(conn, pick_name, game)
                if not pick_team_id:
                    print(f"  ⚠️  Can't resolve pick '{pick_name}' for game {leg.get('game_id')}")
                    all_final = False
                    leg_results.append('unknown')
                    continue

                won = game['winner_id'] == pick_team_id
                if won:
                    legs_won += 1
                    leg_results.append('won')
                else:
                    legs_lost += 1
                    leg_results.append('lost')

            elif leg_type == 'Total':
                total = (game['home_score'] or 0) + (game['away_score'] or 0)
                label = pick_name.upper()
                if 'OVER' in label:
                    line = float(label.split()[-1])
                    won = total > line
                elif 'UNDER' in label:
                    line = float(label.split()[-1])
                    won = total < line
                else:
                    won = False

                if won:
                    legs_won += 1
                    leg_results.append('won')
                else:
                    legs_lost += 1
                    leg_results.append('lost')

        if not all_final:
            continue

        parlay_won = legs_lost == 0
        if parlay_won:
            profit = round(r['payout'] - r['bet_amount'], 2)
        else:
            profit = round(-r['bet_amount'], 2)

        result_str = f"{'WON' if parlay_won else 'LOST'} ({legs_won}/{len(legs)} legs)"
        print(f"  Parlay #{r['id']} ({r['date']}): {result_str} (${profit:+.2f})")
        for i, (leg, res) in enumerate(zip(legs, leg_results)):
            icon = '✅' if res == 'won' else '❌'
            print(f"    {icon} {leg.get('pick', '?')} ({leg.get('matchup', '')})")

        if not dry_run:
            conn.execute(
                "UPDATE tracked_parlays SET won = ?, profit = ?, legs_won = ?, updated_at = ? WHERE id = ?",
                (1 if parlay_won else 0, profit, legs_won, datetime.utcnow().isoformat(), r['id'])
            )
        graded += 1

    return graded


def main():
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("DRY RUN — no database changes\n")

    conn = get_connection()

    print("Grading EV bets...")
    ev = grade_ev_bets(conn, dry_run)

    print("\nGrading consensus bets...")
    cons = grade_consensus_bets(conn, dry_run)

    print("\nGrading parlays...")
    par = grade_parlays(conn, dry_run)

    if not dry_run:
        conn.commit()

    conn.close()

    total = ev + cons + par
    print(f"\nDone: {total} bets graded ({ev} EV, {cons} consensus, {par} parlays)")


if __name__ == '__main__':
    main()
