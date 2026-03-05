#!/usr/bin/env python3
"""
Backfill predictions for new models (venue, rest_travel, upset)
on all graded games so the meta-ensemble has training data.
"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.venue_model import VenueModel
from models.rest_travel_model import RestTravelModel
from models.upset_model import UpsetModel


def backfill(model_name, model_cls, batch_size=100):
    """Backfill predictions for a single model on all graded games."""
    conn = get_connection()
    c = conn.cursor()

    # Get all completed games that DON'T have this model's prediction yet
    c.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.date
        FROM games g
        WHERE g.status = 'final'
        AND g.id NOT IN (
            SELECT game_id FROM model_predictions WHERE model_name = ?
        )
        ORDER BY g.date
    """, (model_name,))
    games = c.fetchall()

    if not games:
        print(f"  {model_name}: All games already have predictions")
        return 0

    print(f"  {model_name}: Backfilling {len(games)} games...")
    model = model_cls()
    success = 0
    errors = 0

    for i, (game_id, home_id, away_id, date) in enumerate(games):
        try:
            result = model.predict_game(home_id, away_id)
            home_prob = result.get('home_win_probability', 0.5)
            home_prob = min(max(home_prob, 0.001), 0.999)

            c.execute("""
                INSERT INTO model_predictions
                (game_id, model_name, predicted_home_prob, raw_home_prob, prediction_source, prediction_context)
                VALUES (?, ?, ?, ?, 'backfill', 'backfill_new_models')
                ON CONFLICT(game_id, model_name) DO UPDATE SET
                    predicted_home_prob = excluded.predicted_home_prob,
                    raw_home_prob = excluded.raw_home_prob,
                    prediction_source = excluded.prediction_source,
                    prediction_context = excluded.prediction_context,
                    predicted_at = CURRENT_TIMESTAMP
            """, (game_id, model_name, home_prob, home_prob))
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error on {game_id}: {e}")

        if (i + 1) % batch_size == 0:
            conn.commit()
            print(f"    {i+1}/{len(games)} done ({success} ok, {errors} errors)")

    conn.commit()
    conn.close()
    print(f"  {model_name}: Done — {success} predictions, {errors} errors")
    return success


def main():
    print("Backfilling new model predictions for all graded games...\n")

    models = [
        ('venue', VenueModel),
        ('rest_travel', RestTravelModel),
        ('upset', UpsetModel),
    ]

    total = 0
    for name, cls in models:
        total += backfill(name, cls)

    print(f"\nTotal backfilled: {total} predictions")

    # Also backfill scheduled games (for today's predictions)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.date
        FROM games g
        WHERE g.status = 'scheduled' AND g.date = date('now')
        AND g.id NOT IN (
            SELECT game_id FROM model_predictions WHERE model_name = 'venue'
        )
    """)
    today_games = c.fetchall()
    conn.close()

    if today_games:
        print(f"\nBackfilling {len(today_games)} scheduled games for today...")
        for name, cls in models:
            model = cls()
            conn = get_connection()
            c = conn.cursor()
            for game_id, home_id, away_id, date in today_games:
                try:
                    result = model.predict_game(home_id, away_id)
                    home_prob = min(max(result.get('home_win_probability', 0.5), 0.001), 0.999)
                    c.execute("""
                        INSERT INTO model_predictions
                        (game_id, model_name, predicted_home_prob, raw_home_prob, prediction_source, prediction_context)
                        VALUES (?, ?, ?, ?, 'backfill', 'backfill_new_models')
                        ON CONFLICT(game_id, model_name) DO UPDATE SET
                            predicted_home_prob = excluded.predicted_home_prob,
                            raw_home_prob = excluded.raw_home_prob,
                            prediction_source = excluded.prediction_source,
                            prediction_context = excluded.prediction_context,
                            predicted_at = CURRENT_TIMESTAMP
                    """, (game_id, name, home_prob, home_prob))
                except Exception:
                    pass
            conn.commit()
            conn.close()


if __name__ == '__main__':
    main()
