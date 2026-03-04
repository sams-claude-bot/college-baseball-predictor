"""DB recording logic for selected bets."""

import json
import sqlite3

from .risk import RISK_ENGINE_MODE, american_to_prob

DB_PATH = 'data/baseball.db'


def build_parlay(results: dict) -> dict:
    """Build a 3-leg parlay from the day's highest-confidence picks.
    
    v2 strategy changes (2026-03-04):
    - Kept at 4 legs (more fun, bigger payouts)
    - Use meta_ensemble probability instead of old consensus (which was 38% accurate)
    - Require minimum 72% model probability per leg (was 62%)
    - Minimum 8% edge per leg (was 5%)
    - No totals legs (they're harder to predict and drag down parlay probability)
    - Score by meta_ensemble_prob * edge (higher = more confident + more value)
    """
    PARLAY_ML_CAP = -250       # Skip heavy favorites (juiced lines)
    PARLAY_MIN_PROB = 0.72     # Higher floor — each leg must be strong
    PARLAY_MAX_PROB = 0.92     # Cap to avoid overfit high-confidence busts
    PARLAY_MIN_EDGE = 8.0      # Meaningful edge required
    PARLAY_BET = 25
    PARLAY_LEGS = 4

    ml_candidates = []
    for b in results['bets']:
        if b['type'] not in ('CONSENSUS', 'ML'):
            continue
        ml = b.get('moneyline')
        if ml is None or ml < PARLAY_ML_CAP:
            continue
        # Use meta_ensemble prob if available, fall back to model_prob
        prob = b.get('meta_prob', b.get('model_prob', 0.5))
        if not (PARLAY_MIN_PROB <= prob <= PARLAY_MAX_PROB):
            continue
        if b.get('edge', 0) < PARLAY_MIN_EDGE:
            continue
        # Score: probability × edge — want legs that are both confident AND have value
        parlay_score = prob * b.get('edge', 0)
        ml_candidates.append({**b, 'parlay_score': parlay_score, 'parlay_prob': prob})
    ml_candidates.sort(key=lambda x: x['parlay_score'], reverse=True)

    # v2: ML-only legs, no totals (totals legs were unreliable in parlays)
    legs = []
    used_ids = set()
    for c in ml_candidates:
        if len(legs) >= PARLAY_LEGS:
            break
        if c['game_id'] not in used_ids:
            legs.append({
                'type': c['type'],
                'game_id': c['game_id'],
                'date': c.get('date', ''),
                'pick': c.get('pick_team_name', ''),
                'matchup': f"{c.get('opponent_name', '')} vs {c.get('pick_team_name', '')}",
                'odds': c['moneyline'],
                'prob': c.get('parlay_prob', c['model_prob']),
                'edge': c['edge'],
            })
            used_ids.add(c['game_id'])

    if len(legs) < 3:  # Need at least 3 legs; prefer 4
        return None

    def ml_to_decimal(ml):
        return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)

    decimal_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        decimal_odds *= ml_to_decimal(leg['odds'])
        combined_prob *= leg['prob']

    american = round((decimal_odds - 1) * 100) if decimal_odds > 2 else round(-100 / (decimal_odds - 1))

    return {
        'legs': legs,
        'num_legs': len(legs),
        'american_odds': american,
        'decimal_odds': round(decimal_odds, 4),
        'model_prob': round(combined_prob, 4),
        'bet_amount': PARLAY_BET,
        'payout': round(PARLAY_BET * decimal_odds, 2),
    }


def record_bets(results: dict):
    """Record recommended bets to database."""
    if not results['bets']:
        print("No bets to record")
        return

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    recorded = 0
    for bet in results['bets']:
        if bet['type'] == 'CONSENSUS':
            c.execute(
                '''
                INSERT OR IGNORE INTO tracked_confident_bets
                (game_id, date, pick_team_id, pick_team_name, opponent_name,
                 is_home, moneyline, models_agree, models_total, avg_prob,
                 confidence, bet_amount, risk_mode, risk_score, kelly_fraction_used, suggested_stake)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 10, ?, ?, ?, ?, ?, ?, ?)
            ''',
                (
                    bet['game_id'],
                    bet['date'],
                    bet['pick_team_id'],
                    bet['pick_team_name'],
                    bet['opponent_name'],
                    bet['is_home'],
                    bet['moneyline'],
                    bet['models_agree'],
                    bet['model_prob'],
                    bet['model_prob'],
                    bet['bet_amount'],
                    RISK_ENGINE_MODE,
                    bet.get('risk_score'),
                    bet.get('kelly_fraction_used'),
                    bet.get('suggested_stake', bet['bet_amount']),
                ),
            )
            if c.rowcount > 0:
                recorded += 1

        elif bet['type'] == 'ML':
            c.execute(
                '''
                INSERT OR IGNORE INTO tracked_bets
                (game_id, date, pick_team_id, pick_team_name, opponent_name,
                 is_home, moneyline, model_prob, dk_implied, edge, bet_amount,
                 risk_mode, risk_score, kelly_fraction_used, suggested_stake)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
                (
                    bet['game_id'],
                    bet['date'],
                    bet['pick_team_id'],
                    bet['pick_team_name'],
                    bet['opponent_name'],
                    bet['is_home'],
                    bet['moneyline'],
                    bet['model_prob'],
                    bet.get('dk_implied', american_to_prob(bet['moneyline'])),
                    bet['edge'],
                    bet['bet_amount'],
                    RISK_ENGINE_MODE,
                    bet.get('risk_score'),
                    bet.get('kelly_fraction_used'),
                    bet.get('suggested_stake', bet['bet_amount']),
                ),
            )
            if c.rowcount > 0:
                recorded += 1

        elif bet['type'] == 'TOTAL':
            c.execute(
                '''
                INSERT OR IGNORE INTO tracked_bets_spreads
                (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount,
                 risk_mode, risk_score, kelly_fraction_used, suggested_stake)
                VALUES (?, ?, 'total', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
                (
                    bet['game_id'],
                    bet['date'],
                    bet['pick'],
                    bet['line'],
                    bet['odds'],
                    bet['model_projection'],
                    bet['edge'],
                    bet['bet_amount'],
                    RISK_ENGINE_MODE,
                    bet.get('risk_score'),
                    bet.get('kelly_fraction_used'),
                    bet.get('suggested_stake', bet['bet_amount']),
                ),
            )
            if c.rowcount > 0:
                recorded += 1

    parlay = build_parlay(results)
    if parlay:
        date = results['date']
        c.execute(
            '''
            INSERT OR IGNORE INTO tracked_parlays
            (date, legs_json, num_legs, american_odds, decimal_odds, model_prob, bet_amount, payout)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
            (
                date,
                json.dumps(parlay['legs']),
                parlay['num_legs'],
                parlay['american_odds'],
                parlay['decimal_odds'],
                parlay['model_prob'],
                parlay['bet_amount'],
                parlay['payout'],
            ),
        )
        if c.rowcount > 0:
            recorded += 1
            print(
                f"\n🎰 Parlay recorded: {parlay['num_legs']} legs, +{parlay['american_odds']}, "
                f"${parlay['bet_amount']} to win ${parlay['payout']}"
            )

    conn.commit()
    conn.close()

    print(f"\n✅ Recorded {recorded} bets")

