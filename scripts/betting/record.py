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

    # Use pre-correlation-cap bets if available (parlay is flat $25, independent of Kelly sizing)
    source_bets = results.get('parlay_candidates', results['bets'])
    ml_candidates = []
    for b in source_bets:
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


def build_longshot_parlay(date_str=None) -> dict:
    """Build an untracked longshot parlay — highest probability at +1500 or better.

    Pulls from ALL games with betting lines and model predictions (not just
    bet-selected games). Finds the combination of legs that maximizes model
    probability while achieving at least +1500 combined odds.

    Rules:
    - Minimum combined American odds: +1500
    - Uses meta_ensemble probabilities for all games
    - Can mix favorites and underdogs from any game
    - Picks the side (home/away) with better model edge per game
    - 3-6 legs
    - $5 fun bet, completely untracked
    """
    import sqlite3
    from itertools import combinations
    from datetime import datetime, timezone, timedelta

    MIN_COMBINED_DECIMAL = 16.0  # +1500 American = 16.0 decimal
    MIN_LEGS = 3
    MAX_LEGS = 6
    LONGSHOT_BET = 5
    MAX_FAVORITE = -400  # Don't include super-heavy favorites (boring legs)
    MIN_MODEL_PROB = 0.35  # Don't include picks the model hates

    if date_str is None:
        utc_now = datetime.now(timezone.utc)
        ct_offset = timedelta(hours=-5) if 3 <= utc_now.month <= 10 else timedelta(hours=-6)
        date_str = (utc_now + ct_offset).strftime('%Y-%m-%d')

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row

    # Get all games with odds AND model predictions
    rows = conn.execute("""
        SELECT g.id as game_id, g.date, g.status,
               bl.home_ml, bl.away_ml,
               h.name as home_name, a.name as away_name,
               g.home_team_id, g.away_team_id,
               mp.predicted_home_prob as meta_prob
        FROM games g
        JOIN betting_lines bl ON g.id = bl.game_id
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        LEFT JOIN model_predictions mp ON g.id = mp.game_id AND mp.model_name = 'meta_ensemble'
        WHERE g.date = ? AND g.status = 'scheduled'
        AND bl.home_ml IS NOT NULL AND bl.away_ml IS NOT NULL
    """, (date_str,)).fetchall()
    conn.close()

    if not rows:
        return None

    def ml_to_decimal(ml):
        return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)

    def american_to_prob(ml):
        if ml > 0:
            return 100 / (ml + 100)
        return abs(ml) / (abs(ml) + 100)

    # Build candidate legs: for each game, pick the side with better model edge
    candidates = []
    for r in rows:
        r = dict(r)
        meta_prob = r['meta_prob']
        if meta_prob is None:
            continue

        home_ml = r['home_ml']
        away_ml = r['away_ml']

        # Home side
        if home_ml >= MAX_FAVORITE and meta_prob >= MIN_MODEL_PROB:
            implied = american_to_prob(home_ml)
            edge = (meta_prob - implied) * 100
            candidates.append({
                'game_id': r['game_id'],
                'date': r['date'],
                'pick': r['home_name'],
                'opponent': r['away_name'],
                'matchup': f"{r['away_name']} @ {r['home_name']}",
                'odds': home_ml,
                'decimal_odds': ml_to_decimal(home_ml),
                'prob': meta_prob,
                'edge': edge,
                'is_dog': home_ml > 0,
                'side': 'home',
            })

        # Away side
        away_prob = 1 - meta_prob
        if away_ml >= MAX_FAVORITE and away_prob >= MIN_MODEL_PROB:
            implied = american_to_prob(away_ml)
            edge = (away_prob - implied) * 100
            candidates.append({
                'game_id': r['game_id'],
                'date': r['date'],
                'pick': r['away_name'],
                'opponent': r['home_name'],
                'matchup': f"{r['away_name']} @ {r['home_name']}",
                'odds': away_ml,
                'decimal_odds': ml_to_decimal(away_ml),
                'prob': away_prob,
                'edge': edge,
                'is_dog': away_ml > 0,
                'side': 'away',
            })

    if len(candidates) < MIN_LEGS:
        return None

    # Sort by probability (greedy: start with highest confidence picks)
    candidates.sort(key=lambda x: x['prob'], reverse=True)

    # Only keep the best side per game
    seen_games = set()
    unique_candidates = []
    for c in candidates:
        if c['game_id'] not in seen_games:
            unique_candidates.append(c)
            seen_games.add(c['game_id'])

    # Also keep high-value underdogs that might not be the "best side"
    for c in candidates:
        if c['game_id'] not in {u['game_id'] for u in unique_candidates} or c['is_dog']:
            if c['game_id'] not in {u['game_id'] for u in unique_candidates if u['side'] == c['side']}:
                unique_candidates.append(c)

    # Dedupe by game_id + side
    seen = set()
    deduped = []
    for c in unique_candidates:
        key = (c['game_id'], c['side'])
        if key not in seen:
            deduped.append(c)
            seen.add(key)

    # Greedy search: find best parlay at each leg count
    best_parlay = None
    best_prob = 0

    # For efficiency, limit candidate pool to top ~20 by prob for combo search
    top_candidates = sorted(deduped, key=lambda x: x['prob'], reverse=True)[:20]

    for n_legs in range(MIN_LEGS, min(MAX_LEGS + 1, len(top_candidates) + 1)):
        for combo in combinations(top_candidates, n_legs):
            # Skip if same game appears twice
            game_ids = [c['game_id'] for c in combo]
            if len(set(game_ids)) != len(game_ids):
                continue

            combined_decimal = 1.0
            combined_prob = 1.0
            for leg in combo:
                combined_decimal *= leg['decimal_odds']
                combined_prob *= leg['prob']

            if combined_decimal >= MIN_COMBINED_DECIMAL and combined_prob > best_prob:
                best_prob = combined_prob
                best_parlay = combo

    if best_parlay is None:
        return None

    # Format result
    legs = []
    decimal_odds = 1.0
    combined_prob = 1.0
    for c in best_parlay:
        decimal_odds *= c['decimal_odds']
        combined_prob *= c['prob']
        legs.append({
            'game_id': c['game_id'],
            'date': c['date'],
            'pick': c['pick'],
            'opponent': c['opponent'],
            'matchup': c['matchup'],
            'odds': c['odds'],
            'prob': round(c['prob'], 4),
            'edge': round(c['edge'], 1),
            'is_dog': c['is_dog'],
        })

    american = round((decimal_odds - 1) * 100) if decimal_odds > 2 else round(-100 / (decimal_odds - 1))

    return {
        'legs': legs,
        'num_legs': len(legs),
        'american_odds': american,
        'decimal_odds': round(decimal_odds, 4),
        'model_prob': round(combined_prob, 4),
        'bet_amount': LONGSHOT_BET,
        'payout': round(LONGSHOT_BET * decimal_odds, 2),
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

    # Claude's Parlay (longshot, tracked separately)
    longshot = build_longshot_parlay(results['date'])
    if longshot:
        c.execute(
            '''
            INSERT OR IGNORE INTO tracked_longshot_parlays
            (date, legs_json, num_legs, american_odds, decimal_odds, model_prob, bet_amount, payout)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
            (
                results['date'],
                json.dumps(longshot['legs']),
                longshot['num_legs'],
                longshot['american_odds'],
                longshot['decimal_odds'],
                longshot['model_prob'],
                longshot['bet_amount'],
                longshot['payout'],
            ),
        )
        if c.rowcount > 0:
            recorded += 1
            print(
                f"\n🔥 Claude's Parlay: {longshot['num_legs']} legs, +{longshot['american_odds']}, "
                f"${longshot['bet_amount']} to win ${longshot['payout']} ({longshot['model_prob']*100:.1f}% model)"
            )

    conn.commit()
    conn.close()

    print(f"\n✅ Recorded {recorded} bets")


def grade_longshot_parlays():
    """Grade longshot parlays for completed dates."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get ungraded longshot parlays
    ungraded = c.execute(
        "SELECT id, date, legs_json FROM tracked_longshot_parlays WHERE won IS NULL"
    ).fetchall()

    graded = 0
    for parlay in ungraded:
        parlay = dict(parlay)
        legs = json.loads(parlay['legs_json'])
        legs_won = 0
        legs_lost = 0
        all_graded = True

        for leg in legs:
            game = c.execute(
                "SELECT status, home_score, away_score, home_team_id, away_team_id, winner_id FROM games WHERE id = ?",
                (leg['game_id'],)
            ).fetchone()

            if not game or game['status'] != 'final':
                all_graded = False
                break

            game = dict(game)
            # Determine if our pick won
            if leg.get('side') == 'home' or leg['pick'] in (game.get('home_team_id', ''), ''):
                # Check by comparing pick name to winner
                pass

            winner = game['winner_id']
            # Match pick to team — the pick name might not match team_id exactly
            # Use the side stored in the leg, or match by game teams
            pick_is_home = leg.get('side') == 'home'
            if pick_is_home is None:
                # Fallback: try matching pick name
                pick_is_home = leg['pick'].lower().replace(' ', '-') in (game['home_team_id'] or '').lower()

            picked_team = game['home_team_id'] if pick_is_home else game['away_team_id']
            if winner == picked_team:
                legs_won += 1
            else:
                legs_lost += 1

        if all_graded:
            won = 1 if legs_won == len(legs) else 0
            c.execute(
                "UPDATE tracked_longshot_parlays SET legs_won=?, legs_lost=?, won=?, graded_at=datetime('now') WHERE id=?",
                (legs_won, legs_lost, won, parlay['id'])
            )
            graded += 1
            result = "✅ HIT" if won else f"❌ {legs_won}/{len(legs)}"
            print(f"Claude's Parlay {parlay['date']}: {result}")

    conn.commit()
    conn.close()
    return graded

