"""Data assembly for the /betting page."""

import sys
from pathlib import Path

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from config import model_config as cfg
from bet_selection_v2 import analyze_games

from web.helpers import get_all_conferences, get_betting_games


def build_betting_page_context(conference=''):
    """Build template context for the betting analysis page."""
    conferences = get_all_conferences()
    games = get_betting_games()

    # Experimental risk engine preview (opt-in via ?view=experimental)
    risk_preview = None
    risk_engine = {
        'mode': getattr(cfg, 'BET_RISK_ENGINE_MODE', 'fixed'),
        'bankroll': getattr(cfg, 'BET_RISK_BANKROLL', 5000.0),
        'kelly_fraction': getattr(cfg, 'BET_RISK_KELLY_FRACTION', 0.25),
        'min_stake': getattr(cfg, 'BET_RISK_MIN_STAKE', 25.0),
        'max_stake': getattr(cfg, 'BET_RISK_MAX_STAKE', 250.0),
    }
    try:
        preview = analyze_games()
        risk_preview = {
            'date': preview.get('date'),
            'bets': preview.get('bets', [])[:8],
            'rejections': len(preview.get('rejections', [])),
            'error': preview.get('error'),
        }
    except Exception as e:
        risk_preview = {'date': None, 'bets': [], 'rejections': 0, 'error': str(e)}

    # === v2 THRESHOLDS ===
    ML_EDGE_FAVORITE = 8.0
    ML_EDGE_UNDERDOG = 15.0
    UNDERDOG_DISCOUNT = 0.5
    CONSENSUS_BONUS_PER_MODEL = 1.0
    TOTALS_EDGE_RUNS = 3.0
    SPREADS_ENABLED = False

    def calc_adjusted_edge(raw_edge, ml, models_agree=5):
        adj = raw_edge
        if ml and ml > 0:  # Underdog discount
            adj = raw_edge * UNDERDOG_DISCOUNT
        bonus = max(0, (models_agree - 5)) * CONSENSUS_BONUS_PER_MODEL
        return adj + bonus

    # Filter by conference if specified
    if conference:
        games = [
            g for g in games
            if g.get('home_conf') == conference or g.get('away_conf') == conference
        ]

    # Build consensus lookup
    consensus_lookup = {}
    for g in games:
        if g.get('model_agreement') and g['model_agreement']['count'] >= 7:
            consensus_lookup[g['game_id']] = g['model_agreement']['count']

    def classify_bucket(game):
        """Classify game into confidence buckets based on pick-side probability."""
        pick_side = game.get('best_pick')

        # Fallback to consensus pick if best_pick isn't set
        if not pick_side and game.get('model_agreement'):
            pick_side = game['model_agreement'].get('pick')

        model_home_prob = game.get('model_home_prob')
        if model_home_prob is None:
            return None

        pick_prob = model_home_prob if pick_side != 'away' else (1 - model_home_prob)

        if pick_prob >= 0.70:
            return {'key': 'heavy_favorite', 'label': 'Heavy Favorite', 'color': 'success', 'prob': pick_prob}
        if pick_prob >= 0.60:
            return {'key': 'moderate_edge', 'label': 'Moderate Edge', 'color': 'primary', 'prob': pick_prob}
        return {'key': 'coin_flip', 'label': 'Coin Flip', 'color': 'secondary', 'prob': pick_prob}

    # Add adjusted_edge + bucket to all games
    for g in games:
        if g.get('best_edge'):
            ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
            models = consensus_lookup.get(g['game_id'], 5)
            g['adjusted_edge'] = calc_adjusted_edge(g['best_edge'], ml, models)
            g['models_agree'] = models
            g['is_underdog'] = ml > 0 if ml else False

        g['bucket'] = classify_bucket(g)

    # Sort by adjusted edge
    games_with_edge = [g for g in games if g.get('adjusted_edge')]
    games_with_edge.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)

    # Confident bets (7/10+ models agree, sorted by adjusted edge)
    # Cap at -300: heavy favorites aren't worth betting even with consensus
    MAX_FAVORITE_ML = -300

    def passes_favorite_cap(g):
        ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
        return ml is None or ml >= MAX_FAVORITE_ML

    confident_candidates = [
        g for g in games
        if g.get('model_agreement')
        and g['model_agreement']['count'] >= 7
        and passes_favorite_cap(g)
    ]
    confident_candidates.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)
    confident_bets = confident_candidates[:6]

    # EV bets — pure raw edge over DK line (exclude consensus picks)
    confident_ids = {g['game_id'] for g in confident_bets}
    ev_candidates = [
        g for g in games_with_edge
        if g['game_id'] not in confident_ids
        and g.get('best_edge', 0) >= ML_EDGE_FAVORITE
        and passes_favorite_cap(g)
    ]
    ev_candidates.sort(key=lambda x: x.get('best_edge', 0), reverse=True)
    ev_bets = ev_candidates[:6]

    # Best totals (3+ runs edge)
    games_with_totals = [g for g in games if g.get('over_under')]
    games_with_totals.sort(key=lambda x: abs(x.get('total_diff', 0)), reverse=True)
    best_totals = [g for g in games_with_totals if abs(g.get('total_diff', 0)) >= TOTALS_EDGE_RUNS]

    # === 4-LEG PARLAY BUILDER ===
    # Mix of ML + Totals picks. Sweet spot: ~80% confidence, -200 range, good edge.
    # Avoid heavy favorites (-300 or worse) — they kill parlay value.
    PARLAY_ML_CAP = -250  # Max favorite ML for parlay legs
    PARLAY_MIN_PROB = 0.62  # Min model confidence
    PARLAY_MAX_PROB = 0.88  # Max — avoid near-locks (low payout)
    PARLAY_MIN_EDGE = 5.0   # Min edge over line

    # ML candidates for parlay
    parlay_ml_candidates = []
    for g in games:
        if not g.get('best_edge') or g['best_edge'] < PARLAY_MIN_EDGE:
            continue
        ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
        if ml is None:
            continue
        # Skip heavy favorites and big underdogs
        if ml < PARLAY_ML_CAP:
            continue
        # Get model probability for the pick side
        prob = g.get('model_home_prob', 0.5)
        if g.get('best_pick') == 'away':
            prob = 1 - prob
        if not (PARLAY_MIN_PROB <= prob <= PARLAY_MAX_PROB):
            continue
        parlay_ml_candidates.append({
            'game': g,
            'type': 'ML',
            'pick_team': g.get('home_team_name') if g.get('best_pick') == 'home' else g.get('away_team_name'),
            'opponent': g.get('away_team_name') if g.get('best_pick') == 'home' else g.get('home_team_name'),
            'pick_label': g.get('home_team_name') if g.get('best_pick') == 'home' else g.get('away_team_name'),
            'odds': ml,
            'prob': prob,
            'edge': g['best_edge'],
            'models_agree': g.get('models_agree', 0),
            'game_id': g['game_id'],
            'matchup': f"{g.get('away_team_name')} @ {g.get('home_team_name')}",
        })
    # Sort by a blend of edge and probability (sweet spot scoring)
    # Prefer ~75-80% range with good edge
    for c in parlay_ml_candidates:
        # Score: penalize extremes, reward ~0.75 prob with high edge
        prob_score = 1.0 - abs(c['prob'] - 0.77) * 3  # Peak at 77%
        c['parlay_score'] = prob_score * c['edge']
    parlay_ml_candidates.sort(key=lambda x: x['parlay_score'], reverse=True)

    # Totals candidates for parlay
    parlay_totals_candidates = []
    for g in games:
        if not g.get('over_under') or not g.get('total_diff'):
            continue
        if abs(g['total_diff']) < 2.0:  # At least 2 runs projected difference
            continue
        total_edge_pct = g.get('total_edge', 0)
        if total_edge_pct < 15:
            continue
        parlay_totals_candidates.append({
            'game': g,
            'type': 'Total',
            'pick_label': f"{g['total_lean']} {g['over_under']}",
            'pick_team': g['total_lean'],
            'odds': -110,  # Standard totals juice
            'prob': min(0.5 + abs(g['total_diff']) * 0.06, 0.85),  # Rough estimate
            'edge': total_edge_pct,
            'total_diff': g['total_diff'],
            'game_id': g['game_id'],
            'matchup': f"{g.get('away_team_name')} @ {g.get('home_team_name')}",
        })
    parlay_totals_candidates.sort(key=lambda x: abs(x.get('total_diff', 0)), reverse=True)

    # Build 4-leg parlay: aim for 2-3 ML + 1-2 totals
    parlay_legs = []
    used_game_ids = set()

    # Take best ML picks first (up to 3)
    for c in parlay_ml_candidates:
        if len(parlay_legs) >= 3:
            break
        if c['game_id'] not in used_game_ids:
            parlay_legs.append(c)
            used_game_ids.add(c['game_id'])

    # Fill remaining with totals (different games preferred)
    for c in parlay_totals_candidates:
        if len(parlay_legs) >= 4:
            break
        # Allow same game for totals + ML combo, or different game
        parlay_legs.append(c)

    # If still short, add more ML
    if len(parlay_legs) < 4:
        for c in parlay_ml_candidates:
            if len(parlay_legs) >= 4:
                break
            if c['game_id'] not in used_game_ids:
                parlay_legs.append(c)
                used_game_ids.add(c['game_id'])

    # Calculate combined parlay odds
    def ml_to_decimal(ml):
        if ml > 0:
            return 1 + ml / 100
        return 1 + 100 / abs(ml)

    parlay_decimal = 1.0
    parlay_combined_prob = 1.0
    for leg in parlay_legs:
        parlay_decimal *= ml_to_decimal(leg['odds'])
        parlay_combined_prob *= leg['prob']

    parlay_american = 0
    if parlay_decimal > 2:
        parlay_american = round((parlay_decimal - 1) * 100)
    elif parlay_decimal > 1:
        parlay_american = round(-100 / (parlay_decimal - 1))

    parlay_payout_per_10 = round(10 * parlay_decimal, 2) if parlay_legs else 0

    return {
        'games': games_with_edge,
        'confident_bets': confident_bets,
        'ev_bets': ev_bets,
        'parlay_legs': parlay_legs[:4],
        'parlay_american': parlay_american,
        'parlay_payout': parlay_payout_per_10,
        'parlay_prob': round(parlay_combined_prob * 100, 1),
        'best_totals': best_totals,
        'conferences': conferences,
        'selected_conference': conference,
        'risk_engine': risk_engine,
        'risk_preview': risk_preview,
        'show_experimental': True,
        'spreads_enabled': SPREADS_ENABLED,
        'v2_thresholds': {
            'ml_favorite': ML_EDGE_FAVORITE,
            'ml_underdog': ML_EDGE_UNDERDOG,
            'totals_runs': TOTALS_EDGE_RUNS,
            'underdog_discount': UNDERDOG_DISCOUNT,
        },
    }

