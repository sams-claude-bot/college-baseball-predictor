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
from web.bet_quality import (
    passes_quality_gate, bet_quality_score, has_vegas_disagreement,
    vegas_implied_prob, MAX_PER_TYPE as QUALITY_MAX_PER_TYPE
)
from web.services.series_probability import compute_series_probs

# Lazy-loaded calibrator
_page_calibrator = None

def _get_calibrator():
    global _page_calibrator
    if _page_calibrator is None:
        try:
            from models.calibration import Calibrator
            cal = Calibrator()
            if cal._load():
                _page_calibrator = cal
            else:
                _page_calibrator = False
        except Exception:
            _page_calibrator = False
    return _page_calibrator if _page_calibrator is not False else None


def _add_calibrated_edge(game, prob, ml):
    """Add calibrated_prob and calibrated_edge to a game dict."""
    cal = _get_calibrator()
    if cal and ml and prob:
        cal_p = cal.calibrate(prob)
        implied = abs(ml)/(abs(ml)+100) if ml < 0 else 100/(100+ml)
        game['calibrated_prob'] = round(cal_p, 4)
        game['calibrated_edge'] = round((cal_p - implied) * 100, 1)


def build_betting_page_context(conference=''):
    """Build template context for the betting analysis page."""
    from datetime import datetime
    import pytz
    ct = pytz.timezone('America/Chicago')
    today_str = datetime.now(ct).strftime('%Y-%m-%d')

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

        # Series probabilities (home-perspective)
        if g.get('model_home_prob') is not None:
            g['series_probs'] = compute_series_probs(g['model_home_prob'])

    # Sort by adjusted edge
    games_with_edge = [g for g in games if g.get('adjusted_edge')]
    games_with_edge.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)

    # === Load tracked bets from DB (source of truth for cards) ===
    # Cards should show exactly what the pipeline locked in, not re-derive.
    from scripts.database import get_connection as _get_tracked_conn
    _tc = _get_tracked_conn()
    _tc_cursor = _tc.cursor()

    # Build game lookup for enriching tracked bets with full game data
    games_by_id = {g['game_id']: g for g in games}

    # EV bets from tracked_bets (loaded FIRST — EV takes priority)
    _tc_cursor.execute(
        'SELECT * FROM tracked_bets WHERE date = ? ORDER BY edge DESC',
        (today_str,)
    )
    tracked_ev = [dict(r) for r in _tc_cursor.fetchall()]
    ev_bets = []
    ev_ids = set()
    for tb in tracked_ev:
        g = games_by_id.get(tb['game_id'])
        if g:
            pick = 'home' if tb['is_home'] else 'away'
            prob = tb.get('model_prob', 0.5)
            ml = tb.get('moneyline')
            g['tracked_ev'] = True
            g['best_pick'] = pick
            g['best_edge'] = tb.get('edge', 0)
            if ml:
                vi = abs(ml)/(abs(ml)+100) if ml < 0 else 100/(100+ml)
                g['vegas_disagreement'] = prob < vi
                g['disagreement_pp'] = round((prob - vi) * 100, 1)
            _add_calibrated_edge(g, prob, ml)
            ev_bets.append(g)
            ev_ids.add(tb['game_id'])

    # Consensus bets from tracked_confident_bets (exclude games already in EV)
    _tc_cursor.execute(
        'SELECT * FROM tracked_confident_bets WHERE date = ? ORDER BY avg_prob DESC',
        (today_str,)
    )
    tracked_consensus = [dict(r) for r in _tc_cursor.fetchall()]
    confident_bets = []
    for tb in tracked_consensus:
        if tb['game_id'] in ev_ids:
            continue
        g = games_by_id.get(tb['game_id'])
        if g:
            pick = 'home' if tb['is_home'] else 'away'
            prob = tb.get('avg_prob', 0.5)
            ml = tb.get('moneyline')
            g['tracked_consensus'] = True
            g['model_agreement'] = g.get('model_agreement') or {
                'count': tb.get('models_agree', 7),
                'total': tb.get('models_total', 10),
                'pick': pick,
                'avg_prob': prob,
                'confidence': prob,
            }
            if ml:
                vi = abs(ml)/(abs(ml)+100) if ml < 0 else 100/(100+ml)
                g['vegas_disagreement'] = prob < vi
                g['disagreement_pp'] = round((prob - vi) * 100, 1)
            _add_calibrated_edge(g, prob, ml)
            confident_bets.append(g)

    # Best totals from tracked_bets_spreads
    _tc_cursor.execute(
        'SELECT * FROM tracked_bets_spreads WHERE date = ? AND bet_type = ? ORDER BY ABS(edge) DESC',
        (today_str, 'total')
    )
    tracked_totals = [dict(r) for r in _tc_cursor.fetchall()]
    best_totals = []
    for tb in tracked_totals:
        g = games_by_id.get(tb['game_id'])
        if g:
            g['tracked_total'] = True
            g['total_lean'] = tb.get('pick', 'OVER')
            g['total_diff'] = tb.get('edge', 0) / 8.0  # approximate runs diff from edge
            best_totals.append(g)

    # Fallback: if no tracked bets exist yet (pre-pipeline), use dynamic calculation
    if not confident_bets and not ev_bets and not best_totals:
        games_with_totals = [g for g in games if g.get('over_under')]
        games_with_totals.sort(key=lambda x: abs(x.get('total_diff', 0)), reverse=True)
        best_totals = [g for g in games_with_totals if abs(g.get('total_diff', 0)) >= 3.0]

    _tc.close()

    # === PARLAY BUILDER ===
    # Mirror morning-pipeline constraints for fallback generation.
    PARLAY_ML_CAP = -250
    PARLAY_MIN_PROB = 0.72
    PARLAY_MAX_PROB = 0.92
    PARLAY_MIN_EDGE = 8.0

    # ML candidates for parlay
    parlay_ml_candidates = []
    for g in games:
        if not g.get('best_edge') or g['best_edge'] < PARLAY_MIN_EDGE:
            continue
        best_pick = g.get('best_pick')
        if best_pick not in ('home', 'away'):
            continue

        ml = g.get('home_ml') if best_pick == 'home' else g.get('away_ml')
        if ml is None:
            continue
        # Skip heavy favorites.
        if ml < PARLAY_ML_CAP:
            continue

        # Use side-aware probability aligned to best_pick.
        agreement = g.get('model_agreement', {})
        if agreement and agreement.get('avg_prob'):
            prob = agreement['avg_prob']
            if agreement.get('pick') in ('home', 'away') and agreement.get('pick') != best_pick:
                prob = 1 - prob
        else:
            prob = g.get('model_home_prob', 0.5)
            if best_pick == 'away':
                prob = 1 - prob
        if not (PARLAY_MIN_PROB <= prob <= PARLAY_MAX_PROB):
            continue
        # Calibrate probability for parlay scoring
        cal = _get_calibrator()
        cal_prob = cal.calibrate(prob) if cal else prob
        implied = abs(ml)/(abs(ml)+100) if ml < 0 else 100/(100+ml)
        cal_edge = round((cal_prob - implied) * 100, 1)

        parlay_ml_candidates.append({
            'game': g,
            'type': 'ML',
            'pick_team': g.get('home_team_name') if best_pick == 'home' else g.get('away_team_name'),
            'opponent': g.get('away_team_name') if best_pick == 'home' else g.get('home_team_name'),
            'pick_label': g.get('home_team_name') if best_pick == 'home' else g.get('away_team_name'),
            'odds': ml,
            'prob': prob,
            'calibrated_prob': round(cal_prob, 4),
            'calibrated_edge': cal_edge,
            'edge': g['best_edge'],
            'models_agree': g.get('models_agree', 0),
            'game_id': g['game_id'],
            'matchup': f"{g.get('away_team_name')} @ {g.get('home_team_name')}",
        })
    # Sort by a blend of edge and probability (sweet spot scoring)
    # Prefer ~75-80% range with good edge
    for c in parlay_ml_candidates:
        # Score: use calibrated prob, penalize extremes, reward ~0.72 range with good calibrated edge
        cp = c.get('calibrated_prob', c['prob'])
        prob_score = 1.0 - abs(cp - 0.72) * 3  # Peak at 72% (calibrated sweet spot)
        c['parlay_score'] = prob_score * max(c.get('calibrated_edge', c['edge']), 0)
    parlay_ml_candidates.sort(key=lambda x: x['parlay_score'], reverse=True)

    # Load parlay from tracked_parlays (set once by morning pipeline).
    # Only fall back to dynamic generation if no stored parlay exists.
    import json as _json
    from scripts.database import get_connection as _get_db
    _parlay_conn = _get_db()

    def ml_to_decimal(ml):
        if ml > 0:
            return 1 + ml / 100
        return 1 + 100 / abs(ml)

    stored_parlay = _parlay_conn.execute(
        "SELECT legs_json, american_odds, decimal_odds, model_prob, payout "
        "FROM tracked_parlays WHERE date = ? LIMIT 1",
        (today_str,)
    ).fetchone()

    if stored_parlay:
        # Use the locked-in parlay from the morning pipeline
        stored_legs = _json.loads(stored_parlay[0] if isinstance(stored_parlay, tuple)
                                  else stored_parlay['legs_json'])
        parlay_legs = []
        for sl in stored_legs:
            leg = {
                'type': sl.get('type', 'CONSENSUS'),
                'game_id': sl.get('game_id', ''),
                'pick': sl.get('pick', ''),
                'pick_label': sl.get('pick', sl.get('pick_label', '')),
                'pick_team': sl.get('pick', sl.get('pick_team', '')),
                'matchup': sl.get('matchup', ''),
                'odds': sl.get('odds', 0),
                'prob': sl.get('prob', 0.5),
                'edge': sl.get('edge', 0),
                'models_agree': sl.get('models_agree'),
                'calibrated_prob': sl.get('calibrated_prob'),
                'calibrated_edge': sl.get('calibrated_edge'),
            }
            parlay_legs.append(leg)

        sp = stored_parlay
        parlay_american = sp[1] if isinstance(sp, tuple) else sp['american_odds']
        parlay_decimal = sp[2] if isinstance(sp, tuple) else sp['decimal_odds']
        parlay_combined_prob = sp[3] if isinstance(sp, tuple) else sp['model_prob']
        parlay_payout_per_10 = round(10 * parlay_decimal, 2) if parlay_decimal else 0
        # Calibrated prob from stored legs
        parlay_calibrated_prob = 1.0
        for leg in parlay_legs:
            cp = leg.get('calibrated_prob')
            parlay_calibrated_prob *= cp if cp is not None else leg['prob']
    else:
        # Fallback: dynamically build ML/CONSENSUS-style parlay if none stored.
        parlay_legs = []
        used_game_ids = set()

        for c in parlay_ml_candidates:
            if len(parlay_legs) >= 4:
                break
            if c['game_id'] not in used_game_ids:
                parlay_legs.append(c)
                used_game_ids.add(c['game_id'])

        if len(parlay_legs) < 3:
            parlay_legs = []
            parlay_decimal = 0
            parlay_combined_prob = 0
            parlay_calibrated_prob = 0
            parlay_american = 0
            parlay_payout_per_10 = 0
        else:
            parlay_decimal = 1.0
            parlay_combined_prob = 1.0
            parlay_calibrated_prob = 1.0
            for leg in parlay_legs:
                parlay_decimal *= ml_to_decimal(leg['odds'])
                parlay_combined_prob *= leg['prob']
                parlay_calibrated_prob *= leg.get('calibrated_prob', leg['prob'])

            parlay_american = 0
            if parlay_decimal > 2:
                parlay_american = round((parlay_decimal - 1) * 100)
            elif parlay_decimal > 1:
                parlay_american = round(-100 / (parlay_decimal - 1))

            parlay_payout_per_10 = round(10 * parlay_decimal, 2)

    _parlay_conn.close()

    # Enrich parlay legs with live scores
    _score_conn = _get_db()
    _score_cur = _score_conn.cursor()
    parlay_legs_won = 0
    parlay_legs_lost = 0
    for leg in parlay_legs:
        gid = leg.get('game_id')
        if gid:
            game_row = _score_cur.execute(
                "SELECT status, home_score, away_score, inning_text, "
                "home_team_id, away_team_id FROM games WHERE id = ?", (gid,)
            ).fetchone()
            if game_row:
                gr = dict(game_row) if hasattr(game_row, 'keys') else {
                    'status': game_row[0], 'home_score': game_row[1],
                    'away_score': game_row[2], 'inning_text': game_row[3],
                    'home_team_id': game_row[4], 'away_team_id': game_row[5],
                }
                leg['game_status'] = gr['status']
                leg['home_score'] = gr['home_score'] or 0
                leg['away_score'] = gr['away_score'] or 0
                leg['inning_text'] = gr['inning_text'] or ''

                # Determine if this leg is winning/losing/pending
                if gr['status'] in ('final',):
                    pick_team = leg.get('pick_team', leg.get('pick', ''))
                    if leg['type'] in ('ML', 'CONSENSUS'):
                        # Check if picked team won
                        home_won = (gr['home_score'] or 0) > (gr['away_score'] or 0)
                        # Is our pick the home team?
                        pick_is_home = pick_team.lower() in (gr.get('home_team_id') or '').lower().replace('-', ' ')
                        if not pick_is_home:
                            # Try matching by name
                            pick_is_home = False  # fallback
                        leg['leg_result'] = 'won' if (home_won == pick_is_home) else 'lost'
                    elif leg['type'] == 'Total':
                        total_runs = (gr['home_score'] or 0) + (gr['away_score'] or 0)
                        ou_line = leg.get('pick_label', '')
                        if 'OVER' in ou_line.upper():
                            leg['leg_result'] = 'won' if total_runs > float(ou_line.split()[-1]) else 'lost'
                        elif 'UNDER' in ou_line.upper():
                            leg['leg_result'] = 'won' if total_runs < float(ou_line.split()[-1]) else 'lost'
                        else:
                            leg['leg_result'] = 'pending'
                    if leg.get('leg_result') == 'won':
                        parlay_legs_won += 1
                    elif leg.get('leg_result') == 'lost':
                        parlay_legs_lost += 1
                elif gr['status'] == 'in-progress':
                    leg['leg_result'] = 'live'
                else:
                    leg['leg_result'] = 'pending'
    _score_conn.close()

    # --- Tag bet results for final games (red/green shading) ---
    def _tag_ml_result(g, pick_side):
        """Tag a ML bet with 'won', 'lost', 'live', or None (pending)."""
        status = g.get('status')
        if status == 'final':
            winner = g.get('winner_id')
            if not winner:
                return None
            picked_id = g.get('home_team_id') if pick_side == 'home' else g.get('away_team_id')
            return 'won' if winner == picked_id else 'lost'
        elif status == 'in-progress':
            return 'live'
        return None

    def _tag_total_result(g):
        """Tag a totals bet with 'won', 'lost', 'live', or None."""
        status = g.get('status')
        if status == 'final':
            actual_total = (g.get('home_score') or 0) + (g.get('away_score') or 0)
            line = g.get('over_under')
            lean = g.get('total_lean', '').upper()
            if not line or not lean:
                return None
            if lean == 'OVER':
                return 'won' if actual_total > line else 'lost'
            elif lean == 'UNDER':
                return 'won' if actual_total < line else 'lost'
            return None
        elif status == 'in-progress':
            return 'live'
        return None

    for g in confident_bets:
        pick = g.get('model_agreement', {}).get('pick', 'home')
        g['bet_result'] = _tag_ml_result(g, pick)

    for g in ev_bets:
        pick = g.get('best_pick', 'home')
        g['bet_result'] = _tag_ml_result(g, pick)

    for g in best_totals:
        g['bet_result'] = _tag_total_result(g)

    # Build risky parlay (untracked, for fun)
    risky_parlay = None
    try:
        from scripts.betting.record import build_risky_parlay
        risky_results = analyze_games()
        risky_parlay = build_risky_parlay(risky_results)
    except Exception:
        pass

    return {
        'games': games_with_edge,
        'confident_bets': confident_bets,
        'ev_bets': ev_bets,
        'parlay_legs': parlay_legs[:4],
        'parlay_american': parlay_american,
        'parlay_payout': parlay_payout_per_10,
        'parlay_prob': round(parlay_combined_prob * 100, 1),
        'parlay_calibrated_prob': round(parlay_calibrated_prob * 100, 1),
        'risky_parlay': risky_parlay,
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
