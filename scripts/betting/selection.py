"""Game analysis and bet filtering/selection logic."""

from typing import Optional

import requests

from web.helpers import calculate_adjusted_edge

from .risk import (
    american_to_prob,
    apply_correlation_caps,
    kelly_fraction,
    suggest_stake_for_bet,
)

API_URL = 'http://localhost:5000/api/best-bets'

# ============ THRESHOLDS ============
ML_EDGE_THRESHOLD = 8.0
ML_EDGE_UNDERDOG = 15.0
ML_CONSENSUS_MIN = 7
ML_MAX_FAVORITE = -200
ML_MAX_FAVORITE_CONSENSUS = -300
ML_MIN_UNDERDOG = 250

ML_MAX_MODEL_PROB = 0.88
ML_MIN_MODEL_PROB = 0.55

TOTALS_EDGE_THRESHOLD = 3.0
TOTALS_MIN_CONFIDENCE = 0.6

SPREADS_ENABLED = False

MAX_CONSENSUS_PER_DAY = 6
MAX_EV_PER_DAY = 6
MAX_TOTALS_PER_DAY = 6


def analyze_games(date_str: Optional[str] = None) -> dict:
    """
    Analyze all games for a date and return bet recommendations.
    Returns dict with 'bets' and 'rejections' for transparency.
    """
    try:
        url = API_URL
        if date_str:
            url += f'?date={date_str}'
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ Failed to fetch from API: {e}")
        return {'bets': [], 'rejections': [], 'error': str(e)}

    date = data['date']
    results = {
        'date': date,
        'bets': [],
        'rejections': [],
        'spreads_disabled': True,
    }

    consensus_lookup = {}
    for g in data.get('confident_bets', []):
        consensus_lookup[g['game_id']] = {
            'models_agree': g['models_agree'],
            'avg_prob': g['avg_prob'],
        }

    for game in data.get('confident_bets', []):
        ml = game.get('moneyline')
        if not ml:
            continue

        models_agree = game['models_agree']
        avg_prob = game['avg_prob']
        rejection_reasons = []

        if models_agree < ML_CONSENSUS_MIN:
            rejection_reasons.append(f"only {models_agree}/10 models agree (need {ML_CONSENSUS_MIN}+)")

        max_fav = ML_MAX_FAVORITE_CONSENSUS if models_agree >= 9 else ML_MAX_FAVORITE
        if ml < max_fav:
            rejection_reasons.append(f"ML {ml} too heavy (max {max_fav} for {models_agree}/10 consensus)")

        if avg_prob > ML_MAX_MODEL_PROB:
            rejection_reasons.append(f"avg prob {avg_prob*100:.0f}% overconfident")

        if rejection_reasons:
            results['rejections'].append({
                'type': 'CONSENSUS',
                'game': f"{game['pick_team_name']} vs {game['opponent_name']}",
                'models_agree': models_agree,
                'ml': ml,
                'reasons': rejection_reasons,
            })
        else:
            kelly_mult = kelly_fraction(avg_prob, ml)
            base_bet = {
                'type': 'CONSENSUS',
                'game_id': game['game_id'],
                'date': game['date'],
                'pick_team_id': game['pick_team_id'],
                'pick_team_name': game['pick_team_name'],
                'opponent_name': game['opponent_name'],
                'is_home': game['is_home'],
                'moneyline': ml,
                'model_prob': avg_prob,
                'models_agree': models_agree,
                'models_total': 10,
                'edge': (avg_prob - american_to_prob(ml)) * 100,
                'kelly_mult': kelly_mult,
            }
            sizing = suggest_stake_for_bet(base_bet)
            base_bet.update(sizing)
            base_bet['bet_amount'] = round(base_bet['suggested_stake'], 0)
            base_bet.setdefault('exposure_bucket', '')
            results['bets'].append(base_bet)

    for game in data.get('moneylines', []):
        if game['game_id'] in [b['game_id'] for b in results['bets'] if b['type'] == 'CONSENSUS']:
            continue
        ml = game['moneyline']
        edge = game['edge']
        model_prob = game['model_prob']
        is_underdog = ml > 0

        consensus = consensus_lookup.get(game['game_id'], {})
        models_agree = consensus.get('models_agree', 5)
        rejection_reasons = []

        edge_threshold = ML_EDGE_UNDERDOG if is_underdog else ML_EDGE_THRESHOLD
        if edge < edge_threshold:
            rejection_reasons.append(
                f"edge {edge:.1f}% < {edge_threshold}% {'underdog ' if is_underdog else ''}threshold"
            )

        if model_prob > ML_MAX_MODEL_PROB:
            rejection_reasons.append(
                f"model {model_prob*100:.0f}% overconfident (max {ML_MAX_MODEL_PROB*100:.0f}%)"
            )
        if model_prob < ML_MIN_MODEL_PROB:
            rejection_reasons.append(
                f"model {model_prob*100:.0f}% too low (min {ML_MIN_MODEL_PROB*100:.0f}%)"
            )

        if ml < ML_MAX_FAVORITE:
            rejection_reasons.append(f"ML {ml} too heavy (max {ML_MAX_FAVORITE})")

        if ml > ML_MIN_UNDERDOG:
            rejection_reasons.append(f"ML +{ml} too long (max +{ML_MIN_UNDERDOG})")

        if rejection_reasons:
            results['rejections'].append({
                'type': 'ML',
                'game': f"{game['pick_team_name']} vs {game['opponent_name']}",
                'edge': edge,
                'ml': ml,
                'reasons': rejection_reasons,
            })
        else:
            kelly_mult = kelly_fraction(model_prob, ml)
            base_bet = {
                'type': 'ML',
                'game_id': game['game_id'],
                'date': game['date'],
                'pick_team_id': game['pick_team_id'],
                'pick_team_name': game['pick_team_name'],
                'opponent_name': game['opponent_name'],
                'is_home': game['is_home'],
                'moneyline': ml,
                'model_prob': model_prob,
                'dk_implied': game['dk_implied'],
                'edge': edge,
                'models_agree': models_agree,
                'kelly_mult': kelly_mult,
                'models_total': 10,
            }
            sizing = suggest_stake_for_bet(base_bet)
            base_bet.update(sizing)
            base_bet['bet_amount'] = round(base_bet['suggested_stake'], 0)
            base_bet.setdefault('exposure_bucket', '')
            results['bets'].append(base_bet)

    for game in data.get('totals', []):
        edge = game['edge']
        rejection_reasons = []

        if edge < TOTALS_EDGE_THRESHOLD:
            rejection_reasons.append(f"edge {edge:.1f} runs < {TOTALS_EDGE_THRESHOLD} threshold")

        if rejection_reasons:
            results['rejections'].append({
                'type': 'TOTAL',
                'game': f"{game['pick']} {game['line']}",
                'edge': edge,
                'reasons': rejection_reasons,
            })
        else:
            base_bet = {
                'type': 'TOTAL',
                'game_id': game['game_id'],
                'date': game['date'],
                'pick': game['pick'],
                'line': game['line'],
                'odds': game['odds'],
                'model_projection': game['model_projection'],
                'edge': edge,
            }
            sizing = suggest_stake_for_bet(base_bet)
            base_bet.update(sizing)
            base_bet['bet_amount'] = round(base_bet['suggested_stake'], 0)
            base_bet.setdefault('exposure_bucket', '')
            results['bets'].append(base_bet)

    if SPREADS_ENABLED:
        pass
    else:
        for game in data.get('spreads', []):
            results['rejections'].append({
                'type': 'SPREAD',
                'game': f"{game['pick']} {game['line']:+.1f}",
                'edge': game['edge'],
                'reasons': ['SPREADS DISABLED - model not calibrated (0/5 historical)'],
            })

    for b in results['bets']:
        ml = b.get('moneyline') or b.get('odds') or -110
        b['adjusted_edge'] = calculate_adjusted_edge(
            b.get('edge', 0),
            moneyline=ml,
            models_agree=b.get('models_agree', 5),
        )

    consensus_bets = [b for b in results['bets'] if b['type'] == 'CONSENSUS']
    ev_bets = [b for b in results['bets'] if b['type'] == 'ML']
    total_bets = [b for b in results['bets'] if b['type'] == 'TOTAL']

    consensus_bets.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)
    ev_bets.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)
    total_bets.sort(key=lambda x: x.get('edge', 0), reverse=True)

    def cap_and_reject(bet_list, limit, label):
        kept = bet_list[:limit]
        for bet in bet_list[limit:]:
            results['rejections'].append({
                'type': bet['type'],
                'game': bet.get('pick_team_name') or bet.get('pick'),
                'edge': bet['edge'],
                'reasons': [f'exceeded {limit} {label}/day limit'],
            })
        return kept

    consensus_bets = cap_and_reject(consensus_bets, MAX_CONSENSUS_PER_DAY, 'consensus')
    ev_bets = cap_and_reject(ev_bets, MAX_EV_PER_DAY, 'EV')
    total_bets = cap_and_reject(total_bets, MAX_TOTALS_PER_DAY, 'totals')

    results['bets'] = consensus_bets + ev_bets + total_bets
    results = apply_correlation_caps(results)

    return results

