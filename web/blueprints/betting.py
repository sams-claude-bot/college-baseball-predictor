"""
Betting Blueprint - Betting analysis and P&L tracker pages
"""

import sys
from pathlib import Path
from flask import Blueprint, render_template, request

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection

from web.helpers import get_all_conferences, get_betting_games

betting_bp = Blueprint('betting', __name__)


@betting_bp.route('/betting')
def betting():
    """Betting analysis page - v2 logic with adjusted edges"""
    conference = request.args.get('conference', '')
    conferences = get_all_conferences()

    games = get_betting_games()

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
        games = [g for g in games
                if g.get('home_conf') == conference or g.get('away_conf') == conference]

    # Build consensus lookup
    consensus_lookup = {}
    for g in games:
        if g.get('model_agreement') and g['model_agreement']['count'] >= 7:
            consensus_lookup[g['game_id']] = g['model_agreement']['count']

    # Add adjusted_edge to all games
    for g in games:
        if g.get('best_edge'):
            ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
            models = consensus_lookup.get(g['game_id'], 5)
            g['adjusted_edge'] = calc_adjusted_edge(g['best_edge'], ml, models)
            g['models_agree'] = models
            g['is_underdog'] = ml > 0 if ml else False

    # Sort by adjusted edge
    games_with_edge = [g for g in games if g.get('adjusted_edge')]
    games_with_edge.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)

    # Confident bets (7/10+ models agree, sorted by adjusted edge)
    # Cap at -300: heavy favorites aren't worth betting even with consensus
    MAX_FAVORITE_ML = -300

    def passes_favorite_cap(g):
        ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
        return ml is None or ml >= MAX_FAVORITE_ML

    confident_candidates = [g for g in games
                           if g.get('model_agreement')
                           and g['model_agreement']['count'] >= 7
                           and passes_favorite_cap(g)]
    confident_candidates.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)
    confident_bets = confident_candidates[:6]

    # EV bets — pure raw edge over DK line (exclude consensus picks)
    confident_ids = {g['game_id'] for g in confident_bets}
    ev_candidates = [g for g in games_with_edge
                     if g['game_id'] not in confident_ids
                     and g.get('best_edge', 0) >= ML_EDGE_FAVORITE
                     and passes_favorite_cap(g)]
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
        else:
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

    return render_template('betting.html',
                          games=games_with_edge,
                          confident_bets=confident_bets,
                          ev_bets=ev_bets,
                          parlay_legs=parlay_legs[:4],
                          parlay_american=parlay_american,
                          parlay_payout=parlay_payout_per_10,
                          parlay_prob=round(parlay_combined_prob * 100, 1),
                          best_totals=best_totals,
                          conferences=conferences,
                          selected_conference=conference,
                          spreads_enabled=SPREADS_ENABLED,
                          v2_thresholds={
                              'ml_favorite': ML_EDGE_FAVORITE,
                              'ml_underdog': ML_EDGE_UNDERDOG,
                              'totals_runs': TOTALS_EDGE_RUNS,
                              'underdog_discount': UNDERDOG_DISCOUNT
                          })


@betting_bp.route('/tracker')
def tracker():
    """P&L Tracker - three categories: Consensus ML, EV ML, and Totals."""
    conn = get_connection()
    c = conn.cursor()

    # --- CONSENSUS ML BETS (from tracked_confident_bets) ---
    c.execute('''
        SELECT tc.*, g.status, g.home_score, g.away_score
        FROM tracked_confident_bets tc
        LEFT JOIN games g ON tc.game_id = g.id
        ORDER BY tc.date, tc.game_id
    ''')
    all_consensus = [dict(r) for r in c.fetchall()]

    consensus_bets = []
    pending_consensus = []
    for b in all_consensus:
        entry = {
            'date': b['date'],
            'game': f"{'vs ' if b['is_home'] else '@ '}{b['opponent_name']}",
            'pick': b['pick_team_name'],
            'moneyline': b['moneyline'],
            'models_agree': b.get('models_agree', 0),
            'edge': round((b.get('avg_prob', 0.5) - (abs(b['moneyline']) / (abs(b['moneyline']) + 100) if b['moneyline'] < 0 else 100 / (100 + b['moneyline']))) * 100, 1) if b['moneyline'] else 0,
            'model_prob': b.get('avg_prob', 0),
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
        }
        if b['won'] is not None:
            consensus_bets.append(entry)
        else:
            pending_consensus.append(entry)

    # --- EV ML BETS (from tracked_bets) ---
    c.execute('''
        SELECT tb.*, g.status, g.home_score, g.away_score
        FROM tracked_bets tb
        LEFT JOIN games g ON tb.game_id = g.id
        ORDER BY tb.date, tb.game_id
    ''')
    all_ev = [dict(r) for r in c.fetchall()]

    ev_bets = []
    pending_ev = []
    for b in all_ev:
        entry = {
            'date': b['date'],
            'game': f"{'vs ' if b['is_home'] else '@ '}{b['opponent_name']}",
            'pick': b['pick_team_name'],
            'moneyline': b['moneyline'],
            'edge': round(b['edge'], 1),
            'model_prob': b['model_prob'],
            'dk_implied': b['dk_implied'],
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
        }
        if b['won'] is not None:
            ev_bets.append(entry)
        else:
            pending_ev.append(entry)

    # --- TOTALS BETS ---
    total_bets_list = []
    pending_totals = []

    c.execute('''
        SELECT tb.*, g.home_score, g.away_score,
               ht.name as home_name,
               at.name as away_name
        FROM tracked_bets_spreads tb
        LEFT JOIN games g ON tb.game_id = g.id
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE tb.bet_type = 'total'
        ORDER BY tb.date, tb.game_id
    ''')
    all_totals = [dict(r) for r in c.fetchall()]
    conn.close()

    for b in all_totals:
        game_label = f"{b.get('away_name', '?')} @ {b.get('home_name', '?')}"
        entry = {
            'date': b['date'],
            'game': game_label,
            'pick': b['pick'],
            'line': b['line'],
            'odds': b['odds'],
            'model_projection': b['model_projection'],
            'edge': round(b['edge'], 1),
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
        }
        if b['won'] is not None:
            total_bets_list.append(entry)
        else:
            pending_totals.append(entry)

    def calc_stats(bet_list):
        total = len(bet_list)
        wins = sum(1 for b in bet_list if b['won'])
        pl = sum(b['profit'] for b in bet_list)
        return {
            'total': total,
            'wins': wins,
            'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
            'pl': round(pl, 2),
            'roi': round(pl / (total * 100) * 100, 1) if total > 0 else 0,
        }

    consensus_stats = calc_stats(consensus_bets)
    ev_stats = calc_stats(ev_bets)
    totals_stats = calc_stats(total_bets_list)

    # Combined P&L
    all_completed = []
    for b in consensus_bets:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'ML', 'pick': b['pick']})
    for b in ev_bets:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'EV', 'pick': b['pick']})
    for b in total_bets_list:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'TOT', 'pick': b['pick']})
    all_completed.sort(key=lambda x: x['date'])

    combined_stats = calc_stats(all_completed)

    # Running P&L for chart (combined)
    running_pl = []
    cumulative = 0
    for b in all_completed:
        cumulative += b['profit']
        running_pl.append({'date': b['date'], 'pl': round(cumulative, 2),
                          'type': b['type'], 'pick': b['pick'], 'profit': b['profit']})

    # Per-type running P&L for individual charts
    def build_running_pl(bet_list):
        result = []
        cum = 0
        sorted_bets = sorted(bet_list, key=lambda x: x['date'])
        for b in sorted_bets:
            cum += b['profit']
            result.append({'date': b['date'], 'pl': round(cum, 2),
                          'pick': b['pick'], 'profit': b['profit']})
        return result

    consensus_running_pl = build_running_pl(consensus_bets)
    ev_running_pl = build_running_pl(ev_bets)
    totals_running_pl = build_running_pl(total_bets_list)

    # --- PARLAY BETS ---
    conn2 = get_connection()
    c2 = conn2.cursor()
    c2.execute('SELECT * FROM tracked_parlays ORDER BY date')
    all_parlays = [dict(r) for r in c2.fetchall()]
    conn2.close()

    import json as _json
    parlay_bets = []
    pending_parlays = []
    for p in all_parlays:
        legs = _json.loads(p['legs_json'])
        entry = {
            'date': p['date'],
            'legs': legs,
            'num_legs': p['num_legs'],
            'american_odds': p['american_odds'],
            'decimal_odds': p['decimal_odds'],
            'model_prob': p['model_prob'],
            'bet_amount': p['bet_amount'],
            'payout': p['payout'],
            'won': p['won'],
            'profit': p.get('profit', 0) or 0,
            'legs_won': p.get('legs_won', 0) or 0,
        }
        if p['won'] is not None:
            parlay_bets.append(entry)
        else:
            pending_parlays.append(entry)

    parlay_stats = calc_stats(parlay_bets)
    parlay_running_pl = build_running_pl([{'date': p['date'], 'pick': f"Parlay +{p['american_odds']}", 'profit': p['profit']} for p in parlay_bets])

    # Update combined stats to include parlays
    for p in parlay_bets:
        all_completed.append({'date': p['date'], 'profit': p['profit'], 'won': p['won'], 'type': 'PARLAY', 'pick': f"Parlay +{p['american_odds']}"})
    all_completed.sort(key=lambda x: x['date'])
    combined_stats = calc_stats(all_completed)

    # Rebuild combined running P&L with parlays included
    running_pl = []
    cumulative = 0
    for b in all_completed:
        cumulative += b['profit']
        running_pl.append({'date': b['date'], 'pl': round(cumulative, 2),
                          'type': b['type'], 'pick': b['pick'], 'profit': b['profit']})

    # Edge buckets (consensus ML)
    buckets = {
        '5-10%': {'bets': [], 'label': '5-10%'},
        '10-20%': {'bets': [], 'label': '10-20%'},
        '20%+': {'bets': [], 'label': '20%+'}
    }
    for b in consensus_bets:
        if b['edge'] >= 20:
            buckets['20%+']['bets'].append(b)
        elif b['edge'] >= 10:
            buckets['10-20%']['bets'].append(b)
        else:
            buckets['5-10%']['bets'].append(b)

    bucket_stats = []
    for key in ('5-10%', '10-20%', '20%+'):
        bb = buckets[key]['bets']
        if bb:
            bw = sum(1 for x in bb if x['won'])
            bp = sum(x['profit'] for x in bb)
            bucket_stats.append({
                'label': key,
                'count': len(bb),
                'wins': bw,
                'win_rate': round(bw / len(bb) * 100, 1),
                'pl': round(bp, 2),
                'roi': round(bp / (len(bb) * 100) * 100, 1)
            })
        else:
            bucket_stats.append({'label': key, 'count': 0, 'wins': 0, 'win_rate': 0, 'pl': 0, 'roi': 0})

    return render_template('tracker.html',
                          consensus_bets=consensus_bets,
                          pending_consensus=pending_consensus,
                          ev_bets=ev_bets,
                          pending_ev=pending_ev,
                          total_bets_list=total_bets_list,
                          pending_totals=pending_totals,
                          consensus_stats=consensus_stats,
                          ev_stats=ev_stats,
                          totals_stats=totals_stats,
                          combined_stats=combined_stats,
                          running_pl=running_pl,
                          consensus_running_pl=consensus_running_pl,
                          ev_running_pl=ev_running_pl,
                          totals_running_pl=totals_running_pl,
                          bucket_stats=bucket_stats,
                          parlay_bets=parlay_bets,
                          pending_parlays=pending_parlays,
                          parlay_stats=parlay_stats,
                          parlay_running_pl=parlay_running_pl)
