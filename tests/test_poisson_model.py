#!/usr/bin/env python3
"""
Unit tests for Poisson model enhancements.

These tests avoid database dependencies by monkeypatching model helpers.
"""

import math

from models import poisson_model as pm


def test_lambda_adjustment_directionality():
    base_home = 6.0
    base_away = 5.0
    league_avg = 5.5

    neutral_home = {'games': 30, 'avg_scored': 5.5, 'avg_allowed': 5.5}
    neutral_away = {'games': 30, 'avg_scored': 5.5, 'avg_allowed': 5.5}

    strong_home_offense = {'games': 30, 'avg_scored': 8.0, 'avg_allowed': 5.5}
    strong_away_defense = {'games': 30, 'avg_scored': 5.5, 'avg_allowed': 3.5}

    neutral_adj_home, _, neutral_meta = pm.apply_opponent_adjusted_lambdas(
        base_home, base_away, neutral_home, neutral_away, league_avg
    )
    boosted_home, _, boosted_meta = pm.apply_opponent_adjusted_lambdas(
        base_home, base_away, strong_home_offense, neutral_away, league_avg
    )
    reduced_home, _, reduced_meta = pm.apply_opponent_adjusted_lambdas(
        base_home, base_away, neutral_home, strong_away_defense, league_avg
    )

    assert neutral_meta['applied'] is True
    assert boosted_meta['applied'] is True
    assert reduced_meta['applied'] is True
    assert boosted_home > neutral_adj_home
    assert reduced_home < neutral_adj_home


def test_lambda_adjustment_fallback_when_stats_missing():
    base_home = 6.25
    base_away = 4.75

    bad_home_stats = {'games': 10, 'avg_scored': 6.0}  # missing avg_allowed
    good_away_stats = {'games': 10, 'avg_scored': 5.2, 'avg_allowed': 5.0}

    adj_home, adj_away, meta = pm.apply_opponent_adjusted_lambdas(
        base_home, base_away, bad_home_stats, good_away_stats, 5.5
    )

    assert meta['fallback'] is True
    assert adj_home == pm.clamp_lambda(base_home)
    assert adj_away == pm.clamp_lambda(base_away)


def test_predict_overdispersion_branch_returns_valid_totals(monkeypatch):
    def fake_team_stats(team_id, last_n_games=None):
        if team_id == 'home':
            return {
                'games': 30,
                'avg_scored': 6.2,
                'avg_allowed': 5.4,
                'avg_scored_home': 6.8,
                'avg_scored_away': 5.7,
                'avg_allowed_home': 5.1,
                'avg_allowed_away': 5.8,
                'total_scored': 186,
                'total_allowed': 162,
            }
        return {
            'games': 30,
            'avg_scored': 5.4,
            'avg_allowed': 6.1,
            'avg_scored_home': 5.9,
            'avg_scored_away': 5.0,
            'avg_allowed_home': 5.7,
            'avg_allowed_away': 6.6,
            'total_scored': 162,
            'total_allowed': 183,
        }

    monkeypatch.setattr(pm.predict_module, 'get_team_run_stats', fake_team_stats)
    monkeypatch.setattr(pm.predict_module, 'get_league_average', lambda: 5.5)
    monkeypatch.setattr(pm.predict_module, 'get_quality_adjustment', lambda team_id, opp_id: (1.0, 1.0))
    monkeypatch.setattr(
        pm.predict_module,
        'get_recent_totals_history',
        lambda team_a, team_b, last_n_games=None: [2, 18, 5, 21, 7, 19, 3, 17, 6, 20, 4, 22]
    )

    pred = pm.predict('home', 'away', team_a_home=True)

    od_meta = pred['poisson_adjustments']['totals_overdispersion']
    assert od_meta['applied'] is True

    for total_line, probs in pred['totals'].items():
        assert isinstance(total_line, float)
        assert 0.0 <= probs['over'] <= 1.0
        assert 0.0 <= probs['under'] <= 1.0
        assert 0.0 <= probs['push'] <= 1.0
        assert math.isclose(probs['over'] + probs['under'] + probs['push'], 1.0, rel_tol=0, abs_tol=2e-3)


def test_opponent_adjustment_strength_blends_toward_neutral(monkeypatch):
    monkeypatch.setattr(pm.lambda_calc, '_calculate_strength_factors', lambda stats, league_avg: stats['factors'])

    def run_with_strength(strength):
        monkeypatch.setattr(
            pm.lambda_calc,
            '_cfg',
            lambda name, default: {
                'POISSON_ENABLE_OPPONENT_ADJUSTMENT': True,
                'POISSON_OPPONENT_ADJUSTMENT_STRENGTH': strength,
            }.get(name, default)
        )
        home_stats = {'factors': (1.10, 0.95)}  # (offense, defense)
        away_stats = {'factors': (0.90, 1.05)}
        return pm.apply_opponent_adjusted_lambdas(6.0, 4.0, home_stats, away_stats, 5.5)

    home_0, away_0, meta_0 = run_with_strength(0.0)
    home_05, away_05, meta_05 = run_with_strength(0.5)
    home_1, away_1, meta_1 = run_with_strength(1.0)

    # strength=0 => neutral multipliers only
    assert home_0 == pm.clamp_lambda(6.0)
    assert away_0 == pm.clamp_lambda(4.0)

    # strength=1 => full original factors
    assert home_1 == pm.clamp_lambda(6.0 * 1.10 * 1.05)
    assert away_1 == pm.clamp_lambda(4.0 * 0.90 * 0.95)

    # strength=0.5 => halfway toward neutral for each factor
    assert home_05 == pm.clamp_lambda(6.0 * 1.05 * 1.025)
    assert away_05 == pm.clamp_lambda(4.0 * 0.95 * 0.975)

    assert meta_0['strength'] == 0.0
    assert meta_05['strength'] == 0.5
    assert meta_1['strength'] == 1.0
