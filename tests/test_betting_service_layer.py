#!/usr/bin/env python3
"""Smoke tests for betting service-layer extraction (Cleanup Sprint A, Step 3)."""

import sys
import sqlite3
from pathlib import Path

import pytest
from flask import Flask

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _init_empty_betting_tables(db_path: Path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE tracked_confident_bets ("
        "date TEXT, game_id TEXT, avg_prob REAL, is_home INTEGER, moneyline INTEGER, "
        "models_agree INTEGER, models_total INTEGER, confidence REAL)"
    )
    c.execute(
        "CREATE TABLE tracked_bets ("
        "date TEXT, game_id TEXT, edge REAL, is_home INTEGER, moneyline INTEGER, model_prob REAL)"
    )
    c.execute(
        "CREATE TABLE tracked_bets_spreads ("
        "date TEXT, game_id TEXT, bet_type TEXT, edge REAL, pick TEXT)"
    )
    c.execute(
        "CREATE TABLE tracked_parlays ("
        "date TEXT, legs_json TEXT, american_odds INTEGER, decimal_odds REAL, "
        "model_prob REAL, payout REAL)"
    )
    c.execute(
        "CREATE TABLE games ("
        "id TEXT, status TEXT, home_score INTEGER, away_score INTEGER, inning_text TEXT, "
        "home_team_id TEXT, away_team_id TEXT)"
    )
    conn.commit()
    conn.close()


def _mock_risk_preview():
    return {'date': '2026-02-24', 'bets': [], 'rejections': [], 'error': None}


def test_betting_page_service_smoke(monkeypatch):
    from web.services import betting_page

    sample_games = [
        {
            'game_id': 'g1',
            'date': '2026-02-24',
            'home_conf': 'SEC',
            'away_conf': 'ACC',
            'home_team_name': 'Home A',
            'away_team_name': 'Away A',
            'best_edge': 12.0,
            'best_pick': 'home',
            'home_ml': -140,
            'away_ml': 120,
            'model_home_prob': 0.72,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'home', 'avg_prob': 0.72},
            'over_under': 10.5,
            'total_diff': 3.4,
            'total_edge': 18.0,
            'total_lean': 'Over',
        },
        {
            'game_id': 'g2',
            'date': '2026-02-24',
            'home_conf': 'Big 12',
            'away_conf': 'SEC',
            'home_team_name': 'Home B',
            'away_team_name': 'Away B',
            'best_edge': 9.0,
            'best_pick': 'away',
            'home_ml': -250,
            'away_ml': 210,
            'model_home_prob': 0.30,
            'model_agreement': {'count': 6, 'total': 12, 'pick': 'away', 'avg_prob': 0.70},
            'over_under': 11.0,
            'total_diff': 1.0,
            'total_edge': 10.0,
            'total_lean': 'Under',
        },
    ]

    monkeypatch.setattr(betting_page, 'get_all_conferences', lambda: ['SEC', 'ACC', 'Big 12'])
    monkeypatch.setattr(betting_page, 'get_betting_games', lambda: [dict(g) for g in sample_games])
    monkeypatch.setattr(
        betting_page,
        'analyze_games',
        lambda: {'date': '2026-02-24', 'bets': [{'id': i} for i in range(10)], 'rejections': [1, 2]},
    )

    ctx = betting_page.build_betting_page_context(conference='SEC')

    assert ctx['selected_conference'] == 'SEC'
    assert 'games' in ctx and isinstance(ctx['games'], list)
    assert 'confident_bets' in ctx and isinstance(ctx['confident_bets'], list)
    assert 'ev_bets' in ctx and isinstance(ctx['ev_bets'], list)
    assert ctx['risk_preview']['date'] == '2026-02-24'
    assert len(ctx['risk_preview']['bets']) == 8  # Preview cap preserved
    assert ctx['risk_preview']['rejections'] == 2
    assert ctx['show_experimental'] is True
    assert ctx['v2_thresholds']['ml_favorite'] == 8.0
    assert all(
        g.get('home_conf') == 'SEC' or g.get('away_conf') == 'SEC'
        for g in (ctx['games'] + ctx['confident_bets'] + ctx['ev_bets'])
    )


def test_parlay_fallback_inverts_opposite_consensus_prob_and_blocks_bad_leg(monkeypatch, tmp_path):
    from web.services import betting_page
    import scripts.database as db_module

    db_path = tmp_path / "betting.db"
    _init_empty_betting_tables(db_path)

    def _new_conn():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    sample_games = [
        {
            'game_id': 'g1',
            'date': '2026-02-24',
            'home_team_name': 'Home A',
            'away_team_name': 'Away A',
            'best_edge': 10.0,
            'best_pick': 'home',
            'home_ml': -140,
            'away_ml': 120,
            'model_home_prob': 0.75,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'home', 'avg_prob': 0.78},
        },
        {
            'game_id': 'g2',
            'date': '2026-02-24',
            'home_team_name': 'Home B',
            'away_team_name': 'Away B',
            'best_edge': 9.5,
            'best_pick': 'away',
            'home_ml': -150,
            'away_ml': 130,
            'model_home_prob': 0.28,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'away', 'avg_prob': 0.76},
        },
        {
            'game_id': 'g3',
            'date': '2026-02-24',
            'home_team_name': 'Home C',
            'away_team_name': 'Away C',
            'best_edge': 11.0,
            'best_pick': 'home',
            'home_ml': -135,
            'away_ml': 115,
            'model_home_prob': 0.74,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'away', 'avg_prob': 0.80},
        },
    ]

    monkeypatch.setattr(db_module, 'get_connection', _new_conn)
    monkeypatch.setattr(betting_page, 'get_all_conferences', lambda: [])
    monkeypatch.setattr(betting_page, 'get_betting_games', lambda: [dict(g) for g in sample_games])
    monkeypatch.setattr(betting_page, 'analyze_games', _mock_risk_preview)
    monkeypatch.setattr(betting_page, '_get_calibrator', lambda: None)

    ctx = betting_page.build_betting_page_context()

    assert ctx['parlay_legs'] == []
    assert ctx['parlay_american'] == 0
    assert ctx['parlay_payout'] == 0
    assert ctx['parlay_prob'] == 0.0
    assert ctx['parlay_calibrated_prob'] == 0.0


def test_parlay_fallback_returns_empty_when_fewer_than_three_legs(monkeypatch, tmp_path):
    from web.services import betting_page
    import scripts.database as db_module

    db_path = tmp_path / "betting.db"
    _init_empty_betting_tables(db_path)

    def _new_conn():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    sample_games = [
        {
            'game_id': 'g1',
            'date': '2026-02-24',
            'home_team_name': 'Home A',
            'away_team_name': 'Away A',
            'best_edge': 10.0,
            'best_pick': 'home',
            'home_ml': -145,
            'away_ml': 125,
            'model_home_prob': 0.75,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'home', 'avg_prob': 0.78},
        },
        {
            'game_id': 'g2',
            'date': '2026-02-24',
            'home_team_name': 'Home B',
            'away_team_name': 'Away B',
            'best_edge': 9.0,
            'best_pick': 'away',
            'home_ml': -140,
            'away_ml': 125,
            'model_home_prob': 0.27,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'away', 'avg_prob': 0.74},
        },
        {
            'game_id': 'g3',
            'date': '2026-02-24',
            'home_team_name': 'Home C',
            'away_team_name': 'Away C',
            'best_edge': 12.0,
            'best_pick': 'home',
            'home_ml': -300,
            'away_ml': 240,
            'model_home_prob': 0.81,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'home', 'avg_prob': 0.82},
        },
    ]

    monkeypatch.setattr(db_module, 'get_connection', _new_conn)
    monkeypatch.setattr(betting_page, 'get_all_conferences', lambda: [])
    monkeypatch.setattr(betting_page, 'get_betting_games', lambda: [dict(g) for g in sample_games])
    monkeypatch.setattr(betting_page, 'analyze_games', _mock_risk_preview)
    monkeypatch.setattr(betting_page, '_get_calibrator', lambda: None)

    ctx = betting_page.build_betting_page_context()

    assert ctx['parlay_legs'] == []
    assert ctx['parlay_american'] == 0
    assert ctx['parlay_payout'] == 0
    assert ctx['parlay_prob'] == 0.0
    assert ctx['parlay_calibrated_prob'] == 0.0


def test_parlay_fallback_builds_sane_parlay_with_three_plus_legs(monkeypatch, tmp_path):
    from web.services import betting_page
    import scripts.database as db_module

    db_path = tmp_path / "betting.db"
    _init_empty_betting_tables(db_path)

    def _new_conn():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    sample_games = [
        {
            'game_id': 'g1',
            'date': '2026-02-24',
            'home_team_name': 'Home A',
            'away_team_name': 'Away A',
            'best_edge': 10.0,
            'best_pick': 'home',
            'home_ml': -140,
            'away_ml': 120,
            'model_home_prob': 0.75,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'home', 'avg_prob': 0.78},
        },
        {
            'game_id': 'g2',
            'date': '2026-02-24',
            'home_team_name': 'Home B',
            'away_team_name': 'Away B',
            'best_edge': 9.0,
            'best_pick': 'away',
            'home_ml': -145,
            'away_ml': 125,
            'model_home_prob': 0.26,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'away', 'avg_prob': 0.74},
        },
        {
            'game_id': 'g3',
            'date': '2026-02-24',
            'home_team_name': 'Home C',
            'away_team_name': 'Away C',
            'best_edge': 11.0,
            'best_pick': 'home',
            'home_ml': -160,
            'away_ml': 140,
            'model_home_prob': 0.79,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'home', 'avg_prob': 0.81},
        },
        {
            'game_id': 'g4',
            'date': '2026-02-24',
            'home_team_name': 'Home D',
            'away_team_name': 'Away D',
            'best_edge': 8.5,
            'best_pick': 'away',
            'home_ml': -135,
            'away_ml': 115,
            'model_home_prob': 0.22,
            'model_agreement': {'count': 8, 'total': 12, 'pick': 'away', 'avg_prob': 0.73},
        },
    ]

    monkeypatch.setattr(db_module, 'get_connection', _new_conn)
    monkeypatch.setattr(betting_page, 'get_all_conferences', lambda: [])
    monkeypatch.setattr(betting_page, 'get_betting_games', lambda: [dict(g) for g in sample_games])
    monkeypatch.setattr(betting_page, 'analyze_games', _mock_risk_preview)
    monkeypatch.setattr(betting_page, '_get_calibrator', lambda: None)

    ctx = betting_page.build_betting_page_context()

    assert len(ctx['parlay_legs']) == 4
    assert ctx['parlay_american'] > 0
    assert ctx['parlay_payout'] > 10
    assert 0.0 < ctx['parlay_prob'] < 100.0
    assert 0.0 < ctx['parlay_calibrated_prob'] < 100.0


def test_risk_engine_page_service_smoke(monkeypatch):
    from web.services import risk_engine_page

    monkeypatch.setattr(
        risk_engine_page,
        'analyze_games',
        lambda: {
            'date': '2026-02-24',
            'bets': [{'type': 'ML', 'pick_team_name': 'Test', 'edge': 5.0}],
            'rejections': [],
            'error': None,
        },
    )

    rows = [
        {'total': 2, 'settled': 2, 'wins': 1, 'losses': 1, 'profit': 10.0, 'staked': 200.0},
        {'total': 1, 'settled': 1, 'wins': 1, 'losses': 0, 'profit': 30.0, 'staked': 100.0},
        {'total': 1, 'settled': 0, 'wins': 0, 'losses': 0, 'profit': 0.0, 'staked': 0.0},
    ]

    class FakeCursor:
        def __init__(self, row_values):
            self._rows = iter(row_values)

        def execute(self, _query):
            return None

        def fetchone(self):
            return next(self._rows)

    class FakeConn:
        def __init__(self, row_values):
            self._cursor = FakeCursor(row_values)
            self.closed = False

        def cursor(self):
            return self._cursor

        def close(self):
            self.closed = True

    fake_conn = FakeConn(rows)
    monkeypatch.setattr(risk_engine_page, 'get_connection', lambda: fake_conn)

    ctx = risk_engine_page.build_risk_engine_page_context()

    assert ctx['risk_preview']['date'] == '2026-02-24'
    assert ctx['risk_pnl']['total_bets'] == 4
    assert ctx['risk_pnl']['settled_bets'] == 3
    assert ctx['risk_pnl']['wins'] == 2
    assert ctx['risk_pnl']['losses'] == 1
    assert ctx['risk_pnl']['profit'] == 40.0
    assert ctx['risk_pnl']['staked'] == 300.0
    assert pytest.approx(ctx['risk_pnl']['roi_pct'], rel=1e-6) == (40.0 / 300.0) * 100.0


def _minimal_betting_context():
    return {
        'games': [],
        'confident_bets': [],
        'ev_bets': [],
        'parlay_legs': [],
        'parlay_american': 0,
        'parlay_payout': 0,
        'parlay_prob': 0.0,
        'best_totals': [],
        'conferences': [],
        'selected_conference': '',
        'risk_engine': {
            'mode': 'fixed',
            'bankroll': 5000.0,
            'kelly_fraction': 0.25,
            'min_stake': 25.0,
            'max_stake': 250.0,
        },
        'risk_preview': {'date': None, 'bets': [], 'rejections': 0, 'error': None},
        'show_experimental': True,
        'spreads_enabled': False,
        'v2_thresholds': {
            'ml_favorite': 8.0,
            'ml_underdog': 15.0,
            'totals_runs': 3.0,
            'underdog_discount': 0.5,
        },
    }


def _minimal_risk_engine_context():
    return {
        'risk_engine': {
            'mode': 'fixed',
            'bankroll': 5000.0,
            'bankroll_peak': 5000.0,
            'kelly_fraction': 0.25,
            'min_stake': 25.0,
            'max_stake': 250.0,
            'drawdown_threshold': 0.1,
            'drawdown_multiplier': 0.5,
        },
        'risk_preview': {'date': None, 'bets': [], 'rejections': [], 'error': None},
        'risk_pnl': {
            'mode': 'fractional_kelly',
            'total_bets': 0,
            'settled_bets': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'roi_pct': 0.0,
            'staked': 0.0,
        },
    }


@pytest.fixture
def betting_test_app(monkeypatch):
    import web.blueprints.betting as betting_blueprint_module

    monkeypatch.setattr(
        betting_blueprint_module,
        'build_betting_page_context',
        lambda conference='': {**_minimal_betting_context(), 'selected_conference': conference},
    )
    monkeypatch.setattr(
        betting_blueprint_module,
        'build_risk_engine_page_context',
        _minimal_risk_engine_context,
    )

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / 'web' / 'templates'),
    )
    app.config['TESTING'] = True
    app.add_template_filter(lambda value: f"+{value}" if value and value > 0 else (str(value) if value is not None else 'N/A'), 'format_odds')
    app.add_template_filter(lambda value: f"{value * 100:.1f}%" if value is not None else 'N/A', 'format_pct')
    app.add_template_filter(lambda value: f"+{value:.1f}%" if value and value > 0 else (f"{value:.1f}%" if value is not None else 'N/A'), 'format_edge')
    app.register_blueprint(betting_blueprint_module.betting_bp)
    return app


def test_betting_route_renders_200(betting_test_app):
    client = betting_test_app.test_client()
    resp = client.get('/betting?conference=SEC&view=experimental')
    assert resp.status_code == 200


def test_risk_engine_route_renders_200(betting_test_app):
    client = betting_test_app.test_client()
    resp = client.get('/experimental/risk-engine')
    assert resp.status_code == 200
