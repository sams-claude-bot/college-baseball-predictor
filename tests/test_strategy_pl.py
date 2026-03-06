#!/usr/bin/env python3
"""
Tests for strategy_pl_report.py

Uses in-memory SQLite to verify:
- Per-strategy P&L calculations across all strategies
- Empty strategy shows 0-0, $0
- Parlay profit = payout - amount when won
- ROI = profit / wagered * 100
- Date filtering works
- --json output is valid
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from strategy_pl_report import (
    build_daily_pl,
    build_report,
    compute_strategy_stats,
    generate_report,
    query_strategy_data,
)


@pytest.fixture
def mem_conn():
    """In-memory SQLite with all four tracking tables populated."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # tracked_bets (EV Moneyline)
    conn.execute("""
        CREATE TABLE tracked_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, date TEXT, pick_team_id TEXT,
            pick_team_name TEXT, opponent_name TEXT, is_home INTEGER,
            moneyline INTEGER, model_prob REAL, dk_implied REAL,
            edge REAL, bet_amount REAL DEFAULT 100,
            won INTEGER, profit REAL, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # tracked_confident_bets (Consensus)
    conn.execute("""
        CREATE TABLE tracked_confident_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, date TEXT, pick_team_id TEXT,
            pick_team_name TEXT, opponent_name TEXT, is_home INTEGER,
            moneyline INTEGER, models_agree INTEGER, models_total INTEGER,
            avg_prob REAL, confidence REAL, bet_amount REAL DEFAULT 100,
            won INTEGER, profit REAL, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # tracked_bets_spreads (Spreads and Totals)
    conn.execute("""
        CREATE TABLE tracked_bets_spreads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, date TEXT, bet_type TEXT,
            pick TEXT, line REAL, odds REAL,
            model_projection REAL, edge REAL,
            bet_amount REAL DEFAULT 100,
            won INTEGER, profit REAL, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # tracked_parlays
    conn.execute("""
        CREATE TABLE tracked_parlays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, legs_json TEXT, num_legs INTEGER,
            american_odds INTEGER, decimal_odds REAL,
            model_prob REAL, bet_amount REAL DEFAULT 25,
            payout REAL, won INTEGER, profit REAL,
            legs_won INTEGER, recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    yield conn
    conn.close()


def _seed_ev(conn, date, won, profit, edge=10.0, bet_amount=100):
    conn.execute(
        "INSERT INTO tracked_bets (game_id, date, pick_team_name, moneyline, model_prob, dk_implied, edge, bet_amount, won, profit) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (f"g-ev-{date}", date, "TeamA", -150, 0.7, 0.6, edge, bet_amount, won, profit),
    )


def _seed_consensus(conn, date, won, profit, edge=None, bet_amount=100):
    conn.execute(
        "INSERT INTO tracked_confident_bets (game_id, date, pick_team_name, moneyline, models_agree, models_total, avg_prob, confidence, bet_amount, won, profit) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"g-con-{date}", date, "TeamB", -200, 9, 12, 0.75, 0.8, bet_amount, won, profit),
    )


def _seed_spread(conn, date, won, profit, edge=3.0, bet_amount=100):
    conn.execute(
        "INSERT INTO tracked_bets_spreads (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount, won, profit) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"g-sp-{date}", date, "spread", "TeamC -3.5", -3.5, -110, -5.0, edge, bet_amount, won, profit),
    )


def _seed_total(conn, date, won, profit, edge=3.5, bet_amount=100):
    conn.execute(
        "INSERT INTO tracked_bets_spreads (game_id, date, bet_type, pick, line, odds, model_projection, edge, bet_amount, won, profit) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"g-tot-{date}", date, "total", "OVER", 12.5, -110, 16.0, edge, bet_amount, won, profit),
    )


def _seed_parlay(conn, date, won, payout, bet_amount=25):
    profit = (payout - bet_amount) if won else -bet_amount
    conn.execute(
        "INSERT INTO tracked_parlays (date, legs_json, num_legs, american_odds, decimal_odds, model_prob, bet_amount, payout, won, profit, legs_won) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (date, '[]', 3, 500, 6.0, 0.2, bet_amount, payout, won, profit, 3 if won else 1),
    )


class TestComputeStrategyStats:
    def test_empty_returns_zeroes(self):
        stats = compute_strategy_stats([])
        assert stats["bets"] == 0
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["profit"] == 0.0
        assert stats["roi"] == 0.0
        assert stats["win_pct"] == 0.0

    def test_single_winning_bet(self):
        rows = [{"won": 1, "profit": 90.91, "bet_amount": 100, "edge": 10.0}]
        stats = compute_strategy_stats(rows)
        assert stats["bets"] == 1
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["win_pct"] == 100.0
        assert stats["profit"] == 90.91
        assert stats["roi"] == round(90.91 / 100 * 100, 1)

    def test_mixed_wins_losses(self):
        rows = [
            {"won": 1, "profit": 100.0, "bet_amount": 100, "edge": 8.0},
            {"won": 0, "profit": -100.0, "bet_amount": 100, "edge": 12.0},
            {"won": 1, "profit": 50.0, "bet_amount": 100, "edge": 6.0},
        ]
        stats = compute_strategy_stats(rows)
        assert stats["bets"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["profit"] == 50.0
        assert stats["win_pct"] == round(2 / 3 * 100, 1)
        assert stats["roi"] == round(50.0 / 300 * 100, 1)
        assert stats["avg_edge"] == round((8.0 + 12.0 + 6.0) / 3, 2)


class TestRoiCalculation:
    def test_roi_formula(self):
        """ROI = profit / wagered * 100"""
        rows = [
            {"won": 1, "profit": 200.0, "bet_amount": 100, "edge": 5.0},
            {"won": 0, "profit": -100.0, "bet_amount": 100, "edge": 5.0},
        ]
        stats = compute_strategy_stats(rows)
        expected_roi = round(100.0 / 200.0 * 100, 1)
        assert stats["roi"] == expected_roi

    def test_roi_negative(self):
        rows = [
            {"won": 0, "profit": -100.0, "bet_amount": 100, "edge": 5.0},
            {"won": 0, "profit": -100.0, "bet_amount": 100, "edge": 5.0},
        ]
        stats = compute_strategy_stats(rows)
        assert stats["roi"] == -100.0


class TestParlayProfit:
    def test_parlay_won_profit(self, mem_conn):
        """Parlay profit = payout - bet_amount when won."""
        _seed_parlay(mem_conn, "2026-02-20", won=1, payout=150.0, bet_amount=25)
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        parlay_rows = raw["Parlays"]
        assert len(parlay_rows) == 1
        assert parlay_rows[0]["profit"] == 125.0  # 150 - 25

    def test_parlay_lost_profit(self, mem_conn):
        """Parlay profit = -bet_amount when lost."""
        _seed_parlay(mem_conn, "2026-02-20", won=0, payout=150.0, bet_amount=25)
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        parlay_rows = raw["Parlays"]
        assert parlay_rows[0]["profit"] == -25.0


class TestQueryStrategyData:
    def test_all_strategies_populated(self, mem_conn):
        _seed_ev(mem_conn, "2026-02-20", won=1, profit=90.0)
        _seed_consensus(mem_conn, "2026-02-20", won=1, profit=80.0)
        _seed_spread(mem_conn, "2026-02-20", won=0, profit=-100.0)
        _seed_total(mem_conn, "2026-02-20", won=1, profit=90.0)
        _seed_parlay(mem_conn, "2026-02-20", won=0, payout=150.0)
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        assert len(raw["EV Moneyline"]) == 1
        assert len(raw["Consensus"]) == 1
        assert len(raw["Spreads"]) == 1
        assert len(raw["Totals"]) == 1
        assert len(raw["Parlays"]) == 1

    def test_correct_per_strategy_stats(self, mem_conn):
        _seed_ev(mem_conn, "2026-02-20", won=1, profit=90.0, edge=10.0)
        _seed_ev(mem_conn, "2026-02-21", won=0, profit=-100.0, edge=8.0)
        _seed_consensus(mem_conn, "2026-02-20", won=1, profit=80.0)
        _seed_consensus(mem_conn, "2026-02-21", won=1, profit=70.0)
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        ev_stats = compute_strategy_stats(raw["EV Moneyline"])
        con_stats = compute_strategy_stats(raw["Consensus"])

        assert ev_stats["bets"] == 2
        assert ev_stats["wins"] == 1
        assert ev_stats["losses"] == 1
        assert ev_stats["profit"] == -10.0

        assert con_stats["bets"] == 2
        assert con_stats["wins"] == 2
        assert con_stats["losses"] == 0
        assert con_stats["profit"] == 150.0

    def test_empty_strategy_zeroes(self, mem_conn):
        """Empty strategy shows 0-0, $0."""
        _seed_ev(mem_conn, "2026-02-20", won=1, profit=90.0)
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        consensus_stats = compute_strategy_stats(raw["Consensus"])
        assert consensus_stats["bets"] == 0
        assert consensus_stats["wins"] == 0
        assert consensus_stats["losses"] == 0
        assert consensus_stats["profit"] == 0.0
        assert consensus_stats["roi"] == 0.0


class TestDateFiltering:
    def test_only_returns_in_range(self, mem_conn):
        _seed_ev(mem_conn, "2026-02-15", won=1, profit=90.0)  # before range
        _seed_ev(mem_conn, "2026-02-20", won=1, profit=80.0)  # in range
        _seed_ev(mem_conn, "2026-03-10", won=1, profit=70.0)  # after range
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        assert len(raw["EV Moneyline"]) == 1
        assert raw["EV Moneyline"][0]["profit"] == 80.0

    def test_boundary_dates_inclusive(self, mem_conn):
        _seed_ev(mem_conn, "2026-02-18", won=1, profit=50.0)  # exact start
        _seed_ev(mem_conn, "2026-03-06", won=1, profit=60.0)  # exact end
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        assert len(raw["EV Moneyline"]) == 2


class TestDailyTimeline:
    def test_daily_pl_aggregation(self, mem_conn):
        _seed_ev(mem_conn, "2026-02-20", won=1, profit=90.0)
        _seed_consensus(mem_conn, "2026-02-20", won=0, profit=-100.0)
        _seed_ev(mem_conn, "2026-02-21", won=1, profit=80.0)
        mem_conn.commit()

        raw = query_strategy_data(mem_conn, "2026-02-18", "2026-03-06")
        timeline = build_daily_pl(raw)

        assert len(timeline) == 2
        assert timeline[0]["date"] == "2026-02-20"
        assert timeline[0]["daily_pl"] == -10.0  # 90 - 100
        assert timeline[0]["cumulative_pl"] == -10.0
        assert timeline[1]["date"] == "2026-02-21"
        assert timeline[1]["daily_pl"] == 80.0
        assert timeline[1]["cumulative_pl"] == 70.0


class TestGenerateReport:
    def test_full_report_flow(self, mem_conn):
        _seed_ev(mem_conn, "2026-02-20", won=1, profit=90.0)
        _seed_consensus(mem_conn, "2026-02-20", won=1, profit=80.0)
        _seed_total(mem_conn, "2026-02-21", won=0, profit=-100.0)
        _seed_parlay(mem_conn, "2026-02-22", won=1, payout=150.0, bet_amount=25)
        mem_conn.commit()

        stats, timeline, md = generate_report(
            conn=mem_conn, start_date="2026-02-18", end_date="2026-03-06"
        )

        assert "EV Moneyline" in stats
        assert "Consensus" in stats
        assert "Spreads" in stats
        assert "Totals" in stats
        assert "Parlays" in stats

        assert stats["EV Moneyline"]["bets"] == 1
        assert stats["Parlays"]["profit"] == 125.0
        assert len(timeline) == 3
        assert "Strategy P&L Report" in md
        assert "TOTAL" in md


class TestJsonOutput:
    def test_json_flag_produces_valid_json(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts" / "strategy_pl_report.py"), "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "period" in data
        assert "strategies" in data
        assert "daily_timeline" in data
        for name in ("EV Moneyline", "Consensus", "Spreads", "Totals", "Parlays"):
            assert name in data["strategies"]


class TestBuildReport:
    def test_markdown_structure(self):
        strategy_stats = {
            "EV Moneyline": compute_strategy_stats([]),
            "Consensus": compute_strategy_stats([
                {"won": 1, "profit": 80.0, "bet_amount": 100, "edge": 12.0},
            ]),
        }
        md = build_report(strategy_stats, [], "2026-02-18", "2026-03-06")
        assert "Strategy Breakdown" in md
        assert "EV Moneyline" in md
        assert "Consensus" in md
        assert "TOTAL" in md
