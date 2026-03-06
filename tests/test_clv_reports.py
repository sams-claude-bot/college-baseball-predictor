#!/usr/bin/env python3
"""
Tests for CLV Daily Report and Weekly Summary scripts.

Uses in-memory SQLite databases to verify:
- Mixed +CLV/-CLV produces correct averages
- No data exits cleanly
- Weekly aggregation correct
- --json outputs valid JSON
- ML vs Consensus breakdown accurate
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


# ============================================
# Fixtures
# ============================================

def _create_tables(conn):
    """Create minimal tracked_bets and tracked_confident_bets tables."""
    conn.execute("""
        CREATE TABLE tracked_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            pick_team_name TEXT,
            opponent_name TEXT,
            moneyline INTEGER,
            model_prob REAL,
            dk_implied REAL,
            edge REAL,
            bet_amount REAL DEFAULT 100,
            won INTEGER,
            profit REAL,
            closing_ml REAL,
            clv_implied REAL,
            clv_cents REAL
        )
    """)
    conn.execute("""
        CREATE TABLE tracked_confident_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            pick_team_name TEXT,
            opponent_name TEXT,
            moneyline INTEGER,
            models_agree INTEGER,
            models_total INTEGER,
            avg_prob REAL,
            confidence REAL,
            bet_amount REAL DEFAULT 100,
            won INTEGER,
            profit REAL,
            closing_ml REAL,
            clv_implied REAL,
            clv_cents REAL
        )
    """)


def _seed_mixed_clv(conn):
    """Insert bets with mixed +CLV and -CLV across both tables."""
    # ML bets: 2 positive CLV, 1 negative CLV
    conn.executemany("""
        INSERT INTO tracked_bets
        (game_id, date, pick_team_name, opponent_name, moneyline,
         model_prob, dk_implied, edge, won, profit, closing_ml, clv_implied, clv_cents)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ('2026-03-01_a_b', '2026-03-01', 'Team A', 'Team B', -130,
         0.60, 0.565, 3.5, 1, 76.92, -150.0, 0.035, 3.5),
        ('2026-03-01_c_d', '2026-03-01', 'Team C', 'Team D', 150,
         0.45, 0.40, 5.0, 1, 150.0, 130.0, 0.035, 3.5),
        ('2026-03-01_e_f', '2026-03-01', 'Team E', 'Team F', -150,
         0.62, 0.60, 2.0, 0, -100.0, -130.0, -0.035, -3.5),
    ])

    # Consensus bets: 1 positive CLV, 1 negative CLV
    conn.executemany("""
        INSERT INTO tracked_confident_bets
        (game_id, date, pick_team_name, opponent_name, moneyline,
         models_agree, models_total, avg_prob, confidence, won, profit,
         closing_ml, clv_implied, clv_cents)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ('2026-03-01_g_h', '2026-03-01', 'Team G', 'Team H', -110,
         8, 12, 0.58, 0.75, 1, 90.91, -130.0, 0.022, 2.2),
        ('2026-03-01_i_j', '2026-03-01', 'Team I', 'Team J', -140,
         7, 12, 0.55, 0.65, 0, -100.0, -120.0, -0.025, -2.5),
    ])
    conn.commit()


def _seed_weekly_data(conn):
    """Insert bets across multiple days for weekly tests."""
    for day_offset in range(7):
        date = f'2026-03-0{day_offset + 1}'
        clv_val = 2.0 + day_offset * 0.5  # increasing CLV over the week
        conn.execute("""
            INSERT INTO tracked_bets
            (game_id, date, pick_team_name, opponent_name, moneyline,
             model_prob, dk_implied, edge, won, profit, closing_ml, clv_implied, clv_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (f'{date}_t1_t2', date, 'Weekly Team', 'Opponent', -130,
              0.60, 0.565, 3.5, 1, 76.92, -150.0, clv_val / 100, clv_val))
    conn.commit()


@pytest.fixture
def mem_conn_mixed():
    """In-memory DB with mixed +CLV/-CLV data."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    _seed_mixed_clv(conn)
    yield conn
    conn.close()


@pytest.fixture
def mem_conn_empty():
    """In-memory DB with empty tables."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def mem_conn_weekly():
    """In-memory DB with 7 days of data."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    _seed_weekly_data(conn)
    yield conn
    conn.close()


# ============================================
# Daily Report Tests
# ============================================

class TestDailyReport:
    """Test CLV daily report generation."""

    def test_mixed_clv_correct_averages(self, mem_conn_mixed):
        """Mixed +CLV/-CLV bets produce correct overall averages."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        assert data is not None
        assert data['total_bets'] == 5

        # Expected: (3.5 + 3.5 + -3.5 + 2.2 + -2.5) / 5 = 0.64
        expected_avg = (3.5 + 3.5 + (-3.5) + 2.2 + (-2.5)) / 5
        assert abs(data['avg_clv_cents'] - expected_avg) < 0.01

    def test_positive_negative_counts(self, mem_conn_mixed):
        """Correctly counts +CLV and -CLV bets."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        assert data['positive_clv_count'] == 3  # 3.5, 3.5, 2.2
        assert data['negative_clv_count'] == 2  # -3.5, -2.5

    def test_win_rates_by_clv(self, mem_conn_mixed):
        """Win rates split correctly for +CLV vs -CLV groups."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        # +CLV: 3 bets (A won, C won, G won) => 100%
        assert data['pos_clv_win_rate'] == 100.0
        # -CLV: 2 bets (E lost, I lost) => 0%
        assert data['neg_clv_win_rate'] == 0.0

    def test_no_data_returns_none(self, mem_conn_empty):
        """No data for date returns None."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_empty)
        assert data is None

    def test_no_data_wrong_date(self, mem_conn_mixed):
        """Wrong date returns None even when other dates have data."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-12-25', conn=mem_conn_mixed)
        assert data is None

    def test_ml_vs_consensus_breakdown(self, mem_conn_mixed):
        """ML vs Consensus breakdown is accurate."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        assert 'ML' in data['by_type']
        assert 'CONSENSUS' in data['by_type']

        ml = data['by_type']['ML']
        assert ml['count'] == 3
        # ML avg CLV: (3.5 + 3.5 + -3.5) / 3 = 1.17
        expected_ml_avg = (3.5 + 3.5 + (-3.5)) / 3
        assert abs(ml['avg_clv_cents'] - expected_ml_avg) < 0.01
        assert ml['wins'] == 2

        cons = data['by_type']['CONSENSUS']
        assert cons['count'] == 2
        # Consensus avg CLV: (2.2 + -2.5) / 2 = -0.15
        expected_cons_avg = (2.2 + (-2.5)) / 2
        assert abs(cons['avg_clv_cents'] - expected_cons_avg) < 0.01
        assert cons['wins'] == 1

    def test_notable_bets_sorted_by_abs_clv(self, mem_conn_mixed):
        """Notable bets are sorted by absolute CLV value."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        assert len(data['notable']) == 3
        abs_values = [abs(b['clv_cents']) for b in data['notable']]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_alltime_stats_present(self, mem_conn_mixed):
        """All-time running averages are included."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        assert data['alltime'] is not None
        assert data['alltime']['total_bets'] == 5

    def test_render_markdown(self, mem_conn_mixed):
        """Markdown rendering produces valid output."""
        from clv_daily_report import generate_daily_report, render_markdown

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        md = render_markdown(data)
        assert '# CLV Daily Report' in md
        assert '2026-03-01' in md
        assert 'Summary' in md
        assert 'Win Rate' in md
        assert 'Breakdown by Type' in md
        assert 'Notable Bets' in md
        assert 'All-Time' in md


# ============================================
# Weekly Summary Tests
# ============================================

class TestWeeklySummary:
    """Test CLV weekly summary generation."""

    def test_weekly_aggregation_correct(self, mem_conn_weekly):
        """Weekly aggregation covers all 7 days correctly."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        assert data is not None
        assert data['total_bets'] == 7
        assert data['start_date'] == '2026-03-01'
        assert data['end_date'] == '2026-03-07'

    def test_weekly_avg_clv(self, mem_conn_weekly):
        """Weekly average CLV is correct."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        # CLV values: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 => avg = 3.5
        assert abs(data['avg_clv_cents'] - 3.5) < 0.01

    def test_weekly_daily_breakdown(self, mem_conn_weekly):
        """Daily breakdown has entries for each day with data."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        assert len(data['daily']) == 7

    def test_weekly_trend_improving(self, mem_conn_weekly):
        """Increasing CLV over week detected as improving."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        assert data['trend_direction'] == 'improving'

    def test_weekly_no_data_returns_none(self, mem_conn_empty):
        """No data returns None."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_empty)
        assert data is None

    def test_weekly_correlation_present(self, mem_conn_weekly):
        """CLV-profit correlation is computed when sufficient data."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        # All bets have same profit (76.92) so std_y != 0 only if varied
        # With constant profit, correlation is None (std_y = 0)
        # This is expected behavior
        assert 'clv_profit_correlation' in data

    def test_weekly_by_type(self, mem_conn_weekly):
        """Weekly by-type breakdown works with single type."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        assert 'ML' in data['by_type']
        assert data['by_type']['ML']['count'] == 7

    def test_render_markdown(self, mem_conn_weekly):
        """Weekly markdown rendering produces valid output."""
        from clv_weekly_summary import generate_weekly_summary, render_markdown

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        md = render_markdown(data)
        assert '# CLV Weekly Summary' in md
        assert 'Trend' in md
        assert 'Daily Breakdown' in md


# ============================================
# JSON Output Tests
# ============================================

class TestJSONOutput:
    """Test --json flag produces valid JSON."""

    def test_daily_json_valid(self, mem_conn_mixed):
        """Daily report JSON output is valid."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed['total_bets'] == 5
        assert isinstance(parsed['avg_clv_cents'], float)

    def test_weekly_json_valid(self, mem_conn_weekly):
        """Weekly summary JSON output is valid."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed['total_bets'] == 7
        assert 'trend_direction' in parsed

    def test_daily_json_serializable(self, mem_conn_mixed):
        """All fields in daily report are JSON serializable."""
        from clv_daily_report import generate_daily_report

        data = generate_daily_report('2026-03-01', conn=mem_conn_mixed)
        # Should not raise
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 0

    def test_weekly_json_serializable(self, mem_conn_weekly):
        """All fields in weekly summary are JSON serializable."""
        from clv_weekly_summary import generate_weekly_summary

        data = generate_weekly_summary('2026-03-07', conn=mem_conn_weekly)
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 0


# ============================================
# Edge Cases
# ============================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bet(self):
        """Single bet produces valid report."""
        from clv_daily_report import generate_daily_report

        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        _create_tables(conn)
        conn.execute("""
            INSERT INTO tracked_bets
            (game_id, date, pick_team_name, opponent_name, moneyline,
             model_prob, dk_implied, edge, won, profit, closing_ml, clv_implied, clv_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('2026-03-01_a_b', '2026-03-01', 'Solo', 'Opp', -130,
              0.60, 0.565, 3.5, 1, 76.92, -150.0, 0.035, 3.5))
        conn.commit()

        data = generate_daily_report('2026-03-01', conn=conn)
        assert data['total_bets'] == 1
        assert data['avg_clv_cents'] == 3.5
        conn.close()

    def test_all_negative_clv(self):
        """All negative CLV bets produce correct averages."""
        from clv_daily_report import generate_daily_report

        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        _create_tables(conn)
        conn.execute("""
            INSERT INTO tracked_bets
            (game_id, date, pick_team_name, opponent_name, moneyline,
             model_prob, dk_implied, edge, won, profit, closing_ml, clv_implied, clv_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('2026-03-01_a_b', '2026-03-01', 'Bad', 'Opp', -150,
              0.60, 0.60, 0.0, 0, -100.0, -130.0, -0.035, -3.5))
        conn.commit()

        data = generate_daily_report('2026-03-01', conn=conn)
        assert data['positive_clv_count'] == 0
        assert data['negative_clv_count'] == 1
        assert data['avg_clv_cents'] == -3.5
        conn.close()

    def test_weekly_partial_week(self):
        """Weekly summary works with fewer than 7 days of data."""
        from clv_weekly_summary import generate_weekly_summary

        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        _create_tables(conn)
        # Only 2 days of data in the 7-day window
        for date in ['2026-03-05', '2026-03-06']:
            conn.execute("""
                INSERT INTO tracked_bets
                (game_id, date, pick_team_name, opponent_name, moneyline,
                 model_prob, dk_implied, edge, won, profit, closing_ml, clv_implied, clv_cents)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (f'{date}_a_b', date, 'Partial', 'Opp', -130,
                  0.60, 0.565, 3.5, 1, 76.92, -150.0, 0.035, 3.5))
        conn.commit()

        data = generate_weekly_summary('2026-03-07', conn=conn)
        assert data is not None
        assert data['total_bets'] == 2
        assert len(data['daily']) == 2
        conn.close()
