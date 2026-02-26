"""Tests for win quality / resume impact service."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from web.services.win_quality import (
    get_quadrant,
    compute_win_quality,
    compute_quadrant_record,
    compute_elo_ranks,
    get_game_resume_impact,
    QUADRANT_COLORS,
)


# ── Quadrant assignment ──────────────────────────────────────────────

class TestGetQuadrant:
    def test_q1_boundaries(self):
        assert get_quadrant(1) == "Q1"
        assert get_quadrant(50) == "Q1"

    def test_q2_boundaries(self):
        assert get_quadrant(51) == "Q2"
        assert get_quadrant(100) == "Q2"

    def test_q3_boundaries(self):
        assert get_quadrant(101) == "Q3"
        assert get_quadrant(200) == "Q3"

    def test_q4_boundaries(self):
        assert get_quadrant(201) == "Q4"
        assert get_quadrant(310) == "Q4"


# ── Win quality scoring ──────────────────────────────────────────────

class TestComputeWinQuality:
    def test_beating_top_team_is_resume_builder(self):
        result = compute_win_quality(team_elo_rank=50, opponent_elo_rank=5, total_teams=310)
        assert result["quadrant"] == "Q1"
        assert result["win_quality"] > 0.6
        assert result["win_label"] == "Resume Builder"

    def test_beating_bottom_team_has_no_value(self):
        result = compute_win_quality(team_elo_rank=10, opponent_elo_rank=300, total_teams=310)
        assert result["quadrant"] == "Q4"
        assert result["win_quality"] < 0
        assert result["win_label"] == "No Value"

    def test_beating_top_beats_bottom(self):
        top = compute_win_quality(team_elo_rank=50, opponent_elo_rank=10, total_teams=310)
        bot = compute_win_quality(team_elo_rank=50, opponent_elo_rank=280, total_teams=310)
        assert top["win_quality"] > bot["win_quality"]

    def test_loss_to_bad_team_worse_than_loss_to_good(self):
        bad = compute_win_quality(team_elo_rank=10, opponent_elo_rank=290, total_teams=310)
        good = compute_win_quality(team_elo_rank=10, opponent_elo_rank=20, total_teams=310)
        assert bad["loss_damage"] < good["loss_damage"]

    def test_loss_to_equal_team_is_slightly_negative(self):
        result = compute_win_quality(team_elo_rank=150, opponent_elo_rank=150, total_teams=310)
        assert result["loss_damage"] < 0

    def test_loss_to_much_better_team_is_understandable(self):
        result = compute_win_quality(team_elo_rank=250, opponent_elo_rank=10, total_teams=310)
        assert result["loss_damage"] >= 0
        assert result["loss_label"] == "Understandable"

    def test_win_quality_clamped(self):
        result = compute_win_quality(team_elo_rank=1, opponent_elo_rank=1, total_teams=310)
        assert result["win_quality"] <= 1.0
        result2 = compute_win_quality(team_elo_rank=310, opponent_elo_rank=310, total_teams=310)
        assert result2["win_quality"] >= -0.5

    def test_loss_damage_clamped(self):
        result = compute_win_quality(team_elo_rank=1, opponent_elo_rank=310, total_teams=310)
        assert result["loss_damage"] >= -1.0
        assert result["loss_damage"] <= 0.5

    def test_same_rank_teams(self):
        result = compute_win_quality(team_elo_rank=100, opponent_elo_rank=100, total_teams=310)
        assert result["quadrant"] == "Q2"
        assert -0.5 <= result["win_quality"] <= 1.0
        assert -1.0 <= result["loss_damage"] <= 0.5

    def test_q2_game_labels(self):
        result = compute_win_quality(team_elo_rank=50, opponent_elo_rank=75, total_teams=310)
        assert result["quadrant"] == "Q2"
        assert result["win_label"] in ("Resume Builder", "Solid Win")


# ── Quadrant record ──────────────────────────────────────────────────

class TestComputeQuadrantRecord:
    def _make_row(self, home_id, away_id, winner_id):
        return {"home_team_id": home_id, "away_team_id": away_id, "winner_id": winner_id}

    @patch("web.services.win_quality.get_connection")
    def test_basic_record(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn

        # team-a beat a Q1 opponent and lost to a Q4 opponent
        rows = [
            self._make_row("team-a", "top-team", "team-a"),    # win vs Q1
            self._make_row("bad-team", "team-a", "bad-team"),  # loss to Q4
        ]
        mock_conn.execute.return_value.fetchall.return_value = rows

        elo_ranks = {"team-a": 50, "top-team": 10, "bad-team": 250}
        record = compute_quadrant_record("team-a", elo_ranks)

        assert record["Q1"] == "1-0"
        assert record["Q4"] == "0-1"
        assert record["Q2"] == "0-0"
        assert record["Q3"] == "0-0"

    @patch("web.services.win_quality.get_connection")
    def test_empty_record(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        record = compute_quadrant_record("team-x", {"team-x": 100})
        for q in ("Q1", "Q2", "Q3", "Q4"):
            assert record[q] == "0-0"

    @patch("web.services.win_quality.get_connection")
    def test_opponent_not_in_rankings_skipped(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn

        rows = [self._make_row("team-a", "unknown-team", "team-a")]
        mock_conn.execute.return_value.fetchall.return_value = rows

        record = compute_quadrant_record("team-a", {"team-a": 50})
        for q in ("Q1", "Q2", "Q3", "Q4"):
            assert record[q] == "0-0"


# ── Full resume impact ───────────────────────────────────────────────

class TestGetGameResumeImpact:
    @patch("web.services.win_quality.compute_quadrant_record")
    @patch("web.services.win_quality.compute_elo_ranks")
    def test_returns_both_teams(self, mock_ranks, mock_qr):
        mock_ranks.return_value = {"team-a": 5, "team-b": 150}
        mock_qr.return_value = {"Q1": "0-0", "Q2": "0-0", "Q3": "0-0", "Q4": "0-0"}

        result = get_game_resume_impact("team-a", "team-b")

        assert "home" in result
        assert "away" in result
        assert result["home"]["elo_rank"] == 5
        assert result["away"]["elo_rank"] == 150
        # Home facing away (rank 150 → Q3), away facing home (rank 5 → Q1)
        assert result["home"]["quadrant"] == "Q3"
        assert result["away"]["quadrant"] == "Q1"

    @patch("web.services.win_quality.compute_elo_ranks")
    def test_returns_none_for_missing_team(self, mock_ranks):
        mock_ranks.return_value = {"team-a": 5}

        result = get_game_resume_impact("team-a", "missing-team")
        assert result is None


# ── Quadrant colors constant ─────────────────────────────────────────

class TestQuadrantColors:
    def test_all_quadrants_have_colors(self):
        for q in ("Q1", "Q2", "Q3", "Q4"):
            assert q in QUADRANT_COLORS
            assert QUADRANT_COLORS[q].startswith("#")
