"""Tests for the Game Quality Index (GQI) service."""

import pytest
from web.services.game_quality import compute_gqi, gqi_label, gqi_color


class TestComputeGQI:
    """Tests for compute_gqi()."""

    def test_bounds_always_1_to_10(self):
        """GQI should always be between 1.0 and 10.0."""
        # Very low Elo teams
        assert 1.0 <= compute_gqi(1200, 1200) <= 10.0
        # Very high Elo teams
        assert 1.0 <= compute_gqi(1800, 1800) <= 10.0
        # Massive mismatch
        assert 1.0 <= compute_gqi(1800, 1200) <= 10.0
        # Both ranked
        assert 1.0 <= compute_gqi(1700, 1700, 1, 2) <= 10.0

    def test_elite_matchup_high_gqi(self):
        """Two highly-ranked elite teams should produce high GQI."""
        gqi = compute_gqi(1650, 1650, home_rank=3, away_rank=5)
        assert gqi >= 9.0

    def test_mismatch_low_gqi(self):
        """A strong team vs a weak team should produce low GQI."""
        gqi = compute_gqi(1650, 1400)
        assert gqi < 5.0

    def test_closeness_matters(self):
        """Same average Elo but different spreads should give different GQI."""
        # Close matchup: 1550 vs 1550 (avg 1550, diff 0)
        close = compute_gqi(1550, 1550)
        # Lopsided matchup: 1700 vs 1400 (avg 1550, diff 300)
        lopsided = compute_gqi(1700, 1400)
        assert close > lopsided

    def test_ranked_boost(self):
        """Having ranked teams should boost GQI."""
        unranked = compute_gqi(1550, 1550)
        one_ranked = compute_gqi(1550, 1550, home_rank=10)
        two_ranked = compute_gqi(1550, 1550, home_rank=10, away_rank=15)
        assert one_ranked > unranked
        assert two_ranked > one_ranked
        assert one_ranked - unranked == pytest.approx(1.0)
        assert two_ranked - one_ranked == pytest.approx(1.0)

    def test_rank_outside_25_no_boost(self):
        """Ranks outside 1-25 should not give a boost."""
        no_rank = compute_gqi(1550, 1550)
        rank_30 = compute_gqi(1550, 1550, home_rank=30)
        assert no_rank == rank_30

    def test_rank_zero_no_boost(self):
        """Rank of 0 should not give a boost."""
        no_rank = compute_gqi(1550, 1550)
        rank_0 = compute_gqi(1550, 1550, home_rank=0)
        assert no_rank == rank_0

    def test_very_low_elo(self):
        """Very low Elo teams still get closeness credit if evenly matched."""
        gqi = compute_gqi(1200, 1200)
        # strength = 0 (below 1400), closeness = 3 (diff=0) → 3.0
        assert gqi == 3.0

    def test_minimum_gqi_is_1(self):
        """Worst case: low Elo + big mismatch → clamped to 1.0."""
        gqi = compute_gqi(1200, 1400)
        # strength = 0, closeness = max(0, 3 - 200/50) = 0 → 0, clamped to 1.0
        assert gqi == 1.0

    def test_very_high_elo_no_rank(self):
        """High Elo with no ranks maxes at strength + closeness = 8."""
        gqi = compute_gqi(1750, 1750)
        # strength = min(5, (1750-1400)/60) = 5.0 (capped)
        # closeness = 3.0 (diff=0)
        # no ranked boost
        assert gqi == 8.0

    def test_max_gqi(self):
        """Maximum possible GQI is 10.0."""
        gqi = compute_gqi(1750, 1750, home_rank=1, away_rank=2)
        assert gqi == 10.0

    def test_return_type_is_float(self):
        """GQI should be a float."""
        gqi = compute_gqi(1500, 1500)
        assert isinstance(gqi, float)


class TestGQILabel:
    """Tests for gqi_label()."""

    def test_must_watch(self):
        assert gqi_label(8.0) == "Must Watch"
        assert gqi_label(10.0) == "Must Watch"

    def test_great(self):
        assert gqi_label(6.5) == "Great"
        assert gqi_label(7.9) == "Great"

    def test_good(self):
        assert gqi_label(5.0) == "Good"
        assert gqi_label(6.4) == "Good"

    def test_average(self):
        assert gqi_label(3.5) == "Average"
        assert gqi_label(4.9) == "Average"

    def test_mismatch(self):
        assert gqi_label(1.0) == "Mismatch"
        assert gqi_label(3.4) == "Mismatch"


class TestGQIColor:
    """Tests for gqi_color()."""

    def test_dark_green_for_must_watch(self):
        assert gqi_color(8.0) == "#1a7431"
        assert gqi_color(10.0) == "#1a7431"

    def test_blue_for_great(self):
        assert gqi_color(6.5) == "#2196F3"
        assert gqi_color(7.9) == "#2196F3"

    def test_blue_gray_for_good(self):
        assert gqi_color(5.0) == "#607D8B"

    def test_orange_for_average(self):
        assert gqi_color(3.5) == "#FF9800"

    def test_red_for_mismatch(self):
        assert gqi_color(1.0) == "#d32f2f"
        assert gqi_color(3.4) == "#d32f2f"
