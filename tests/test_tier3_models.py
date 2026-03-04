#!/usr/bin/env python3
"""
Tests for Tier 3 diverse signal models: Venue, RestTravel, Upset.

Tests:
- Import and instantiation
- predict_game returns valid dict with home_win_probability in [0,1]
- Probabilities sum to ~1.0
- Works with real team IDs from DB
- Missing data handling (unknown venue, new team)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVenueModel:
    """Tests for the VenueModel."""

    def test_import(self):
        from models.venue_model import VenueModel
        assert VenueModel is not None

    def test_instantiation(self):
        from models.venue_model import VenueModel
        m = VenueModel()
        assert m.name == "venue"

    def test_predict_returns_required_keys(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 'home_win_probability' in r
        assert 'away_win_probability' in r

    def test_probability_range(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 0 < r['home_win_probability'] < 1
        assert 0 < r['away_win_probability'] < 1

    def test_probabilities_sum_to_one(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        total = r['home_win_probability'] + r['away_win_probability']
        assert 0.99 <= total <= 1.01

    def test_known_teams(self):
        """Test with well-known team IDs."""
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game('mississippi-state', 'alabama')
        assert 0 < r['home_win_probability'] < 1

    def test_missing_venue_data(self):
        """Model should handle unknown teams gracefully."""
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game('nonexistent-team-xyz', 'another-fake-team')
        assert 0 < r['home_win_probability'] < 1
        assert 0 < r['away_win_probability'] < 1

    def test_neutral_site(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'],
                           neutral_site=True)
        assert 0 < r['home_win_probability'] < 1

    def test_has_projected_runs(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.venue_model import VenueModel
        m = VenueModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 'projected_home_runs' in r
        assert 'projected_away_runs' in r
        assert r['projected_home_runs'] > 0
        assert r['projected_away_runs'] > 0

    def test_haversine(self):
        from models.venue_model import haversine
        # NYC to LA ~ 2,451 miles
        d = haversine(40.7128, -74.0060, 34.0522, -118.2437)
        assert 2400 < d < 2500


class TestRestTravelModel:
    """Tests for the RestTravelModel."""

    def test_import(self):
        from models.rest_travel_model import RestTravelModel
        assert RestTravelModel is not None

    def test_instantiation(self):
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        assert m.name == "rest_travel"

    def test_predict_returns_required_keys(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 'home_win_probability' in r
        assert 'away_win_probability' in r

    def test_probability_range(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 0 < r['home_win_probability'] < 1
        assert 0 < r['away_win_probability'] < 1

    def test_probabilities_sum_to_one(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        total = r['home_win_probability'] + r['away_win_probability']
        assert 0.99 <= total <= 1.01

    def test_known_teams(self):
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game('mississippi-state', 'alabama')
        assert 0 < r['home_win_probability'] < 1

    def test_missing_data(self):
        """Model should handle unknown teams gracefully."""
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game('nonexistent-team-xyz', 'another-fake-team')
        assert 0 < r['home_win_probability'] < 1

    def test_with_explicit_date(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'],
                           game_date='2025-03-01')
        assert 0 < r['home_win_probability'] < 1

    def test_inputs_dict(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.rest_travel_model import RestTravelModel
        m = RestTravelModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 'inputs' in r
        assert 'home_days_rest' in r['inputs']
        assert 'away_days_rest' in r['inputs']


class TestUpsetModel:
    """Tests for the UpsetModel."""

    def test_import(self):
        from models.upset_model import UpsetModel
        assert UpsetModel is not None

    def test_instantiation(self):
        from models.upset_model import UpsetModel
        m = UpsetModel()
        assert m.name == "upset"

    def test_predict_returns_required_keys(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 'home_win_probability' in r
        assert 'away_win_probability' in r

    def test_probability_range(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 0 < r['home_win_probability'] < 1
        assert 0 < r['away_win_probability'] < 1

    def test_probabilities_sum_to_one(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        total = r['home_win_probability'] + r['away_win_probability']
        assert 0.99 <= total <= 1.01

    def test_known_teams(self):
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game('mississippi-state', 'alabama')
        assert 0 < r['home_win_probability'] < 1

    def test_missing_data(self):
        """Model should handle unknown teams gracefully."""
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game('nonexistent-team-xyz', 'another-fake-team')
        assert 0 < r['home_win_probability'] < 1

    def test_upset_probability_in_inputs(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'])
        assert 'inputs' in r
        assert 'p_upset' in r['inputs']
        assert 0 <= r['inputs']['p_upset'] <= 1

    def test_elo_expected(self):
        from models.upset_model import elo_expected
        # Equal ratings should give 50%
        assert abs(elo_expected(1500, 1500) - 0.5) < 0.01
        # Higher rated team should be favored
        assert elo_expected(1600, 1400) > 0.5

    def test_neutral_site(self, sample_team_ids):
        if 'sample_home' not in sample_team_ids:
            pytest.skip("No sample teams available")
        from models.upset_model import UpsetModel
        m = UpsetModel()
        r = m.predict_game(sample_team_ids['sample_home'],
                           sample_team_ids['sample_away'],
                           neutral_site=True)
        assert 0 < r['home_win_probability'] < 1
