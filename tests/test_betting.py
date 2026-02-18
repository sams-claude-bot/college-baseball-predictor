#!/usr/bin/env python3
"""
Betting Logic Tests

Tests for bet_selection_v2.py functions:
- american_to_implied_prob conversion
- Edge calculation bounds
- No crashes with missing odds data

These tests prevent betting logic bugs that could lead to bad bet recommendations.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestOddsConversion:
    """Test American odds to probability conversions."""
    
    def test_american_to_prob_positive_odds(self):
        """Positive odds (underdog) conversion."""
        from bet_selection_v2 import american_to_prob
        
        # +100 = 50% implied
        prob = american_to_prob(100)
        assert abs(prob - 0.5) < 0.001, f"+100 should be 50%, got {prob*100:.1f}%"
        
        # +200 = 33.3% implied
        prob = american_to_prob(200)
        assert abs(prob - 0.333) < 0.01, f"+200 should be ~33.3%, got {prob*100:.1f}%"
        
        # +150 = 40% implied
        prob = american_to_prob(150)
        assert abs(prob - 0.4) < 0.01, f"+150 should be 40%, got {prob*100:.1f}%"
    
    def test_american_to_prob_negative_odds(self):
        """Negative odds (favorite) conversion."""
        from bet_selection_v2 import american_to_prob
        
        # -100 = 50% implied
        prob = american_to_prob(-100)
        assert abs(prob - 0.5) < 0.001, f"-100 should be 50%, got {prob*100:.1f}%"
        
        # -200 = 66.7% implied
        prob = american_to_prob(-200)
        assert abs(prob - 0.667) < 0.01, f"-200 should be ~66.7%, got {prob*100:.1f}%"
        
        # -150 = 60% implied
        prob = american_to_prob(-150)
        assert abs(prob - 0.6) < 0.01, f"-150 should be 60%, got {prob*100:.1f}%"
    
    def test_american_to_prob_extreme_favorite(self):
        """Extreme favorite odds."""
        from bet_selection_v2 import american_to_prob
        
        # -500 = 83.3% implied
        prob = american_to_prob(-500)
        assert 0.8 < prob < 0.9, f"-500 should be ~83%, got {prob*100:.1f}%"
    
    def test_american_to_prob_extreme_underdog(self):
        """Extreme underdog odds."""
        from bet_selection_v2 import american_to_prob
        
        # +500 = 16.7% implied
        prob = american_to_prob(500)
        assert 0.1 < prob < 0.2, f"+500 should be ~16.7%, got {prob*100:.1f}%"
    
    def test_american_to_prob_returns_valid_probability(self):
        """Output should always be a valid probability (0 < p < 1)."""
        from bet_selection_v2 import american_to_prob
        
        test_odds = [-1000, -500, -200, -150, -100, 100, 150, 200, 500, 1000]
        
        for odds in test_odds:
            prob = american_to_prob(odds)
            assert 0 < prob < 1, f"Odds {odds} gave invalid probability {prob}"


class TestKellyFraction:
    """Test Kelly criterion calculations."""
    
    def test_kelly_returns_reasonable_values(self):
        """Kelly fraction should be between 0 and max cap."""
        from bet_selection_v2 import kelly_fraction
        
        # Fair odds with 55% edge
        k = kelly_fraction(0.55, 100, fraction=0.25)
        assert 0 <= k <= 2.0, f"Kelly fraction {k} out of expected range"
        
        # Strong favorite with fair odds
        k = kelly_fraction(0.70, -200, fraction=0.25)
        assert 0 <= k <= 2.0, f"Kelly fraction {k} out of expected range"
    
    def test_kelly_zero_when_no_edge(self):
        """Kelly should be ~0 when model prob equals implied prob."""
        from bet_selection_v2 import kelly_fraction
        
        # +100 = 50% implied, model says 50%
        k = kelly_fraction(0.50, 100, fraction=0.25)
        assert k < 0.05, f"No edge should give near-zero Kelly, got {k}"
    
    def test_kelly_positive_with_positive_edge(self):
        """Kelly should be positive when model has edge."""
        from bet_selection_v2 import kelly_fraction
        
        # +100 = 50% implied, model says 60%
        k = kelly_fraction(0.60, 100, fraction=0.25)
        assert k > 0, f"Positive edge should give positive Kelly, got {k}"
    
    def test_kelly_capped(self):
        """Kelly should not exceed the cap (2.0)."""
        from bet_selection_v2 import kelly_fraction
        
        # Extreme edge
        k = kelly_fraction(0.95, 100, fraction=1.0)  # Full Kelly with huge edge
        assert k <= 2.0, f"Kelly should be capped at 2.0, got {k}"


class TestEdgeCalculation:
    """Test edge calculation logic."""
    
    def test_edge_between_negative_one_and_one(self):
        """Edge values should be bounded."""
        from bet_selection_v2 import american_to_prob
        
        test_cases = [
            (0.55, 100),   # Small edge on even odds
            (0.70, -200),  # Favorite with smaller edge
            (0.40, 200),   # Underdog with negative edge
            (0.80, -150),  # Good edge on favorite
        ]
        
        for model_prob, odds in test_cases:
            implied = american_to_prob(odds)
            edge = model_prob - implied
            
            assert -1 <= edge <= 1, (
                f"Edge {edge:.3f} out of bounds for "
                f"model_prob={model_prob}, odds={odds}"
            )


class TestBetSelectionRobustness:
    """Test that bet selection doesn't crash with edge cases."""
    
    def test_analyze_with_no_api_gracefully_fails(self):
        """analyze_games should handle API failures gracefully."""
        from bet_selection_v2 import analyze_games
        
        # API might not be running, but shouldn't crash
        result = analyze_games("2099-12-31")  # Far future date
        
        # Should return a dict with error or empty bets
        assert isinstance(result, dict)
        assert 'bets' in result or 'error' in result
    
    def test_thresholds_are_sensible(self):
        """Threshold constants should be reasonable."""
        from bet_selection_v2 import (
            ML_EDGE_THRESHOLD,
            ML_EDGE_UNDERDOG,
            ML_CONSENSUS_MIN,
            ML_MAX_MODEL_PROB,
            ML_MIN_MODEL_PROB,
        )
        
        # Edge thresholds should be percentages (0-100 range based on code)
        assert 0 < ML_EDGE_THRESHOLD < 50, f"ML_EDGE_THRESHOLD={ML_EDGE_THRESHOLD} seems wrong"
        assert 0 < ML_EDGE_UNDERDOG < 50, f"ML_EDGE_UNDERDOG={ML_EDGE_UNDERDOG} seems wrong"
        
        # Consensus should require majority
        assert ML_CONSENSUS_MIN >= 5, f"Consensus min {ML_CONSENSUS_MIN} too low"
        assert ML_CONSENSUS_MIN <= 10, f"Consensus min {ML_CONSENSUS_MIN} too high"
        
        # Model prob bounds should be valid probabilities
        assert 0.5 < ML_MIN_MODEL_PROB < 1.0
        assert 0.5 < ML_MAX_MODEL_PROB <= 1.0
        assert ML_MIN_MODEL_PROB < ML_MAX_MODEL_PROB


class TestBettingDatabase:
    """Test betting-related database integrity."""
    
    def test_tracked_bets_structure(self, db_connection):
        """tracked_bets table should exist and have expected columns."""
        c = db_connection.cursor()
        
        try:
            c.execute("PRAGMA table_info(tracked_bets)")
            columns = {row['name'] for row in c.fetchall()}
            
            # Should have at least these columns
            expected = ['game_id']  # Minimal expectation
            for col in expected:
                assert col in columns, f"tracked_bets missing column: {col}"
        except Exception as e:
            pytest.fail(f"tracked_bets table error: {e}")
    
    def test_tracked_confident_bets_structure(self, db_connection):
        """tracked_confident_bets table should exist."""
        c = db_connection.cursor()
        
        try:
            c.execute("SELECT COUNT(*) FROM tracked_confident_bets")
            # Just checking it exists and is queryable
        except Exception as e:
            pytest.fail(f"tracked_confident_bets table error: {e}")
    
    def test_betting_lines_structure(self, db_connection):
        """betting_lines table should exist with odds data."""
        c = db_connection.cursor()
        
        try:
            c.execute("PRAGMA table_info(betting_lines)")
            columns = {row['name'] for row in c.fetchall()}
            
            # Should have game_id at minimum
            assert 'game_id' in columns, "betting_lines missing game_id column"
        except Exception as e:
            pytest.fail(f"betting_lines table error: {e}")
