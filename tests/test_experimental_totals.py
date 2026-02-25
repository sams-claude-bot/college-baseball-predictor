"""Tests for experimental V2 totals models."""

import sys
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNegBinModel:
    """Test the Negative Binomial distribution model."""
    
    def test_negbin_pmf_sums_to_one(self):
        """NegBin PMF should sum to approximately 1.0."""
        from models.negbin_model import negbin_team_pmf
        
        mu = 6.5  # Typical team runs
        r = 6.0   # Dispersion
        total = sum(negbin_team_pmf(k, mu, r) for k in range(50))
        assert abs(total - 1.0) < 0.01, f"PMF sum = {total}, expected ~1.0"
    
    def test_negbin_pmf_zero_mean(self):
        """Zero mean should give all probability at 0."""
        from models.negbin_model import negbin_team_pmf
        assert negbin_team_pmf(0, 0.0, 5.0) == 1.0
        assert negbin_team_pmf(1, 0.0, 5.0) == 0.0
    
    def test_negbin_ou_probabilities_sum(self):
        """Over + under + push should sum to approximately 1.0."""
        from models.negbin_model import negbin_over_under
        
        result = negbin_over_under(6.0, 5.5, 11.5, dispersion=6.0)
        total = result['over'] + result['under'] + result['push']
        assert abs(total - 1.0) < 0.05, f"O/U probs sum = {total}"
    
    def test_negbin_ou_returns_dispersion(self):
        """Should return dispersion info in result."""
        from models.negbin_model import negbin_over_under
        
        result = negbin_over_under(6.0, 5.5, 11.5, dispersion=6.0)
        assert 'dispersion' in result
        assert result['dispersion'] == 6.0
    
    def test_variance_boost_increases_spread(self):
        """Higher variance_boost should increase over probability for high lines."""
        from models.negbin_model import negbin_over_under
        
        # High line: more variance should push more probability into tails
        normal = negbin_over_under(5.0, 5.0, 15.0, dispersion=6.0, variance_boost=1.0)
        boosted = negbin_over_under(5.0, 5.0, 15.0, dispersion=6.0, variance_boost=1.5)
        
        # With more variance, over prob for a high line should increase
        assert boosted['over'] >= normal['over'] - 0.01  # Allow small tolerance


class TestRunsEnsembleV2:
    """Test the V2 ensemble model."""
    
    def test_predict_returns_required_keys(self):
        """V2 predict should return same structure as production."""
        from models.experimental.runs_ensemble_v2 import predict
        
        # Use teams that likely exist in DB
        result = predict('mississippi-state', 'ole-miss', total_line=11.5)
        
        required_keys = [
            'home_team', 'away_team', 'projected_home_runs',
            'projected_away_runs', 'projected_total',
            'confidence_interval', 'std_dev', 'model_agreement',
            'model_breakdown'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_no_pitching_component(self):
        """V2 should not include pitching in weights."""
        from models.experimental.runs_ensemble_v2 import DEFAULT_RUN_WEIGHTS_V2
        
        assert 'pitching' not in DEFAULT_RUN_WEIGHTS_V2
        assert 'poisson' in DEFAULT_RUN_WEIGHTS_V2
        assert 'advanced' in DEFAULT_RUN_WEIGHTS_V2
    
    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        from models.experimental.runs_ensemble_v2 import DEFAULT_RUN_WEIGHTS_V2
        
        total = sum(DEFAULT_RUN_WEIGHTS_V2.values())
        assert abs(total - 1.0) < 0.001
    
    def test_over_under_includes_negbin(self):
        """When line is provided, should include negbin probabilities."""
        from models.experimental.runs_ensemble_v2 import predict
        
        result = predict('mississippi-state', 'ole-miss', total_line=11.5)
        
        if 'over_under' in result:
            ou = result['over_under']
            assert 'negbin_over' in ou
            assert 'negbin_under' in ou
            assert 'dispersion' in ou
    
    def test_model_version_tag(self):
        """Should include v2 model version tag."""
        from models.experimental.runs_ensemble_v2 import predict
        
        result = predict('mississippi-state', 'ole-miss')
        assert result.get('model_version') == 'v2_experimental'


class TestOverConfidenceGate:
    """Test the OVER confidence gate."""
    
    def test_over_low_edge_downgraded(self):
        """OVER with low edge should get confidence downgraded."""
        from models.experimental.runs_ensemble_v2 import apply_over_confidence_gate
        
        pred, conf, gated = apply_over_confidence_gate('OVER', 5.0, 0.8)
        assert gated is True
        assert conf < 0.8  # Should be reduced
    
    def test_over_high_edge_kept(self):
        """OVER with high edge should keep confidence."""
        from models.experimental.runs_ensemble_v2 import apply_over_confidence_gate
        
        pred, conf, gated = apply_over_confidence_gate('OVER', 12.0, 0.8)
        assert gated is False
        assert conf == 0.8
    
    def test_under_not_gated(self):
        """UNDER predictions should never be gated."""
        from models.experimental.runs_ensemble_v2 import apply_over_confidence_gate
        
        pred, conf, gated = apply_over_confidence_gate('UNDER', 3.0, 0.8)
        assert gated is False
        assert conf == 0.8
    
    def test_gate_threshold_is_8pct(self):
        """Gate should trigger at exactly 8% edge."""
        from models.experimental.runs_ensemble_v2 import apply_over_confidence_gate, OVER_EDGE_THRESHOLD
        
        assert OVER_EDGE_THRESHOLD == 8.0
        
        _, _, gated_at = apply_over_confidence_gate('OVER', 8.0, 0.8)
        assert gated_at is True  # 8.0 <= 8.0
        
        _, _, gated_above = apply_over_confidence_gate('OVER', 8.1, 0.8)
        assert gated_above is False  # 8.1 > 8.0


class TestGameContext:
    """Test game context adjustment."""
    
    def test_context_returns_variance_boost(self):
        """Should return a variance_boost field."""
        from models.experimental.runs_ensemble_v2 import get_game_context
        
        ctx = get_game_context('mississippi-state', 'ole-miss')
        assert 'variance_boost' in ctx
        assert ctx['variance_boost'] >= 1.0
