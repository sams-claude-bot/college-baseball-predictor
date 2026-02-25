"""
Negative Binomial over/under model for college baseball.

Key advantage over Poisson: captures overdispersion in college baseball
(variance >> mean for run totals). Empirical data shows variance/mean ≈ 3.2,
far exceeding the Poisson assumption of variance = mean.
"""

import math
from typing import Dict, Optional, Tuple
from scipy import stats as sp_stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

# Cache the fitted dispersion so we don't re-query every call
_cached_dispersion: Optional[float] = None
_cached_league_mean: Optional[float] = None
_cached_league_var: Optional[float] = None


def fit_dispersion_from_db(min_games: int = 50) -> Tuple[float, float, float]:
    """
    Fit dispersion parameter from historical game totals.
    
    Returns (dispersion_r, league_mean, league_variance).
    For NegBin parameterization: r = mu^2 / (var - mu).
    """
    global _cached_dispersion, _cached_league_mean, _cached_league_var
    
    if _cached_dispersion is not None:
        return _cached_dispersion, _cached_league_mean, _cached_league_var
    
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT home_score + away_score as total
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL
    ''')
    totals = [row['total'] for row in c.fetchall()]
    conn.close()
    
    if len(totals) < min_games:
        # Fallback: use reasonable defaults for college baseball
        _cached_dispersion = 6.0
        _cached_league_mean = 13.0
        _cached_league_var = 41.0
        return _cached_dispersion, _cached_league_mean, _cached_league_var
    
    mean = sum(totals) / len(totals)
    var = sum((t - mean) ** 2 for t in totals) / len(totals)
    
    # NegBin dispersion: r = mu^2 / (var - mu)
    if var > mean:
        r = (mean * mean) / (var - mean)
    else:
        r = 100.0  # Effectively Poisson (no overdispersion)
    
    _cached_dispersion = r
    _cached_league_mean = mean
    _cached_league_var = var
    
    return r, mean, var


def reset_cache():
    """Reset cached dispersion (for testing)."""
    global _cached_dispersion, _cached_league_mean, _cached_league_var
    _cached_dispersion = None
    _cached_league_mean = None
    _cached_league_var = None


def negbin_team_pmf(k: int, mu: float, r: float) -> float:
    """
    NegBin PMF for a single team's runs.
    
    Parameterized by mean (mu) and dispersion (r).
    As r -> inf, this converges to Poisson(mu).
    """
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    
    p = r / (r + mu)  # success probability
    # scipy uses (n, p) parameterization where n=r, p=p
    return sp_stats.nbinom.pmf(k, r, p)


def negbin_over_under(home_runs: float, away_runs: float,
                      total_line: float,
                      dispersion: Optional[float] = None,
                      variance_boost: float = 1.0) -> Dict[str, float]:
    """
    Calculate over/under probabilities using Negative Binomial distribution.
    
    Same interface as poisson_over_under() in runs_ensemble.py.
    
    Args:
        home_runs: Expected home team runs
        away_runs: Expected away team runs
        total_line: The O/U line
        dispersion: Override dispersion parameter (None = fit from DB)
        variance_boost: Multiplier for variance (>1 = more spread, for blowout risk)
    """
    if dispersion is None:
        dispersion, _, _ = fit_dispersion_from_db()
    
    # Scale dispersion per-team based on their expected runs vs league average
    # Teams with higher expected runs tend to have proportionally higher variance
    r_home = dispersion * (home_runs / max(home_runs + away_runs, 1.0)) * 2.0
    r_away = dispersion * (away_runs / max(home_runs + away_runs, 1.0)) * 2.0
    
    # Apply variance boost (for context adjustments like non-conf blowout risk)
    if variance_boost > 1.0:
        r_home /= variance_boost  # Lower r = more variance
        r_away /= variance_boost
    
    # Ensure minimum dispersion
    r_home = max(r_home, 0.5)
    r_away = max(r_away, 0.5)
    
    MAX_RUNS = 25
    over_prob = 0.0
    under_prob = 0.0
    push_prob = 0.0
    
    for h in range(MAX_RUNS):
        h_prob = negbin_team_pmf(h, home_runs, r_home)
        for a in range(MAX_RUNS):
            a_prob = negbin_team_pmf(a, away_runs, r_away)
            combined = h_prob * a_prob
            total = h + a
            if total > total_line:
                over_prob += combined
            elif total < total_line:
                under_prob += combined
            else:
                push_prob += combined
    
    return {
        'over': round(over_prob, 4),
        'under': round(under_prob, 4),
        'push': round(push_prob, 4),
        'dispersion': round(dispersion, 4),
        'r_home': round(r_home, 4),
        'r_away': round(r_away, 4),
    }
