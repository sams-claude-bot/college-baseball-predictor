#!/usr/bin/env python3
"""Poisson/NB distribution and outcome probability helpers."""

import math
from functools import lru_cache
from typing import Dict, List, Tuple

from .lambda_calc import _cfg

MAX_RUNS = 25

@lru_cache(maxsize=1000)
def poisson_pmf(k: int, lambda_: float) -> float:
    """Poisson probability mass function: P(X = k) given mean lambda."""
    if lambda_ <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    # Use log to avoid overflow for large values
    log_prob = k * math.log(lambda_) - lambda_ - math.lgamma(k + 1)
    return math.exp(log_prob)

def poisson_cdf(k: int, lambda_: float) -> float:
    """Cumulative distribution: P(X <= k)."""
    return sum(poisson_pmf(i, lambda_) for i in range(k + 1))

def negative_binomial_pmf(k: int, mean_: float, variance_: float) -> float:
    """
    NB-style PMF parameterized by mean and variance.
    Falls back to Poisson when variance <= mean.
    """
    if k < 0:
        return 0.0
    if mean_ <= 0:
        return 1.0 if k == 0 else 0.0
    if variance_ <= mean_ + 1e-9:
        return poisson_pmf(k, mean_)

    # For NB(mean=mu, var=mu+mu^2/r): solve r = mu^2 / (var-mu)
    r = (mean_ * mean_) / max(variance_ - mean_, 1e-9)
    p = r / (r + mean_)  # success probability
    log_prob = (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1.0 - p)
    )
    return math.exp(log_prob)

def build_probability_matrix(lambda_a: float, lambda_b: float) -> List[List[float]]:
    """
    Build joint probability matrix P(A=i, B=j) for all i,j in [0, MAX_RUNS].
    
    Returns 2D list where matrix[i][j] = P(Team A scores i, Team B scores j)
    """
    matrix = []
    for i in range(MAX_RUNS + 1):
        row = []
        for j in range(MAX_RUNS + 1):
            # Assuming independence (simplification - could add correlation)
            prob = poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
            row.append(prob)
        matrix.append(row)
    return matrix

def build_total_pmf_from_matrix(matrix: List[List[float]]) -> List[float]:
    """Collapse joint score matrix into total-runs PMF."""
    total_pmf = [0.0] * (2 * MAX_RUNS + 1)
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            total_pmf[i + j] += matrix[i][j]
    return total_pmf

def _mean_variance(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_ = sum(values) / len(values)
    var_ = sum((x - mean_) ** 2 for x in values) / len(values)
    return mean_, var_

def maybe_apply_overdispersion_to_total_pmf(
    base_total_pmf: List[float],
    expected_total: float,
    recent_totals: List[float]
) -> Tuple[List[float], Dict]:
    """
    Optionally widen the totals PMF when empirical totals are overdispersed.
    This only affects totals probabilities; win/spread matrix remains unchanged.
    """
    meta = {
        'enabled': _cfg('POISSON_ENABLE_OVERDISPERSION', True),
        'applied': False,
        'sample_size': len(recent_totals or []),
        'method': 'poisson'
    }
    if not meta['enabled']:
        return base_total_pmf, meta

    recent_totals = recent_totals or []
    min_samples = _cfg('POISSON_OVERDISPERSION_MIN_SAMPLES', 8)
    if len(recent_totals) < min_samples:
        return base_total_pmf, meta

    emp_mean, emp_var = _mean_variance(recent_totals)
    if emp_mean <= 0:
        return base_total_pmf, meta

    ratio = emp_var / emp_mean
    meta.update({
        'empirical_mean': round(emp_mean, 3),
        'empirical_variance': round(emp_var, 3),
        'variance_mean_ratio': round(ratio, 3)
    })

    threshold = _cfg('POISSON_OVERDISPERSION_VAR_MEAN_RATIO', 1.35)
    if ratio < threshold:
        return base_total_pmf, meta

    max_var_mult = _cfg('POISSON_OVERDISPERSION_MAX_VAR_MULTIPLIER', 2.5)
    target_var = min(emp_var, max_var_mult * max(expected_total, 1e-6))

    widened = [
        negative_binomial_pmf(k, expected_total, target_var)
        for k in range(2 * MAX_RUNS + 1)
    ]
    total_mass = sum(widened)
    if total_mass <= 0:
        return base_total_pmf, meta
    widened = [p / total_mass for p in widened]

    meta.update({
        'applied': True,
        'method': 'negative_binomial_variance_match',
        'target_variance': round(target_var, 3)
    })
    return widened, meta

def calculate_win_probability(matrix: List[List[float]]) -> Tuple[float, float, float]:
    """
    Calculate win/loss/tie probabilities from joint probability matrix.
    
    Returns (P(A wins), P(B wins), P(tie))
    """
    p_a_wins = 0.0
    p_b_wins = 0.0
    p_tie = 0.0
    
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            prob = matrix[i][j]
            if i > j:
                p_a_wins += prob
            elif j > i:
                p_b_wins += prob
            else:
                p_tie += prob
    
    # In baseball, ties go to extra innings - redistribute tie probability
    # Assume 50/50 split in extras (slight simplification)
    p_a_wins += p_tie * 0.5
    p_b_wins += p_tie * 0.5
    
    return p_a_wins, p_b_wins, p_tie

def calculate_total_probability(matrix: List[List[float]], total: float) -> Dict:
    """
    Calculate over/under probabilities for a total line.
    
    Returns probabilities for over, under, and push.
    """
    p_over = 0.0
    p_under = 0.0
    p_push = 0.0
    
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            prob = matrix[i][j]
            combined = i + j
            if combined > total:
                p_over += prob
            elif combined < total:
                p_under += prob
            else:
                p_push += prob
    
    return {
        'over': round(p_over, 4),
        'under': round(p_under, 4),
        'push': round(p_push, 4),
        'total_line': total
    }

def calculate_total_probability_from_pmf(total_pmf: List[float], total: float) -> Dict:
    """Calculate over/under probabilities from a total-runs PMF."""
    p_over = 0.0
    p_under = 0.0
    p_push = 0.0
    for combined, prob in enumerate(total_pmf):
        if combined > total:
            p_over += prob
        elif combined < total:
            p_under += prob
        else:
            p_push += prob
    return {
        'over': round(p_over, 4),
        'under': round(p_under, 4),
        'push': round(p_push, 4),
        'total_line': total
    }

def calculate_spread_probability(matrix: List[List[float]], spread: float) -> Dict:
    """
    Calculate probability Team A covers the spread.
    
    spread > 0 means A is underdog (e.g., +1.5)
    spread < 0 means A is favorite (e.g., -1.5)
    """
    p_cover = 0.0
    p_not_cover = 0.0
    p_push = 0.0
    
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            prob = matrix[i][j]
            margin = i - j + spread  # A's margin adjusted by spread
            if margin > 0:
                p_cover += prob
            elif margin < 0:
                p_not_cover += prob
            else:
                p_push += prob
    
    return {
        'cover': round(p_cover, 4),
        'not_cover': round(p_not_cover, 4),
        'push': round(p_push, 4),
        'spread': spread
    }

def get_most_likely_scores(matrix: List[List[float]], top_n: int = 10) -> List[Dict]:
    """Get the most likely final scores."""
    scores = []
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            scores.append({
                'team_a_runs': i,
                'team_b_runs': j,
                'probability': matrix[i][j]
            })
    
    scores.sort(key=lambda x: x['probability'], reverse=True)
    return scores[:top_n]
