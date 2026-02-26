"""3-game series probability calculator.

Given a single-game win probability for the home team, compute the
probabilities of each team winning at least 1, at least 2, or all 3 games
in a 3-game series (assuming independent, identically-distributed games).
"""


def compute_series_probs(home_win_prob: float) -> dict:
    """
    Given single-game home win probability, compute 3-game series probabilities.

    Returns dict with keys:
      home_win_1plus, home_win_2plus, home_sweep,
      away_win_1plus, away_win_2plus, away_sweep
    All as floats 0-1.
    """
    p = max(0.0, min(1.0, home_win_prob))
    q = 1.0 - p

    return {
        'home_win_1plus': 1.0 - q ** 3,
        'home_win_2plus': p ** 3 + 3 * p ** 2 * q,
        'home_sweep': p ** 3,
        'away_win_1plus': 1.0 - p ** 3,
        'away_win_2plus': q ** 3 + 3 * q ** 2 * p,
        'away_sweep': q ** 3,
    }
