"""
Game Quality Index (GQI) — rates how compelling a matchup is (1.0–10.0).

Components:
  - strength (0–5): average Elo of both teams, normalized
  - closeness (0–3): reward evenly matched games
  - ranked_boost (0–2): bonus for Top 25 teams
"""


def compute_gqi(home_elo, away_elo, home_rank=None, away_rank=None):
    """Compute Game Quality Index (1.0 - 10.0)."""
    avg_elo = (home_elo + away_elo) / 2
    elo_diff = abs(home_elo - away_elo)

    # Strength component (0-5): how good are both teams?
    strength = max(0, min(5, (avg_elo - 1400) / 60))

    # Closeness component (0-3): reward close matchups
    closeness = max(0, 3 - (elo_diff / 50))

    # Ranked boost (0-2)
    ranked_boost = 0
    if home_rank and 1 <= home_rank <= 25:
        ranked_boost += 1.0
    if away_rank and 1 <= away_rank <= 25:
        ranked_boost += 1.0

    gqi = strength + closeness + ranked_boost
    return round(max(1.0, min(10.0, gqi)), 1)


def gqi_label(gqi):
    """Human-readable label for a GQI score."""
    if gqi >= 8.0:
        return "Must Watch"
    elif gqi >= 6.5:
        return "Great"
    elif gqi >= 5.0:
        return "Good"
    elif gqi >= 3.5:
        return "Average"
    else:
        return "Mismatch"


def gqi_color(gqi):
    """Hex color for a GQI score."""
    if gqi >= 8.0:
        return "#1a7431"
    elif gqi >= 6.5:
        return "#2196F3"
    elif gqi >= 5.0:
        return "#607D8B"
    elif gqi >= 3.5:
        return "#FF9800"
    else:
        return "#d32f2f"
