"""
Win Quality / Resume Impact — rates how much a win or loss in this game
affects each team's NCAA tournament resume.

Uses Elo-based quadrants (like basketball's NET quadrants):
  Q1: opponent Elo rank 1-50    (elite)
  Q2: opponent Elo rank 51-100  (good)
  Q3: opponent Elo rank 101-200 (okay)
  Q4: opponent Elo rank 201+    (weak)
"""

import sys
from pathlib import Path

base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection


def compute_elo_ranks():
    """Return {team_id: rank} for all teams ordered by Elo descending."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT team_id, rating FROM elo_ratings ORDER BY rating DESC"
    ).fetchall()
    conn.close()
    return {row["team_id"]: i + 1 for i, row in enumerate(rows)}


def get_quadrant(elo_rank):
    """Map an Elo rank to its quadrant string."""
    if elo_rank <= 50:
        return "Q1"
    elif elo_rank <= 100:
        return "Q2"
    elif elo_rank <= 200:
        return "Q3"
    else:
        return "Q4"


QUADRANT_COLORS = {
    "Q1": "#1a7431",
    "Q2": "#2196F3",
    "Q3": "#FF9800",
    "Q4": "#d32f2f",
}


def compute_win_quality(team_elo_rank, opponent_elo_rank, total_teams=310):
    """
    Compute win quality and loss damage for a team facing this opponent.

    Returns dict with:
      quadrant, win_quality, win_label, win_color,
      loss_damage, loss_label, loss_color
    """
    quadrant = get_quadrant(opponent_elo_rank)

    # Win quality: based on opponent's strength percentile
    opp_pct = 1 - (opponent_elo_rank / total_teams)  # 1.0 = best, 0.0 = worst
    win_quality = round(max(-0.5, min(1.0, (opp_pct * 2) - 0.5)), 2)

    # Loss damage: based on how far below you the opponent is
    rank_gap = (team_elo_rank - opponent_elo_rank) / total_teams
    loss_damage = round(max(-1.0, min(0.5, rank_gap - 0.2)), 2)

    # Labels
    if win_quality >= 0.6:
        win_label = "Resume Builder"
        win_color = "#1a7431"
    elif win_quality >= 0.2:
        win_label = "Solid Win"
        win_color = "#2196F3"
    elif win_quality >= -0.1:
        win_label = "Expected"
        win_color = "#FF9800"
    else:
        win_label = "No Value"
        win_color = "#d32f2f"

    if loss_damage >= 0.0:
        loss_label = "Understandable"
        loss_color = "#1a7431"
    elif loss_damage >= -0.2:
        loss_label = "Concerning"
        loss_color = "#FF9800"
    elif loss_damage >= -0.5:
        loss_label = "Bad Loss"
        loss_color = "#d32f2f"
    else:
        loss_label = "Catastrophic"
        loss_color = "#8b0000"

    return {
        "quadrant": quadrant,
        "win_quality": win_quality,
        "win_label": win_label,
        "win_color": win_color,
        "loss_damage": loss_damage,
        "loss_label": loss_label,
        "loss_color": loss_color,
    }


def compute_quadrant_record(team_id, elo_ranks=None):
    """
    Compute a team's W/L record bucketed by opponent quadrant.
    Returns {"Q1": "2-1", "Q2": "3-0", "Q3": "5-1", "Q4": "8-0"}.
    """
    if elo_ranks is None:
        elo_ranks = compute_elo_ranks()

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT home_team_id, away_team_id, winner_id
        FROM games
        WHERE status = 'final'
          AND (home_team_id = ? OR away_team_id = ?)
        """,
        (team_id, team_id),
    ).fetchall()
    conn.close()

    record = {"Q1": [0, 0], "Q2": [0, 0], "Q3": [0, 0], "Q4": [0, 0]}

    for row in rows:
        opponent_id = (
            row["away_team_id"]
            if row["home_team_id"] == team_id
            else row["home_team_id"]
        )
        opp_rank = elo_ranks.get(opponent_id)
        if opp_rank is None:
            continue
        quad = get_quadrant(opp_rank)
        if row["winner_id"] == team_id:
            record[quad][0] += 1
        else:
            record[quad][1] += 1

    return {q: f"{w}-{l}" for q, (w, l) in record.items()}


def get_game_resume_impact(home_id, away_id):
    """
    Main entry point: compute resume impact for both teams in a matchup.
    Returns dict with 'home' and 'away' sub-dicts.
    """
    elo_ranks = compute_elo_ranks()
    total_teams = len(elo_ranks)

    home_rank = elo_ranks.get(home_id)
    away_rank = elo_ranks.get(away_id)

    if home_rank is None or away_rank is None:
        return None

    home_impact = compute_win_quality(home_rank, away_rank, total_teams)
    home_impact["quad_record"] = compute_quadrant_record(home_id, elo_ranks)
    home_impact["elo_rank"] = home_rank

    away_impact = compute_win_quality(away_rank, home_rank, total_teams)
    away_impact["quad_record"] = compute_quadrant_record(away_id, elo_ranks)
    away_impact["elo_rank"] = away_rank

    return {"home": home_impact, "away": away_impact}
