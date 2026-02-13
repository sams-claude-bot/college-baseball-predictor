#!/usr/bin/env python3
"""
Generate PDF reports with charts for college baseball predictions.

Includes all 7 models + ensemble with dynamic data from the database.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile

# Add paths
_scripts_dir = Path(__file__).parent
_models_dir = _scripts_dir.parent / "models"
_reports_dir = _scripts_dir.parent / "reports"
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_models_dir))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from compare_models import MODELS, normalize_team_id
from database import get_connection, get_current_top_25
from betting_lines import american_to_implied_prob, implied_prob_to_american

# Ensure reports directory exists
_reports_dir.mkdir(parents=True, exist_ok=True)

# Display names for teams
TEAM_DISPLAY_NAMES = {
    "mississippi-state": "Mississippi State",
    "mississippi-st": "Mississippi State",
}

def display_name(team_id):
    """Get a pretty display name for a team id"""
    if team_id in TEAM_DISPLAY_NAMES:
        return TEAM_DISPLAY_NAMES[team_id]
    return team_id.replace("-", " ").title()


class BaseballReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'College Baseball Predictions', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(128, 0, 0)  # Maroon
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(128, 0, 0)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def mono_line(self, text):
        self.set_font('Courier', '', 9)
        self.cell(0, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def create_model_comparison_chart(home_team, away_team, predictions, output_path):
    """Create bar chart comparing model predictions"""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Sort by home win prob, ensemble last
    items = sorted(predictions.items(),
                   key=lambda x: (x[0] == 'ensemble', x[1]['home_win_probability']),
                   reverse=True)
    models = [name for name, _ in items]
    home_probs = [predictions[m]['home_win_probability'] * 100 for m in models]
    away_probs = [predictions[m]['away_win_probability'] * 100 for m in models]

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], home_probs, width, label=home_team, color='#800000')
    bars2 = ax.bar([i + width / 2 for i in x], away_probs, width, label=away_team, color='#4a4a4a')

    ax.set_ylabel('Win Probability (%)')
    ax.set_title(f'Model Predictions: {away_team} @ {home_team}')
    ax.set_xticks(list(x))
    labels = [f'{m} *' if m == 'ensemble' else m for m in models]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_runs_projection_chart(home_team, away_team, predictions, output_path):
    """Create chart showing projected runs"""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    items = sorted(predictions.items(),
                   key=lambda x: (x[0] == 'ensemble', x[1]['projected_home_runs']),
                   reverse=True)
    models = [name for name, _ in items]
    home_runs = [predictions[m]['projected_home_runs'] for m in models]
    away_runs = [predictions[m]['projected_away_runs'] for m in models]

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], home_runs, width, label=home_team, color='#800000')
    bars2 = ax.bar([i + width / 2 for i in x], away_runs, width, label=away_team, color='#4a4a4a')

    ax.set_ylabel('Projected Runs')
    ax.set_title(f'Runs Projection: {away_team} @ {home_team}')
    ax.set_xticks(list(x))
    labels = [f'{m} *' if m == 'ensemble' else m for m in models]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_ensemble_weights_chart(ensemble, output_path):
    """Create pie/bar chart of current ensemble weights"""
    if not hasattr(ensemble, 'weights'):
        return False

    weights = ensemble.weights
    fig, ax = plt.subplots(figsize=(8, 4))

    names = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    vals = [weights[n] * 100 for n in names]
    colors = ['#800000', '#c0392b', '#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#95a5a6']

    bars = ax.barh(names, vals, color=colors[:len(names)])
    ax.set_xlabel('Weight (%)')
    ax.set_title('Dynamic Ensemble Model Weights')
    ax.set_xlim(0, max(vals) * 1.3)

    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def create_value_chart(value_picks, output_path):
    """Create chart showing betting value/edge"""
    if not value_picks:
        return False

    fig, ax = plt.subplots(figsize=(8, 4))

    picks = [v[0] for v in value_picks]
    edges = [float(str(v[1]).replace('%', '')) for v in value_picks]

    colors = ['#2ecc71' if e > 10 else '#f39c12' if e > 5 else '#95a5a6' for e in edges]
    bars = ax.barh(picks, edges, color=colors)
    ax.set_xlabel('Edge vs Sportsbook (%)')
    ax.set_title('Value Picks - Model Edge')
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='10% threshold')

    for bar, edge in zip(bars, edges):
        ax.annotate(f'{edge:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def get_betting_lines_for_report(days=4):
    """Fetch betting lines for upcoming games"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT home_team_id, away_team_id, home_ml, away_ml, over_under, date
        FROM betting_lines
        WHERE date >= date('now')
          AND date <= date('now', ? || ' days')
        ORDER BY date
    ''', (str(days),))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_ev_chart(value_picks, output_path):
    """Create EV/edge chart for value picks"""
    if not value_picks:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    picks = [v[0][:25] for v in value_picks]
    edges = [v[1] for v in value_picks]
    evs = [v[2] for v in value_picks]

    # Edge chart
    colors = ['#2ecc71' if e > 10 else '#f39c12' if e > 5 else '#3498db' for e in edges]
    bars1 = ax1.barh(picks, edges, color=colors)
    ax1.set_xlabel('Edge vs DraftKings (%)')
    ax1.set_title('Model Edge')
    ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(x=10, color='green', linestyle='--', alpha=0.7)
    for bar, val in zip(bars1, edges):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=8)

    # EV chart
    ev_colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in evs]
    bars2 = ax2.barh(picks, evs, color=ev_colors)
    ax2.set_xlabel('Expected Value per $100 Bet')
    ax2.set_title('EV per $100')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    for bar, val in zip(bars2, evs):
        ax2.annotate(f'${val:.1f}', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def get_model_disclaimer():
    """Return contextual disclaimer based on season progress"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM games WHERE status='final'")
    games_played = c.fetchone()[0]
    conn.close()

    if games_played < 20:
        return (
            "EARLY SEASON DISCLAIMER: Most models have limited game data. "
            "The prior and Elo models carry the ensemble right now. Stats-based models "
            "(Pythagorean, Log5, Advanced, Pitching) are near 50/50 on everything, which inflates "
            "underdog edges. Treat these as directional, not actionable, until ~3 weeks of data accumulate. "
            "The model will sharpen significantly as results come in."
        )
    elif games_played < 100:
        return (
            "MID-SEASON NOTE: Models are building confidence with accumulating data. "
            "Ensemble weights are adjusting based on accuracy. Edges above 10% are worth attention."
        )
    else:
        return (
            "Full-season model with substantial data. Ensemble weights tuned by accuracy tracking. "
            "Strong edges (>10%) have historically been profitable."
        )


def generate_top5_picks(betting_games):
    """Generate top 5 picks with commentary"""
    all_picks = []
    for bl in betting_games:
        home_id = bl['home_team_id']
        away_id = bl['away_team_id']
        home_ml = bl['home_ml']
        away_ml = bl['away_ml']
        over_under = bl.get('over_under')

        h_name = display_name(home_id)
        a_name = display_name(away_id)

        dk_h = american_to_implied_prob(home_ml)
        dk_a = american_to_implied_prob(away_ml)
        vig = dk_h + dk_a
        dk_h /= vig
        dk_a /= vig

        try:
            pred = MODELS['ensemble'].predict_game(home_id, away_id)
            mp = pred['home_win_probability']
            he = (mp - dk_h) * 100
            ae = ((1 - mp) - dk_a) * 100

            if he > ae:
                side, edge, odds, prob, dk = h_name, he, home_ml, mp, dk_h
                payout = home_ml / 100 if home_ml > 0 else 100 / abs(home_ml)
                ev = mp * payout - (1 - mp)
            else:
                side, edge, odds, prob, dk = a_name, ae, away_ml, 1 - mp, dk_a
                payout = away_ml / 100 if away_ml > 0 else 100 / abs(away_ml)
                ev = (1 - mp) * payout - mp

            proj = pred['projected_total']
            t_diff = (proj - over_under) if over_under else 0
            t_lean = "OVER" if t_diff > 0.5 else "UNDER" if t_diff < -0.5 else None

            all_picks.append({
                'game': f"{a_name} @ {h_name}",
                'pick': side, 'odds': odds, 'edge': edge, 'ev': ev * 100,
                'model_prob': prob, 'dk_prob': dk,
                'total_lean': t_lean, 'total_diff': t_diff,
                'ou': over_under, 'proj': proj,
                'home_id': home_id, 'away_id': away_id,
                'home_name': h_name, 'away_name': a_name,
                'home_ml': home_ml, 'away_ml': away_ml,
                'commentary': '',
            })
        except Exception:
            pass

    all_picks.sort(key=lambda x: abs(x['edge']), reverse=True)

    # Add commentary to top picks
    for p in all_picks:
        p['commentary'] = generate_pick_commentary(p)

    return all_picks


def generate_pick_commentary(pick):
    """Generate contextual commentary for a pick"""
    edge = pick['edge']
    odds = pick['odds']
    ev = pick['ev']
    home = pick['home_name']
    away = pick['away_name']
    side = pick['pick']
    is_dog = odds > 0

    # Check how many games played
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM games WHERE status='final'")
    total_games = c.fetchone()[0]
    conn.close()

    early_season = total_games < 20

    if early_season and is_dog and edge > 15:
        return (
            f"Big edge on paper but driven largely by cold-start models defaulting to ~50/50. "
            f"DK has {home if side == away else away} as a heavy favorite for good reason. "
            f"Monitor this matchup as data accumulates -- if the edge persists in week 3, it's real."
        )
    elif early_season and is_dog and edge > 5:
        return (
            f"Moderate underdog value. The line looks exploitable but early-season model uncertainty "
            f"means this edge could evaporate once stats-based models calibrate. "
            f"Worth tracking, not yet worth betting heavy."
        )
    elif is_dog and edge > 10:
        return (
            f"Significant model disagreement with the market. "
            f"At {odds:+d}, the payout justifies the risk if our model is even close to right. "
            f"EV of ${ev:+.0f} per $100 is hard to ignore."
        )
    elif is_dog and edge > 5:
        return (
            f"Solid value play. The market may be underrating {side} here. "
            f"Not a slam dunk but the kind of +EV spot that adds up over a season."
        )
    elif not is_dog and edge > 5:
        return (
            f"Model likes the favorite even more than DK does. "
            f"Laying {odds:+d} isn't glamorous but the model sees this as more lopsided than the line suggests."
        )
    elif pick.get('total_lean'):
        lean = pick['total_lean']
        diff = pick['total_diff']
        return (
            f"Side edge is thin but the model sees {lean} value -- projecting {pick['proj']:.1f} total "
            f"vs the {pick['ou']} line ({abs(diff):.1f} run gap). Could be the better play."
        )
    else:
        return (
            f"Small edge. Market has this one roughly right. "
            f"Pass unless you have a strong situational lean."
        )


def create_best_bets_chart(edges, output_path):
    """Create horizontal bar chart of best bets by edge size"""
    if not edges:
        return False

    fig, ax = plt.subplots(figsize=(12, max(4, len(edges) * 0.5)))

    labels = [f"{e['pick']} ({e['odds']:+d})" for e in reversed(edges)]
    edge_vals = [e['edge'] for e in reversed(edges)]
    ev_vals = [e['ev'] for e in reversed(edges)]

    colors = []
    for e in reversed(edges):
        if abs(e['edge']) > 10:
            colors.append('#2ecc71')
        elif abs(e['edge']) > 5:
            colors.append('#f39c12')
        else:
            colors.append('#3498db')

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, edge_vals, color=colors, height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Edge vs DraftKings (%)')
    ax.set_title('Best Bets - Model Edge vs DraftKings (Ensemble)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=-5, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=-10, color='green', linestyle='--', alpha=0.5, linewidth=0.8)

    for bar, val, ev in zip(bars, edge_vals, ev_vals):
        x_pos = bar.get_width()
        ax.annotate(f'{val:+.1f}% (${ev:+.0f})',
                     xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                     xytext=(5 if x_pos >= 0 else -5, 0),
                     textcoords="offset points",
                     ha='left' if x_pos >= 0 else 'right',
                     va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def get_upcoming_games(days=3):
    """Fetch upcoming games from the database"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT g.home_team_id, g.away_team_id, g.date, g.time, g.venue,
               g.is_conference_game, g.is_neutral_site
        FROM games g
        WHERE g.status = 'scheduled'
          AND g.date >= date('now')
          AND g.date <= date('now', ? || ' days')
        ORDER BY g.date, g.time
    ''', (str(days),))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sec_games(days=3):
    """Fetch upcoming SEC games"""
    conn = get_connection()
    c = conn.cursor()

    sec_teams = [
        'mississippi-state', 'ole-miss', 'alabama', 'auburn', 'arkansas',
        'florida', 'georgia', 'kentucky', 'lsu', 'missouri',
        'oklahoma', 'south-carolina', 'tennessee', 'texas',
        'texas-a&m', 'vanderbilt'
    ]
    placeholders = ','.join('?' for _ in sec_teams)

    c.execute(f'''
        SELECT g.home_team_id, g.away_team_id, g.date, g.time, g.venue,
               g.is_conference_game, g.is_neutral_site
        FROM games g
        WHERE g.status = 'scheduled'
          AND g.date >= date('now')
          AND g.date <= date('now', ? || ' days')
          AND (g.home_team_id IN ({placeholders}) OR g.away_team_id IN ({placeholders}))
        ORDER BY g.date, g.time
    ''', (str(days), *sec_teams, *sec_teams))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def generate_predictions_for_game(home_id, away_id, neutral=False):
    """Run all models for a single game"""
    predictions = {}
    for name, model in MODELS.items():
        try:
            predictions[name] = model.predict_game(home_id, away_id, neutral)
        except Exception:
            pass
    return predictions


def generate_weekend_preview(output_path=None):
    """Generate full weekend preview PDF with all 7 models"""

    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = _reports_dir / f"weekend_preview_{date_str}.pdf"

    pdf = BaseballReport()
    pdf.add_page()

    # ─── Title ───
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'WEEKEND PREVIEW', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font('Helvetica', '', 12)
    now = datetime.now()
    pdf.cell(0, 10, now.strftime('%B %d, %Y'), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    # ─── Featured Game: Mississippi State ───
    home_id = normalize_team_id("Mississippi State")
    # Find MS State's next opponent from the DB
    upcoming = get_upcoming_games(days=4)
    ms_game = None
    for g in upcoming:
        if home_id in (g['home_team_id'], g['away_team_id']):
            ms_game = g
            break

    if ms_game:
        is_home = ms_game['home_team_id'] == home_id
        opp_id = ms_game['away_team_id'] if is_home else ms_game['home_team_id']
        neutral = bool(ms_game.get('is_neutral_site'))
        game_home_id = ms_game['home_team_id']
        game_away_id = ms_game['away_team_id']
    else:
        # Fallback: first scheduled game
        game_home_id = home_id
        game_away_id = normalize_team_id("Hofstra")
        opp_id = game_away_id
        neutral = False

    home_name = display_name(game_home_id)
    away_name = display_name(game_away_id)

    pdf.chapter_title(f'FEATURED: {away_name.upper()} @ {home_name.upper()}')
    pdf.set_font('Helvetica', '', 10)
    if ms_game and ms_game.get('venue'):
        pdf.body_text(ms_game['venue'])
    if ms_game and ms_game.get('date'):
        pdf.body_text(f"Date: {ms_game['date']}  Time: {ms_game.get('time', 'TBD')}")
    pdf.ln(5)

    # Get predictions from all 7 models
    predictions = generate_predictions_for_game(game_home_id, game_away_id, neutral)

    if predictions:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Model comparison chart
            chart1 = os.path.join(tmpdir, "model_comp.png")
            create_model_comparison_chart(home_name, away_name, predictions, chart1)
            pdf.image(chart1, x=5, w=200)
            pdf.ln(3)

            # Runs projection chart
            chart2 = os.path.join(tmpdir, "runs_proj.png")
            create_runs_projection_chart(home_name, away_name, predictions, chart2)
            pdf.image(chart2, x=5, w=200)
            pdf.ln(5)

        # Predictions table (all 7 + ensemble)
        pdf.section_title('Model Predictions Summary')
        pdf.set_font('Courier', '', 8)

        header = f"  {'Model':<14} {'Home Win':>8}  {'Away Win':>8}  {'Home R':>6}  {'Away R':>6}  {'Total':>5}  {'Pick'}"
        pdf.mono_line(header)
        pdf.mono_line("  " + "-" * 72)

        # Sort by home win probability descending
        sorted_preds = sorted(predictions.items(),
                              key=lambda x: x[1]['home_win_probability'],
                              reverse=True)

        for name, pred in sorted_preds:
            star = " *" if name == "ensemble" else "  "
            pick = home_name if pred['home_win_probability'] > 0.5 else away_name
            row = (f"  {name:<14} {pred['home_win_probability']*100:>7.1f}%"
                   f"  {pred['away_win_probability']*100:>7.1f}%"
                   f"  {pred['projected_home_runs']:>5.1f}"
                   f"  {pred['projected_away_runs']:>5.1f}"
                   f"  {pred['projected_total']:>5.1f}"
                   f"  {pick}{star}")
            pdf.mono_line(row)

        pdf.ln(5)

        # Series prediction
        ensemble = MODELS.get('ensemble')
        if ensemble:
            try:
                series = ensemble.predict_series(game_home_id, game_away_id)
                pdf.section_title('Series Prediction (Best of 3)')
                pdf.body_text(f"{home_name} wins series: {series['home_series_probability']*100:.0f}%")
                pdf.body_text(f"{away_name} wins series: {series['away_series_probability']*100:.0f}%")
                pdf.body_text(f"Per-game probability: {series['per_game_probability']*100:.1f}%")
            except Exception:
                pass

        pdf.ln(5)

        # Run line analysis
        pdf.section_title('Run Line Analysis (-1.5)')
        pdf.set_font('Courier', '', 8)
        pdf.mono_line(f"  {'Model':<14} {'Home -1.5':>10}  {'Away +1.5':>10}  {'Pick'}")
        pdf.mono_line("  " + "-" * 50)
        for name, pred in sorted_preds:
            if 'run_line' in pred:
                rl = pred['run_line']
                hc = rl.get('home_cover_prob', rl.get('home_minus_1_5', 0))
                ac = rl.get('away_cover_prob', rl.get('away_plus_1_5', 0))
                pick = f"{home_name} -1.5" if hc > 0.5 else f"{away_name} +1.5"
                pdf.mono_line(f"  {name:<14} {hc*100:>9.1f}%  {ac*100:>9.1f}%  {pick}")

    # ─── Ensemble Weights Page ───
    pdf.add_page()
    pdf.chapter_title('DYNAMIC ENSEMBLE WEIGHTS')
    pdf.body_text('Weights auto-adjust based on rolling prediction accuracy.')
    pdf.body_text('Models that predict better get more influence over time.')
    pdf.ln(3)

    ensemble = MODELS.get('ensemble')
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_chart = os.path.join(tmpdir, "weights.png")
        if create_ensemble_weights_chart(ensemble, weights_chart):
            pdf.image(weights_chart, x=10, w=190)
            pdf.ln(5)

    # Weight details
    if ensemble and hasattr(ensemble, 'weights'):
        pdf.section_title('Current Weights')
        pdf.set_font('Courier', '', 9)
        for name, weight in sorted(ensemble.weights.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(weight * 50)
            pdf.mono_line(f"  {name:<14} {weight*100:>5.1f}%  {bar}")

    pdf.ln(5)
    pdf.section_title('Model Descriptions')
    model_descriptions = {
        "prior": "Preseason rankings + Bayesian blending (cold start solver)",
        "elo": "FiveThirtyEight-style Elo ratings, margin-of-victory adjusted",
        "advanced": "Opponent-adjusted, recency-weighted, SOS-aware",
        "pitching": "Starting pitcher matchup + bullpen state + fatigue",
        "conference": "Conference strength adjustments (SEC boost, etc.)",
        "log5": "Bill James Log5 head-to-head formula",
        "pythagorean": "Bill James Pythagorean runs scored/allowed",
    }
    pdf.set_font('Helvetica', '', 9)
    for name, desc in model_descriptions.items():
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(30, 5, name, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ─── SEC Games Page ───
    pdf.add_page()
    pdf.chapter_title('SEC GAMES THIS WEEKEND')

    sec_games = get_sec_games(days=4)
    if sec_games:
        pdf.set_font('Courier', '', 8)
        pdf.mono_line(f"  {'Matchup':<40} {'Date':<12} {'Ensemble':>8}")
        pdf.mono_line("  " + "-" * 62)

        for g in sec_games:
            h = display_name(g['home_team_id'])
            a = display_name(g['away_team_id'])
            matchup = f"{a} @ {h}"
            date = g.get('date', '')

            # Quick ensemble prediction
            try:
                ep = MODELS['ensemble'].predict_game(
                    g['home_team_id'], g['away_team_id'],
                    bool(g.get('is_neutral_site')))
                home_pct = f"{ep['home_win_probability']*100:.0f}%"
                fav = h if ep['home_win_probability'] > 0.5 else a
                result = f"{fav} {home_pct}"
            except Exception:
                result = "N/A"

            pdf.mono_line(f"  {matchup:<40} {date:<12} {result:>8}")
    else:
        pdf.body_text("No SEC games found in the database for this weekend.")
        pdf.body_text("Games will populate as schedules are tracked.")

    pdf.ln(10)

    # ─── Other Top 25 Games ───
    pdf.section_title('TOP 25 MATCHUPS')
    top25 = get_current_top_25()
    if top25:
        ranked_ids = {t['id'] for t in top25}
        ranked_games = []
        all_games = get_upcoming_games(days=4)
        for g in all_games:
            if g['home_team_id'] in ranked_ids or g['away_team_id'] in ranked_ids:
                ranked_games.append(g)

        if ranked_games:
            pdf.set_font('Courier', '', 8)
            for g in ranked_games[:15]:  # Cap at 15
                h = display_name(g['home_team_id'])
                a = display_name(g['away_team_id'])
                try:
                    ep = MODELS['ensemble'].predict_game(
                        g['home_team_id'], g['away_team_id'],
                        bool(g.get('is_neutral_site')))
                    fav = h if ep['home_win_probability'] > 0.5 else a
                    pct = max(ep['home_win_probability'], ep['away_win_probability'])
                    pdf.mono_line(f"  {a} @ {h:<25} → {fav} ({pct*100:.0f}%)")
                except Exception:
                    pdf.mono_line(f"  {a} @ {h}")
        else:
            pdf.body_text("No ranked team games found for this weekend.")
    else:
        pdf.body_text("No Top 25 rankings loaded yet.")

    # ─── Expected Value vs DraftKings ───
    ev_page_added = False
    betting_games = get_betting_lines_for_report(days=4)
    if betting_games:
        pdf.add_page()
        pdf.chapter_title('EXPECTED VALUE vs DRAFTKINGS')
        pdf.body_text('Model probability vs implied odds. Positive EV = value bet.')
        pdf.body_text('Edge = Model Prob - DK Implied Prob. EV = Edge * Payout.')
        pdf.ln(3)

        value_picks = []  # Collect for summary chart

        for bl in betting_games:
            home_id = bl['home_team_id']
            away_id = bl['away_team_id']
            home_ml = bl['home_ml']
            away_ml = bl['away_ml']
            over_under = bl.get('over_under')

            h_name = display_name(home_id)
            a_name = display_name(away_id)

            # DK implied probs (vig-removed)
            dk_home_raw = american_to_implied_prob(home_ml)
            dk_away_raw = american_to_implied_prob(away_ml)
            total_vig = dk_home_raw + dk_away_raw
            dk_home = dk_home_raw / total_vig
            dk_away = dk_away_raw / total_vig

            pdf.section_title(f'{a_name} @ {h_name}')
            pdf.set_font('Courier', '', 8)
            pdf.mono_line(f"  DraftKings:  {h_name} {home_ml:+d} ({dk_home*100:.1f}%)  |  {a_name} {away_ml:+d} ({dk_away*100:.1f}%)")
            if over_under:
                pdf.mono_line(f"  O/U: {over_under}")
            pdf.mono_line("")

            # Header
            pdf.mono_line(f"  {'Model':<14} {'Prob':>6} {'DK':>6} {'Edge':>7} {'ML EV':>8} {'Rating'}")
            pdf.mono_line("  " + "-" * 58)

            for model_name, model in MODELS.items():
                try:
                    pred = model.predict_game(home_id, away_id)
                    model_prob = pred['home_win_probability']

                    # Home side edge
                    home_edge = (model_prob - dk_home) * 100
                    # Away side edge
                    away_edge = ((1 - model_prob) - dk_away) * 100

                    # EV calculation: if we bet $100 on the side with edge
                    if home_edge > away_edge:
                        # Home is the value side
                        edge_pct = home_edge
                        if home_ml > 0:
                            payout = home_ml / 100
                        else:
                            payout = 100 / abs(home_ml)
                        ev = model_prob * payout - (1 - model_prob)
                        ev_dollars = ev * 100  # Per $100 bet
                        pick_side = h_name
                    else:
                        edge_pct = away_edge
                        if away_ml > 0:
                            payout = away_ml / 100
                        else:
                            payout = 100 / abs(away_ml)
                        ev = (1 - model_prob) * payout - model_prob
                        ev_dollars = ev * 100
                        pick_side = a_name

                    # Rating
                    if edge_pct > 10:
                        rating = "STRONG"
                    elif edge_pct > 5:
                        rating = "GOOD"
                    elif edge_pct > 2:
                        rating = "LEAN"
                    else:
                        rating = "--"

                    pdf.mono_line(
                        f"  {model_name:<14} {model_prob*100:>5.1f}% {dk_home*100:>5.1f}% {home_edge:>+6.1f}% "
                        f"${ev_dollars:>+6.1f}  {rating}"
                    )

                    # Track for value picks summary
                    if model_name == 'ensemble' and abs(edge_pct) > 3:
                        value_picks.append((
                            f"{pick_side} ({home_ml:+d}/{away_ml:+d})",
                            edge_pct,
                            ev_dollars,
                            f"{a_name} @ {h_name}"
                        ))
                except Exception:
                    pass

            # Totals comparison
            if over_under:
                pdf.mono_line("")
                pdf.mono_line(f"  {'Model':<14} {'Proj Total':>10} {'DK O/U':>8} {'Lean':>8}")
                pdf.mono_line("  " + "-" * 42)
                for model_name, model in MODELS.items():
                    try:
                        pred = model.predict_game(home_id, away_id)
                        proj = pred['projected_total']
                        diff = proj - over_under
                        lean = "OVER" if diff > 0.5 else "UNDER" if diff < -0.5 else "PUSH"
                        pdf.mono_line(f"  {model_name:<14} {proj:>9.1f} {over_under:>8.1f} {lean:>8}")
                    except Exception:
                        pass

            pdf.ln(5)

        # Value picks summary with chart
        if value_picks:
            pdf.add_page()
            pdf.chapter_title('VALUE PICKS SUMMARY')
            pdf.body_text('Ensemble model picks with >3% edge vs DraftKings.')
            pdf.ln(3)

            with tempfile.TemporaryDirectory() as tmpdir:
                ev_chart = os.path.join(tmpdir, "ev_chart.png")
                if create_ev_chart(value_picks, ev_chart):
                    pdf.image(ev_chart, x=10, w=190)
                    pdf.ln(5)

            pdf.set_font('Courier', '', 9)
            pdf.mono_line(f"  {'Pick':<35} {'Edge':>7} {'EV/$100':>9} {'Game'}")
            pdf.mono_line("  " + "-" * 65)
            for pick, edge, ev, game in sorted(value_picks, key=lambda x: x[1], reverse=True):
                pdf.mono_line(f"  {pick:<35} {edge:>+6.1f}% ${ev:>+7.1f}  {game}")

        ev_page_added = True

    # ─── Best Bets: Largest Model vs DK Disagreements ───
    if betting_games and len(betting_games) > 1:
        all_edges = []
        for bl in betting_games:
            home_id = bl['home_team_id']
            away_id = bl['away_team_id']
            home_ml = bl['home_ml']
            away_ml = bl['away_ml']
            over_under = bl.get('over_under')

            h_name = display_name(home_id)
            a_name = display_name(away_id)

            dk_home_raw = american_to_implied_prob(home_ml)
            dk_away_raw = american_to_implied_prob(away_ml)
            total_vig = dk_home_raw + dk_away_raw
            dk_home = dk_home_raw / total_vig
            dk_away = dk_away_raw / total_vig

            # Get ensemble prediction
            try:
                pred = MODELS['ensemble'].predict_game(home_id, away_id)
                model_prob = pred['home_win_probability']
                home_edge = (model_prob - dk_home) * 100
                away_edge = ((1 - model_prob) - dk_away) * 100

                if home_edge > away_edge:
                    best_side = h_name
                    best_edge = home_edge
                    best_odds = home_ml
                    best_prob = model_prob
                    dk_prob = dk_home
                    if home_ml > 0:
                        payout = home_ml / 100
                    else:
                        payout = 100 / abs(home_ml)
                    ev = model_prob * payout - (1 - model_prob)
                else:
                    best_side = a_name
                    best_edge = away_edge
                    best_odds = away_ml
                    best_prob = 1 - model_prob
                    dk_prob = dk_away
                    if away_ml > 0:
                        payout = away_ml / 100
                    else:
                        payout = 100 / abs(away_ml)
                    ev = (1 - model_prob) * payout - model_prob

                ev_dollars = ev * 100

                # Totals edge
                total_edge = None
                total_lean = None
                if over_under:
                    proj_total = pred['projected_total']
                    total_diff = proj_total - over_under
                    if abs(total_diff) > 0.5:
                        total_lean = "OVER" if total_diff > 0 else "UNDER"
                        total_edge = abs(total_diff)

                all_edges.append({
                    'game': f"{a_name} @ {h_name}",
                    'pick': best_side,
                    'odds': best_odds,
                    'model_prob': best_prob,
                    'dk_prob': dk_prob,
                    'edge': best_edge,
                    'ev': ev_dollars,
                    'total_lean': total_lean,
                    'total_edge': total_edge,
                    'over_under': over_under,
                    'proj_total': pred['projected_total'],
                })
            except Exception:
                pass

        # Sort by absolute edge descending
        all_edges.sort(key=lambda x: abs(x['edge']), reverse=True)

        pdf.add_page()
        pdf.chapter_title('BEST BETS - TOP 25 GAMES')
        pdf.body_text('Games ranked by largest ensemble model disagreement with DraftKings.')
        pdf.body_text('Bigger edge = more model confidence the line is wrong.')
        pdf.ln(5)

        # Best bets chart
        with tempfile.TemporaryDirectory() as tmpdir:
            bb_chart = os.path.join(tmpdir, "best_bets.png")
            if create_best_bets_chart(all_edges[:12], bb_chart):
                pdf.image(bb_chart, x=5, w=200)
                pdf.ln(5)

        # Table
        pdf.set_font('Courier', '', 7)
        pdf.mono_line(f"  {'#':<3} {'Game':<30} {'Pick':<18} {'Odds':>6} {'Model':>6} {'DK':>6} {'Edge':>7} {'EV/$100':>8}")
        pdf.mono_line("  " + "-" * 88)

        for i, e in enumerate(all_edges, 1):
            rating = "***" if abs(e['edge']) > 10 else "** " if abs(e['edge']) > 5 else "*  " if abs(e['edge']) > 2 else "   "
            pdf.mono_line(
                f"  {i:<3} {e['game']:<30} {e['pick']:<18} {e['odds']:>+5d} "
                f"{e['model_prob']*100:>5.1f}% {e['dk_prob']*100:>5.1f}% "
                f"{e['edge']:>+6.1f}% ${e['ev']:>+6.1f} {rating}"
            )

        pdf.ln(5)
        pdf.set_font('Helvetica', '', 9)
        pdf.body_text("*** = STRONG (>10% edge)  ** = GOOD (>5%)  * = LEAN (>2%)")

        # Totals best bets
        total_plays = [e for e in all_edges if e['total_lean'] and e['total_edge'] and e['total_edge'] > 1.0]
        if total_plays:
            total_plays.sort(key=lambda x: x['total_edge'], reverse=True)
            pdf.ln(5)
            pdf.section_title('TOTALS PLAYS')
            pdf.set_font('Courier', '', 8)
            pdf.mono_line(f"  {'Game':<30} {'Lean':<6} {'Model':>6} {'DK O/U':>7} {'Diff':>6}")
            pdf.mono_line("  " + "-" * 58)
            for e in total_plays:
                pdf.mono_line(
                    f"  {e['game']:<30} {e['total_lean']:<6} {e['proj_total']:>5.1f} "
                    f"{e['over_under']:>7.1f} {e['total_edge']:>+5.1f}"
                )

    # ─── Top 5 Picks of the Weekend ───
    if betting_games and len(betting_games) > 1:
        top5 = generate_top5_picks(betting_games)
        if top5:
            pdf.add_page()
            pdf.chapter_title('TOP 5 PICKS OF THE WEEKEND')
            pdf.ln(3)

            for i, pick in enumerate(top5[:5], 1):
                # Pick header
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 8, f"#{i}  {pick['pick']} ({pick['odds']:+d})",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(0, 6, pick['game'],
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                # Stats line
                pdf.set_font('Courier', '', 9)
                pdf.mono_line(
                    f"  Model: {pick['model_prob']*100:.1f}%  |  DK Implied: {pick['dk_prob']*100:.1f}%  |  "
                    f"Edge: {pick['edge']:+.1f}%  |  EV/$100: ${pick['ev']:+.1f}"
                )
                if pick.get('total_lean'):
                    pdf.mono_line(
                        f"  Totals: {pick['total_lean']} {pick['ou']} "
                        f"(model projects {pick['proj']:.1f})"
                    )

                # Commentary
                pdf.set_font('Helvetica', 'I', 9)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 5, f"  {pick['commentary']}")
                pdf.set_text_color(0, 0, 0)

                # Confidence bar
                conf = min(abs(pick['edge']), 30) / 30  # Normalize to 0-1
                bar_width = 150
                pdf.set_fill_color(46, 204, 113) if pick['edge'] > 10 else \
                    pdf.set_fill_color(243, 156, 18) if pick['edge'] > 5 else \
                    pdf.set_fill_color(52, 152, 219)
                pdf.set_font('Helvetica', '', 8)
                pdf.cell(25, 5, "Confidence:", new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.cell(int(bar_width * conf), 5, '', fill=True,
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(6)

            # Disclaimer
            pdf.ln(5)
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(120, 120, 120)
            pdf.multi_cell(0, 4,
                get_model_disclaimer()
            )
            pdf.set_text_color(0, 0, 0)

    # ─── Footer ───
    pdf.ln(15)
    pdf.set_font('Helvetica', '', 9)
    pdf.body_text(f"Report generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.body_text(f"Models: {len(MODELS)} ({', '.join(MODELS.keys())})")
    pdf.body_text("Next report: Monday 11 PM CT (Weekend Recap + Midweek Preview)")

    pdf.output(str(output_path))
    print(f"✓ Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    output = generate_weekend_preview()
    print(f"\nGenerated: {output}")
