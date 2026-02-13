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
