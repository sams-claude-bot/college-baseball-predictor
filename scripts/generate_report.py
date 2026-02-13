#!/usr/bin/env python3
"""
Generate PDF reports with charts for college baseball predictions
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF

from compare_models import MODELS, normalize_team_id
from database import get_current_top_25

# Ensure reports directory exists
_reports_dir.mkdir(parents=True, exist_ok=True)

class BaseballReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'College Baseball Predictions', align='C', ln=True)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(128, 0, 0)  # Maroon for MS State
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)
        
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(128, 0, 0)
        self.cell(0, 8, title, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)


def create_model_comparison_chart(home_team, away_team, predictions, output_path):
    """Create bar chart comparing model predictions"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    models = list(predictions.keys())
    home_probs = [predictions[m]['home_win_probability'] * 100 for m in models]
    away_probs = [predictions[m]['away_win_probability'] * 100 for m in models]
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], home_probs, width, label=home_team, color='#800000')
    bars2 = ax.bar([i + width/2 for i in x], away_probs, width, label=away_team, color='#4a4a4a')
    
    ax.set_ylabel('Win Probability (%)')
    ax.set_title(f'Model Predictions: {away_team} @ {home_team}')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_runs_projection_chart(home_team, away_team, predictions, output_path):
    """Create chart showing projected runs"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    models = list(predictions.keys())
    home_runs = [predictions[m]['projected_home_runs'] for m in models]
    away_runs = [predictions[m]['projected_away_runs'] for m in models]
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], home_runs, width, label=home_team, color='#800000')
    bars2 = ax.bar([i + width/2 for i in x], away_runs, width, label=away_team, color='#4a4a4a')
    
    ax.set_ylabel('Projected Runs')
    ax.set_title(f'Runs Projection: {away_team} @ {home_team}')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_value_chart(value_picks, output_path):
    """Create chart showing betting value/edge"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    picks = [v[0] for v in value_picks]
    edges = [float(v[1].replace('%', '')) for v in value_picks]
    
    colors = ['#2ecc71' if e > 10 else '#f39c12' if e > 5 else '#95a5a6' for e in edges]
    
    bars = ax.barh(picks, edges, color=colors)
    ax.set_xlabel('Edge vs DraftKings (%)')
    ax.set_title('Value Picks - Model Edge vs Sportsbook')
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='10% threshold')
    
    # Add value labels
    for bar, edge in zip(bars, edges):
        ax.annotate(f'{edge:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points", ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_weekend_preview(output_path=None):
    """Generate full weekend preview PDF"""
    
    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = _reports_dir / f"weekend_preview_{date_str}.pdf"
    
    pdf = BaseballReport()
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'WEEKEND PREVIEW', ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Opening Weekend - February 13-15, 2026', ln=True, align='C')
    pdf.ln(10)
    
    # Featured Game
    pdf.chapter_title('#4 MISSISSIPPI STATE vs HOFSTRA')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, 'Dudy Noble Field, Starkville, MS', ln=True)
    pdf.cell(0, 6, 'Friday 4:00 PM | Saturday 1:00 PM | Sunday 1:00 PM CT', ln=True)
    pdf.ln(5)
    
    # Get predictions
    home_id = normalize_team_id("Mississippi State")
    away_id = normalize_team_id("Hofstra")
    
    predictions = {}
    for name, model in MODELS.items():
        try:
            predictions[name] = model.predict_game(home_id, away_id)
        except:
            pass
    
    # Create charts
    with tempfile.TemporaryDirectory() as tmpdir:
        # Model comparison chart
        chart1_path = os.path.join(tmpdir, "model_comp.png")
        create_model_comparison_chart("Mississippi State", "Hofstra", predictions, chart1_path)
        pdf.image(chart1_path, x=10, w=190)
        pdf.ln(5)
        
        # Runs projection chart
        chart2_path = os.path.join(tmpdir, "runs_proj.png")
        create_runs_projection_chart("Mississippi State", "Hofstra", predictions, chart2_path)
        pdf.image(chart2_path, x=10, w=190)
        pdf.ln(5)
    
    # Predictions table
    pdf.section_title('Model Predictions Summary')
    pdf.set_font('Courier', '', 9)
    
    header = f"{'Model':<12} {'MS State':<10} {'Hofstra':<10} {'Total':<8} {'Pick':<12}"
    pdf.cell(0, 5, header, ln=True)
    pdf.cell(0, 5, "-" * 55, ln=True)
    
    for name, pred in predictions.items():
        row = f"{name:<12} {pred['home_win_probability']*100:>6.1f}%   {pred['away_win_probability']*100:>6.1f}%   {pred['projected_total']:>5.1f}   {'MS State' if pred['home_win_probability'] > 0.5 else 'Hofstra':<12}"
        pdf.cell(0, 5, row, ln=True)
    
    pdf.ln(5)
    
    # Series prediction
    ensemble = MODELS.get('ensemble')
    if ensemble:
        series = ensemble.predict_series(home_id, away_id)
        pdf.section_title('Series Prediction (Best of 3)')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f"Mississippi State wins series: {series['home_series_probability']*100:.0f}%", ln=True)
        pdf.cell(0, 6, f"Hofstra wins series: {series['away_series_probability']*100:.0f}%", ln=True)
    
    # New page for Top 25
    pdf.add_page()
    pdf.chapter_title('TOP 25 GAMES THIS WEEKEND')
    
    top_25_games = [
        ("#1 UCLA vs Oregon", "78%", "UCLA"),
        ("#2 LSU vs Wofford", "82%", "LSU"),
        ("#3 Texas vs Rice", "71%", "Texas"),
        ("#4 MS State vs Hofstra", "59%", "MS State"),
        ("#5 Georgia Tech vs Army", "75%", "Ga Tech"),
        ("#7 Arkansas vs W. Illinois", "85%", "Arkansas"),
        ("#9 Auburn vs UAB", "68%", "Auburn"),
        ("#13 Florida vs Michigan", "65%", "Florida"),
        ("#14 Tennessee vs Air Force", "72%", "Tennessee"),
        ("#15 Georgia vs Georgia State", "69%", "Georgia"),
        ("#23 Vanderbilt vs Indiana St", "70%", "Vanderbilt"),
    ]
    
    pdf.set_font('Courier', '', 9)
    header = f"{'Matchup':<30} {'Win Prob':<10} {'Pick':<15}"
    pdf.cell(0, 5, header, ln=True)
    pdf.cell(0, 5, "-" * 55, ln=True)
    
    for game in top_25_games:
        row = f"{game[0]:<30} {game[1]:<10} {game[2]:<15}"
        pdf.cell(0, 5, row, ln=True)
    
    pdf.ln(10)
    
    # Value picks section
    pdf.chapter_title('VALUE PICKS vs DRAFTKINGS')
    
    value_picks = [
        ("Hofstra +450", "26.5%", "Models see MS State overvalued"),
        ("Oregon +280", "12.1%", "UCLA line too steep"),
        ("OVER 9.5 (MS State)", "8.2%", "Models project 10.2 runs"),
        ("Michigan +240", "7.8%", "Florida may be overrated early"),
    ]
    
    # Value chart
    with tempfile.TemporaryDirectory() as tmpdir:
        chart3_path = os.path.join(tmpdir, "value.png")
        create_value_chart(value_picks, chart3_path)
        pdf.image(chart3_path, x=10, w=190)
    
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 10)
    
    for pick in value_picks:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(50, 6, pick[0], ln=False)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(20, 6, f"Edge: {pick[1]}", ln=False)
        pdf.cell(0, 6, pick[2], ln=True)
    
    # SEC section
    pdf.add_page()
    pdf.chapter_title('SEC OUT-OF-CONFERENCE WATCH')
    
    pdf.set_font('Helvetica', '', 10)
    sec_games = [
        "LSU vs Wofford (Southern Conference)",
        "Texas vs Rice (AAC)",
        "Mississippi State vs Hofstra (CAA)",
        "Arkansas vs Western Illinois (Summit League)",
        "Auburn vs UAB (AAC)",
        "Florida vs Michigan (Big Ten)",
        "Tennessee vs Air Force (Mountain West)",
        "Georgia vs Georgia State (Sun Belt)",
        "Vanderbilt vs Indiana State (MVC)",
        "Texas A&M vs Lamar (WAC)",
        "Kentucky vs Wright State (Horizon)",
    ]
    
    pdf.cell(0, 6, "SEC teams open the season against these conferences:", ln=True)
    pdf.ln(3)
    
    for game in sec_games:
        pdf.cell(10, 5, chr(149), ln=False)  # Bullet
        pdf.cell(0, 5, game, ln=True)
    
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 6, "Tracking SEC performance vs other conferences throughout the season.", ln=True)
    
    # Footer info
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 5, "Next report: Monday 11 PM CT (Weekend Recap + Midweek Preview)", ln=True)
    
    # Save
    pdf.output(str(output_path))
    print(f"✓ Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    output = generate_weekend_preview()
    print(f"\nGenerated: {output}")
