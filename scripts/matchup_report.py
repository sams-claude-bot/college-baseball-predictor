#!/usr/bin/env python3
"""
Generate detailed matchup reports for Mississippi State games

Includes:
- Pitching matchup with stats
- Top hitters comparison
- Key stats and trends
"""

import sys
from pathlib import Path
from datetime import datetime
import tempfile

_scripts_dir = Path(__file__).parent
_models_dir = _scripts_dir.parent / "models"
_reports_dir = _scripts_dir.parent / "reports"
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_models_dir))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from player_stats import get_starting_pitchers, get_top_hitters, get_team_roster
from compare_models import MODELS, normalize_team_id
from database import get_connection, get_team_record, get_team_runs

_reports_dir.mkdir(parents=True, exist_ok=True)


def create_hitting_comparison_chart(home_team, away_team, home_hitters, away_hitters, output_path):
    """Create batting comparison chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Home team hitters
    if home_hitters:
        names = [h['name'].split()[-1][:8] for h in home_hitters[:6]]  # Last names, truncated
        avgs = [h['batting_avg'] for h in home_hitters[:6]]
        
        colors = ['#800000' if avg >= .300 else '#a04040' if avg >= .250 else '#c08080' for avg in avgs]
        axes[0].barh(names, avgs, color=colors)
        axes[0].set_xlabel('Batting Average')
        axes[0].set_title(f'{home_team} Top Hitters')
        axes[0].set_xlim(0, 0.5)
        axes[0].axvline(x=0.300, color='green', linestyle='--', alpha=0.5, label='.300')
        for i, avg in enumerate(avgs):
            axes[0].annotate(f'.{int(avg*1000):03d}', xy=(avg + 0.01, i), va='center')
    else:
        axes[0].text(0.5, 0.5, 'No hitting data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f'{home_team} Top Hitters')
    
    # Away team hitters
    if away_hitters:
        names = [h['name'].split()[-1][:8] for h in away_hitters[:6]]
        avgs = [h['batting_avg'] for h in away_hitters[:6]]
        
        colors = ['#4a4a4a' if avg >= .300 else '#6a6a6a' if avg >= .250 else '#9a9a9a' for avg in avgs]
        axes[1].barh(names, avgs, color=colors)
        axes[1].set_xlabel('Batting Average')
        axes[1].set_title(f'{away_team} Top Hitters')
        axes[1].set_xlim(0, 0.5)
        axes[1].axvline(x=0.300, color='green', linestyle='--', alpha=0.5)
        for i, avg in enumerate(avgs):
            axes[1].annotate(f'.{int(avg*1000):03d}', xy=(avg + 0.01, i), va='center')
    else:
        axes[1].text(0.5, 0.5, 'No hitting data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'{away_team} Top Hitters')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_pitching_comparison_chart(home_starters, away_starters, home_team, away_team, output_path):
    """Create pitching staff comparison"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Combine data
    pitchers = []
    eras = []
    colors = []
    
    for p in (home_starters or [])[:3]:
        pitchers.append(f"{p['name'].split()[-1][:8]}\n({home_team[:3]})")
        eras.append(p['era'] if p['era'] else 0)
        colors.append('#800000')
    
    for p in (away_starters or [])[:3]:
        pitchers.append(f"{p['name'].split()[-1][:8]}\n({away_team[:3]})")
        eras.append(p['era'] if p['era'] else 0)
        colors.append('#4a4a4a')
    
    if pitchers:
        bars = ax.bar(pitchers, eras, color=colors)
        ax.set_ylabel('ERA')
        ax.set_title('Starting Pitchers - ERA Comparison')
        ax.axhline(y=3.00, color='green', linestyle='--', alpha=0.5, label='Elite (3.00)')
        ax.axhline(y=4.50, color='orange', linestyle='--', alpha=0.5, label='Avg (4.50)')
        
        for bar, era in zip(bars, eras):
            ax.annotate(f'{era:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No pitching data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_matchup_pdf(home_team, away_team, game_date=None, output_path=None):
    """Generate full matchup PDF report"""
    
    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)
    
    if output_path is None:
        date_str = game_date or datetime.now().strftime("%Y-%m-%d")
        output_path = _reports_dir / f"matchup_{home_id}_vs_{away_id}_{date_str}.pdf"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_fill_color(128, 0, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, f'MATCHUP: {away_team} @ {home_team}', ln=True, fill=True, align='C')
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, f'Game Date: {game_date or "TBD"}', ln=True, align='C')
    pdf.ln(5)
    
    # Get data
    home_starters = get_starting_pitchers(home_id)
    away_starters = get_starting_pitchers(away_id)
    home_hitters = get_top_hitters(home_id, 6)
    away_hitters = get_top_hitters(away_id, 6)
    
    # Pitching Matchup Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 10, 'PITCHING MATCHUP', ln=True, fill=True)
    pdf.ln(3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pitching chart
        pitch_chart = f"{tmpdir}/pitching.png"
        create_pitching_comparison_chart(home_starters, away_starters, home_team, away_team, pitch_chart)
        pdf.image(pitch_chart, x=10, w=190)
    
    pdf.ln(5)
    
    # Pitching details table
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(95, 8, f'{home_team} Probable Starter', border=1)
    pdf.cell(95, 8, f'{away_team} Probable Starter', border=1, ln=True)
    
    pdf.set_font('Helvetica', '', 10)
    
    home_sp = home_starters[0] if home_starters else None
    away_sp = away_starters[0] if away_starters else None
    
    if home_sp:
        home_sp_text = f"{home_sp['name']} ({home_sp['throws']}HP)\n{home_sp['wins']}-{home_sp['losses']}, {home_sp['era']:.2f} ERA"
    else:
        home_sp_text = "TBA"
    
    if away_sp:
        away_sp_text = f"{away_sp['name']} ({away_sp['throws']}HP)\n{away_sp['wins']}-{away_sp['losses']}, {away_sp['era']:.2f} ERA"
    else:
        away_sp_text = "TBA"
    
    pdf.multi_cell(95, 6, home_sp_text, border=1)
    pdf.set_xy(pdf.get_x() + 95, pdf.get_y() - 12)
    pdf.multi_cell(95, 6, away_sp_text, border=1)
    
    pdf.ln(10)
    
    # Hitting Comparison
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'HITTING COMPARISON', ln=True, fill=True)
    pdf.ln(3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hit_chart = f"{tmpdir}/hitting.png"
        create_hitting_comparison_chart(home_team, away_team, home_hitters, away_hitters, hit_chart)
        pdf.image(hit_chart, x=10, w=190)
    
    pdf.ln(5)
    
    # Hitting stats tables
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, f'{home_team} Top Hitters', ln=True)
    
    if home_hitters:
        pdf.set_font('Courier', '', 8)
        header = f"{'Name':<18} {'AVG':<6} {'HR':<4} {'RBI':<4} {'OPS':<6}"
        pdf.cell(0, 5, header, ln=True)
        pdf.cell(0, 3, "-" * 45, ln=True)
        
        for h in home_hitters[:5]:
            row = f"{h['name'][:17]:<18} {h['batting_avg']:.3f}  {h['home_runs']:<4} {h['rbi']:<4} {h['ops']:.3f}"
            pdf.cell(0, 5, row, ln=True)
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 6, "No hitting data available - season opener", ln=True)
    
    pdf.ln(5)
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, f'{away_team} Top Hitters', ln=True)
    
    if away_hitters:
        pdf.set_font('Courier', '', 8)
        pdf.cell(0, 5, header, ln=True)
        pdf.cell(0, 3, "-" * 45, ln=True)
        
        for h in away_hitters[:5]:
            row = f"{h['name'][:17]:<18} {h['batting_avg']:.3f}  {h['home_runs']:<4} {h['rbi']:<4} {h['ops']:.3f}"
            pdf.cell(0, 5, row, ln=True)
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 6, "No hitting data available", ln=True)
    
    # Model Predictions
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'MODEL PREDICTIONS', ln=True, fill=True)
    pdf.ln(5)
    
    pdf.set_font('Courier', '', 9)
    header = f"{'Model':<12} {'Home Win':<10} {'Away Win':<10} {'Total':<8} {'Pick'}"
    pdf.cell(0, 5, header, ln=True)
    pdf.cell(0, 3, "-" * 55, ln=True)
    
    for name, model in MODELS.items():
        try:
            pred = model.predict_game(home_id, away_id)
            pick = home_team if pred['home_win_probability'] > 0.5 else away_team
            row = f"{name:<12} {pred['home_win_probability']*100:>6.1f}%   {pred['away_win_probability']*100:>6.1f}%   {pred['projected_total']:>5.1f}   {pick}"
            pdf.cell(0, 5, row, ln=True)
        except Exception as e:
            pdf.cell(0, 5, f"{name:<12} Error: {str(e)[:30]}", ln=True)
    
    # Footer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 5, "Stats will update as season progresses", ln=True)
    
    pdf.output(str(output_path))
    print(f"✓ Matchup report saved: {output_path}")
    return output_path


def print_matchup_text(home_team, away_team):
    """Print text matchup report to console"""
    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)
    
    print(f"\n{'='*60}")
    print(f"⚾ MATCHUP: {away_team} @ {home_team}")
    print('='*60)
    
    # Pitching
    print(f"\n🎯 PITCHING MATCHUP")
    print("-" * 40)
    
    home_sp = get_starting_pitchers(home_id)
    away_sp = get_starting_pitchers(away_id)
    
    if home_sp:
        sp = home_sp[0]
        print(f"\n{home_team} STARTER:")
        print(f"   {sp['name']} ({sp['position']})")
        print(f"   {sp['wins']}-{sp['losses']} | {sp['era']:.2f} ERA | {sp['whip']:.2f} WHIP")
    else:
        print(f"\n{home_team} STARTER: TBA")
    
    if away_sp:
        sp = away_sp[0]
        print(f"\n{away_team} STARTER:")
        print(f"   {sp['name']} ({sp['position']})")
        print(f"   {sp['wins']}-{sp['losses']} | {sp['era']:.2f} ERA")
    else:
        print(f"\n{away_team} STARTER: TBA")
    
    # Hitting
    print(f"\n🏏 TOP HITTERS")
    print("-" * 40)
    
    home_hit = get_top_hitters(home_id, 5)
    if home_hit:
        print(f"\n{home_team}:")
        print(f"  {'Name':<18} {'AVG':<6} {'HR':<4} {'RBI':<4}")
        for h in home_hit:
            print(f"  {h['name'][:17]:<18} .{int(h['batting_avg']*1000):03d}  {h['home_runs']:<4} {h['rbi']:<4}")
    
    away_hit = get_top_hitters(away_id, 5)
    if away_hit:
        print(f"\n{away_team}:")
        print(f"  {'Name':<18} {'AVG':<6} {'HR':<4} {'RBI':<4}")
        for h in away_hit:
            print(f"  {h['name'][:17]:<18} .{int(h['batting_avg']*1000):03d}  {h['home_runs']:<4} {h['rbi']:<4}")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python matchup_report.py text <home> <away>  - Text report")
        print("  python matchup_report.py pdf <home> <away>   - PDF report")
        return
    
    cmd = sys.argv[1]
    home = sys.argv[2] if len(sys.argv) > 2 else "Mississippi State"
    away = sys.argv[3] if len(sys.argv) > 3 else "Hofstra"
    
    if cmd == "text":
        print_matchup_text(home, away)
    elif cmd == "pdf":
        generate_matchup_pdf(home, away)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
