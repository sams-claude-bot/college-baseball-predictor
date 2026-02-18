#!/usr/bin/env python3
"""
Power Rankings - Multi-Model Round-Robin Simulation

Runs every team against every other team using ALL models (not just ensemble),
then ranks by composite score. Processes one model at a time to stay memory-lean.

Tables:
  power_rankings        — composite rankings (backward compatible)
  power_rankings_detail — per-model scores for each team/date

Usage:
    python3 scripts/power_rankings.py [--top N] [--conference SEC] [--min-games 5]
    python3 scripts/power_rankings.py --top 25 --store -v
    python3 scripts/power_rankings.py --model elo --top 50
"""

import sys
import gc
import argparse
from pathlib import Path
from datetime import datetime

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

import warnings
import logging

# Suppress noisy model warnings during bulk simulation
logging.getLogger().setLevel(logging.ERROR)

from database import get_connection

# Models that produce meaningful win probabilities (not totals/spread-only)
WIN_PROB_MODELS = [
    "pythagorean", "elo", "log5", "advanced", "pitching",
    "conference", "prior", "poisson", "neural", "xgboost", "lightgbm",
]

# Ensemble weights for composite power score (mirrors ensemble_model defaults)
MODEL_WEIGHTS = {
    "prior": 0.16,
    "elo": 0.15,
    "conference": 0.12,
    "advanced": 0.12,
    "log5": 0.10,
    "poisson": 0.08,
    "pythagorean": 0.08,
    "pitching": 0.05,
    "lightgbm": 0.08,
    "xgboost": 0.06,
    "neural": 0.00,  # tracked independently, not in ensemble
}

# Short names for display
ABBREV = {
    'pythagorean': 'Pyth', 'elo': 'Elo', 'log5': 'Log5',
    'advanced': 'Adv', 'pitching': 'Pitch', 'conference': 'Conf',
    'prior': 'Prior', 'poisson': 'Pois', 'neural': 'NN',
    'xgboost': 'XGB', 'lightgbm': 'LGB', 'ensemble': 'Ens',
}


def get_eligible_teams(min_games=3):
    """Get teams with enough completed games for meaningful predictions."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT t.id, t.name, t.conference, COUNT(g.id) as games_played
        FROM teams t
        JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id)
            AND g.status = 'final'
        GROUP BY t.id
        HAVING games_played >= ?
        ORDER BY t.name
    ''', (min_games,))
    teams = [dict(r) for r in c.fetchall()]
    conn.close()
    return teams


def load_single_model(model_name):
    """
    Import and instantiate a single model. Returns (name, instance) or None.
    Done inside a function so we can delete references and free memory after each model.
    """
    try:
        if model_name == "pythagorean":
            from models.pythagorean_model import PythagoreanModel
            return PythagoreanModel()
        elif model_name == "elo":
            from models.elo_model import EloModel
            return EloModel()
        elif model_name == "log5":
            from models.log5_model import Log5Model
            return Log5Model()
        elif model_name == "advanced":
            from models.advanced_model import AdvancedModel
            return AdvancedModel()
        elif model_name == "pitching":
            from models.pitching_model import PitchingModel
            return PitchingModel()
        elif model_name == "conference":
            from models.conference_model import ConferenceModel
            return ConferenceModel()
        elif model_name == "prior":
            from models.prior_model import PriorModel
            return PriorModel()
        elif model_name == "poisson":
            from models.ensemble_model import PoissonModelWrapper
            return PoissonModelWrapper()
        elif model_name == "neural":
            from models.neural_model import NeuralModel
            return NeuralModel(use_model_predictions=False)
        elif model_name == "xgboost":
            from models.xgboost_model import XGBMoneylineModel
            m = XGBMoneylineModel(use_model_predictions=False)
            return m if m.is_trained() else None
        elif model_name == "lightgbm":
            from models.lightgbm_model import LGBMoneylineModel
            m = LGBMoneylineModel(use_model_predictions=False)
            return m if m.is_trained() else None
    except Exception as e:
        print(f"  ⚠️  Could not load {model_name}: {e}")
    return None


def run_round_robin_single_model(model, teams, verbose=False):
    """
    Run round-robin for ONE model. Returns dict of team_id -> stats.
    Memory usage: O(n) where n = number of teams.
    """
    team_ids = [t['id'] for t in teams]
    n = len(team_ids)
    total_matchups = n * (n - 1) // 2

    # Initialize accumulators
    acc = {}
    for t in teams:
        acc[t['id']] = {
            'team_id': t['id'],
            'team_name': t['name'],
            'conference': t['conference'],
            'total_win_prob': 0.0,
            'total_projected_runs': 0.0,
            'total_projected_runs_against': 0.0,
            'matchups': 0,
            'dominant_wins': 0,
            'close_matchups': 0,
            'errors': 0,
        }

    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            home_id = team_ids[i]
            away_id = team_ids[j]

            try:
                pred = model.predict_game(home_id, away_id, neutral_site=True)
                home_prob = pred.get('home_win_probability', 0.5)
                away_prob = pred.get('away_win_probability', 1.0 - home_prob)
                home_runs = pred.get('projected_home_runs', 0)
                away_runs = pred.get('projected_away_runs', 0)
            except Exception:
                home_prob = away_prob = 0.5
                home_runs = away_runs = 0
                acc[home_id]['errors'] += 1
                acc[away_id]['errors'] += 1

            for tid, wp, rf, ra in [
                (home_id, home_prob, home_runs, away_runs),
                (away_id, away_prob, away_runs, home_runs),
            ]:
                r = acc[tid]
                r['total_win_prob'] += wp
                r['total_projected_runs'] += rf
                r['total_projected_runs_against'] += ra
                r['matchups'] += 1
                if wp > 0.70:
                    r['dominant_wins'] += 1
                if 0.45 <= wp <= 0.55:
                    r['close_matchups'] += 1

            done += 1
            if verbose and done % 2000 == 0:
                print(f"    {done}/{total_matchups} ({done * 100 // total_matchups}%)")

    # Calculate averages
    for tid, r in acc.items():
        m = max(r['matchups'], 1)
        r['avg_win_prob'] = r['total_win_prob'] / m
        r['avg_projected_runs'] = r['total_projected_runs'] / m
        r['avg_projected_runs_against'] = r['total_projected_runs_against'] / m
        r['avg_run_diff'] = r['avg_projected_runs'] - r['avg_projected_runs_against']
        r['dominance_pct'] = r['dominant_wins'] / m

    return acc


def ensure_tables():
    """Create DB tables if needed."""
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS power_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            power_score REAL NOT NULL,
            avg_win_prob REAL,
            avg_run_diff REAL,
            avg_projected_runs REAL,
            dominance_pct REAL,
            prev_rank INTEGER,
            rank_change INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_power_rankings_date ON power_rankings(date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_power_rankings_team ON power_rankings(team_id)')

    c.execute('''
        CREATE TABLE IF NOT EXISTS power_rankings_detail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            avg_win_prob REAL,
            avg_run_diff REAL,
            avg_projected_runs REAL,
            dominance_pct REAL,
            model_rank INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, model_name, date)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_prd_date ON power_rankings_detail(date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_prd_team ON power_rankings_detail(team_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_prd_model ON power_rankings_detail(model_name)')

    conn.commit()
    conn.close()


def store_model_detail(model_name, model_results, date_str):
    """Store one model's round-robin results into power_rankings_detail."""
    conn = get_connection()
    c = conn.cursor()

    # Compute per-model ranks
    sorted_tids = sorted(model_results.keys(),
                         key=lambda t: -model_results[t]['avg_win_prob'])
    model_ranks = {tid: i + 1 for i, tid in enumerate(sorted_tids)}

    # Clear this model's rows for the date
    c.execute('DELETE FROM power_rankings_detail WHERE model_name = ? AND date = ?',
              (model_name, date_str))

    for tid, r in model_results.items():
        c.execute('''
            INSERT INTO power_rankings_detail
            (team_id, model_name, avg_win_prob, avg_run_diff,
             avg_projected_runs, dominance_pct, model_rank, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tid, model_name, r['avg_win_prob'], r['avg_run_diff'],
            r['avg_projected_runs'], r['dominance_pct'],
            model_ranks[tid], date_str
        ))

    conn.commit()
    conn.close()


def load_all_model_scores(date_str):
    """Load per-model avg_win_prob from power_rankings_detail for composite calculation."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT team_id, model_name, avg_win_prob, avg_run_diff,
               avg_projected_runs, dominance_pct, model_rank
        FROM power_rankings_detail WHERE date = ?
    ''', (date_str,))
    rows = c.fetchall()
    conn.close()

    # Restructure: model_name -> {team_id -> row}
    results = {}
    for row in rows:
        mn = row['model_name']
        tid = row['team_id']
        if mn not in results:
            results[mn] = {}
        results[mn][tid] = {
            'avg_win_prob': row['avg_win_prob'],
            'avg_run_diff': row['avg_run_diff'],
            'avg_projected_runs': row['avg_projected_runs'],
            'dominance_pct': row['dominance_pct'],
            'model_rank': row['model_rank'],
        }
    return results


def compute_composite(model_scores, team_ids):
    """
    Compute weighted composite score from individual model scores.
    Uses MODEL_WEIGHTS. Ensemble is excluded from composite (it IS a composite).
    """
    composite = {}
    model_names = [m for m in model_scores.keys() if m != 'ensemble']

    for tid in team_ids:
        weighted_sum = 0.0
        weight_sum = 0.0
        for mname in model_names:
            if tid not in model_scores.get(mname, {}):
                continue
            w = MODEL_WEIGHTS.get(mname, 0.05)
            if w <= 0:
                continue
            weighted_sum += model_scores[mname][tid]['avg_win_prob'] * w
            weight_sum += w
        composite[tid] = weighted_sum / weight_sum if weight_sum > 0 else 0.5

    return composite


def get_previous_rankings():
    """Get the most recent power rankings for movement tracking."""
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT team_id, rank FROM power_rankings
            WHERE date = (SELECT MAX(date) FROM power_rankings)
            ORDER BY rank
        ''')
        prev = {row['team_id']: row['rank'] for row in c.fetchall()}
    except Exception:
        prev = {}
    conn.close()
    return prev


def store_composite_rankings(composite, model_scores, teams, date_str):
    """Store the final composite rankings into power_rankings table."""
    prev = get_previous_rankings()
    team_info = {t['id']: t for t in teams}
    ranked_ids = sorted(composite.keys(), key=lambda tid: -composite[tid])

    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM power_rankings WHERE date = ?', (date_str,))

    for rank_idx, tid in enumerate(ranked_ids):
        rank = rank_idx + 1
        prev_rank = prev.get(tid)
        rank_change = (prev_rank - rank) if prev_rank else None

        # Use ensemble stats if available
        ens = model_scores.get('ensemble', {}).get(tid, {})
        run_diff = ens.get('avg_run_diff')
        proj_runs = ens.get('avg_projected_runs')
        dom_pct = ens.get('dominance_pct')

        c.execute('''
            INSERT INTO power_rankings
            (team_id, rank, power_score, avg_win_prob, avg_run_diff,
             avg_projected_runs, dominance_pct, prev_rank, rank_change, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tid, rank, composite[tid], composite[tid],
            run_diff, proj_runs, dom_pct,
            prev_rank, rank_change, date_str
        ))

    conn.commit()
    conn.close()
    print(f"✅ Stored {len(ranked_ids)} composite power rankings for {date_str}")


def print_rankings(model_scores, composite, teams, top_n=25, conference=None, single_model=None):
    """Pretty-print the rankings."""
    team_info = {t['id']: t for t in teams}
    prev = get_previous_rankings()

    if single_model:
        ms = model_scores.get(single_model)
        if not ms:
            print(f"Model '{single_model}' not found. Available: {list(model_scores.keys())}")
            return
        ranked_ids = sorted(ms.keys(), key=lambda t: -ms[t]['avg_win_prob'])
        if conference:
            ranked_ids = [t for t in ranked_ids
                          if team_info.get(t, {}).get('conference', '').upper() == conference.upper()]
        display_ids = ranked_ids[:top_n]

        title = f"Power Rankings — {single_model}"
        print(f"\n{'':=<75}")
        print(f"  {title}")
        print(f"{'':=<75}")
        header = f"{'Rank':<6}{'Team':<28}{'Conf':<8}{'Score':<8}{'RunDiff':<9}{'Dom%':<6}"
        print(header)
        print("-" * len(header))
        for tid in display_ids:
            overall_rank = ranked_ids.index(tid) + 1
            r = ms[tid]
            t = team_info.get(tid, {})
            print(f"#{overall_rank:<5}{t.get('name','?'):<28}{t.get('conference',''):<8}"
                  f"{r['avg_win_prob']:.3f}   {r['avg_run_diff']:+.2f}    "
                  f"{r['dominance_pct']*100:.0f}%")
        return

    # Multi-model composite view
    ranked_ids = sorted(composite.keys(), key=lambda tid: -composite[tid])
    if conference:
        ranked_ids = [t for t in ranked_ids
                      if team_info.get(t, {}).get('conference', '').upper() == conference.upper()]
    display_ids = ranked_ids[:top_n]

    # Model columns sorted by weight
    model_names = [m for m in model_scores.keys() if m != 'ensemble']
    model_names.sort(key=lambda m: -MODEL_WEIGHTS.get(m, 0))

    print(f"\n{'':=<120}")
    print(f"  Power Rankings — Composite (weighted blend of {len(model_names)} models)")
    print(f"{'':=<120}")

    cols = f"{'Rank':<6}{'Team':<26}{'Conf':<7}{'Score':<8}"
    for mn in model_names:
        cols += f"{ABBREV.get(mn, mn[:4]):<7}"
    cols += "  Mv"
    print(cols)
    print("-" * len(cols))

    for tid in display_ids:
        overall_rank = ranked_ids.index(tid) + 1
        t = team_info.get(tid, {})
        score = composite[tid]

        prev_rank = prev.get(tid)
        if prev_rank:
            change = prev_rank - overall_rank
            arrow = f"↑{change}" if change > 0 else f"↓{abs(change)}" if change < 0 else "–"
        else:
            arrow = "NEW"

        row = f"#{overall_rank:<5}{t.get('name','?'):<26}{t.get('conference',''):<7}{score:.3f}   "
        for mn in model_names:
            mr = model_scores.get(mn, {}).get(tid)
            if mr:
                row += f"{mr['avg_win_prob']:.3f}  "
            else:
                row += f"{'—':>5}  "
        row += f" {arrow}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Generate Multi-Model Power Rankings')
    parser.add_argument('--top', type=int, default=25, help='Show top N teams')
    parser.add_argument('--conference', type=str, help='Filter by conference')
    parser.add_argument('--min-games', type=int, default=3, help='Minimum games played')
    parser.add_argument('--store', action='store_true', help='Store results in database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show progress')
    parser.add_argument('--all', action='store_true', help='Show all teams')
    parser.add_argument('--model', type=str, help='Show rankings for a single model')
    args = parser.parse_args()

    date_str = datetime.now().strftime('%Y-%m-%d')

    print(f"🏟️  Generating Multi-Model Power Rankings...")
    print(f"   Min games: {args.min_games}")

    teams = get_eligible_teams(args.min_games)
    print(f"   Eligible teams: {len(teams)}")
    n = len(teams)
    total = n * (n - 1) // 2

    # All models to run (individual only — ensemble is redundant since
    # the composite score already blends individual model results)
    all_model_names = [m for m in WIN_PROB_MODELS]
    print(f"   Models: {', '.join(all_model_names)} ({len(all_model_names)} total)")
    print(f"   Matchups per model: {total:,}")
    print(f"   Total predictions: {total * len(all_model_names):,}")
    print()

    ensure_tables()

    # Process one model at a time to keep memory low
    completed_models = []
    for model_name in all_model_names:
        print(f"  📊 Running {model_name}...")
        model = load_single_model(model_name)
        if model is None:
            print(f"     Skipped (not available/trained)")
            continue

        model_results = run_round_robin_single_model(model, teams, verbose=args.verbose)
        errors = sum(r['errors'] for r in model_results.values())
        if errors:
            print(f"     ⚠️  {errors} prediction errors (used 0.5 fallback)")

        if args.store:
            store_model_detail(model_name, model_results, date_str)
            print(f"     ✅ Stored to DB")

        completed_models.append(model_name)

        # Free memory before next model
        del model
        del model_results
        gc.collect()

    print(f"\n  Completed {len(completed_models)}/{len(all_model_names)} models")

    # Now load scores back from DB (or from memory if not storing)
    if args.store:
        model_scores = load_all_model_scores(date_str)
    else:
        # Re-run without storing — need to keep in memory
        # For non-store mode, do a lighter pass
        print("  (Re-running for display — use --store to avoid this)")
        model_scores = {}
        for model_name in completed_models:
            model = load_single_model(model_name)
            if model is None:
                continue
            model_scores[model_name] = run_round_robin_single_model(model, teams, verbose=False)
            del model
            gc.collect()

    # Composite score
    team_ids = [t['id'] for t in teams]
    composite = compute_composite(model_scores, team_ids)

    if args.store:
        store_composite_rankings(composite, model_scores, teams, date_str)

    top_n = len(teams) if args.all else args.top
    print_rankings(model_scores, composite, teams, top_n=top_n,
                   conference=args.conference, single_model=args.model)

    return composite


if __name__ == '__main__':
    main()
