#!/usr/bin/env python3
"""
XGBoost vs LightGBM Training Data Comparison Study

Compares model performance when trained on:
1. Historical only (2024-2025): ~5,868 games
2. 2026 only: current season (~280 train, ~100 test)
3. Combined: historical + 2026 train

Uses the same holdout test set (last 100 games of 2026) for fair comparison.

Metrics:
- Moneyline: Accuracy, Log Loss
- Totals: MAE, RMSE
- Spread: MAE, Run Line Accuracy (within 1.5 runs)
"""

import sys
import json
import pickle
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: LightGBM not available")

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
DB_PATH = DATA_DIR / "baseball.db"

# Ensure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)


def get_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_team_name_to_id_map():
    """Build mapping from historical team names to team IDs."""
    conn = get_connection()
    c = conn.cursor()
    
    # Direct name -> id mapping from teams table
    c.execute("SELECT id, name, nickname FROM teams")
    mapping = {}
    for row in c.fetchall():
        team_id = row['id']
        name = row['name']
        nickname = row['nickname'] or ''
        
        # Map variations
        full_name = f"{name} {nickname}".strip()
        mapping[full_name.lower()] = team_id
        mapping[name.lower()] = team_id
        if nickname:
            mapping[f"{name.lower()} {nickname.lower()}"] = team_id
            
    # Also load from team_aliases
    c.execute("SELECT alias, team_id FROM team_aliases")
    for row in c.fetchall():
        mapping[row['alias'].lower()] = row['team_id']
    
    conn.close()
    return mapping


def normalize_team_name(name, mapping):
    """Try to map a team name to a team ID."""
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in mapping:
        return mapping[name_lower]
    
    # Try removing common suffixes
    for suffix in [' bulldogs', ' tigers', ' crimson tide', ' razorbacks', ' gators',
                   ' aggies', ' longhorns', ' wildcats', ' bears', ' cowboys',
                   ' falcons', ' zips', ' cougars', ' bearcats', ' eagles',
                   ' cardinals', ' blue devils', ' demon deacons', ' seminoles',
                   ' hurricanes', ' yellow jackets', ' cavaliers', ' hokies',
                   ' tar heels', ' wolfpack', ' gamecocks', ' volunteers',
                   ' commodores', ' rebels', ' fighting irish', ' golden gophers',
                   ' hawkeyes', ' spartans', ' wolverines', ' buckeyes',
                   ' nittany lions', ' hoosiers', ' badgers', ' fighting illini',
                   ' boilermakers', ' cornhuskers', ' jayhawks', ' cyclones',
                   ' sooners', ' horned frogs', ' red raiders', ' mountaineers',
                   ' sun devils', ' bruins', ' trojans', ' beavers', ' ducks',
                   ' huskies', ' cougars', ' buffaloes', ' utes', ' golden bears',
                   ' cardinal', ' gaels', ' toreros', ' waves', ' broncos',
                   ' lions', ' panthers', ' thundering herd', ' rainbow warriors',
                   ' red storm', ' musketeers', ' friars', ' hoyas', ' colonels',
                   ' pirates', ' owls', ' terrapins', ' midshipmen', ' black knights',
                   ' mocs', ' titans', ' redhawks', ' rockets', ' bobcats',
                   ' cardinals', ' red wolves', ' warhawks', ' rajun cajuns',
                   ' colonels', ' dukes', ' pride', ' jaspers', ' rams']:
        base = name_lower.replace(suffix, '').strip()
        if base in mapping:
            return mapping[base]
    
    # Try hyphenated version
    hyphenated = name_lower.replace(' ', '-')
    if hyphenated in mapping:
        return mapping[hyphenated]
    
    # Try just the first word(s) before the mascot
    parts = name_lower.split()
    if len(parts) >= 2:
        school_name = ' '.join(parts[:-1])  # Drop last word (mascot)
        if school_name in mapping:
            return mapping[school_name]
        hyphenated = school_name.replace(' ', '-')
        if hyphenated in mapping:
            return mapping[hyphenated]
    
    return None


def load_historical_games(team_mapping):
    """Load historical games (2024-2025) with team ID mapping."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT date, away_team, home_team, away_score, home_score,
               neutral_site, season
        FROM historical_games
        WHERE season IN (2024, 2025)
        AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date
    """)
    
    games = []
    unmapped = set()
    
    for row in c.fetchall():
        home_id = normalize_team_name(row['home_team'], team_mapping)
        away_id = normalize_team_name(row['away_team'], team_mapping)
        
        if home_id is None:
            unmapped.add(row['home_team'])
            continue
        if away_id is None:
            unmapped.add(row['away_team'])
            continue
        
        games.append({
            'date': row['date'],
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'neutral_site': bool(row['neutral_site']),
            'home_win': row['home_score'] > row['away_score'],
            'total_runs': row['home_score'] + row['away_score'],
            'margin': row['home_score'] - row['away_score'],
            'source': 'historical',
            'season': row['season'],
        })
    
    conn.close()
    
    if unmapped:
        print(f"Warning: Could not map {len(unmapped)} historical team names")
        if len(unmapped) <= 20:
            for name in sorted(unmapped):
                print(f"  - {name}")
    
    return games


def load_2026_games():
    """Load 2026 games from games table."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT id, date, home_team_id, away_team_id, home_score, away_score,
               is_neutral_site, is_conference_game
        FROM games
        WHERE status = 'final'
        AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date, id
    """)
    
    games = []
    for row in c.fetchall():
        games.append({
            'id': row['id'],
            'date': row['date'],
            'home_team_id': row['home_team_id'],
            'away_team_id': row['away_team_id'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'neutral_site': bool(row['is_neutral_site']),
            'is_conference': bool(row['is_conference_game']),
            'home_win': row['home_score'] > row['away_score'],
            'total_runs': row['home_score'] + row['away_score'],
            'margin': row['home_score'] - row['away_score'],
            'source': '2026',
            'season': 2026,
        })
    
    conn.close()
    return games


def compute_simple_features(game, team_stats):
    """
    Compute simplified features for training.
    Uses team aggregate stats rather than the full NN feature pipeline.
    This allows training on historical data without needing elo_ratings.
    """
    home_id = game['home_team_id']
    away_id = game['away_team_id']
    
    home_stats = team_stats.get(home_id, {})
    away_stats = team_stats.get(away_id, {})
    
    features = []
    
    # Home team features
    features.append(home_stats.get('win_pct', 0.5))
    features.append(home_stats.get('runs_per_game', 5.0))
    features.append(home_stats.get('runs_allowed_per_game', 5.0))
    features.append(home_stats.get('run_diff_per_game', 0.0))
    features.append(home_stats.get('pythagorean_pct', 0.5))
    features.append(home_stats.get('games', 0) / 60.0)  # Normalized by ~season length
    
    # Away team features
    features.append(away_stats.get('win_pct', 0.5))
    features.append(away_stats.get('runs_per_game', 5.0))
    features.append(away_stats.get('runs_allowed_per_game', 5.0))
    features.append(away_stats.get('run_diff_per_game', 0.0))
    features.append(away_stats.get('pythagorean_pct', 0.5))
    features.append(away_stats.get('games', 0) / 60.0)
    
    # Differentials (home - away)
    features.append(home_stats.get('win_pct', 0.5) - away_stats.get('win_pct', 0.5))
    features.append(home_stats.get('runs_per_game', 5.0) - away_stats.get('runs_per_game', 5.0))
    features.append(home_stats.get('run_diff_per_game', 0.0) - away_stats.get('run_diff_per_game', 0.0))
    
    # Home/away breakdown if available
    features.append(home_stats.get('home_win_pct', 0.5))
    features.append(away_stats.get('away_win_pct', 0.5))
    
    # Neutral site indicator
    features.append(1.0 if game.get('neutral_site') else 0.0)
    
    return np.array(features, dtype=np.float32)


def build_cumulative_stats(games):
    """
    Build cumulative team stats from games list.
    Returns a dict: team_id -> stats dict
    Processes games in chronological order.
    """
    team_stats = defaultdict(lambda: {
        'games': 0,
        'wins': 0,
        'losses': 0,
        'runs_scored': 0,
        'runs_allowed': 0,
        'home_games': 0,
        'home_wins': 0,
        'away_games': 0,
        'away_wins': 0,
    })
    
    # Sort by date
    sorted_games = sorted(games, key=lambda g: g['date'])
    
    # We'll compute stats dynamically as we process
    stats_snapshots = {}  # game_idx -> team_stats copy at that point
    
    for idx, game in enumerate(sorted_games):
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        
        # Snapshot stats BEFORE this game for feature computation
        stats_snapshots[idx] = {
            tid: dict(stats) for tid, stats in team_stats.items()
        }
        
        # Update home team
        h = team_stats[home_id]
        h['games'] += 1
        h['home_games'] += 1
        h['runs_scored'] += game['home_score']
        h['runs_allowed'] += game['away_score']
        if game['home_win']:
            h['wins'] += 1
            h['home_wins'] += 1
        else:
            h['losses'] += 1
        
        # Update away team
        a = team_stats[away_id]
        a['games'] += 1
        a['away_games'] += 1
        a['runs_scored'] += game['away_score']
        a['runs_allowed'] += game['home_score']
        if not game['home_win']:
            a['wins'] += 1
            a['away_wins'] += 1
        else:
            a['losses'] += 1
    
    return sorted_games, stats_snapshots


def enrich_stats(stats):
    """Add derived stats to a stats dict."""
    games = stats.get('games', 0)
    if games == 0:
        return {
            'games': 0,
            'win_pct': 0.5,
            'runs_per_game': 5.0,
            'runs_allowed_per_game': 5.0,
            'run_diff_per_game': 0.0,
            'pythagorean_pct': 0.5,
            'home_win_pct': 0.5,
            'away_win_pct': 0.5,
        }
    
    wins = stats.get('wins', 0)
    rs = stats.get('runs_scored', 0)
    ra = stats.get('runs_allowed', 0)
    home_games = stats.get('home_games', 0)
    home_wins = stats.get('home_wins', 0)
    away_games = stats.get('away_games', 0)
    away_wins = stats.get('away_wins', 0)
    
    # Pythagorean expectation
    rs2 = rs ** 2
    ra2 = ra ** 2
    pythag = rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5
    
    return {
        'games': games,
        'win_pct': wins / games,
        'runs_per_game': rs / games,
        'runs_allowed_per_game': ra / games,
        'run_diff_per_game': (rs - ra) / games,
        'pythagorean_pct': pythag,
        'home_win_pct': home_wins / home_games if home_games > 0 else 0.5,
        'away_win_pct': away_wins / away_games if away_games > 0 else 0.5,
    }


def prepare_dataset(games, min_games=5):
    """
    Prepare feature matrix and labels from games.
    Uses cumulative stats up to each game.
    Returns X, y_moneyline, y_totals, y_spread
    """
    sorted_games, stats_snapshots = build_cumulative_stats(games)
    
    X = []
    y_ml = []  # 1 = home win, 0 = away win
    y_totals = []  # total runs
    y_spread = []  # home margin
    
    for idx, game in enumerate(sorted_games):
        # Get stats snapshot before this game
        snapshot = stats_snapshots[idx]
        
        # Enrich stats for both teams
        home_stats = enrich_stats(snapshot.get(game['home_team_id'], {}))
        away_stats = enrich_stats(snapshot.get(game['away_team_id'], {}))
        
        # Skip if either team has too few games
        if home_stats['games'] < min_games or away_stats['games'] < min_games:
            continue
        
        # Build temporary stats dict for feature computation
        temp_stats = {
            game['home_team_id']: home_stats,
            game['away_team_id']: away_stats,
        }
        
        features = compute_simple_features(game, temp_stats)
        X.append(features)
        y_ml.append(1 if game['home_win'] else 0)
        y_totals.append(game['total_runs'])
        y_spread.append(game['margin'])
    
    return (
        np.array(X, dtype=np.float32),
        np.array(y_ml, dtype=np.int32),
        np.array(y_totals, dtype=np.float32),
        np.array(y_spread, dtype=np.float32),
    )


def train_xgb_classifier(X_train, y_train, X_val, y_val, use_gpu=True):
    """Train XGBoost classifier with early stopping."""
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
    }
    
    if use_gpu:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_xgb_regressor(X_train, y_train, X_val, y_val, use_gpu=True):
    """Train XGBoost regressor with early stopping."""
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    if use_gpu:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_classifier(X_train, y_train, X_val, y_val, use_gpu=True):
    """Train LightGBM classifier with early stopping."""
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_use_dp'] = False
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
    ]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


def train_lgb_regressor(X_train, y_train, X_val, y_val, use_gpu=True):
    """Train LightGBM regressor with early stopping."""
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_use_dp'] = False
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
    ]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


def evaluate_classifier(model, X_test, y_test):
    """Evaluate classifier and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    
    return {
        'accuracy': round(float(acc) * 100, 2),
        'log_loss': round(float(ll), 4),
    }


def evaluate_regressor(model, X_test, y_test, task='totals'):
    """Evaluate regressor and return metrics."""
    y_pred = model.predict(X_test)
    errors = y_pred - y_test
    
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    result = {
        'mae': round(float(mae), 3),
        'rmse': round(float(rmse), 3),
    }
    
    if task == 'spread':
        # Run line accuracy: % within 1.5 runs
        within_1_5 = np.abs(errors) <= 1.5
        result['run_line_acc'] = round(float(within_1_5.mean()) * 100, 2)
    
    return result


def run_comparison():
    """Run the full comparison study."""
    print("=" * 60)
    print("XGBoost vs LightGBM Training Data Comparison Study")
    print("=" * 60)
    print()
    
    # Check GPU availability
    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            use_gpu = False
    except ImportError:
        print("Warning: PyTorch not available for GPU check, attempting GPU anyway")
    
    # Load data
    print("Loading data...")
    team_mapping = build_team_name_to_id_map()
    print(f"  Team mapping entries: {len(team_mapping)}")
    
    historical_games = load_historical_games(team_mapping)
    print(f"  Historical games (2024-2025): {len(historical_games)}")
    
    games_2026 = load_2026_games()
    print(f"  2026 games: {len(games_2026)}")
    
    # Split 2026 data into train and test
    # Last ~100 games for test
    test_size = min(100, len(games_2026) // 3)
    games_2026_train = games_2026[:-test_size]
    games_2026_test = games_2026[-test_size:]
    
    print(f"  2026 train: {len(games_2026_train)}, 2026 test (holdout): {len(games_2026_test)}")
    print()
    
    # Prepare datasets
    print("Preparing datasets...")
    
    # Historical only
    X_hist, y_ml_hist, y_tot_hist, y_spr_hist = prepare_dataset(historical_games, min_games=3)
    print(f"  Historical dataset: {len(X_hist)} samples")
    
    # 2026 train only
    X_2026, y_ml_2026, y_tot_2026, y_spr_2026 = prepare_dataset(games_2026_train, min_games=2)
    print(f"  2026 train dataset: {len(X_2026)} samples")
    
    # Combined
    combined_games = historical_games + games_2026_train
    X_comb, y_ml_comb, y_tot_comb, y_spr_comb = prepare_dataset(combined_games, min_games=3)
    print(f"  Combined dataset: {len(X_comb)} samples")
    
    # Test set from 2026 holdout
    # For test, we use cumulative stats from all prior games (including historical)
    all_prior = historical_games + games_2026_train
    X_test, y_ml_test, y_tot_test, y_spr_test = prepare_dataset(
        all_prior + games_2026_test, min_games=2
    )
    # Extract just the test portion
    test_offset = len(X_test) - len(games_2026_test)
    # Actually, we need a cleaner approach - let's prepare test separately
    
    # Rebuild test features using combined training data as prior
    sorted_prior, prior_stats = build_cumulative_stats(all_prior)
    # Get final stats after all prior games
    final_stats = {}
    for tid, s in prior_stats.get(len(sorted_prior) - 1, {}).items():
        final_stats[tid] = enrich_stats(s)
    # Also add any teams that played in the last game
    if sorted_prior:
        last_game = sorted_prior[-1]
        for tid in [last_game['home_team_id'], last_game['away_team_id']]:
            if tid not in final_stats:
                final_stats[tid] = enrich_stats({})
    
    # Now compute test features
    X_test_list = []
    y_ml_test_list = []
    y_tot_test_list = []
    y_spr_test_list = []
    
    for game in games_2026_test:
        home_stats = final_stats.get(game['home_team_id'], enrich_stats({}))
        away_stats = final_stats.get(game['away_team_id'], enrich_stats({}))
        
        temp_stats = {
            game['home_team_id']: home_stats,
            game['away_team_id']: away_stats,
        }
        
        features = compute_simple_features(game, temp_stats)
        X_test_list.append(features)
        y_ml_test_list.append(1 if game['home_win'] else 0)
        y_tot_test_list.append(game['total_runs'])
        y_spr_test_list.append(game['margin'])
    
    X_test = np.array(X_test_list, dtype=np.float32)
    y_ml_test = np.array(y_ml_test_list, dtype=np.int32)
    y_tot_test = np.array(y_tot_test_list, dtype=np.float32)
    y_spr_test = np.array(y_spr_test_list, dtype=np.float32)
    
    print(f"  Test set: {len(X_test)} samples")
    print()
    
    # Split training data for validation
    def split_with_val(X, y, val_ratio=0.15):
        return train_test_split(X, y, test_size=val_ratio, random_state=42)
    
    # Results storage
    results = {
        'moneyline': {},
        'totals': {},
        'spread': {},
    }
    
    training_sets = {
        'historical': (X_hist, y_ml_hist, y_tot_hist, y_spr_hist),
        '2026_only': (X_2026, y_ml_2026, y_tot_2026, y_spr_2026),
        'combined': (X_comb, y_ml_comb, y_tot_comb, y_spr_comb),
    }
    
    # Train and evaluate each combination
    print("Training models...")
    print("-" * 60)
    
    for train_name, (X_tr, y_ml_tr, y_tot_tr, y_spr_tr) in training_sets.items():
        print(f"\n[{train_name.upper()}] Training set: {len(X_tr)} samples")
        
        if len(X_tr) < 50:
            print(f"  Skipping - too few samples")
            continue
        
        # Split for validation
        X_train, X_val, y_ml_train, y_ml_val = split_with_val(X_tr, y_ml_tr)
        _, _, y_tot_train, y_tot_val = split_with_val(X_tr, y_tot_tr)
        _, _, y_spr_train, y_spr_val = split_with_val(X_tr, y_spr_tr)
        
        # ---- XGBoost ----
        if XGB_AVAILABLE:
            print(f"  Training XGBoost...")
            
            # Moneyline
            model = train_xgb_classifier(X_train, y_ml_train, X_val, y_ml_val, use_gpu)
            metrics = evaluate_classifier(model, X_test, y_ml_test)
            results['moneyline'][f'xgb_{train_name}'] = metrics
            print(f"    Moneyline: Acc={metrics['accuracy']:.1f}%, LogLoss={metrics['log_loss']:.4f}")
            
            # Totals
            model = train_xgb_regressor(X_train, y_tot_train, X_val, y_tot_val, use_gpu)
            metrics = evaluate_regressor(model, X_test, y_tot_test, 'totals')
            results['totals'][f'xgb_{train_name}'] = metrics
            print(f"    Totals: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
            
            # Spread
            model = train_xgb_regressor(X_train, y_spr_train, X_val, y_spr_val, use_gpu)
            metrics = evaluate_regressor(model, X_test, y_spr_test, 'spread')
            results['spread'][f'xgb_{train_name}'] = metrics
            print(f"    Spread: MAE={metrics['mae']:.3f}, RunLineAcc={metrics['run_line_acc']:.1f}%")
        
        # ---- LightGBM ----
        if LGB_AVAILABLE:
            print(f"  Training LightGBM...")
            
            # Moneyline
            model = train_lgb_classifier(X_train, y_ml_train, X_val, y_ml_val, use_gpu)
            metrics = evaluate_classifier(model, X_test, y_ml_test)
            results['moneyline'][f'lgb_{train_name}'] = metrics
            print(f"    Moneyline: Acc={metrics['accuracy']:.1f}%, LogLoss={metrics['log_loss']:.4f}")
            
            # Totals
            model = train_lgb_regressor(X_train, y_tot_train, X_val, y_tot_val, use_gpu)
            metrics = evaluate_regressor(model, X_test, y_tot_test, 'totals')
            results['totals'][f'lgb_{train_name}'] = metrics
            print(f"    Totals: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
            
            # Spread
            model = train_lgb_regressor(X_train, y_spr_train, X_val, y_spr_val, use_gpu)
            metrics = evaluate_regressor(model, X_test, y_spr_test, 'spread')
            results['spread'][f'lgb_{train_name}'] = metrics
            print(f"    Spread: MAE={metrics['mae']:.3f}, RunLineAcc={metrics['run_line_acc']:.1f}%")
    
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Generate markdown report
    generate_report(results, len(X_test), {
        'historical': len(X_hist),
        '2026_only': len(X_2026),
        'combined': len(X_comb),
    })
    
    return results


def generate_report(results, test_size, train_sizes):
    """Generate markdown report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    report = f"""# Gradient Boosting Training Data Comparison

**Generated:** {timestamp}

## Study Design

Compared XGBoost and LightGBM model performance when trained on different datasets:

| Training Set | Samples | Description |
|--------------|---------|-------------|
| Historical | {train_sizes.get('historical', 'N/A')} | 2024-2025 season games |
| 2026 Only | {train_sizes.get('2026_only', 'N/A')} | Current season (excluding test) |
| Combined | {train_sizes.get('combined', 'N/A')} | Historical + 2026 train |

**Test Set:** {test_size} games (last ~100 games of 2026 season)

---

## Moneyline Results

| Model | Training Data | Accuracy | Log Loss |
|-------|---------------|----------|----------|
"""
    
    # Sort by accuracy
    ml_results = sorted(
        results['moneyline'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    for name, metrics in ml_results:
        model_type = 'XGBoost' if name.startswith('xgb') else 'LightGBM'
        train_type = name.split('_', 1)[1].replace('_', ' ').title()
        report += f"| {model_type} | {train_type} | {metrics['accuracy']:.1f}% | {metrics['log_loss']:.4f} |\n"
    
    report += """
---

## Totals Results

| Model | Training Data | MAE | RMSE |
|-------|---------------|-----|------|
"""
    
    # Sort by MAE (lower is better)
    tot_results = sorted(
        results['totals'].items(),
        key=lambda x: x[1]['mae']
    )
    
    for name, metrics in tot_results:
        model_type = 'XGBoost' if name.startswith('xgb') else 'LightGBM'
        train_type = name.split('_', 1)[1].replace('_', ' ').title()
        report += f"| {model_type} | {train_type} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} |\n"
    
    report += """
---

## Spread Results

| Model | Training Data | MAE | Run Line Acc |
|-------|---------------|-----|--------------|
"""
    
    # Sort by MAE
    spr_results = sorted(
        results['spread'].items(),
        key=lambda x: x[1]['mae']
    )
    
    for name, metrics in spr_results:
        model_type = 'XGBoost' if name.startswith('xgb') else 'LightGBM'
        train_type = name.split('_', 1)[1].replace('_', ' ').title()
        report += f"| {model_type} | {train_type} | {metrics['mae']:.3f} | {metrics['run_line_acc']:.1f}% |\n"
    
    # Analysis section
    report += """
---

## Analysis

### Best Performers by Task

"""
    
    # Find best for each task
    if ml_results:
        best_ml = ml_results[0]
        model_type = 'XGBoost' if best_ml[0].startswith('xgb') else 'LightGBM'
        train_type = best_ml[0].split('_', 1)[1].replace('_', ' ').title()
        report += f"**Moneyline:** {model_type} trained on {train_type} ({best_ml[1]['accuracy']:.1f}% accuracy)\n\n"
    
    if tot_results:
        best_tot = tot_results[0]
        model_type = 'XGBoost' if best_tot[0].startswith('xgb') else 'LightGBM'
        train_type = best_tot[0].split('_', 1)[1].replace('_', ' ').title()
        report += f"**Totals:** {model_type} trained on {train_type} ({best_tot[1]['mae']:.3f} MAE)\n\n"
    
    if spr_results:
        best_spr = spr_results[0]
        model_type = 'XGBoost' if best_spr[0].startswith('xgb') else 'LightGBM'
        train_type = best_spr[0].split('_', 1)[1].replace('_', ' ').title()
        report += f"**Spread:** {model_type} trained on {train_type} ({best_spr[1]['mae']:.3f} MAE, {best_spr[1]['run_line_acc']:.1f}% run line)\n\n"
    
    report += """### Key Findings

"""
    
    # Compare XGBoost vs LightGBM
    xgb_wins = 0
    lgb_wins = 0
    
    for task_results in [ml_results, tot_results, spr_results]:
        if task_results:
            winner = task_results[0][0]
            if winner.startswith('xgb'):
                xgb_wins += 1
            else:
                lgb_wins += 1
    
    if xgb_wins > lgb_wins:
        report += f"1. **XGBoost** slightly outperforms LightGBM ({xgb_wins}/{xgb_wins+lgb_wins} tasks)\n"
    elif lgb_wins > xgb_wins:
        report += f"1. **LightGBM** slightly outperforms XGBoost ({lgb_wins}/{xgb_wins+lgb_wins} tasks)\n"
    else:
        report += "1. **XGBoost and LightGBM** perform similarly across tasks\n"
    
    # Compare training data
    hist_best = 0
    curr_best = 0
    comb_best = 0
    
    for task_results in [ml_results, tot_results, spr_results]:
        if task_results:
            winner = task_results[0][0]
            if 'historical' in winner:
                hist_best += 1
            elif '2026' in winner:
                curr_best += 1
            elif 'combined' in winner:
                comb_best += 1
    
    report += "\n2. **Training Data Impact:**\n"
    if comb_best >= hist_best and comb_best >= curr_best:
        report += "   - Combined historical + current season data works best\n"
        report += "   - More data generally helps gradient boosting models\n"
    elif hist_best > curr_best:
        report += "   - Historical data alone performs surprisingly well\n"
        report += "   - Suggests patterns from past seasons transfer to 2026\n"
    else:
        report += "   - Current season data is most predictive for recent games\n"
        report += "   - May indicate team dynamics differ from historical patterns\n"
    
    report += """
### Recommendations

"""
    
    if comb_best >= 1:
        report += """1. **Use combined training data** for production models
   - Historical games provide volume for stable feature learning
   - Current season games capture recent team performance
   
2. **Retrain periodically** as more 2026 games are played
   - Weekly retraining recommended during active season
   
3. **Consider ensemble** of XGBoost + LightGBM for robustness
"""
    else:
        report += """1. **Prioritize recent data** if historical performance is poor
   - Team compositions change significantly year-to-year
   
2. **Expand 2026 dataset** as season progresses
   - Current models may improve with more in-season data
   
3. **Investigate feature engineering** to better capture temporal patterns
"""
    
    report += """
---

*Report generated by `scripts/compare_gb_training.py`*
"""
    
    # Save report
    report_path = REPORTS_DIR / "gradient_boosting_training_comparison.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    
    # Also save JSON results
    json_path = DATA_DIR / "gb_training_comparison.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_size': test_size,
            'train_sizes': train_sizes,
            'results': results,
        }, f, indent=2)
    
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    run_comparison()
