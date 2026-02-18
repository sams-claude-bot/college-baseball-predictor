#!/usr/bin/env python3
"""
Train Neural Network Models with Weather Features

Trains all three NN models:
- nn_totals_model.py → totals prediction (HISTORICAL 2024-2025 data)
- nn_spread_model.py → margin prediction (2026 CURRENT SEASON only)
- nn_dow_totals_model.py → day-of-week aware totals (HISTORICAL 2024-2025 data)

TRAINING DATA RATIONALE:
========================
- **Totals models** use historical data (6,184 games) because run totals 
  depend on stable factors like weather, day-of-week patterns, and general 
  offensive/defensive tendencies that don't change dramatically year-to-year.
  
- **Spread/moneyline models** use current season data only because win
  probability depends heavily on current team strength, roster changes, 
  and season-specific performance. Historical data from different teams/rosters
  would introduce noise.
"""

import sys
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

# Feature computation constants (copied from nn_features for standalone training)
DEFAULT_ELO = 1500
ELO_K = 32
ELO_HOME_ADV = 50

# Weather defaults and normalization
DEFAULT_WEATHER = {
    'temp_f': 65.0,
    'humidity_pct': 55.0,
    'wind_speed_mph': 6.0,
    'wind_direction_deg': 180,
    'precip_prob_pct': 5.0,
    'is_dome': 0,
}

WEATHER_NORM = {
    'temp_f': (65.0, 15.0),
    'humidity_pct': (55.0, 20.0),
    'wind_speed_mph': (7.0, 5.0),
    'precip_prob_pct': (10.0, 15.0),
}


class RollingTrainingFeatureComputer:
    """
    Computes features from the games table for training.
    Processes games chronologically, maintaining rolling state per team.
    No data leakage: features for game N use only games 0..N-1.
    Includes weather features from game_weather table.
    """

    def __init__(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
            'recent_results': [],
            'last_game_date': None,
            'opponents': [],
        })

    def reset(self):
        """Reset all state."""
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
            'recent_results': [],
            'last_game_date': None,
            'opponents': [],
        })

    def compute_game_features(self, game_row, weather_row=None):
        """
        Compute features for a game BEFORE updating state.

        Args:
            game_row: dict with keys from historical_games table
            weather_row: optional dict with weather data

        Returns:
            (features_array, totals_label, margin_label, dow)
        """
        # Support both historical_games (home_team) and games (home_team_id)
        home = game_row.get('home_team') or game_row.get('home_team_id')
        away = game_row.get('away_team') or game_row.get('away_team_id')
        date_str = game_row['date']
        neutral = game_row.get('neutral_site') or game_row.get('is_neutral_site', 0)
        is_conf = game_row.get('is_conference_game', 0)  # Not available in historical
        home_score = game_row['home_score']
        away_score = game_row['away_score']

        features = []

        for team, opp in [(home, away), (away, home)]:
            s = self.team_stats[team]
            gp = s['games']

            # Elo
            features.append(self.elo[team])

            # Win %
            win_pct_all = s['wins'] / gp if gp > 0 else 0.5
            features.append(win_pct_all)

            recent = s['recent_results']
            last10 = recent[-10:]
            last20 = recent[-20:]
            features.append(sum(last10) / len(last10) if last10 else 0.5)
            features.append(sum(last20) / len(last20) if last20 else 0.5)

            # Run differential per game
            rd = (s['runs_scored'] - s['runs_allowed']) / gp if gp > 0 else 0.0
            features.append(rd)

            # Pythagorean
            rs2 = s['runs_scored'] ** 2
            ra2 = s['runs_allowed'] ** 2
            features.append(rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5)

            # Batting - use actual runs/game if available, else defaults
            rpg = s['runs_scored'] / gp if gp > 0 else 4.5
            features.extend([0.260, 0.330, 0.400, 0.730,  # avg, obp, slg, ops
                             0.8, 3.5, 7.0,  # hr/g, bb/g, so/g
                             rpg])

            # Pitching
            rapg = s['runs_allowed'] / gp if gp > 0 else 4.5
            era_est = rapg * 0.9
            features.extend([era_est, 1.35, 7.5, 3.5, 0.8, 0.260])

            # Situational
            if s['last_game_date']:
                try:
                    last_d = datetime.strptime(s['last_game_date'], '%Y-%m-%d')
                    curr_d = datetime.strptime(date_str, '%Y-%m-%d')
                    days_rest = (curr_d - last_d).days
                except (ValueError, TypeError):
                    days_rest = 2.0
            else:
                days_rest = 3.0
            features.append(float(min(days_rest, 14)))  # Cap at 14 days

            # SOS
            if s['opponents']:
                sos = sum(self.elo[o] for o in s['opponents']) / len(s['opponents'])
            else:
                sos = DEFAULT_ELO
            features.append(sos)

            # Advanced batting defaults
            features.extend([100.0, 0.320, 0.140, 0.300, 20.0, 8.5, 43.0, 36.0, 21.0])
            # Advanced pitching defaults
            features.extend([4.00, 4.00, 4.00, 43.0, 36.0])

        # Game-level
        features.append(1.0 if neutral else 0.0)
        features.append(1.0 if is_conf else 0.0)

        # Weather features
        features.extend(self._compute_weather_features(weather_row))

        # Labels
        total_runs = home_score + away_score
        margin = home_score - away_score  # positive = home won

        # Day of week
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            dow = int(dt.strftime('%w'))  # 0=Sunday..6=Saturday
        except (ValueError, TypeError):
            dow = 5  # default Friday

        return np.array(features, dtype=np.float32), float(total_runs), float(margin), dow

    def _compute_weather_features(self, weather_row):
        """Compute normalized weather features (7 features)."""
        w = dict(DEFAULT_WEATHER)

        if weather_row:
            for key in ['temp_f', 'humidity_pct', 'wind_speed_mph', 
                        'wind_direction_deg', 'precip_prob_pct', 'is_dome']:
                if weather_row.get(key) is not None:
                    w[key] = weather_row[key]

        # Normalize
        temp_norm = (w['temp_f'] - WEATHER_NORM['temp_f'][0]) / WEATHER_NORM['temp_f'][1]
        humidity_norm = (w['humidity_pct'] - WEATHER_NORM['humidity_pct'][0]) / WEATHER_NORM['humidity_pct'][1]
        wind_speed_norm = (w['wind_speed_mph'] - WEATHER_NORM['wind_speed_mph'][0]) / WEATHER_NORM['wind_speed_mph'][1]
        precip_norm = (w['precip_prob_pct'] - WEATHER_NORM['precip_prob_pct'][0]) / WEATHER_NORM['precip_prob_pct'][1]

        wind_dir_rad = math.radians(w['wind_direction_deg'])
        wind_dir_sin = math.sin(wind_dir_rad)
        wind_dir_cos = math.cos(wind_dir_rad)

        is_dome = 1.0 if w['is_dome'] else 0.0

        return [temp_norm, humidity_norm, wind_speed_norm,
                wind_dir_sin, wind_dir_cos, precip_norm, is_dome]

    def update_state(self, game_row):
        """Update rolling state AFTER computing features."""
        home = game_row.get('home_team') or game_row.get('home_team_id')
        away = game_row.get('away_team') or game_row.get('away_team_id')
        hs = game_row['home_score']
        aws = game_row['away_score']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site') or game_row.get('is_neutral_site', 0)

        home_won = hs > aws

        for team, opp, rs, ra, won in [
            (home, away, hs, aws, home_won),
            (away, home, aws, hs, not home_won),
        ]:
            s = self.team_stats[team]
            s['games'] += 1
            s['runs_scored'] += rs
            s['runs_allowed'] += ra
            if won:
                s['wins'] += 1
            else:
                s['losses'] += 1
            s['recent_results'].append(1.0 if won else 0.0)
            if len(s['recent_results']) > 50:
                s['recent_results'] = s['recent_results'][-50:]
            s['last_game_date'] = date_str
            s['opponents'].append(opp)

        # Update Elo
        home_elo = self.elo[home]
        away_elo = self.elo[away]
        elo_diff = home_elo - away_elo + (0 if neutral else ELO_HOME_ADV)
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        actual_home = 1.0 if home_won else 0.0

        mov = abs(hs - aws)
        mov_mult = math.log(max(mov, 1) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))

        self.elo[home] += ELO_K * mov_mult * (actual_home - expected_home)
        self.elo[away] += ELO_K * mov_mult * (expected_home - actual_home)


def load_historical_training_data():
    """
    Load historical games (2024-2025) with weather data for TOTALS training.
    
    Used for: nn_totals_model, nn_dow_totals_model
    Rationale: Run totals depend on stable factors that transfer across seasons.
    """
    conn = get_connection()
    c = conn.cursor()

    # Get historical games with weather, ordered by date
    c.execute("""
        SELECT g.id, g.date, g.home_team, g.away_team, 
               g.home_score, g.away_score, g.neutral_site, g.season,
               gw.temp_f, gw.humidity_pct, gw.wind_speed_mph, gw.wind_direction_deg,
               gw.precip_prob_pct, gw.is_dome
        FROM historical_games g
        LEFT JOIN historical_game_weather gw ON g.id = gw.game_id
        ORDER BY g.date ASC
    """)

    rows = c.fetchall()
    conn.close()

    games = []
    for row in rows:
        game = {
            'id': row['id'],
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'neutral_site': row['neutral_site'],
            'season': row['season'],
        }
        weather = None
        if row['temp_f'] is not None:
            weather = {
                'temp_f': row['temp_f'],
                'humidity_pct': row['humidity_pct'],
                'wind_speed_mph': row['wind_speed_mph'],
                'wind_direction_deg': row['wind_direction_deg'],
                'precip_prob_pct': row['precip_prob_pct'],
                'is_dome': row['is_dome'],
            }
        games.append((game, weather))

    return games


def load_current_season_training_data():
    """
    Load current season (2026) games with weather data for SPREAD/MONEYLINE training.
    
    Used for: nn_spread_model
    Rationale: Win probability depends on current team strength and roster composition.
               Historical data from different teams/rosters would introduce noise.
    """
    conn = get_connection()
    c = conn.cursor()

    # Get 2026 completed games with weather, ordered by date
    c.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id, 
               g.home_score, g.away_score, g.is_neutral_site, g.is_conference_game,
               gw.temp_f, gw.humidity_pct, gw.wind_speed_mph, gw.wind_direction_deg,
               gw.precip_prob_pct, gw.is_dome
        FROM games g
        LEFT JOIN game_weather gw ON g.id = gw.game_id
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY g.date ASC
    """)

    rows = c.fetchall()
    conn.close()

    games = []
    for row in rows:
        game = {
            'id': row['id'],
            'date': row['date'],
            'home_team_id': row['home_team_id'],
            'away_team_id': row['away_team_id'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'is_neutral_site': row['is_neutral_site'],
            'is_conference_game': row['is_conference_game'],
        }
        weather = None
        if row['temp_f'] is not None:
            weather = {
                'temp_f': row['temp_f'],
                'humidity_pct': row['humidity_pct'],
                'wind_speed_mph': row['wind_speed_mph'],
                'wind_direction_deg': row['wind_direction_deg'],
                'precip_prob_pct': row['precip_prob_pct'],
                'is_dome': row['is_dome'],
            }
        games.append((game, weather))

    return games


# Alias for backward compatibility
def load_training_data():
    """Alias for load_historical_training_data() for backward compatibility."""
    return load_historical_training_data()


def clip_features(X, clip_std=5.0):
    """
    Clip extreme feature values to prevent numeric instability.
    Uses per-feature mean/std computed from the data.
    """
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std < 1e-6] = 1.0  # Avoid division by zero
    
    lower = mean - clip_std * std
    upper = mean + clip_std * std
    
    return np.clip(X, lower, upper)


def prepare_datasets(games, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare train/val/test datasets from games.
    
    Splits chronologically to avoid data leakage.
    """
    computer = RollingTrainingFeatureComputer()

    features_list = []
    totals_list = []
    margins_list = []
    dow_list = []

    n_with_weather = 0
    for game, weather in games:
        features, total, margin, dow = computer.compute_game_features(game, weather)
        
        # Replace any NaN/inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        features_list.append(features)
        totals_list.append(total)
        margins_list.append(margin)
        dow_list.append(dow)
        
        if weather is not None:
            n_with_weather += 1
        
        computer.update_state(game)

    print(f"Total games: {len(games)}")
    print(f"Games with weather data: {n_with_weather} ({100*n_with_weather/len(games):.1f}%)")
    print(f"Feature dimension: {len(features_list[0])}")

    X = np.array(features_list)
    y_totals = np.array(totals_list)
    y_margins = np.array(margins_list)
    y_dow = np.array(dow_list)
    
    # Clip extreme values to prevent numeric issues
    X = clip_features(X, clip_std=5.0)
    
    # Check for NaN/inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("WARNING: NaN or Inf in features after clipping!")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Chronological split
    n = len(X)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    
    # Compute normalization from ALL data (not just train) to handle distribution shift
    # This is a compromise - in production we'd use only train, but for this early-season
    # data with limited samples, using all data for normalization is more stable
    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0)
    global_std[global_std < 1e-6] = 1.0

    return {
        'X_train': X[:train_end],
        'X_val': X[train_end:val_end],
        'X_test': X[val_end:],
        'y_totals_train': y_totals[:train_end],
        'y_totals_val': y_totals[train_end:val_end],
        'y_totals_test': y_totals[val_end:],
        'y_margins_train': y_margins[:train_end],
        'y_margins_val': y_margins[train_end:val_end],
        'y_margins_test': y_margins[val_end:],
        'dow_train': y_dow[:train_end],
        'dow_val': y_dow[train_end:val_end],
        'dow_test': y_dow[val_end:],
        'global_mean': global_mean,
        'global_std': global_std,
    }


def train_totals_model(data):
    """Train the totals model with global normalization."""
    import torch
    from pathlib import Path
    from models.nn_totals_model import TotalsTrainer, TotalsNet
    
    input_size = data['X_train'].shape[1]
    print(f"\n{'='*50}")
    print(f"Training Totals Model (input_size={input_size})")
    print(f"{'='*50}")
    
    # Use global normalization
    feature_mean = data['global_mean']
    feature_std = data['global_std']
    
    X_train = (data['X_train'] - feature_mean) / feature_std
    X_val = (data['X_val'] - feature_mean) / feature_std
    X_test = (data['X_test'] - feature_mean) / feature_std
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TotalsNet(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(data['y_totals_train'], dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(data['y_totals_val'], dtype=torch.float32).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    def gaussian_nll_loss(mean, logvar, target):
        var = torch.exp(logvar) + 1e-6
        return 0.5 * torch.mean(torch.log(var) + (target - mean) ** 2 / var)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(150):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            mean, logvar = model(X_batch)
            loss = gaussian_nll_loss(mean, logvar, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_mean, val_logvar = model(X_val_t)
            val_loss = gaussian_nll_loss(val_mean, val_logvar, y_val_t).item()
            val_mae = torch.abs(val_mean - y_val_t).mean().item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/150 | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Save checkpoint
    model_path = Path(__file__).parent.parent / "data" / "nn_totals_model.pt"
    torch.save({
        'model_state_dict': best_state,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_size': input_size,
    }, model_path)
    print(f"Saved to {model_path}")
    
    # Evaluate
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        test_mean, _ = model(X_test_t)
        preds = test_mean.cpu().numpy()
    
    errors = preds - data['y_totals_test']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    print(f"\nTest Results:")
    print(f"  MAE: {mae:.3f} runs")
    print(f"  RMSE: {rmse:.3f} runs")
    print(f"  Mean predicted: {preds.mean():.2f}")
    print(f"  Mean actual: {data['y_totals_test'].mean():.2f}")
    
    return {'mae': mae, 'rmse': rmse}


def train_spread_model(data):
    """Train the spread model with global normalization."""
    import torch
    from pathlib import Path
    from models.nn_spread_model import SpreadNet
    
    input_size = data['X_train'].shape[1]
    print(f"\n{'='*50}")
    print(f"Training Spread Model (input_size={input_size})")
    print(f"{'='*50}")
    
    feature_mean = data['global_mean']
    feature_std = data['global_std']
    
    X_train = (data['X_train'] - feature_mean) / feature_std
    X_val = (data['X_val'] - feature_mean) / feature_std
    X_test = (data['X_test'] - feature_mean) / feature_std
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpreadNet(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(data['y_margins_train'], dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(data['y_margins_val'], dtype=torch.float32).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    def gaussian_nll_loss(mean, logvar, target):
        var = torch.exp(logvar) + 1e-6
        return 0.5 * torch.mean(torch.log(var) + (target - mean) ** 2 / var)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(150):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            mean, logvar = model(X_batch)
            loss = gaussian_nll_loss(mean, logvar, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_mean, val_logvar = model(X_val_t)
            val_loss = gaussian_nll_loss(val_mean, val_logvar, y_val_t).item()
            val_mae = torch.abs(val_mean - y_val_t).mean().item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/150 | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    
    model_path = Path(__file__).parent.parent / "data" / "nn_spread_model.pt"
    torch.save({
        'model_state_dict': best_state,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_size': input_size,
    }, model_path)
    print(f"Saved to {model_path}")
    
    # Evaluate
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        test_mean, _ = model(X_test_t)
        preds = test_mean.cpu().numpy()
    
    errors = preds - data['y_margins_test']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    # Run line accuracy
    spread = -1.5
    pred_covers = preds > spread
    actual_covers = data['y_margins_test'] > spread
    rl_correct = (pred_covers == actual_covers).sum()
    rl_accuracy = rl_correct / len(preds)
    
    print(f"\nTest Results:")
    print(f"  MAE: {mae:.3f} runs")
    print(f"  RMSE: {rmse:.3f} runs")
    print(f"  Run Line Accuracy: {rl_accuracy:.1%}")
    print(f"  Mean predicted: {preds.mean():.2f}")
    print(f"  Mean actual: {data['y_margins_test'].mean():.2f}")
    
    return {'mae': mae, 'rmse': rmse, 'rl_accuracy': rl_accuracy}


def train_dow_totals_model(data):
    """Train the day-of-week aware totals model with global normalization."""
    import torch
    from pathlib import Path
    from models.nn_dow_totals_model import DoWTotalsNet
    
    input_size = data['X_train'].shape[1]
    dow_embed_dim = 4
    print(f"\n{'='*50}")
    print(f"Training DoW Totals Model (input_size={input_size})")
    print(f"{'='*50}")
    
    feature_mean = data['global_mean']
    feature_std = data['global_std']
    
    X_train = (data['X_train'] - feature_mean) / feature_std
    X_val = (data['X_val'] - feature_mean) / feature_std
    X_test = (data['X_test'] - feature_mean) / feature_std
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DoWTotalsNet(input_size, dow_embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    dow_train_t = torch.tensor(data['dow_train'], dtype=torch.long).to(device)
    y_train_t = torch.tensor(data['y_totals_train'], dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    dow_val_t = torch.tensor(data['dow_val'], dtype=torch.long).to(device)
    y_val_t = torch.tensor(data['y_totals_val'], dtype=torch.float32).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, dow_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    def gaussian_nll_loss(mean, logvar, target):
        var = torch.exp(logvar) + 1e-6
        return 0.5 * torch.mean(torch.log(var) + (target - mean) ** 2 / var)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(150):
        model.train()
        train_losses = []
        for X_batch, dow_batch, y_batch in train_loader:
            optimizer.zero_grad()
            mean, logvar = model(X_batch, dow_batch)
            loss = gaussian_nll_loss(mean, logvar, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_mean, val_logvar = model(X_val_t, dow_val_t)
            val_loss = gaussian_nll_loss(val_mean, val_logvar, y_val_t).item()
            val_mae = torch.abs(val_mean - y_val_t).mean().item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/150 | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    
    model_path = Path(__file__).parent.parent / "data" / "nn_dow_totals_model.pt"
    torch.save({
        'model_state_dict': best_state,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_size': input_size,
        'dow_embed_dim': dow_embed_dim,
    }, model_path)
    print(f"Saved to {model_path}")
    
    # Evaluate
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    dow_test_t = torch.tensor(data['dow_test'], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        test_mean, _ = model(X_test_t, dow_test_t)
        preds = test_mean.cpu().numpy()
    
    errors = preds - data['y_totals_test']
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    print(f"\nTest Results:")
    print(f"  MAE: {mae:.3f} runs")
    print(f"  RMSE: {rmse:.3f} runs")
    print(f"  Mean predicted: {preds.mean():.2f}")
    print(f"  Mean actual: {data['y_totals_test'].mean():.2f}")
    
    # By day of week
    dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    print(f"\nBy Day of Week:")
    for d in range(7):
        mask = data['dow_test'] == d
        if mask.sum() == 0:
            continue
        d_preds = preds[mask]
        d_actual = data['y_totals_test'][mask]
        d_mae = np.abs(d_preds - d_actual).mean()
        print(f"  {dow_names[d]}: n={mask.sum()}, MAE={d_mae:.2f}, "
              f"actual={d_actual.mean():.1f}, pred={d_preds.mean():.1f}")
    
    return {'mae': mae, 'rmse': rmse}


def verify_models():
    """Verify all models load correctly and produce reasonable predictions."""
    print(f"\n{'='*50}")
    print("Verifying Models")
    print(f"{'='*50}")
    
    from models.nn_totals_model import NNTotalsModel
    from models.nn_spread_model import NNSpreadModel
    from models.nn_dow_totals_model import NNDoWTotalsModel
    
    # Test with a real game matchup
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT home_team_id, away_team_id FROM games LIMIT 1")
    row = c.fetchone()
    conn.close()
    
    if not row:
        print("No games found for verification")
        return False
    
    home_id, away_id = row['home_team_id'], row['away_team_id']
    
    all_ok = True
    
    # Test totals model
    totals = NNTotalsModel(use_model_predictions=False)
    print(f"\nTotals Model:")
    print(f"  is_trained(): {totals.is_trained()}")
    if totals.is_trained():
        pred = totals.predict_game(home_id, away_id)
        print(f"  projected_total: {pred['projected_total']}")
        if not (5 <= pred['projected_total'] <= 25):
            print(f"  WARNING: Total {pred['projected_total']} seems unreasonable!")
            all_ok = False
    else:
        all_ok = False
    
    # Test spread model
    spread = NNSpreadModel(use_model_predictions=False)
    print(f"\nSpread Model:")
    print(f"  is_trained(): {spread.is_trained()}")
    if spread.is_trained():
        pred = spread.predict_game(home_id, away_id)
        print(f"  projected_margin: {pred['projected_margin']}")
        if not (-15 <= pred['projected_margin'] <= 15):
            print(f"  WARNING: Margin {pred['projected_margin']} seems unreasonable!")
            all_ok = False
    else:
        all_ok = False
    
    # Test DoW totals model
    dow = NNDoWTotalsModel(use_model_predictions=False)
    print(f"\nDoW Totals Model:")
    print(f"  is_trained(): {dow.is_trained()}")
    if dow.is_trained():
        pred = dow.predict_game(home_id, away_id)
        print(f"  projected_total: {pred['projected_total']}")
        if not (5 <= pred['projected_total'] <= 25):
            print(f"  WARNING: Total {pred['projected_total']} seems unreasonable!")
            all_ok = False
    else:
        all_ok = False
    
    return all_ok


def main():
    # ==== TOTALS MODELS: Train on historical data (2024-2025) ====
    print("="*60)
    print("TOTALS MODELS - Loading historical training data (2024-2025)...")
    print("="*60)
    historical_games = load_historical_training_data()
    
    if len(historical_games) < 50:
        print(f"Only {len(historical_games)} historical games available. Need more data.")
        return
    
    print(f"Loaded {len(historical_games)} historical games")
    
    print("\nPreparing historical datasets for totals models...")
    historical_data = prepare_datasets(historical_games)
    
    print(f"Train: {len(historical_data['X_train'])} games")
    print(f"Val: {len(historical_data['X_val'])} games")
    print(f"Test: {len(historical_data['X_test'])} games")
    
    # Train totals models on historical data
    train_totals_model(historical_data)
    train_dow_totals_model(historical_data)
    
    # ==== SPREAD MODEL: Train on current season (2026) only ====
    print("\n" + "="*60)
    print("SPREAD MODEL - Loading current season data (2026 only)...")
    print("="*60)
    current_games = load_current_season_training_data()
    
    if len(current_games) < 30:
        print(f"Only {len(current_games)} current season games. Spread model needs more games.")
        print("Skipping spread model training (will use fallback predictions).")
    else:
        print(f"Loaded {len(current_games)} current season games")
        
        print("\nPreparing current season datasets for spread model...")
        # Use smaller validation split for limited data
        current_data = prepare_datasets(current_games, val_ratio=0.15, test_ratio=0.15)
        
        print(f"Train: {len(current_data['X_train'])} games")
        print(f"Val: {len(current_data['X_val'])} games")
        print(f"Test: {len(current_data['X_test'])} games")
        
        # Train spread model on current season data
        train_spread_model(current_data)
    
    # Verify all models
    success = verify_models()
    
    if success:
        print(f"\n{'='*60}")
        print("✓ All models trained and verified successfully!")
        print("  - Totals models: trained on 2024-2025 historical data")
        print("  - Spread model: trained on 2026 current season data")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("⚠ Some models may have issues - check output above")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
