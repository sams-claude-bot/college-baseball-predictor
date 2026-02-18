#!/usr/bin/env python3
"""
Data-Driven Weather Impact Model

Learns weather effects on scoring from historical game data rather than
hardcoding assumptions. Uses regression to determine coefficients.

Team-agnostic: learns overall weather impact on college baseball scoring,
not team-specific effects (rosters change year to year).
"""

import sys
import math
import json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection

# Path to store learned coefficients
COEFFICIENTS_FILE = Path(__file__).parent.parent / "data" / "weather_coefficients.json"

# Default coefficients (used before training)
DEFAULT_COEFFICIENTS = {
    'intercept': 0.0,
    'temp_coef': 0.0,        # Effect of temperature (centered)
    'wind_out_coef': 0.0,    # Effect of wind blowing out
    'humidity_coef': 0.0,    # Effect of humidity (centered)
    'dome_coef': 0.0,        # Effect of dome (vs outdoor)
    'temp_mean': 67.0,       # Centering values
    'humidity_mean': 50.0,
    'trained_on': None,
    'n_games': 0,
    'r_squared': 0.0,
}


def compute_wind_out_component(wind_speed: float, wind_direction_deg: float) -> float:
    """
    Compute the "wind blowing out" component.
    
    Convention: 0° = North, 180° = South
    Standard park: home plate south, CF north
    Wind from south (180°) blows OUT toward CF (positive)
    Wind from north (0°) blows IN toward home plate (negative)
    
    Returns: wind_speed * cos_component, where positive = out
    """
    if wind_speed is None or wind_direction_deg is None:
        return 0.0
    
    # Convert direction to radians
    rad = math.radians(wind_direction_deg)
    
    # cos(180°) = -1, so -cos gives us +1 for south wind (out)
    # cos(0°) = 1, so -cos gives us -1 for north wind (in)
    out_component = -math.cos(rad)
    
    return wind_speed * out_component


def load_coefficients() -> dict:
    """Load learned coefficients from file, or return defaults."""
    if COEFFICIENTS_FILE.exists():
        try:
            with open(COEFFICIENTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_COEFFICIENTS.copy()


def save_coefficients(coefficients: dict):
    """Save learned coefficients to file."""
    COEFFICIENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COEFFICIENTS_FILE, 'w') as f:
        json.dump(coefficients, f, indent=2)


def train_weather_model(seasons: list = None, save: bool = True) -> dict:
    """
    Train weather model using regression on historical data.
    
    Model: total_runs_residual = β0 + β1*temp_centered + β2*wind_out + β3*humidity_centered + β4*dome
    
    Where total_runs_residual = actual_total - league_avg_for_season
    
    Args:
        seasons: List of seasons to train on (default: [2024, 2025])
        save: Whether to save coefficients to file
    
    Returns:
        Dictionary of learned coefficients
    """
    if seasons is None:
        seasons = [2024, 2025]
    
    conn = get_connection()
    c = conn.cursor()
    
    # Get games with weather data
    placeholders = ','.join('?' * len(seasons))
    c.execute(f'''
        SELECT 
            hg.season,
            hg.home_score + hg.away_score as total_runs,
            hw.temp_f,
            hw.wind_speed_mph,
            hw.wind_direction_deg,
            hw.humidity_pct,
            hw.is_dome
        FROM historical_games hg
        JOIN historical_game_weather hw ON hg.id = hw.game_id
        WHERE hg.season IN ({placeholders})
        AND hw.temp_f IS NOT NULL
        AND hw.wind_speed_mph IS NOT NULL
        AND hw.humidity_pct IS NOT NULL
    ''', seasons)
    
    rows = c.fetchall()
    conn.close()
    
    if len(rows) < 100:
        print(f"Warning: Only {len(rows)} games with complete weather data")
        return DEFAULT_COEFFICIENTS.copy()
    
    # Calculate season averages for residualization
    season_totals = {}
    for row in rows:
        season = row[0]
        total = row[1]
        if season not in season_totals:
            season_totals[season] = []
        season_totals[season].append(total)
    
    season_avgs = {s: np.mean(t) for s, t in season_totals.items()}
    
    # Calculate feature means for centering
    temps = [r[2] for r in rows if r[2] is not None]
    humidities = [r[5] for r in rows if r[5] is not None]
    
    temp_mean = np.mean(temps)
    humidity_mean = np.mean(humidities)
    
    # Build feature matrix X and target vector y
    X = []
    y = []
    
    for row in rows:
        season, total, temp, wind, wind_dir, humidity, is_dome = row
        
        # Target: residual from season average
        residual = total - season_avgs[season]
        
        # Features (centered)
        temp_centered = temp - temp_mean
        wind_out = compute_wind_out_component(wind, wind_dir)
        humidity_centered = humidity - humidity_mean
        dome = 1.0 if is_dome else 0.0
        
        X.append([1.0, temp_centered, wind_out, humidity_centered, dome])
        y.append(residual)
    
    X = np.array(X)
    y = np.array(y)
    
    # Ordinary Least Squares regression: β = (X'X)^-1 X'y
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        print("Warning: Regression failed, using defaults")
        return DEFAULT_COEFFICIENTS.copy()
    
    # Calculate R-squared
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Calculate standard errors for coefficient significance
    n = len(y)
    p = X.shape[1]
    mse = ss_res / (n - p)
    var_beta = mse * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se
    
    coefficients = {
        'intercept': float(beta[0]),
        'temp_coef': float(beta[1]),
        'wind_out_coef': float(beta[2]),
        'humidity_coef': float(beta[3]),
        'dome_coef': float(beta[4]),
        'temp_mean': float(temp_mean),
        'humidity_mean': float(humidity_mean),
        'trained_on': datetime.now().isoformat(),
        'n_games': len(rows),
        'r_squared': float(r_squared),
        'seasons': seasons,
        't_statistics': {
            'intercept': float(t_stats[0]),
            'temp': float(t_stats[1]),
            'wind_out': float(t_stats[2]),
            'humidity': float(t_stats[3]),
            'dome': float(t_stats[4]),
        },
        'standard_errors': {
            'intercept': float(se[0]),
            'temp': float(se[1]),
            'wind_out': float(se[2]),
            'humidity': float(se[3]),
            'dome': float(se[4]),
        }
    }
    
    if save:
        save_coefficients(coefficients)
    
    return coefficients


def calculate_weather_adjustment(weather_data: dict, coefficients: dict = None, 
                                  apply_adjustment: bool = True) -> tuple:
    """
    Calculate weather adjustment using learned coefficients.
    
    Args:
        weather_data: dict with temp_f, wind_speed_mph, wind_direction_deg, humidity_pct, is_dome
        coefficients: Learned coefficients (loads from file if None)
        apply_adjustment: If False, returns 0 adjustment but still computes components
                         (for informational display without affecting predictions)
    
    Note: Backtest shows weather improves MAE by only 0.018 runs (not significant).
    Consider setting apply_adjustment=False to show weather info without adjustment.
    
    Returns:
        (adjustment, components) where adjustment is runs to add/subtract from prediction
    """
    if coefficients is None:
        coefficients = load_coefficients()
    
    # Use defaults for missing weather data
    temp = weather_data.get('temp_f') if weather_data.get('temp_f') is not None else coefficients.get('temp_mean', 67.0)
    wind_speed = weather_data.get('wind_speed_mph') if weather_data.get('wind_speed_mph') is not None else 0.0
    wind_dir = weather_data.get('wind_direction_deg') if weather_data.get('wind_direction_deg') is not None else 180
    humidity = weather_data.get('humidity_pct') if weather_data.get('humidity_pct') is not None else coefficients.get('humidity_mean', 50.0)
    is_dome = weather_data.get('is_dome', 0)
    
    # Calculate centered features
    temp_centered = temp - coefficients.get('temp_mean', 67.0)
    wind_out = compute_wind_out_component(wind_speed, wind_dir)
    humidity_centered = humidity - coefficients.get('humidity_mean', 50.0)
    dome = 1.0 if is_dome else 0.0
    
    # Calculate individual contributions
    temp_effect = coefficients.get('temp_coef', 0) * temp_centered
    wind_effect = coefficients.get('wind_out_coef', 0) * wind_out
    humidity_effect = coefficients.get('humidity_coef', 0) * humidity_centered
    dome_effect = coefficients.get('dome_coef', 0) * dome
    
    # Total adjustment (intercept should be ~0 if training data was residualized)
    raw_adjustment = temp_effect + wind_effect + humidity_effect + dome_effect
    
    # Optionally disable adjustment (show info only, don't affect predictions)
    total_adjustment = raw_adjustment if apply_adjustment else 0.0
    
    components = {
        'temp_f': temp,
        'temp_centered': round(temp_centered, 1),
        'temp_effect': round(temp_effect, 3),
        'wind_speed_mph': wind_speed,
        'wind_direction_deg': wind_dir,
        'wind_out_component': round(wind_out, 2),
        'wind_effect': round(wind_effect, 3),
        'humidity_pct': humidity,
        'humidity_centered': round(humidity_centered, 1),
        'humidity_effect': round(humidity_effect, 3),
        'is_dome': is_dome,
        'dome_effect': round(dome_effect, 3),
        'raw_adjustment': round(raw_adjustment, 3),
        'total_adjustment': round(total_adjustment, 3),
        'adjustment_applied': apply_adjustment,
        'model_r_squared': coefficients.get('r_squared', 0),
    }
    
    return total_adjustment, components


def print_model_summary(coefficients: dict = None):
    """Print a summary of the learned weather model."""
    if coefficients is None:
        coefficients = load_coefficients()
    
    print("=" * 60)
    print("WEATHER MODEL - LEARNED COEFFICIENTS")
    print("=" * 60)
    
    if coefficients.get('trained_on'):
        print(f"Trained: {coefficients['trained_on']}")
        print(f"Games: {coefficients.get('n_games', 'N/A')}")
        print(f"Seasons: {coefficients.get('seasons', 'N/A')}")
        print(f"R²: {coefficients.get('r_squared', 0):.4f}")
    else:
        print("Model not yet trained - using defaults")
    
    print()
    print("Coefficients (effect on total runs):")
    print("-" * 60)
    
    t_stats = coefficients.get('t_statistics', {})
    
    # Temperature
    temp_coef = coefficients.get('temp_coef', 0)
    temp_t = t_stats.get('temp', 0)
    sig_temp = '*' if abs(temp_t) > 1.96 else ''
    print(f"  Temperature: {temp_coef:+.4f} runs per °F from {coefficients.get('temp_mean', 67):.1f}°F")
    print(f"               (t={temp_t:.2f}){sig_temp}")
    print(f"               → 20°F warmer = {temp_coef * 20:+.2f} runs")
    
    # Wind
    wind_coef = coefficients.get('wind_out_coef', 0)
    wind_t = t_stats.get('wind_out', 0)
    sig_wind = '*' if abs(wind_t) > 1.96 else ''
    print(f"  Wind out:    {wind_coef:+.4f} runs per mph·cos(dir)")
    print(f"               (t={wind_t:.2f}){sig_wind}")
    print(f"               → 15mph blowing out = {wind_coef * 15:+.2f} runs")
    
    # Humidity
    hum_coef = coefficients.get('humidity_coef', 0)
    hum_t = t_stats.get('humidity', 0)
    sig_hum = '*' if abs(hum_t) > 1.96 else ''
    print(f"  Humidity:    {hum_coef:+.4f} runs per % from {coefficients.get('humidity_mean', 50):.1f}%")
    print(f"               (t={hum_t:.2f}){sig_hum}")
    print(f"               → 30% higher = {hum_coef * 30:+.2f} runs")
    
    # Dome
    dome_coef = coefficients.get('dome_coef', 0)
    dome_t = t_stats.get('dome', 0)
    sig_dome = '*' if abs(dome_t) > 1.96 else ''
    print(f"  Dome:        {dome_coef:+.4f} runs vs outdoor")
    print(f"               (t={dome_t:.2f}){sig_dome}")
    
    print()
    print("* = statistically significant (|t| > 1.96, p < 0.05)")
    print("=" * 60)


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Weather Impact Model')
    parser.add_argument('action', nargs='?', choices=['train', 'show', 'predict'],
                       default='show', help='Action to perform')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2024, 2025],
                       help='Seasons to train on')
    parser.add_argument('--temp', type=float, help='Temperature (°F)')
    parser.add_argument('--wind', type=float, help='Wind speed (mph)')
    parser.add_argument('--wind-dir', type=int, help='Wind direction (degrees)')
    parser.add_argument('--humidity', type=float, help='Humidity (%)')
    parser.add_argument('--dome', action='store_true', help='Dome game')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        print(f"Training weather model on seasons: {args.seasons}")
        coefficients = train_weather_model(seasons=args.seasons)
        print_model_summary(coefficients)
        
    elif args.action == 'predict':
        weather = {}
        if args.temp is not None:
            weather['temp_f'] = args.temp
        if args.wind is not None:
            weather['wind_speed_mph'] = args.wind
        if args.wind_dir is not None:
            weather['wind_direction_deg'] = args.wind_dir
        if args.humidity is not None:
            weather['humidity_pct'] = args.humidity
        if args.dome:
            weather['is_dome'] = 1
        
        if not weather:
            print("No weather data provided. Use --temp, --wind, --wind-dir, --humidity, --dome")
        else:
            adj, components = calculate_weather_adjustment(weather)
            print(f"Weather adjustment: {adj:+.2f} runs")
            print("\nBreakdown:")
            print(f"  Temperature ({components['temp_f']:.0f}°F): {components['temp_effect']:+.3f}")
            print(f"  Wind ({components['wind_speed_mph']:.0f}mph @ {components['wind_direction_deg']}°): {components['wind_effect']:+.3f}")
            print(f"  Humidity ({components['humidity_pct']:.0f}%): {components['humidity_effect']:+.3f}")
            if components['is_dome']:
                print(f"  Dome: {components['dome_effect']:+.3f}")
    
    else:  # show
        print_model_summary()
