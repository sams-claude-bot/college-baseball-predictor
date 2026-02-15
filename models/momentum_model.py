#!/usr/bin/env python3
"""
Momentum Model - Predicts based on recent team form/streaks.

Theory: Recent performance (last 5-7 games) is more predictive than season averages,
especially for detecting hot/cold streaks that standard models miss.

Factors tracked:
- Win streak / losing streak
- Recent run differential (weighted by recency)
- Recent batting performance vs season average
- Recent pitching performance vs season average
- Home/away form

Output: Momentum score (-1.0 to +1.0) that can adjust win probability.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection


# Configuration
LOOKBACK_GAMES = 7  # Number of recent games to analyze
RECENCY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # Most recent first


def get_recent_games(team_id: str, limit: int = LOOKBACK_GAMES, before_date: str = None) -> List[Dict]:
    """Get team's most recent completed games."""
    conn = get_connection()
    cur = conn.cursor()
    
    if before_date is None:
        before_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get games where this team played and game is complete
    cur.execute('''
        SELECT 
            g.id, g.date, g.home_team_id, g.away_team_id,
            g.home_score, g.away_score, g.winner_id
        FROM games g
        WHERE (g.home_team_id = ? OR g.away_team_id = ?)
        AND g.home_score IS NOT NULL 
        AND g.away_score IS NOT NULL
        AND g.winner_id IS NOT NULL
        AND g.date <= ?
        ORDER BY g.date DESC, g.id DESC
        LIMIT ?
    ''', (team_id, team_id, before_date, limit))
    
    games = []
    for row in cur.fetchall():
        game_id, date, home_id, away_id, home_score, away_score, winner_id = row
        
        is_home = (home_id == team_id)
        team_score = home_score if is_home else away_score
        opp_score = away_score if is_home else home_score
        won = (winner_id == team_id)
        
        games.append({
            'game_id': game_id,
            'date': date,
            'is_home': is_home,
            'team_score': team_score,
            'opp_score': opp_score,
            'run_diff': team_score - opp_score,
            'won': won,
            'opponent_id': away_id if is_home else home_id
        })
    
    conn.close()
    return games


def calculate_streak(games: List[Dict]) -> Tuple[int, str]:
    """Calculate current win/loss streak. Returns (streak_length, 'W'|'L')."""
    if not games:
        return 0, 'N'
    
    streak_type = 'W' if games[0]['won'] else 'L'
    streak_length = 0
    
    for game in games:
        if (game['won'] and streak_type == 'W') or (not game['won'] and streak_type == 'L'):
            streak_length += 1
        else:
            break
    
    return streak_length, streak_type


def calculate_weighted_run_diff(games: List[Dict]) -> float:
    """Calculate recency-weighted run differential."""
    if not games:
        return 0.0
    
    total_weight = 0.0
    weighted_diff = 0.0
    
    for i, game in enumerate(games):
        weight = RECENCY_WEIGHTS[i] if i < len(RECENCY_WEIGHTS) else 0.3
        weighted_diff += game['run_diff'] * weight
        total_weight += weight
    
    return weighted_diff / total_weight if total_weight > 0 else 0.0


def calculate_recent_win_pct(games: List[Dict]) -> float:
    """Calculate win percentage over recent games."""
    if not games:
        return 0.5
    
    wins = sum(1 for g in games if g['won'])
    return wins / len(games)


def calculate_home_away_form(games: List[Dict], home: bool) -> float:
    """Calculate form specifically for home or away games."""
    filtered = [g for g in games if g['is_home'] == home]
    if not filtered:
        return 0.5
    
    wins = sum(1 for g in filtered if g['won'])
    return wins / len(filtered)


def get_momentum_score(team_id: str, before_date: str = None) -> Dict:
    """
    Calculate comprehensive momentum score for a team.
    
    Returns dict with:
    - momentum_score: -1.0 to +1.0 (negative = cold, positive = hot)
    - components: breakdown of factors
    - games_analyzed: number of games used
    """
    games = get_recent_games(team_id, LOOKBACK_GAMES, before_date)
    
    if len(games) < 3:
        # Not enough data, return neutral
        return {
            'momentum_score': 0.0,
            'confidence': 'low',
            'games_analyzed': len(games),
            'components': {}
        }
    
    # Calculate components
    streak_len, streak_type = calculate_streak(games)
    weighted_run_diff = calculate_weighted_run_diff(games)
    recent_win_pct = calculate_recent_win_pct(games)
    
    # Normalize components to -1 to +1 scale
    
    # Streak component: W5+ = +0.3, L5+ = -0.3, scaled
    streak_score = (streak_len / 7) * (1 if streak_type == 'W' else -1)
    streak_score = max(-0.3, min(0.3, streak_score))
    
    # Run differential component: +3 per game avg = +0.3, scaled
    run_diff_score = weighted_run_diff / 10  # +10 run diff = +1.0
    run_diff_score = max(-0.4, min(0.4, run_diff_score))
    
    # Win percentage component: deviation from 0.5
    win_pct_score = (recent_win_pct - 0.5) * 0.6  # 100% = +0.3, 0% = -0.3
    
    # Combine components
    momentum_score = streak_score * 0.3 + run_diff_score * 0.4 + win_pct_score * 0.3
    momentum_score = max(-1.0, min(1.0, momentum_score))
    
    # Determine confidence
    confidence = 'high' if len(games) >= 6 else 'medium' if len(games) >= 4 else 'low'
    
    return {
        'momentum_score': round(momentum_score, 3),
        'confidence': confidence,
        'games_analyzed': len(games),
        'components': {
            'streak': f"{streak_type}{streak_len}",
            'streak_score': round(streak_score, 3),
            'weighted_run_diff': round(weighted_run_diff, 2),
            'run_diff_score': round(run_diff_score, 3),
            'recent_win_pct': round(recent_win_pct, 3),
            'win_pct_score': round(win_pct_score, 3)
        },
        'recent_games': [
            {'date': g['date'], 'vs': g['opponent_id'], 
             'result': f"{'W' if g['won'] else 'L'} {g['team_score']}-{g['opp_score']}"}
            for g in games[:5]
        ]
    }


def predict_with_momentum(team_a: str, team_b: str, 
                          base_prob_a: float = 0.5,
                          game_date: str = None) -> Dict:
    """
    Adjust a base win probability using momentum factors.
    
    Args:
        team_a: First team ID
        team_b: Second team ID  
        base_prob_a: Base probability of team_a winning (from other model)
        game_date: Date to calculate momentum as of (for backtesting)
    
    Returns:
        Adjusted probability and momentum details
    """
    mom_a = get_momentum_score(team_a, game_date)
    mom_b = get_momentum_score(team_b, game_date)
    
    # Momentum differential
    mom_diff = mom_a['momentum_score'] - mom_b['momentum_score']
    
    # Adjustment factor: +/-0.05 to +/-0.15 based on momentum differential
    # Max adjustment of 15% to avoid overweighting momentum
    adjustment = mom_diff * 0.075  # +1.0 diff = +7.5% swing
    adjustment = max(-0.15, min(0.15, adjustment))
    
    # Apply adjustment
    adjusted_prob = base_prob_a + adjustment
    adjusted_prob = max(0.05, min(0.95, adjusted_prob))  # Keep in reasonable bounds
    
    return {
        'base_probability': base_prob_a,
        'adjusted_probability': round(adjusted_prob, 3),
        'adjustment': round(adjustment, 3),
        'momentum_differential': round(mom_diff, 3),
        'team_a_momentum': mom_a,
        'team_b_momentum': mom_b
    }


def get_hottest_teams(conference: str = None, limit: int = 10) -> List[Dict]:
    """Get teams with highest momentum scores."""
    conn = get_connection()
    cur = conn.cursor()
    
    if conference:
        cur.execute('SELECT id FROM teams WHERE conference = ?', (conference,))
    else:
        cur.execute('SELECT id FROM teams')
    
    team_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    
    results = []
    for team_id in team_ids:
        mom = get_momentum_score(team_id)
        if mom['games_analyzed'] >= 3:
            results.append({
                'team_id': team_id,
                'momentum_score': mom['momentum_score'],
                'streak': mom['components'].get('streak', 'N/A'),
                'recent_win_pct': mom['components'].get('recent_win_pct', 0),
                'games_analyzed': mom['games_analyzed']
            })
    
    # Sort by momentum score
    results.sort(key=lambda x: x['momentum_score'], reverse=True)
    return results[:limit]


def get_coldest_teams(conference: str = None, limit: int = 10) -> List[Dict]:
    """Get teams with lowest momentum scores."""
    hot = get_hottest_teams(conference, limit=100)
    hot.sort(key=lambda x: x['momentum_score'])
    return hot[:limit]


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Momentum Model')
    parser.add_argument('action', choices=['team', 'matchup', 'hot', 'cold'],
                        help='Action to perform')
    parser.add_argument('--team', help='Team ID for single team analysis')
    parser.add_argument('--team-a', help='First team for matchup')
    parser.add_argument('--team-b', help='Second team for matchup')
    parser.add_argument('--base-prob', type=float, default=0.5,
                        help='Base probability for team A')
    parser.add_argument('--conference', help='Filter by conference')
    parser.add_argument('--limit', type=int, default=10, help='Number of results')
    args = parser.parse_args()
    
    if args.action == 'team':
        if not args.team:
            print("Error: --team required")
            sys.exit(1)
        result = get_momentum_score(args.team)
        print(f"\nMomentum Analysis: {args.team}")
        print(f"{'='*50}")
        print(f"Momentum Score: {result['momentum_score']:+.3f}")
        print(f"Confidence: {result['confidence']}")
        print(f"Games Analyzed: {result['games_analyzed']}")
        print(f"\nComponents:")
        for k, v in result['components'].items():
            print(f"  {k}: {v}")
        print(f"\nRecent Games:")
        for g in result['recent_games']:
            print(f"  {g['date']}: {g['result']} vs {g['vs']}")
    
    elif args.action == 'matchup':
        if not args.team_a or not args.team_b:
            print("Error: --team-a and --team-b required")
            sys.exit(1)
        result = predict_with_momentum(args.team_a, args.team_b, args.base_prob)
        print(f"\nMomentum-Adjusted Prediction")
        print(f"{'='*50}")
        print(f"{args.team_a} vs {args.team_b}")
        print(f"\nBase Probability ({args.team_a}): {result['base_probability']:.1%}")
        print(f"Momentum Adjustment: {result['adjustment']:+.1%}")
        print(f"Adjusted Probability: {result['adjusted_probability']:.1%}")
        print(f"\nMomentum Differential: {result['momentum_differential']:+.3f}")
        print(f"\n{args.team_a}: {result['team_a_momentum']['momentum_score']:+.3f} "
              f"({result['team_a_momentum']['components'].get('streak', 'N/A')})")
        print(f"{args.team_b}: {result['team_b_momentum']['momentum_score']:+.3f} "
              f"({result['team_b_momentum']['components'].get('streak', 'N/A')})")
    
    elif args.action == 'hot':
        results = get_hottest_teams(args.conference, args.limit)
        print(f"\n🔥 Hottest Teams" + (f" ({args.conference})" if args.conference else ""))
        print(f"{'='*50}")
        for i, r in enumerate(results, 1):
            print(f"{i:2}. {r['team_id']:20} {r['momentum_score']:+.3f} "
                  f"({r['streak']}, {r['recent_win_pct']:.0%} L{r['games_analyzed']})")
    
    elif args.action == 'cold':
        results = get_coldest_teams(args.conference, args.limit)
        print(f"\n❄️ Coldest Teams" + (f" ({args.conference})" if args.conference else ""))
        print(f"{'='*50}")
        for i, r in enumerate(results, 1):
            print(f"{i:2}. {r['team_id']:20} {r['momentum_score']:+.3f} "
                  f"({r['streak']}, {r['recent_win_pct']:.0%} L{r['games_analyzed']})")
