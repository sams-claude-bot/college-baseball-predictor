#!/usr/bin/env python3
"""
Daily data collection orchestrator

Run this daily to:
1. Collect NCAA stats
2. Update game results
3. Collect box scores for completed games
4. Update player stats
5. Generate predictions for upcoming games using ensemble model
6. Save daily snapshot
7. Report on model performance
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Set up paths - we're in scripts/core/, need to go up to project root
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR / "core"))
sys.path.insert(0, str(SCRIPTS_DIR / "data"))
sys.path.insert(0, str(BASE_DIR / "models"))

from database import get_current_top_25, get_recent_games

# Import predictor with new ensemble
from predictor_db import Predictor

# Import box score collector
try:
    from collect_box_scores import collect_box_scores
except ImportError:
    collect_box_scores = None

def main():
    """Main collection routine"""
    print(f"=== Daily Collection: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    
    # 1. Show current Top 25
    print("\n--- Current Top 25 ---")
    top25 = get_current_top_25()
    for rank, team in top25[:10]:
        print(f"  #{rank}: {team}")
    
    # 2. Show recent games
    print("\n--- Recent Games ---")
    games = get_recent_games(days=1)
    for game in games[:10]:
        print(f"  {game}")
    
    # 3. Collect box scores if available
    if collect_box_scores:
        print("\n--- Collecting Box Scores ---")
        try:
            collect_box_scores()
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n=== Collection Complete ===")

if __name__ == "__main__":
    main()
