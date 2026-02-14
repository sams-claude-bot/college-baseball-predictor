#!/usr/bin/env python3
"""
Generate preliminary model results for Feb 14-16 weekend games
"""

import sqlite3
import json
import sys
import os
from datetime import datetime, timedelta

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

class WeekendPredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir, 'data', 'baseball.db')
        self.predictions = {}
        
    def get_weekend_games(self):
        """Get games for Feb 14-16 weekend"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get games for the weekend
        weekend_dates = ['2026-02-14', '2026-02-15', '2026-02-16']
        
        all_games = []
        for date in weekend_dates:
            cursor.execute("""
                SELECT id, date, home_team_id, away_team_id, venue, status
                FROM games 
                WHERE date = ? AND (status = 'scheduled' OR status IS NULL)
                ORDER BY date, id
            """, (date,))
            
            games = cursor.fetchall()
            all_games.extend(games)
        
        conn.close()
        
        print(f"Found {len(all_games)} games for weekend Feb 14-16")
        return all_games
    
    def get_team_name(self, team_id):
        """Get team name from ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, nickname FROM teams WHERE id = ?", (team_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            name, nickname = result
            return f"{name} {nickname}".strip()
        return team_id
    
    def run_elo_predictions(self, games):
        """Run ELO model predictions"""
        print("\n=== ELO MODEL PREDICTIONS ===")
        
        try:
            from models.elo_model import EloModel
            
            elo_model = EloModel()
            predictions = []
            
            for game in games:
                game_id, date, home_team, away_team, venue, status = game
                
                try:
                    prediction = elo_model.predict_game(home_team, away_team, game_id)
                    
                    if prediction:
                        predictions.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_win_prob': prediction.get('home_win_prob', 0.5),
                            'away_win_prob': prediction.get('away_win_prob', 0.5),
                            'predicted_winner': home_team if prediction.get('home_win_prob', 0.5) > 0.5 else away_team
                        })
                        
                        home_name = self.get_team_name(home_team)
                        away_name = self.get_team_name(away_team)
                        home_prob = prediction.get('home_win_prob', 0.5)
                        
                        print(f"  {date}: {away_name} @ {home_name} - {home_name} {home_prob:.1%}")
                
                except Exception as e:
                    print(f"    Error predicting game {game_id}: {e}")
            
            return predictions
            
        except ImportError as e:
            print(f"  Could not import ELO model: {e}")
            return []
        except Exception as e:
            print(f"  Error running ELO predictions: {e}")
            return []
    
    def run_log5_predictions(self, games):
        """Run Log5 model predictions"""
        print("\n=== LOG5 MODEL PREDICTIONS ===")
        
        try:
            from models.log5_model import Log5Model
            
            log5_model = Log5Model()
            predictions = []
            
            for game in games:
                game_id, date, home_team, away_team, venue, status = game
                
                try:
                    prediction = log5_model.predict_game(home_team, away_team)
                    
                    if prediction:
                        predictions.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_win_prob': prediction.get('home_win_prob', 0.5),
                            'away_win_prob': prediction.get('away_win_prob', 0.5),
                            'predicted_winner': home_team if prediction.get('home_win_prob', 0.5) > 0.5 else away_team
                        })
                        
                        home_name = self.get_team_name(home_team)
                        away_name = self.get_team_name(away_team)
                        home_prob = prediction.get('home_win_prob', 0.5)
                        
                        print(f"  {date}: {away_name} @ {home_name} - {home_name} {home_prob:.1%}")
                
                except Exception as e:
                    print(f"    Error predicting game {game_id}: {e}")
            
            return predictions
            
        except ImportError as e:
            print(f"  Could not import Log5 model: {e}")
            return []
        except Exception as e:
            print(f"  Error running Log5 predictions: {e}")
            return []
    
    def run_pythagorean_predictions(self, games):
        """Run Pythagorean model predictions"""
        print("\n=== PYTHAGOREAN MODEL PREDICTIONS ===")
        
        try:
            from models.pythagorean_model import PythagoreanModel
            
            pyth_model = PythagoreanModel()
            predictions = []
            
            for game in games:
                game_id, date, home_team, away_team, venue, status = game
                
                try:
                    prediction = pyth_model.predict_game(home_team, away_team)
                    
                    if prediction:
                        predictions.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_win_prob': prediction.get('home_win_prob', 0.5),
                            'away_win_prob': prediction.get('away_win_prob', 0.5),
                            'predicted_winner': home_team if prediction.get('home_win_prob', 0.5) > 0.5 else away_team
                        })
                        
                        home_name = self.get_team_name(home_team)
                        away_name = self.get_team_name(away_team)
                        home_prob = prediction.get('home_win_prob', 0.5)
                        
                        print(f"  {date}: {away_name} @ {home_name} - {home_name} {home_prob:.1%}")
                
                except Exception as e:
                    print(f"    Error predicting game {game_id}: {e}")
            
            return predictions
            
        except ImportError as e:
            print(f"  Could not import Pythagorean model: {e}")
            return []
        except Exception as e:
            print(f"  Error running Pythagorean predictions: {e}")
            return []
    
    def run_advanced_predictions(self, games):
        """Run Advanced model predictions"""
        print("\n=== ADVANCED MODEL PREDICTIONS ===")
        
        try:
            from models.advanced_model import AdvancedModel
            
            adv_model = AdvancedModel()
            predictions = []
            
            for game in games:
                game_id, date, home_team, away_team, venue, status = game
                
                try:
                    prediction = adv_model.predict_game(home_team, away_team)
                    
                    if prediction:
                        predictions.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_win_prob': prediction.get('home_win_prob', 0.5),
                            'away_win_prob': prediction.get('away_win_prob', 0.5),
                            'predicted_winner': home_team if prediction.get('home_win_prob', 0.5) > 0.5 else away_team
                        })
                        
                        home_name = self.get_team_name(home_team)
                        away_name = self.get_team_name(away_team)
                        home_prob = prediction.get('home_win_prob', 0.5)
                        
                        print(f"  {date}: {away_name} @ {home_name} - {home_name} {home_prob:.1%}")
                
                except Exception as e:
                    print(f"    Error predicting game {game_id}: {e}")
            
            return predictions
            
        except ImportError as e:
            print(f"  Could not import Advanced model: {e}")
            return []
        except Exception as e:
            print(f"  Error running Advanced predictions: {e}")
            return []
    
    def run_ensemble_predictions(self, games):
        """Run Ensemble model predictions"""
        print("\n=== ENSEMBLE MODEL PREDICTIONS ===")
        
        try:
            from models.ensemble_model import EnsembleModel
            
            ensemble_model = EnsembleModel()
            predictions = []
            
            for game in games:
                game_id, date, home_team, away_team, venue, status = game
                
                try:
                    prediction = ensemble_model.predict_game(home_team, away_team)
                    
                    if prediction:
                        predictions.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_win_prob': prediction.get('home_win_prob', 0.5),
                            'away_win_prob': prediction.get('away_win_prob', 0.5),
                            'predicted_winner': home_team if prediction.get('home_win_prob', 0.5) > 0.5 else away_team
                        })
                        
                        home_name = self.get_team_name(home_team)
                        away_name = self.get_team_name(away_team)
                        home_prob = prediction.get('home_win_prob', 0.5)
                        
                        print(f"  {date}: {away_name} @ {home_name} - {home_name} {home_prob:.1%}")
                
                except Exception as e:
                    print(f"    Error predicting game {game_id}: {e}")
            
            return predictions
            
        except ImportError as e:
            print(f"  Could not import Ensemble model: {e}")
            return []
        except Exception as e:
            print(f"  Error running Ensemble predictions: {e}")
            return []
    
    def generate_weekend_report(self):
        """Generate comprehensive weekend predictions report"""
        
        print("=== WEEKEND PREDICTIONS REPORT - FEB 14-16, 2026 ===")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S CST')}")
        
        # Get weekend games
        games = self.get_weekend_games()
        
        if not games:
            print("No games found for the weekend")
            return
        
        # Run all models
        model_results = {
            'elo': self.run_elo_predictions(games),
            'log5': self.run_log5_predictions(games),
            'pythagorean': self.run_pythagorean_predictions(games),
            'advanced': self.run_advanced_predictions(games),
            'ensemble': self.run_ensemble_predictions(games)
        }
        
        # Create comprehensive report
        report = {
            'generation_time': datetime.now().isoformat(),
            'weekend_dates': ['2026-02-14', '2026-02-15', '2026-02-16'],
            'total_games': len(games),
            'model_results': model_results,
            'model_summary': {}
        }
        
        # Generate model summary
        for model_name, predictions in model_results.items():
            if predictions:
                report['model_summary'][model_name] = {
                    'predictions_generated': len(predictions),
                    'avg_confidence': sum([abs(p['home_win_prob'] - 0.5) * 2 for p in predictions]) / len(predictions) if predictions else 0
                }
            else:
                report['model_summary'][model_name] = {
                    'predictions_generated': 0,
                    'status': 'failed_or_unavailable'
                }
        
        # Save report
        report_file = os.path.join(self.base_dir, 'data', 'weekend_predictions_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary text
        summary = f"""
=== WEEKEND PREDICTIONS SUMMARY ===
Date Range: Feb 14-16, 2026
Total Games: {len(games)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S CST')}

MODEL PERFORMANCE:
"""
        
        for model_name, summary_data in report['model_summary'].items():
            predictions_count = summary_data['predictions_generated']
            if predictions_count > 0:
                avg_conf = summary_data.get('avg_confidence', 0)
                summary += f"- {model_name.upper()}: {predictions_count} predictions (avg confidence: {avg_conf:.1%})\n"
            else:
                summary += f"- {model_name.upper()}: No predictions generated\n"
        
        summary += f"\nDetailed results saved to: {report_file}\n"
        
        # Save summary
        summary_file = os.path.join(self.base_dir, 'data', 'weekend_predictions_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(summary)
        print(f"Summary saved to: {summary_file}")
        
        return report

if __name__ == "__main__":
    predictor = WeekendPredictor()
    report = predictor.generate_weekend_report()
    
    print("\n=== WEEKEND PREDICTIONS COMPLETE ===")
    print("Sam can review the predictions in the morning!")