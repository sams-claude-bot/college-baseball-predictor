"""CLI flow and console output for bet selection."""

import sys

from .record import record_bets
from .risk import RISK_BANKROLL, RISK_ENGINE_MODE, RISK_KELLY_FRACTION
from .selection import (
    MAX_CONSENSUS_PER_DAY,
    MAX_EV_PER_DAY,
    MAX_TOTALS_PER_DAY,
    ML_EDGE_THRESHOLD,
    ML_MAX_FAVORITE,
    ML_MIN_UNDERDOG,
    SPREADS_ENABLED,
    TOTALS_EDGE_THRESHOLD,
    analyze_games,
)


def print_analysis(results: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"BET SELECTION ANALYSIS - {results['date']}")
    print(f"{'='*60}")
    print(f"Risk mode: {RISK_ENGINE_MODE} | Bankroll: ${RISK_BANKROLL:.0f} | Kelly frac: {RISK_KELLY_FRACTION:.2f}")

    print(f"\n🎯 RECOMMENDED BETS ({len(results['bets'])})")
    print("-" * 40)

    if not results['bets']:
        print("  No bets meet the criteria today")
    else:
        for bet in results['bets']:
            if bet['type'] == 'CONSENSUS':
                sign = '+' if bet['moneyline'] > 0 else ''
                bonus = bet['adjusted_edge'] - bet['edge']
                print(f"  🎯 {bet['pick_team_name']} ({sign}{bet['moneyline']})")
                print(
                    f"       Models: {bet['models_agree']}/10 | Edge: {bet['edge']:.1f}% + "
                    f"{bonus:.1f}% bonus = {bet['adjusted_edge']:.1f}%"
                )
                print(
                    f"       Bet: ${bet['bet_amount']:.0f} (suggested ${bet.get('suggested_stake', bet['bet_amount']):.2f}) "
                    f"| risk {bet.get('risk_score', 0):.2f} | kelly {bet.get('kelly_fraction_used', 0):.4f}"
                )
            elif bet['type'] == 'ML':
                sign = '+' if bet['moneyline'] > 0 else ''
                bonus = bet.get('adjusted_edge', bet['edge']) - bet['edge']
                adj_str = f" + {bonus:.1f}% bonus = {bet['adjusted_edge']:.1f}%" if bonus > 0 else ""
                print(f"  💰 {bet['pick_team_name']} ({sign}{bet['moneyline']})")
                print(f"       Edge: {bet['edge']:.1f}%{adj_str} | Model: {bet['model_prob']*100:.0f}%")
                print(
                    f"       Bet: ${bet['bet_amount']:.0f} (suggested ${bet.get('suggested_stake', bet['bet_amount']):.2f}) "
                    f"| risk {bet.get('risk_score', 0):.2f} | kelly {bet.get('kelly_fraction_used', 0):.4f}"
                )
            elif bet['type'] == 'TOTAL':
                print(f"  📊 TOTAL: {bet['pick']} {bet['line']} ({bet['odds']:+d})")
                print(f"       Edge: {bet['edge']:.1f} runs | Proj: {bet['model_projection']:.1f}")
                print(
                    f"       Bet: ${bet['bet_amount']:.0f} (suggested ${bet.get('suggested_stake', bet['bet_amount']):.2f}) "
                    f"| risk {bet.get('risk_score', 0):.2f}"
                )
            if bet.get('exposure_bucket'):
                print(f"       Exposure: {bet['exposure_bucket']}")
            print()

    print(f"\n❌ REJECTIONS ({len(results['rejections'])})")
    print("-" * 40)

    by_reason = {}
    for rej in results['rejections']:
        for reason in rej['reasons']:
            key = reason.split(' ')[0]
            by_reason.setdefault(key, []).append(rej)

    for reason_cat, rejs in by_reason.items():
        print(f"  {reason_cat}: {len(rejs)} games")

    if results.get('spreads_disabled'):
        print("\n  ⚠️  SPREADS DISABLED (0/5 historical, needs recalibration)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bet_selection_v2.py [analyze|record]")
        print("\nThresholds:")
        print(f"  ML: {ML_EDGE_THRESHOLD}%+ edge, ML between {ML_MAX_FAVORITE} and +{ML_MIN_UNDERDOG}")
        print(f"  Totals: {TOTALS_EDGE_THRESHOLD}+ runs edge")
        print(f"  Spreads: {'ENABLED' if SPREADS_ENABLED else 'DISABLED'}")
        print(f"  Max bets/day: consensus={MAX_CONSENSUS_PER_DAY}, ev={MAX_EV_PER_DAY}, totals={MAX_TOTALS_PER_DAY}")
        print(f"  Risk engine: {RISK_ENGINE_MODE}")
        sys.exit(1)

    cmd = sys.argv[1]
    date_arg = sys.argv[2] if len(sys.argv) > 2 else None

    results = analyze_games(date_arg)

    if 'error' in results:
        sys.exit(1)

    print_analysis(results)

    if cmd == 'record':
        record_bets(results)
    elif cmd != 'analyze':
        print(f"Unknown command: {cmd}")
        sys.exit(1)

