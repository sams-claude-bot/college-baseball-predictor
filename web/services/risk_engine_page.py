"""Data assembly for the /experimental/risk-engine page."""

import sys
from pathlib import Path

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection
from config import model_config as cfg
from bet_selection_v2 import analyze_games


def build_risk_engine_page_context():
    """Build template context for the experimental risk engine page."""
    risk_engine = {
        'mode': getattr(cfg, 'BET_RISK_ENGINE_MODE', 'fixed'),
        'bankroll': getattr(cfg, 'BET_RISK_BANKROLL', 5000.0),
        'bankroll_peak': getattr(cfg, 'BET_RISK_BANKROLL_PEAK', getattr(cfg, 'BET_RISK_BANKROLL', 5000.0)),
        'kelly_fraction': getattr(cfg, 'BET_RISK_KELLY_FRACTION', 0.25),
        'min_stake': getattr(cfg, 'BET_RISK_MIN_STAKE', 25.0),
        'max_stake': getattr(cfg, 'BET_RISK_MAX_STAKE', 250.0),
        'drawdown_threshold': getattr(cfg, 'BET_RISK_DRAWDOWN_THRESHOLD', 0.10),
        'drawdown_multiplier': getattr(cfg, 'BET_RISK_DRAWDOWN_FRACTION_MULTIPLIER', 0.5),
    }

    preview = {'date': None, 'bets': [], 'rejections': [], 'error': None}
    try:
        preview = analyze_games()
    except Exception as e:
        preview['error'] = str(e)

    # Experimental P&L tracking for fractional Kelly mode
    pnl = {
        'mode': 'fractional_kelly',
        'total_bets': 0,
        'settled_bets': 0,
        'wins': 0,
        'losses': 0,
        'profit': 0.0,
        'roi_pct': 0.0,
        'staked': 0.0,
    }
    try:
        conn = get_connection()
        c = conn.cursor()

        # ML + consensus in tracked_bets / tracked_confident_bets
        c.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN won IS NOT NULL THEN 1 ELSE 0 END) as settled,
                   SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN won = 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(CASE WHEN won IS NOT NULL THEN profit ELSE 0 END), 0) as profit,
                   COALESCE(SUM(CASE WHEN won IS NOT NULL THEN COALESCE(suggested_stake, bet_amount, 0) ELSE 0 END), 0) as staked
            FROM tracked_bets
            WHERE risk_mode = 'fractional_kelly'
        """)
        a = c.fetchone()

        c.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN won IS NOT NULL THEN 1 ELSE 0 END) as settled,
                   SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN won = 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(CASE WHEN won IS NOT NULL THEN profit ELSE 0 END), 0) as profit,
                   COALESCE(SUM(CASE WHEN won IS NOT NULL THEN COALESCE(suggested_stake, bet_amount, 0) ELSE 0 END), 0) as staked
            FROM tracked_confident_bets
            WHERE risk_mode = 'fractional_kelly'
        """)
        b = c.fetchone()

        c.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN won IS NOT NULL THEN 1 ELSE 0 END) as settled,
                   SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN won = 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(CASE WHEN won IS NOT NULL THEN profit ELSE 0 END), 0) as profit,
                   COALESCE(SUM(CASE WHEN won IS NOT NULL THEN COALESCE(suggested_stake, bet_amount, 0) ELSE 0 END), 0) as staked
            FROM tracked_bets_spreads
            WHERE risk_mode = 'fractional_kelly' AND bet_type = 'total'
        """)
        d = c.fetchone()
        conn.close()

        for row in (a, b, d):
            pnl['total_bets'] += row['total'] or 0
            pnl['settled_bets'] += row['settled'] or 0
            pnl['wins'] += row['wins'] or 0
            pnl['losses'] += row['losses'] or 0
            pnl['profit'] += row['profit'] or 0.0
            pnl['staked'] += row['staked'] or 0.0

        if pnl['staked'] > 0:
            pnl['roi_pct'] = (pnl['profit'] / pnl['staked']) * 100.0
    except Exception:
        pass

    return {
        'risk_engine': risk_engine,
        'risk_preview': preview,
        'risk_pnl': pnl,
    }
