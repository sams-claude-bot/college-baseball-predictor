"""Risk engine helpers and configuration for bet selection."""

from typing import List, Optional, Tuple

from config import model_config as cfg

# ============ RISK ENGINE (v1) ============
BASE_BET = int(getattr(cfg, 'BET_RISK_FIXED_STAKE', 100))  # Legacy fixed unit

RISK_ENGINE_MODE = getattr(cfg, 'BET_RISK_ENGINE_MODE', 'fixed')
RISK_BANKROLL = float(getattr(cfg, 'BET_RISK_BANKROLL', 5000.0))
RISK_BANKROLL_PEAK = float(getattr(cfg, 'BET_RISK_BANKROLL_PEAK', RISK_BANKROLL))
RISK_MIN_STAKE = float(getattr(cfg, 'BET_RISK_MIN_STAKE', BASE_BET))
RISK_MAX_STAKE = float(getattr(cfg, 'BET_RISK_MAX_STAKE', BASE_BET))
RISK_FIXED_STAKE = float(getattr(cfg, 'BET_RISK_FIXED_STAKE', BASE_BET))
RISK_KELLY_FRACTION = float(getattr(cfg, 'BET_RISK_KELLY_FRACTION', 0.25))
RISK_DRAWDOWN_THRESHOLD = float(getattr(cfg, 'BET_RISK_DRAWDOWN_THRESHOLD', 0.10))
RISK_DRAWDOWN_FRACTION_MULTIPLIER = float(
    getattr(cfg, 'BET_RISK_DRAWDOWN_FRACTION_MULTIPLIER', 0.50)
)
RISK_CORRELATION_CAP_ENABLED = bool(getattr(cfg, 'BET_RISK_CORRELATION_CAP_ENABLED', True))
RISK_MAX_EXPOSURE_PER_TEAM = float(getattr(cfg, 'BET_RISK_MAX_EXPOSURE_PER_TEAM', 200.0))
RISK_MAX_EXPOSURE_PER_CONFERENCE = float(getattr(cfg, 'BET_RISK_MAX_EXPOSURE_PER_CONFERENCE', 350.0))
RISK_MAX_EXPOSURE_PER_DAY = float(getattr(cfg, 'BET_RISK_MAX_EXPOSURE_PER_DAY', 800.0))

RISK_WEIGHT_CONSENSUS = float(getattr(cfg, 'BET_RISK_EDGE_QUALITY_WEIGHT_CONSENSUS', 0.50))
RISK_WEIGHT_CALIBRATION = float(getattr(cfg, 'BET_RISK_EDGE_QUALITY_WEIGHT_CALIBRATION', 0.25))
RISK_WEIGHT_COVERAGE = float(getattr(cfg, 'BET_RISK_EDGE_QUALITY_WEIGHT_COVERAGE', 0.25))
RISK_CALIBRATION_PROXY_DEFAULT = float(getattr(cfg, 'BET_RISK_CALIBRATION_PROXY_DEFAULT', 0.50))
RISK_COVERAGE_PROXY_DEFAULT = float(getattr(cfg, 'BET_RISK_COVERAGE_PROXY_DEFAULT', 0.50))


def kelly_fraction(win_prob: float, odds: int, fraction: float = 0.25) -> float:
    """
    Calculate Kelly bet size as fraction of bankroll.
    Uses fractional Kelly (default 25%) to reduce variance.

    Returns multiplier for base bet (0 to ~2x).
    """
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))

    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b if b > 0 else 0
    kelly_adj = kelly * fraction
    return max(0, min(2.0, kelly_adj))


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (100 + odds)
    return abs(odds) / (abs(odds) + 100)


def raw_kelly_fraction(win_prob: float, odds: int) -> float:
    """Full Kelly bankroll fraction (can be >1 for extreme edges, capped at 0+ here)."""
    if odds == 0:
        return 0.0
    if odds > 0:
        decimal_odds = 1 + (odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(odds))
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    p = win_prob
    q = 1 - p
    return max(0.0, (b * p - q) / b)


def get_drawdown_pct(bankroll: float, peak_bankroll: float) -> float:
    """Return drawdown as fraction of peak bankroll (0-1)."""
    if peak_bankroll <= 0:
        return 0.0
    return max(0.0, min(1.0, (peak_bankroll - bankroll) / peak_bankroll))


def drawdown_kelly_multiplier(
    drawdown_pct: float,
    threshold: float = RISK_DRAWDOWN_THRESHOLD,
    multiplier_below_threshold: float = RISK_DRAWDOWN_FRACTION_MULTIPLIER,
) -> float:
    """Step throttle for Kelly fraction once drawdown exceeds threshold."""
    if drawdown_pct > threshold:
        return max(0.0, min(1.0, multiplier_below_threshold))
    return 1.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def consensus_strength_score(models_agree: Optional[int], models_total: int = 10) -> float:
    """Normalize model consensus strength to 0-1; neutral if unavailable."""
    if models_agree is None or models_total <= 0:
        return 0.5
    return clamp01(models_agree / models_total)


def proxy_calibration_confidence(bet: dict) -> float:
    """
    Calibration confidence proxy (0-1).
    TODO: replace default with calibrated model confidence/uncertainty signal.
    """
    for key in ('calibration_confidence', 'calibration_score'):
        if key in bet and bet[key] is not None:
            return clamp01(bet[key])
    return clamp01(RISK_CALIBRATION_PROXY_DEFAULT)


def proxy_feature_coverage(bet: dict) -> float:
    """
    Feature coverage/data completeness proxy (0-1).
    TODO: wire to feature availability / missingness metrics from prediction pipeline.
    """
    if 'feature_coverage' in bet and bet['feature_coverage'] is not None:
        return clamp01(bet['feature_coverage'])
    if 'data_completeness' in bet and bet['data_completeness'] is not None:
        return clamp01(bet['data_completeness'])
    if 'missing_feature_pct' in bet and bet['missing_feature_pct'] is not None:
        return clamp01(1.0 - float(bet['missing_feature_pct']))
    return clamp01(RISK_COVERAGE_PROXY_DEFAULT)


def edge_quality_score(bet: dict) -> float:
    """Composite edge quality score (0-1) used as a risk scaler."""
    consensus = consensus_strength_score(bet.get('models_agree'), bet.get('models_total', 10))
    calibration = proxy_calibration_confidence(bet)
    coverage = proxy_feature_coverage(bet)

    total_w = RISK_WEIGHT_CONSENSUS + RISK_WEIGHT_CALIBRATION + RISK_WEIGHT_COVERAGE
    if total_w <= 0:
        return 0.5
    score = (
        consensus * RISK_WEIGHT_CONSENSUS
        + calibration * RISK_WEIGHT_CALIBRATION
        + coverage * RISK_WEIGHT_COVERAGE
    ) / total_w
    return clamp01(score)


def suggest_stake_for_bet(
    bet: dict,
    *,
    risk_mode: str = RISK_ENGINE_MODE,
    bankroll: float = RISK_BANKROLL,
    peak_bankroll: float = RISK_BANKROLL_PEAK,
    min_stake: float = RISK_MIN_STAKE,
    max_stake: float = RISK_MAX_STAKE,
    fixed_stake: float = RISK_FIXED_STAKE,
    kelly_fraction_cfg: float = RISK_KELLY_FRACTION,
) -> dict:
    """
    Return sizing details for a bet. Uses fractional Kelly for ML-style bets when enabled.
    Totals fall back to fixed stake unless a probability is available.
    """
    risk_score = edge_quality_score(bet)
    drawdown_pct = get_drawdown_pct(bankroll, peak_bankroll)
    drawdown_mult = drawdown_kelly_multiplier(drawdown_pct)
    kelly_fraction_used = 0.0

    if risk_mode != 'fractional_kelly':
        stake = fixed_stake
    else:
        odds = bet.get('moneyline', bet.get('odds'))
        model_prob = bet.get('model_prob')
        if odds is None or model_prob is None:
            stake = fixed_stake
        else:
            full_kelly = raw_kelly_fraction(float(model_prob), int(odds))
            kelly_fraction_used = full_kelly * kelly_fraction_cfg * drawdown_mult * risk_score
            stake = bankroll * kelly_fraction_used

    if stake > 0:
        stake = max(min_stake, min(max_stake, stake))
    stake = round(stake, 2) if stake > 0 else 0.0

    return {
        'risk_score': risk_score,
        'drawdown_pct': drawdown_pct,
        'drawdown_kelly_multiplier': drawdown_mult,
        'kelly_fraction_used': round(kelly_fraction_used, 6),
        'suggested_stake': stake,
    }


def bet_exposure_clusters(bet: dict) -> List[Tuple[str, str, float]]:
    """Return correlation exposure buckets for a bet."""
    clusters = []
    date_key = bet.get('date') or 'unknown-date'
    clusters.append(('day', f"day:{date_key}", RISK_MAX_EXPOSURE_PER_DAY))

    if bet['type'] in ('ML', 'CONSENSUS'):
        team_id = bet.get('pick_team_id')
        team_name = bet.get('pick_team_name')
        if team_id is not None or team_name:
            clusters.append(('team', f"team:{team_id or team_name}", RISK_MAX_EXPOSURE_PER_TEAM))

        conference = (
            bet.get('pick_conference')
            or bet.get('conference')
            or bet.get('pick_team_conference')
        )
        if conference:
            clusters.append(('conference', f"conference:{conference}", RISK_MAX_EXPOSURE_PER_CONFERENCE))

    return clusters


def apply_correlation_caps(results: dict) -> dict:
    """Reduce/reject bets that exceed aggregate exposure caps by cluster."""
    if not RISK_CORRELATION_CAP_ENABLED:
        for bet in results['bets']:
            bet['exposure_bucket'] = '|'.join([c[1] for c in bet_exposure_clusters(bet)])
        return results

    exposure_totals = {}
    kept = []

    for bet in results['bets']:
        requested = float(bet.get('suggested_stake', bet.get('bet_amount', 0)) or 0)
        clusters = bet_exposure_clusters(bet)
        bet['exposure_bucket'] = '|'.join([cluster_key for _, cluster_key, _ in clusters])

        remaining_caps = []
        for _, cluster_key, cluster_cap in clusters:
            used = exposure_totals.get(cluster_key, 0.0)
            remaining_caps.append(cluster_cap - used)
        max_allowable = min(remaining_caps) if remaining_caps else requested

        if max_allowable <= 0:
            results['rejections'].append({
                'type': bet['type'],
                'game': bet.get('pick_team_name') or bet.get('pick') or bet.get('game_id'),
                'edge': bet.get('edge'),
                'reasons': [f'correlation cap reached ({bet["exposure_bucket"]})'],
            })
            continue

        final_stake = min(requested, max_allowable)
        if RISK_ENGINE_MODE == 'fractional_kelly' and 0 < final_stake < RISK_MIN_STAKE:
            results['rejections'].append({
                'type': bet['type'],
                'game': bet.get('pick_team_name') or bet.get('pick') or bet.get('game_id'),
                'edge': bet.get('edge'),
                'reasons': [f'stake ${final_stake:.2f} below min ${RISK_MIN_STAKE:.2f} after correlation cap'],
            })
            continue

        bet['suggested_stake'] = round(final_stake, 2)
        bet['bet_amount'] = round(final_stake, 0)
        bet['correlation_capped'] = final_stake < requested

        for _, cluster_key, _ in clusters:
            exposure_totals[cluster_key] = exposure_totals.get(cluster_key, 0.0) + final_stake

        kept.append(bet)

    results['bets'] = kept
    return results

