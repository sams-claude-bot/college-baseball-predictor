from scripts.replay_uplift_benchmark import (
    apply_leak_safe_filter,
    build_trusted_comparison_rows,
    compute_flip_analysis,
    compute_probability_metrics,
)


def _row(**overrides):
    base = {
        "game_id": "g1",
        "model_name": "meta_ensemble",
        "predicted_home_prob": 0.60,
        "predicted_at": "2026-03-01 14:00:00",
        "prediction_source": "live",
        "game_date": "2026-03-01",
        "game_time": "15:00:00",
        "home_team_id": "home-a",
        "away_team_id": "away-a",
        "is_neutral_site": 0,
        "home_won": 1,
    }
    base.update(overrides)
    return base


def test_cohort_filter_sanity_excludes_backfill_and_late():
    rows = [
        _row(game_id="g1", model_name="meta_ensemble", prediction_source="backfill"),
        _row(game_id="g2", model_name="meta_ensemble", predicted_at="2026-03-01 14:55:01"),
        _row(game_id="g3", model_name="meta_ensemble", predicted_at="2026-03-01 14:54:59"),
    ]

    kept, stats = apply_leak_safe_filter(rows)

    assert [r["game_id"] for r in kept] == ["g3"]
    assert stats.excluded_backfill == 1
    assert stats.excluded_late == 1


def test_baseline_candidate_alignment_on_same_game_ids():
    filtered_rows = [
        _row(game_id="g1", model_name="meta_ensemble", predicted_home_prob=0.62, home_won=1),
        _row(game_id="g1", model_name="elo", predicted_home_prob=0.70, home_won=1),
        _row(game_id="g1", model_name="pythagorean", predicted_home_prob=0.55, home_won=1),
        _row(game_id="g2", model_name="meta_ensemble", predicted_home_prob=0.48, home_won=0),
        _row(game_id="g2", model_name="elo", predicted_home_prob=0.40, home_won=0),
    ]

    def stub_score(probs):
        # deterministic candidate from stored base probs only
        return probs.get("elo", 0.5)

    pairs = build_trusted_comparison_rows(filtered_rows, stub_score)

    assert [p["game_id"] for p in pairs] == ["g1", "g2"]
    assert [p["baseline_meta"] for p in pairs] == [0.62, 0.48]
    assert [p["candidate_meta"] for p in pairs] == [0.70, 0.40]


def test_metric_calculations_deterministic():
    pairs = [
        {"game_id": "g1", "home_won": 1, "baseline_meta": 0.9, "candidate_meta": 0.8},
        {"game_id": "g2", "home_won": 1, "baseline_meta": 0.8, "candidate_meta": 0.7},
        {"game_id": "g3", "home_won": 0, "baseline_meta": 0.7, "candidate_meta": 0.6},
        {"game_id": "g4", "home_won": 0, "baseline_meta": 0.2, "candidate_meta": 0.1},
    ]

    baseline = compute_probability_metrics(pairs, "baseline_meta")
    candidate = compute_probability_metrics(pairs, "candidate_meta")

    assert baseline["n"] == 4
    assert baseline["accuracy"] == 0.75
    assert round(baseline["brier"], 4) == 0.1450
    assert round(baseline["log_loss"], 4) == 0.4389

    assert candidate["n"] == 4
    assert candidate["accuracy"] == 0.75
    assert round(candidate["brier"], 4) == 0.1250
    assert round(candidate["log_loss"], 4) == 0.4004


def test_flip_accounting_correctness():
    pairs = [
        # flip and improve (wrong -> correct)
        {"game_id": "g1", "home_won": 1, "baseline_meta": 0.49, "candidate_meta": 0.51},
        # flip and worsen (correct -> wrong)
        {"game_id": "g2", "home_won": 1, "baseline_meta": 0.52, "candidate_meta": 0.48},
        # no flip
        {"game_id": "g3", "home_won": 0, "baseline_meta": 0.30, "candidate_meta": 0.20},
    ]

    flips = compute_flip_analysis(pairs)

    assert flips["side_flips"] == 2
    assert flips["net_correct_change_from_flips"] == 0
