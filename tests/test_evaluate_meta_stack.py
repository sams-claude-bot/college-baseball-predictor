from scripts.evaluate_meta_stack import (
    apply_leak_safe_filter,
    build_intersection,
    compute_metrics,
)


def _base_row(**overrides):
    row = {
        "game_id": "g1",
        "model_name": "meta_ensemble",
        "predicted_home_prob": 0.6,
        "predicted_at": "2026-03-01 14:00:00",
        "prediction_source": "live",
        "game_date": "2026-03-01",
        "game_time": "15:00:00",
        "home_won": 1,
    }
    row.update(overrides)
    return row


def test_cohort_filter_excludes_backfill():
    rows = [
        _base_row(game_id="g1", prediction_source="backfill"),
        _base_row(game_id="g2", prediction_source="live"),
    ]
    kept, stats = apply_leak_safe_filter(rows)
    assert len(kept) == 1
    assert kept[0]["game_id"] == "g2"
    assert stats.excluded_backfill == 1


def test_cohort_filter_excludes_late_predictions():
    # kickoff at 15:00, cutoff is 14:55:00, so 14:55:01 is too late
    late = "2026-03-01 14:55:01"
    rows = [
        _base_row(game_id="g1", predicted_at=late),
        _base_row(game_id="g2", predicted_at="2026-03-01 14:54:59"),
    ]
    kept, stats = apply_leak_safe_filter(rows)
    assert len(kept) == 1
    assert kept[0]["game_id"] == "g2"
    assert stats.excluded_late == 1


def test_intersection_cohort_builder_correctness():
    rows = [
        _base_row(game_id="g1", model_name="meta_ensemble"),
        _base_row(game_id="g2", model_name="meta_ensemble"),
        _base_row(game_id="g1", model_name="elo"),
        _base_row(game_id="g3", model_name="elo"),
        _base_row(game_id="g1", model_name="pythagorean"),
        _base_row(game_id="g2", model_name="pythagorean"),
    ]
    by_model = build_intersection(rows, ["meta_ensemble", "elo", "pythagorean"])
    assert sorted(r["game_id"] for r in by_model["meta_ensemble"]) == ["g1"]
    assert sorted(r["game_id"] for r in by_model["elo"]) == ["g1"]
    assert sorted(r["game_id"] for r in by_model["pythagorean"]) == ["g1"]


def test_metric_computation_sanity():
    rows = [
        _base_row(game_id="g1", predicted_home_prob=0.9, home_won=1),
        _base_row(game_id="g2", predicted_home_prob=0.8, home_won=1),
        _base_row(game_id="g3", predicted_home_prob=0.7, home_won=0),
        _base_row(game_id="g4", predicted_home_prob=0.2, home_won=0),
    ]
    metrics = compute_metrics(rows)
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.75
    assert round(metrics["brier"], 4) == 0.1450
    assert round(metrics["log_loss"], 4) == 0.4389
