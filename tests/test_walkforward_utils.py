import numpy as np

from scripts.walkforward_utils import build_strict_date_folds, aggregate_binary_oof


def test_folds_chronological_and_non_leaky():
    dates = np.array([
        "2026-01-01", "2026-01-01",
        "2026-01-02",
        "2026-01-03", "2026-01-03",
        "2026-01-04",
    ], dtype=object)

    folds = build_strict_date_folds(dates, min_warmup=2)
    assert [f.test_date for f in folds] == ["2026-01-02", "2026-01-03", "2026-01-04"]

    for fold in folds:
        train_dates = dates[fold.train_idx]
        test_dates = dates[fold.test_idx]
        assert np.all(train_dates < fold.test_date)
        assert np.all(test_dates == fold.test_date)


def test_no_train_row_has_date_gte_test_date():
    dates = np.array([
        "2026-02-01", "2026-02-01", "2026-02-02", "2026-02-02", "2026-02-03"
    ], dtype=object)

    folds = build_strict_date_folds(dates, min_warmup=1)
    assert len(folds) >= 1

    for fold in folds:
        for tr_i in fold.train_idx:
            for te_i in fold.test_idx:
                assert dates[tr_i] < dates[te_i]


def test_metric_aggregation_deterministic_on_toy_data():
    y_true = np.array([0, 1, 1, 0], dtype=np.int32)
    y_prob = np.array([0.1, 0.8, 0.7, 0.3], dtype=np.float64)

    m1 = aggregate_binary_oof(y_true, y_prob)
    m2 = aggregate_binary_oof(y_true, y_prob)

    assert m1 == m2
    assert m1["n"] == 4
    assert abs(m1["accuracy"] - 1.0) < 1e-12
    expected_brier = np.mean((y_prob - y_true) ** 2)
    assert abs(m1["brier"] - expected_brier) < 1e-12
    assert m1["logloss"] > 0
