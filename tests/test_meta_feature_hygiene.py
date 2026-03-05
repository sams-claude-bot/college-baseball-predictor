import inspect
import sqlite3

import numpy as np

from models.meta_ensemble import MetaEnsemble, MODEL_NAMES


def test_feature_names_are_leak_safe_only():
    meta = MetaEnsemble()
    expected = [f"{m}_prob" for m in MODEL_NAMES] + [
        "models_predicting_home",
        "avg_home_prob",
        "prob_spread",
    ]

    columns = [
        "game_id", "date", "home_team_id", "away_team_id",
        "elo_prob", "pythagorean_prob", "lightgbm_prob", "poisson_prob",
        "xgboost_prob", "pitching_prob", "pear_prob", "quality_prob",
        "neural_prob", "venue_prob", "rest_travel_prob", "upset_prob",
        "home_won",
    ]
    row = (
        "g1", "2026-03-01", "h", "a",
        0.55, 0.51, 0.5, 0.49, 0.57, 0.53, 0.59, 0.52, 0.56, 0.54, 0.5, 0.47,
        1,
    )
    _, _, _, feature_names = meta._build_features([row], columns)

    assert feature_names == expected


def test_extract_query_has_no_current_state_dependencies():
    source = inspect.getsource(MetaEnsemble._extract_training_data)

    assert "elo_ratings" not in source
    assert "pear_ratings" not in source
    assert "team_rpi" not in source
    assert "current_rank" not in source


def test_predict_defaults_missing_base_probs_to_half(monkeypatch, tmp_path):
    db_path = tmp_path / "meta_predict.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("CREATE TABLE games (id TEXT PRIMARY KEY, date TEXT, home_team_id TEXT, away_team_id TEXT)")
    c.execute(
        """
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            model_name TEXT,
            predicted_home_prob REAL,
            predicted_at TEXT
        )
        """
    )
    c.execute("INSERT INTO games VALUES ('g1', '2026-03-01', 'h1', 'a1')")
    c.execute(
        "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at) VALUES ('g1', 'elo', 0.7, '2026-03-01 10:00:00')"
    )
    c.execute(
        "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at) VALUES ('g1', 'pear', 0.4, '2026-03-01 10:00:00')"
    )
    conn.commit()
    conn.close()

    class DummyModel:
        def __init__(self):
            self.seen = None

        def predict_proba(self, arr):
            self.seen = arr
            return np.array([[0.3, 0.7]])

    meta = MetaEnsemble()
    dummy = DummyModel()
    meta.xgb_model = dummy
    meta.lr_model = None
    meta.feature_names = [f"{m}_prob" for m in MODEL_NAMES] + [
        "models_predicting_home",
        "avg_home_prob",
        "prob_spread",
    ]
    monkeypatch.setattr(meta, "_load", lambda: True)

    def _new_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(meta, "_get_connection", _new_conn)

    prob = meta.predict(game_id="g1")
    assert 0 <= prob <= 1
    assert dummy.seen is not None
    assert dummy.seen.shape == (1, len(MODEL_NAMES) + 3)
    assert dummy.seen[0, MODEL_NAMES.index("elo")] == 0.7
    assert dummy.seen[0, MODEL_NAMES.index("pear")] == 0.4
    assert dummy.seen[0, MODEL_NAMES.index("pythagorean")] == 0.5
    assert dummy.seen[0, MODEL_NAMES.index("upset")] == 0.5
