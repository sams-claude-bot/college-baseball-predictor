import sqlite3
from pathlib import Path

from scripts import database
from scripts import predict_and_track
from scripts import backfill_new_models


def _base_conn(tmp_path: Path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("CREATE TABLE teams (id TEXT PRIMARY KEY, name TEXT)")
    c.execute(
        """
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT,
            time TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            status TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            model_name TEXT,
            predicted_home_prob REAL,
            raw_home_prob REAL,
            predicted_home_runs REAL,
            predicted_away_runs REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            prediction_source TEXT NOT NULL DEFAULT 'live',
            prediction_context TEXT,
            was_correct INTEGER,
            UNIQUE(game_id, model_name)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            over_under REAL,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE totals_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            over_under_line REAL,
            projected_total REAL,
            prediction TEXT,
            model_name TEXT,
            over_prob REAL,
            under_prob REAL
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def test_migration_adds_provenance_and_conservative_backfill(tmp_path):
    conn = sqlite3.connect(tmp_path / "legacy.db")
    c = conn.cursor()
    c.execute("CREATE TABLE games (id TEXT PRIMARY KEY, date TEXT)")
    c.execute(
        """
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            model_name TEXT,
            predicted_home_prob REAL,
            predicted_at TEXT,
            UNIQUE(game_id, model_name)
        )
        """
    )
    c.execute("INSERT INTO games (id, date) VALUES ('g1', '2026-03-01')")
    c.execute("INSERT INTO games (id, date) VALUES ('g2', '2026-03-01')")
    c.execute(
        "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at) VALUES ('g1','venue',0.6,'2026-03-03 12:00:00')"
    )
    c.execute(
        "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at) VALUES ('g2','elo',0.6,'2026-03-01 10:00:00')"
    )

    database._ensure_prediction_provenance_schema(conn)
    conn.commit()

    cols = {r[1] for r in conn.execute("PRAGMA table_info(model_predictions)").fetchall()}
    assert "prediction_source" in cols
    assert "prediction_context" in cols

    rows = conn.execute(
        "SELECT game_id, model_name, prediction_source FROM model_predictions ORDER BY game_id"
    ).fetchall()
    assert rows[0][2] == "backfill"  # delayed venue row
    assert rows[1][2] == "live"      # leave non-backfill model untouched

    idx = {r[1] for r in conn.execute("PRAGMA index_list(model_predictions)").fetchall()}
    assert "idx_model_pred_source" in idx
    assert "idx_model_pred_train_filter" in idx
    conn.close()


def test_predict_and_backfill_tag_sources(monkeypatch, tmp_path):
    db_path = _base_conn(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("INSERT INTO teams (id, name) VALUES ('home', 'Home')")
    c.execute("INSERT INTO teams (id, name) VALUES ('away', 'Away')")
    c.execute(
        "INSERT INTO games (id, date, time, home_team_id, away_team_id, status) VALUES ('g1','2099-01-01','12:00:00','home','away','scheduled')"
    )
    conn.commit()

    def _new_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(predict_and_track, "get_connection", _new_conn)
    monkeypatch.setattr(predict_and_track, "MODEL_NAMES", ["elo", "meta_ensemble"])

    class DummyPredictor:
        def __init__(self, model=None):
            self.model = model

        def predict_game(self, home_id, away_id):
            return {"home_win_probability": 0.6, "projected_home_runs": 5.0, "projected_away_runs": 4.0}

    class DummyMeta:
        def predict(self, game_id=None, home_team_id=None, away_team_id=None):
            return 0.55

    class DummySlim:
        def is_trained(self):
            return False

    monkeypatch.setattr(predict_and_track, "Predictor", DummyPredictor)
    monkeypatch.setattr(predict_and_track, "SlimTotalsModel", DummySlim)
    monkeypatch.setattr("models.meta_ensemble.MetaEnsemble", DummyMeta)

    predict_and_track.predict_games(date="2099-01-01", refresh_existing=False)
    rows = conn.execute("SELECT model_name, prediction_source FROM model_predictions ORDER BY model_name").fetchall()
    assert all(r[1] == "live" for r in rows)

    predict_and_track.predict_games(date="2099-01-01", refresh_existing=True)
    rows = conn.execute("SELECT model_name, prediction_source FROM model_predictions ORDER BY model_name").fetchall()
    assert all(r[1] == "refresh" for r in rows)

    monkeypatch.setattr(backfill_new_models, "get_connection", _new_conn)

    class BFModel:
        def predict_game(self, home_id, away_id):
            return {"home_win_probability": 0.42}

    # force backfill by deleting one model row
    conn.execute("DELETE FROM model_predictions WHERE model_name='elo'")
    conn.execute("UPDATE games SET status='final' WHERE id='g1'")
    conn.commit()

    backfill_new_models.backfill("elo", BFModel, batch_size=1)
    src = conn.execute("SELECT prediction_source FROM model_predictions WHERE game_id='g1' AND model_name='elo'").fetchone()[0]
    assert src == "backfill"
