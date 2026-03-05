import sqlite3

from models.meta_ensemble import MetaEnsemble, MODEL_NAMES


def _create_meta_db(tmp_path, with_provenance=True):
    db_path = tmp_path / ("meta_with_prov.db" if with_provenance else "meta_legacy.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("CREATE TABLE games (id TEXT PRIMARY KEY, date TEXT, time TEXT, home_team_id TEXT, away_team_id TEXT, home_score INTEGER, away_score INTEGER, status TEXT)")
    c.execute("CREATE TABLE teams (id TEXT PRIMARY KEY, conference TEXT, current_rank INTEGER)")
    c.execute("CREATE TABLE elo_ratings (team_id TEXT PRIMARY KEY, rating REAL)")
    c.execute("CREATE TABLE pear_ratings (team_id TEXT PRIMARY KEY, rating REAL)")
    c.execute("CREATE TABLE team_rpi (team_id TEXT PRIMARY KEY, sams_rpi REAL)")

    if with_provenance:
        c.execute(
            """
            CREATE TABLE model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                model_name TEXT,
                predicted_home_prob REAL,
                predicted_at TEXT,
                prediction_source TEXT,
                was_correct INTEGER
            )
            """
        )
    else:
        c.execute(
            """
            CREATE TABLE model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                model_name TEXT,
                predicted_home_prob REAL,
                predicted_at TEXT,
                was_correct INTEGER
            )
            """
        )

    for t in ["h1", "a1", "h2", "a2", "h3", "a3"]:
        c.execute("INSERT INTO teams (id, conference, current_rank) VALUES (?, 'SEC', NULL)", (t,))
        c.execute("INSERT INTO elo_ratings (team_id, rating) VALUES (?, 1500)", (t,))
        c.execute("INSERT INTO pear_ratings (team_id, rating) VALUES (?, 0.5)", (t,))
        c.execute("INSERT INTO team_rpi (team_id, sams_rpi) VALUES (?, 0.5)", (t,))

    c.execute("INSERT INTO games VALUES ('g_live', '2026-03-01', '18:00:00', 'h1', 'a1', 5, 3, 'final')")
    c.execute("INSERT INTO games VALUES ('g_backfill', '2026-03-01', '18:00:00', 'h2', 'a2', 2, 4, 'final')")
    c.execute("INSERT INTO games VALUES ('g_nogametime', '2026-03-01', NULL, 'h3', 'a3', 6, 2, 'final')")

    def insert_row(game_id, model_name, p, ts, src="live"):
        if with_provenance:
            c.execute(
                "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at, prediction_source, was_correct) VALUES (?,?,?,?,?,1)",
                (game_id, model_name, p, ts, src),
            )
        else:
            c.execute(
                "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at, was_correct) VALUES (?,?,?,?,1)",
                (game_id, model_name, p, ts),
            )

    for i, m in enumerate(MODEL_NAMES[:7]):
        insert_row("g_live", m, 0.51 + i * 0.001, "2026-03-01 12:00:00", "live")
        insert_row("g_backfill", m, 0.45 + i * 0.001, "2026-03-03 12:00:00", "backfill")
        insert_row("g_nogametime", m, 0.62, "2026-03-01 20:00:00", "live")

    conn.commit()
    conn.close()
    return db_path


def test_extract_training_data_excludes_backfill_and_postgame(monkeypatch, tmp_path):
    db_path = _create_meta_db(tmp_path, with_provenance=True)
    m = MetaEnsemble()

    def _new_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(m, "_get_connection", _new_conn)

    rows, columns = m._extract_training_data()
    game_ids = {r[0] for r in rows}

    assert "g_live" in game_ids
    assert "g_backfill" not in game_ids
    assert "g_nogametime" in game_ids  # fallback to end-of-day works

    report = m.cohort_integrity_report()
    assert report["excluded_by_source"] > 0
    assert report["final_training_games"] == 2


def test_extract_training_data_legacy_schema_without_provenance(monkeypatch, tmp_path):
    db_path = _create_meta_db(tmp_path, with_provenance=False)

    conn = sqlite3.connect(db_path)
    # Add a clear postgame leak row set (no provenance column exists)
    conn.execute("INSERT INTO games VALUES ('g_late', '2026-03-01', '10:00:00', 'h1', 'a2', 1, 0, 'final')")
    for i, mname in enumerate(MODEL_NAMES[:7]):
        conn.execute(
            "INSERT INTO model_predictions (game_id, model_name, predicted_home_prob, predicted_at, was_correct) VALUES (?,?,?,?,1)",
            ("g_late", mname, 0.55 + i * 0.001, "2026-03-01 11:30:00"),
        )
    conn.commit()
    conn.close()

    m = MetaEnsemble()

    def _new_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(m, "_get_connection", _new_conn)

    rows, _ = m._extract_training_data()
    ids = {r[0] for r in rows}
    assert "g_late" not in ids  # timestamp leak guard still works on legacy schema
    assert "g_live" in ids
