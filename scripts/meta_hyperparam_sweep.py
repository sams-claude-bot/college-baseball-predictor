#!/usr/bin/env python3
"""Meta-ensemble XGB hyperparameter sweep with walk-forward validation."""
import numpy as np
import sqlite3
import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

MODEL_NAMES = ['prior', 'neural', 'elo', 'ensemble', 'pythagorean', 'lightgbm',
               'poisson', 'conference', 'xgboost', 'advanced', 'log5', 'pitching', 'pear', 'quality']

conn = sqlite3.connect('/home/sam/college-baseball-predictor/data/baseball.db')
conn.row_factory = sqlite3.Row

rows = list(conn.execute("""
    SELECT g.date,
        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_won,
        """ + ",\n        ".join(
            f"MAX(CASE WHEN mp.model_name='{m}' THEN mp.predicted_home_prob END) as {m}_prob"
            for m in MODEL_NAMES
        ) + """,
        eh.rating as home_elo, ea.rating as away_elo
    FROM games g
    JOIN model_predictions mp ON g.id = mp.game_id
    LEFT JOIN elo_ratings eh ON g.home_team_id = eh.team_id
    LEFT JOIN elo_ratings ea ON g.away_team_id = ea.team_id
    WHERE g.status = 'final' AND g.home_score IS NOT NULL
    GROUP BY g.id HAVING COUNT(DISTINCT mp.model_name) >= 10
    ORDER BY g.date
"""))

# Minimal feature set: 14 probs + elo_diff
X = np.array([[r[f'{m}_prob'] or 0.5 for m in MODEL_NAMES] +
              [(r['home_elo'] or 1500) - (r['away_elo'] or 1500)] for r in rows])
y = np.array([r['home_won'] for r in rows])
dates = [r['date'] for r in rows]
ud = sorted(set(dates))

# Pre-compute masks
masks_train = []
masks_test = []
for i in range(3, len(ud)):
    tm = np.array([d < ud[i] for d in dates])
    vm = np.array([d == ud[i] for d in dates])
    if tm.sum() >= 20 and vm.sum() > 0:
        masks_train.append(tm)
        masks_test.append(vm)

print(f"Games: {len(rows)}, Features: {X.shape[1]}, Folds: {len(masks_train)}")

def test(params):
    px, tr = [], []
    for tm, vm in zip(masks_train, masks_test):
        xgb = XGBClassifier(**params, eval_metric='logloss', verbosity=0,
                           random_state=42, n_jobs=1)
        xgb.fit(X[tm], y[tm])
        px.extend(xgb.predict_proba(X[vm])[:, 1])
        tr.extend(y[vm])
    t, p = np.array(tr), np.array(px)
    return accuracy_score(t, (p > .5).astype(int)), log_loss(t, p)

configs = []
for n in [30, 50, 75, 100, 150, 200]:
    for d in [2, 3, 4]:
        for lr in [0.01, 0.03, 0.05, 0.1]:
            for ss in [0.7, 0.85, 1.0]:
                configs.append({'n_estimators': n, 'max_depth': d,
                               'learning_rate': lr, 'subsample': ss})

print(f"Testing {len(configs)} configs...")
start = time.time()

results = []
for i, cfg in enumerate(configs):
    acc, ll = test(cfg)
    results.append((acc, ll, cfg))
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (len(configs) - i - 1)
        print(f"  {i+1}/{len(configs)} done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

print(f"\nDone in {time.time()-start:.0f}s\n")

results.sort(key=lambda x: x[1])
print("=== TOP 10 by Log Loss ===")
for acc, ll, p in results[:10]:
    print(f"  {acc:.1%} / {ll:.4f} | n={p['n_estimators']} d={p['max_depth']} lr={p['learning_rate']} ss={p['subsample']}")

results.sort(key=lambda x: -x[0])
print("\n=== TOP 10 by Accuracy ===")
for acc, ll, p in results[:10]:
    print(f"  {acc:.1%} / {ll:.4f} | n={p['n_estimators']} d={p['max_depth']} lr={p['learning_rate']} ss={p['subsample']}")

print("\n=== CURRENT DEFAULT ===")
acc, ll = test({'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1})
print(f"  {acc:.1%} / {ll:.4f}")
