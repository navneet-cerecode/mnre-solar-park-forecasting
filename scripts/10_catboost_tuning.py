"""Step 10 / E10 - CatBoost on the GPU and a small LightGBM tuning grid.

Tuning uses only the calib split (never test); the chosen setting is then scored once on test.
Output: reports/results/e10_catboost_tuning.csv
"""
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from importlib import import_module

from solarpark import config
from solarpark.features import FEATURES

t6 = import_module("06_train_models")
DAY = 5.0
GRID = [dict(num_leaves=n, learning_rate=lr, min_data_in_leaf=m) for n in (31, 127, 255) for lr in (0.02, 0.05) for m in (50, 200)]


def nrmse(y, p):
    return 100 * float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


rows = []
for h in config.HORIZONS_H:
    df = pd.read_parquet(config.PROCESSED / f"model_h{h}.parquet")
    tr, va, te = df[df.split == "train"], df[df.split == "calib"], df[df.split == "test"]
    vd, td = va[va.elevation > DAY], te[te.elevation > DAY]

    t0 = time.time()
    cb = CatBoostRegressor(iterations=4000, learning_rate=0.05, depth=8, loss_function="RMSE", task_type="GPU",
                           od_type="Iter", od_wait=100, random_seed=42, verbose=False)
    cb.fit(tr[FEATURES], tr.y, cat_features=["park_code"], eval_set=(va[FEATURES], va.y))
    rows.append({"h": h, "model": "CatBoost (GPU)", "setting": f"depth 8, {cb.get_best_iteration()} trees",
                 "calib_nRMSE_%": nrmse(vd.y, t6.clip(cb.predict(vd[FEATURES]), vd)),
                 "test_nRMSE_%": nrmse(td.y, t6.clip(cb.predict(td[FEATURES]), td)), "seconds": time.time() - t0})

    best = None
    for g in GRID:
        m = t6.fit_lgbm(tr, va, FEATURES, dict(t6.LGB_PARAMS, **g))
        score = nrmse(vd.y, t6.clip(m.predict(vd[FEATURES], num_iteration=m.best_iteration), vd))
        if best is None or score < best[0]:
            best = (score, g, m)
    score, g, m = best
    rows.append({"h": h, "model": "LightGBM tuned (12-setting grid on calib)", "setting": str(g),
                 "calib_nRMSE_%": score, "test_nRMSE_%": nrmse(td.y, t6.clip(m.predict(td[FEATURES], num_iteration=m.best_iteration), td)),
                 "seconds": np.nan})
    d = t6.fit_lgbm(tr, va, FEATURES)
    rows.append({"h": h, "model": "LightGBM default (E1)", "setting": "num_leaves 127, lr 0.05, min_data 50",
                 "calib_nRMSE_%": nrmse(vd.y, t6.clip(d.predict(vd[FEATURES], num_iteration=d.best_iteration), vd)),
                 "test_nRMSE_%": nrmse(td.y, t6.clip(d.predict(td[FEATURES], num_iteration=d.best_iteration), td)), "seconds": np.nan})
    print(pd.DataFrame(rows[-3:]).round(3).to_string(index=False), flush=True)

pd.DataFrame(rows).to_csv(config.REPORTS / "results" / "e10_catboost_tuning.csv", index=False)
