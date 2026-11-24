"""Step 6 / Experiment E1 - baselines vs Ridge vs Random Forest vs XGBoost (GPU) vs LightGBM.

Train on the train split, early-stop on calib, report on test (daylight hours).
Outputs: data/processed/preds_h{h}.parquet, models/lgbm_h{h}.txt, reports/results/e1_models.csv
"""
import sys
import time

import _bootstrap  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solarpark import config, evaluate
from solarpark.features import FEATURES

RES = config.REPORTS / "results"; RES.mkdir(parents=True, exist_ok=True)
MOD = config.ROOT / "models"; MOD.mkdir(exist_ok=True)
KEEP = ["park_id", "loc_id", "state", "region", "tier", "capacity_mw", "time_utc", "split", "elevation", "pcs", "y",
        "bl_persistence", "bl_smart", "bl_nwp_physics"]

LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)


def fit_lgbm(tr, va, feats, params=LGB_PARAMS, rounds=3000):
    dtr = lgb.Dataset(tr[feats], tr.y, categorical_feature=["park_code"] if "park_code" in feats else "auto")
    dva = lgb.Dataset(va[feats], va.y, reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def clip(p, d):
    return np.where(d.pcs > 0, np.clip(p, 0, 1), 0.0)


if __name__ == "__main__":
    allres = []
    HS = config.horizons_from_argv(sys.argv)
    for h in HS:
        df = pd.read_parquet(config.PROCESSED / f"model_h{h}.parquet")
        tr, va = df[df.split == "train"], df[df.split == "calib"]
        ev = df[df.split != "train"].copy()
        out = ev[KEEP].copy()

        # climatology: mean output for (park, month, hour) in the training period
        key = lambda d: [d.park_id, d.time_utc.dt.month, d.time_utc.dt.hour]
        clim = tr.groupby(key(tr)).y.mean()
        out["bl_climatology"] = clim.reindex(pd.MultiIndex.from_arrays(key(ev))).to_numpy()

        X = lambda d: d[FEATURES].fillna(-1)
        t0 = time.time()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X(tr), tr.y)
        out["ridge"] = clip(ridge.predict(X(ev)), ev)
        rf = RandomForestRegressor(n_estimators=150, max_samples=0.3, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=42)
        rf.fit(X(tr), tr.y)
        out["random_forest"] = clip(rf.predict(X(ev)), ev)
        t_rf = time.time() - t0

        t0 = time.time()
        xg = xgb.train(dict(device="cuda", tree_method="hist", max_depth=8, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=20, eval_metric="rmse", seed=42),
                       xgb.DMatrix(tr[FEATURES], tr.y), 3000, evals=[(xgb.DMatrix(va[FEATURES], va.y), "val")],
                       early_stopping_rounds=100, verbose_eval=False)
        out["xgboost_gpu"] = clip(xg.predict(xgb.DMatrix(ev[FEATURES]), iteration_range=(0, xg.best_iteration + 1)), ev)
        t_xgb = time.time() - t0

        t0 = time.time()
        gbm = fit_lgbm(tr, va, FEATURES)
        out["lightgbm"] = clip(gbm.predict(ev[FEATURES], num_iteration=gbm.best_iteration), ev)
        gbm.save_model(str(MOD / f"lgbm_h{h}.txt"))
        t_lgb = time.time() - t0

        out.to_parquet(config.PROCESSED / f"preds_h{h}.parquet", index=False)
        models = ["bl_persistence", "bl_smart", "bl_climatology", "bl_nwp_physics", "ridge", "random_forest", "xgboost_gpu", "lightgbm"]
        test = out[out.split == "test"]
        r = evaluate.table(test, models); r["h"] = h; r["subset"] = "all 43 parks"
        rc = evaluate.table(test[test.tier == "U2_core"], models); rc["h"] = h; rc["subset"] = "18 core parks"
        allres += [r, rc]
        print(f"\n=== h = {h} h  (RF+ridge {t_rf:.0f}s, XGBoost-GPU {t_xgb:.0f}s [{xg.best_iteration} rounds], LightGBM {t_lgb:.0f}s [{gbm.best_iteration} rounds])")
        print(r[["model", "nMAE_%", "nRMSE_%", "MBE_%", "skill_vs_smart"]].round(3).to_string(index=False), flush=True)

    res = pd.concat(allres)
    old = RES / "e1_models.csv"
    if old.exists():   # keep results of horizons not re-run
        prev = pd.read_csv(old)
        res = pd.concat([prev[~prev.h.isin(HS)], res])
    res.sort_values(["h", "subset"]).to_csv(RES / "e1_models.csv", index=False)
    print("\nwrote", RES / "e1_models.csv")
