"""Step 7 - experiments E2-E9 with LightGBM (the planned main model).

E2 per-park vs global | E3 regional vs global | E4 which inputs matter (ablation + SHAP)
E5 aggregation park -> state -> region -> all | E6 ramps and weather regimes
E7 unseen locations (leave-one-location-out) | E8 prediction intervals (CQR) | E9 perfect-weather upper bound
Usage: python scripts/07_experiments.py [E2 E3 ...]   (default: all)
"""
import sys

import _bootstrap  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd

from solarpark import config, evaluate
from solarpark.features import FEATURES, GEOM, META, NWP, ORACLE, PAST, NWP_VARS
from importlib import import_module

train_mod = import_module("06_train_models")
fit_lgbm, clip, LGB_PARAMS = train_mod.fit_lgbm, train_mod.clip, train_mod.LGB_PARAMS
RES = config.REPORTS / "results"
H = config.horizons_from_argv(sys.argv)
WANT = {a for a in sys.argv[1:] if not a.startswith("h=")} or {f"E{i}" for i in range(2, 10)}
DAY = evaluate.DAYLIGHT_ELEV


def data(h):
    return pd.read_parquet(config.PROCESSED / f"model_h{h}.parquet")


def preds(h):
    return pd.read_parquet(config.PROCESSED / f"preds_h{h}.parquet")


def nrmse(y, p):
    return 100 * float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def save(df, name):
    """Merge with an existing table so re-running one horizon keeps the others (long tables with an h column)."""
    p = RES / f"{name}.csv"
    if "h" in df.columns and p.exists():
        prev = pd.read_csv(p)
        if "h" in prev.columns and set(prev.columns) == set(df.columns):
            df = pd.concat([prev[~prev.h.isin(df.h.unique())], df[prev.columns]]).sort_values("h", kind="stable")
    df.to_csv(p, index=False)
    print(f"\n--- {name} ---\n{df.round(3).to_string(index=False)}", flush=True)


# ------------------------------------------------------------------ E2 / E3
def per_group_models(group_col, label):
    rows = []
    for h in H:
        df, pr = data(h), preds(h)
        feats = [f for f in FEATURES if f not in ("park_code", "region_code")] if group_col == "park_id" else [f for f in FEATURES if f != "region_code"]
        for g, d in df.groupby(group_col):
            tr, va, te = d[d.split == "train"], d[d.split == "calib"], d[d.split == "test"]
            m = fit_lgbm(tr, va, feats, dict(LGB_PARAMS, num_leaves=31, min_data_in_leaf=20))
            te = te[te.elevation > DAY]
            local = clip(m.predict(te[feats], num_iteration=m.best_iteration), te)
            glob = pr.set_index(["park_id", "time_utc"]).loc[pd.MultiIndex.from_frame(te[["park_id", "time_utc"]]), "lightgbm"].to_numpy()
            rows.append({group_col: g, "h": h, f"{label}_nRMSE_%": nrmse(te.y, local), "global_nRMSE_%": nrmse(te.y, glob), "n": len(te)})
    return pd.DataFrame(rows)


if "E2" in WANT:
    e2 = per_group_models("park_id", "per_park")
    save(e2, "e2_per_park_vs_global_detail")
    e2["global_wins"] = e2["global_nRMSE_%"] < e2["per_park_nRMSE_%"]
    save(e2.groupby("h").agg(per_park=("per_park_nRMSE_%", "mean"), global_model=("global_nRMSE_%", "mean"),
                             global_wins=("global_wins", "sum"), parks=("park_id", "count")).reset_index(), "e2_per_park_vs_global")

if "E3" in WANT:
    e3 = per_group_models("region", "regional")
    save(e3, "e3_regional_vs_global")

# ------------------------------------------------------------------ E4
if "E4" in WANT:
    groups = {"full (all inputs)": FEATURES, "no recent power (weather only)": [f for f in FEATURES if f not in PAST],
              "no NWP forecast (history only)": [f for f in FEATURES if f not in NWP],
              "no park metadata": [f for f in FEATURES if f not in META]}
    rows, shap_rows = [], []
    for h in H:
        df = data(h)
        tr, va, te = df[df.split == "train"], df[df.split == "calib"], df[df.split == "test"]
        te = te[te.elevation > DAY]
        for name, feats in groups.items():
            m = fit_lgbm(tr, va, feats)
            rows.append({"h": h, "inputs": name, "nRMSE_%": nrmse(te.y, clip(m.predict(te[feats], num_iteration=m.best_iteration), te))})
            if name.startswith("full"):
                sample = te.sample(20000, random_state=0)
                contrib = np.abs(m.predict(sample[feats], num_iteration=m.best_iteration, pred_contrib=True)[:, :-1]).mean(0)
                c = pd.Series(contrib, index=feats)
                for gname, gf in {"recent power": PAST, "sun position & time": GEOM, "NWP forecast": NWP, "park metadata": META}.items():
                    shap_rows.append({"h": h, "group": gname, "mean_abs_SHAP": c[[f for f in gf if f in c]].sum()})
    save(pd.DataFrame(rows).pivot(index="inputs", columns="h", values="nRMSE_%").reset_index(), "e4_ablation")
    s = pd.DataFrame(shap_rows)
    s["share_%"] = 100 * s.mean_abs_SHAP / s.groupby("h").mean_abs_SHAP.transform("sum")
    save(s.pivot(index="group", columns="h", values="share_%").reset_index(), "e4_shap_share")

# ------------------------------------------------------------------ E5
if "E5" in WANT:
    rows, corr_rows = [], []
    for h in H:
        pr = preds(h)
        te = pr[pr.split == "test"]
        for subset, d in (("all 43 parks", te), ("18 core parks", te[te.tier == "U2_core"])):
            def agg_err(model, level):
                d2 = d.assign(y_mw=d.y * d.capacity_mw, f_mw=d[model] * d.capacity_mw)
                keys = ["time_utc"] + ([level] if level else [])
                a = d2.groupby(keys)[["y_mw", "f_mw"]].sum()
                caps = d.drop_duplicates("park_id").groupby(level).capacity_mw.sum() if level else d.drop_duplicates("park_id").capacity_mw.sum()
                a = a[a.y_mw > 0]
                err = (a.f_mw - a.y_mw)
                if level:
                    c = caps.reindex(a.index.get_level_values(level)).to_numpy()
                    per = (err / c).groupby(level=level).apply(lambda e: 100 * np.sqrt(np.mean(e ** 2)))
                    w = caps.reindex(per.index)
                    return float((per * w).sum() / w.sum())          # capacity-weighted across groups
                return 100 * float(np.sqrt(np.mean((err / caps) ** 2)))
            for model in ("lightgbm", "bl_smart"):
                dd = d[d.elevation > DAY]
                park = dd.groupby("park_id").apply(lambda g: nrmse(g.y, g[model]), include_groups=False)
                w = dd.drop_duplicates("park_id").set_index("park_id").capacity_mw.reindex(park.index)
                rows.append({"h": h, "subset": subset, "model": model, "park": float((park * w).sum() / w.sum()),
                             "state": agg_err(model, "state"), "region": agg_err(model, "region"), "all parks": agg_err(model, None)})
        if h == 1:   # error correlation vs distance between weather locations (LightGBM)
            locs = pd.read_csv(config.LOCATIONS_FILE).set_index("loc_id")
            e = te.assign(err=te.lightgbm - te.y)[te.elevation > DAY].groupby(["time_utc", "loc_id"]).err.mean().unstack()
            cm = e.corr(min_periods=500)
            for i, a in enumerate(cm.index):
                for b in cm.index[i + 1:]:
                    la, lb = np.radians(locs.loc[[a, b], ["latitude", "longitude"]].to_numpy())
                    dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lb[0] - la[0]) / 2) ** 2 + np.cos(la[0]) * np.cos(lb[0]) * np.sin((lb[1] - la[1]) / 2) ** 2))
                    corr_rows.append({"km": dist, "corr": cm.loc[a, b]})
    save(pd.DataFrame(rows), "e5_aggregation")
    c = pd.DataFrame(corr_rows).dropna()
    c["band_km"] = pd.cut(c.km, [0, 100, 300, 600, 1000, 1500, 3000])
    save(c.groupby("band_km", observed=True)["corr"].agg(["mean", "count"]).reset_index().astype({"band_km": str}), "e5_error_corr_vs_distance")

# ------------------------------------------------------------------ E6
if "E6" in WANT:
    rows = []
    for h in H:
        pr = preds(h).sort_values(["park_id", "time_utc"])
        te = pr[(pr.split == "test")].copy()
        k = (te.y / te.pcs).where(te.pcs > 0.1)
        kh = (te.lightgbm / te.pcs).where(te.pcs > 0.1)
        prev_ok = te.groupby("park_id").time_utc.diff() == pd.Timedelta(hours=1)
        dk = (k - k.groupby(te.park_id).shift(1)).where(prev_ok)
        dkh = (kh - kh.groupby(te.park_id).shift(1)).where(prev_ok)
        te["ramp"] = dk.abs() > 0.3
        te["ramp_pred"] = dkh.abs() > 0.3
        te = te[(te.elevation > DAY) & dk.notna()]
        ist = te.time_utc + pd.Timedelta(hours=5, minutes=30)
        daily_k = (te.y.groupby([te.park_id, ist.dt.date]).transform("sum") / te.pcs.groupby([te.park_id, ist.dt.date]).transform("sum"))
        te["regime"] = np.select([daily_k > 0.8, daily_k < 0.3], ["clear", "overcast"], "partly cloudy")
        te["season"] = np.where(ist.dt.month.isin([6, 7, 8, 9]), "monsoon (Jun-Sep)", "other months")
        for name, mask in [("all daylight hours", te.index == te.index), ("ramp hours (|dk|>0.3)", te.ramp), ("non-ramp hours", ~te.ramp),
                           ("clear days", te.regime == "clear"), ("partly cloudy days", te.regime == "partly cloudy"),
                           ("overcast days", te.regime == "overcast"), ("monsoon", te.season.str.startswith("monsoon")), ("non-monsoon", ~te.season.str.startswith("monsoon"))]:
            g = te[mask]
            rows.append({"h": h, "condition": name, "share_%": 100 * len(g) / len(te), "lightgbm_nRMSE_%": nrmse(g.y, g.lightgbm),
                         "smart_nRMSE_%": nrmse(g.y, g.bl_smart)})
        hit = (te.ramp & te.ramp_pred).sum(); miss = (te.ramp & ~te.ramp_pred).sum(); fa = (~te.ramp & te.ramp_pred).sum()
        rows.append({"h": h, "condition": f"ramp detection: POD={hit / (hit + miss):.2f} FAR={fa / max(1, hit + fa):.2f} CSI={hit / (hit + miss + fa):.2f}",
                     "share_%": np.nan, "lightgbm_nRMSE_%": np.nan, "smart_nRMSE_%": np.nan})
    save(pd.DataFrame(rows), "e6_ramps_regimes")

# ------------------------------------------------------------------ E7
if "E7" in WANT:
    rows = []
    feats = [f for f in FEATURES if f != "park_code"]          # zero-shot: no park identity
    for h in (1, 6):
        df = data(h)
        tr_all, va_all = df[df.split == "train"], df[df.split == "calib"]
        base = fit_lgbm(tr_all, va_all, feats)                  # same features, location seen in training
        for loc, d in df.groupby("loc_id"):
            m = fit_lgbm(tr_all[tr_all.loc_id != loc], va_all[va_all.loc_id != loc], feats)
            te = d[(d.split == "test") & (d.elevation > DAY)]
            rows.append({"h": h, "loc_id": loc, "parks": ",".join(sorted(d.park_id.unique())), "state": d.state.iat[0],
                         "seen_nRMSE_%": nrmse(te.y, clip(base.predict(te[feats], num_iteration=base.best_iteration), te)),
                         "unseen_nRMSE_%": nrmse(te.y, clip(m.predict(te[feats], num_iteration=m.best_iteration), te))})
        print(f"E7 h={h} done", flush=True)
    r = pd.DataFrame(rows)
    save(r, "e7_unseen_locations_detail")
    r["penalty_%"] = 100 * (r["unseen_nRMSE_%"] / r["seen_nRMSE_%"] - 1)
    save(r.groupby("h")[["seen_nRMSE_%", "unseen_nRMSE_%", "penalty_%"]].agg(["mean", "max"]).reset_index(), "e7_unseen_locations")

# ------------------------------------------------------------------ E8
if "E8" in WANT:
    rows, monthly = [], []
    alpha = 0.2
    for h in H:
        df = data(h)
        tr, va, te = df[df.split == "train"], df[df.split == "calib"], df[df.split == "test"]
        q = {}
        for a in (0.1, 0.9):
            m = fit_lgbm(tr, va, FEATURES, dict(LGB_PARAMS, objective="quantile", alpha=a, metric="quantile"))
            q[a] = m
        lo = lambda d: clip(q[0.1].predict(d[FEATURES], num_iteration=q[0.1].best_iteration), d)
        hi = lambda d: clip(q[0.9].predict(d[FEATURES], num_iteration=q[0.9].best_iteration), d)
        vd = va[va.elevation > DAY]; td = te[te.elevation > DAY]
        s = np.maximum(lo(vd) - vd.y, vd.y - hi(vd))           # CQR conformity scores on calibration period
        qhat = np.quantile(s, min(1, (1 - alpha) * (1 + 1 / len(s))))
        L, U = lo(td), hi(td)
        # per-hour CQR intervals for the dashboard / API (all test rows, incl. low sun)
        pd.DataFrame({"park_id": te.park_id, "time_utc": te.time_utc, "lo": np.clip(lo(te) - qhat, 0, 1), "hi": np.clip(hi(te) + qhat, 0, 1)}
                     ).to_parquet(config.PROCESSED / f"intervals_h{h}.parquet", index=False)
        for name, (l, u) in {"LightGBM quantile (raw)": (L, U), "CQR (conformal)": (np.clip(L - qhat, 0, 1), np.clip(U + qhat, 0, 1))}.items():
            cov = (td.y >= l) & (td.y <= u)
            rows.append({"h": h, "method": name, "target_coverage_%": 80, "PICP_%": 100 * cov.mean(), "mean_width_%": 100 * np.mean(u - l)})
            if name.startswith("CQR"):
                mon = (td.time_utc + pd.Timedelta(hours=5, minutes=30)).dt.month
                monthly += [{"h": h, "month": int(mth), "PICP_%": 100 * cv.mean()} for mth, cv in cov.groupby(mon.to_numpy())]
        # all-parks total: split conformal on the aggregate MW error
        pr = preds(h)
        agg = pr.assign(y_mw=pr.y * pr.capacity_mw, f_mw=pr.lightgbm * pr.capacity_mw).groupby(["split", "time_utc"])[["y_mw", "f_mw"]].sum()
        cal, tst = agg.loc["calib"], agg.loc["test"]
        cal, tst = cal[cal.y_mw > 0], tst[tst.y_mw > 0]
        qa = np.quantile(np.abs(cal.f_mw - cal.y_mw), 0.8 * (1 + 1 / len(cal)))
        cov = (np.abs(tst.f_mw - tst.y_mw) <= qa)
        cap = pr.drop_duplicates("park_id").capacity_mw.sum()
        rows.append({"h": h, "method": "All-parks total, split conformal (MW)", "target_coverage_%": 80, "PICP_%": 100 * cov.mean(),
                     "mean_width_%": 100 * 2 * qa / cap, "halfwidth_mw": qa})
    save(pd.DataFrame(rows), "e8_intervals")
    save(pd.DataFrame(monthly).pivot(index="month", columns="h", values="PICP_%").reset_index(), "e8_cqr_monthly_coverage")

# ------------------------------------------------------------------ E9
if "E9" in WANT:
    rows = []
    oracle = [f for f in FEATURES if f not in NWP] + ORACLE   # actual weather at the valid time, no physics power
    for h in H:
        df = data(h)
        tr, va, te = df[df.split == "train"], df[df.split == "calib"], df[df.split == "test"]
        te = te[te.elevation > DAY]
        for name, feats in (("NWP forecast inputs (realistic)", FEATURES), ("actual weather inputs (oracle)", oracle)):
            m = fit_lgbm(tr, va, feats)
            rows.append({"h": h, "inputs": name, "nRMSE_%": nrmse(te.y, clip(m.predict(te[feats], num_iteration=m.best_iteration), te))})
    save(pd.DataFrame(rows).pivot(index="inputs", columns="h", values="nRMSE_%").reset_index(), "e9_oracle_vs_nwp")
