"""Step 13 / E3b - climate-based grouping instead of grid regions.

1. Koppen-Geiger class of every park (kgcpy: Rubel et al. 2017 map, 1986-2010, via the kgc package).
2. Data-driven weather clusters: k-means on each location's monthly climatology (clear-sky index, cloud cover,
   temperature, rain) from the TRAINING period only; k chosen by silhouette score.
3. One LightGBM per group vs the single global model (as E3).
Outputs: data/interim/park_climate.csv, climate_zone column in the master table, reports/results/e3b_climate_groups*.csv
"""
import _bootstrap  # noqa: F401
import kgcpy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from importlib import import_module

from solarpark import config, openmeteo, registry
from solarpark.features import FEATURES

t6 = import_module("06_train_models")
RES = config.REPORTS / "results"
KOPPEN = {"BWh": "hot desert", "BSh": "hot semi-arid", "Aw": "tropical savanna", "Am": "tropical monsoon",
          "Cwa": "humid subtropical, dry winter", "Cwb": "subtropical highland, dry winter", "Cfa": "humid subtropical",
          "As": "tropical savanna, dry summer", "Csa": "Mediterranean, hot summer"}
TRAIN = (pd.Timestamp(config.SPLITS["train"][0], tz="UTC"), pd.Timestamp(config.SPLITS["train"][1] + " 23:00", tz="UTC"))

reg = registry.load_registry()
parks = reg[reg.tier != "excluded"].copy()
parks["koppen"] = [kgcpy.lookupCZ(la, lo) for la, lo in zip(parks.latitude, parks.longitude)]
parks["koppen_name"] = parks.koppen.map(KOPPEN).fillna(parks.koppen)

# monthly climatology per weather location, training period only
tg = pd.read_parquet(config.TARGETS_FILE, columns=["park_id", "time_utc", "p_norm", "p_cs_norm"])
tg = tg[(tg.time_utc >= TRAIN[0]) & (tg.time_utc <= TRAIN[1]) & (tg.p_cs_norm > 0.1)]
tg["k"] = (tg.p_norm / tg.p_cs_norm).clip(0, 1.2)
rows = []
for loc, g in parks.groupby("loc_id"):
    era, _ = openmeteo.load_dataset("era5", loc)
    era = openmeteo.rename_short(era).loc[TRAIN[0]:TRAIN[1]]
    m = era.index.month
    k = tg[tg.park_id == g.park_id.iat[0]].set_index("time_utc").k
    row = {"loc_id": loc}
    for mon in range(1, 13):
        row[f"k_{mon}"] = k[k.index.month == mon].mean()
        row[f"cc_{mon}"] = era.cc[m == mon].mean()
        row[f"t_{mon}"] = era.t2m[m == mon].mean()
        row[f"tp_{mon}"] = era.tp[m == mon].sum() / max(1, len(set(era.index[m == mon].year)))
    rows.append(row)
clim = pd.DataFrame(rows).set_index("loc_id")
X = StandardScaler().fit_transform(clim.fillna(clim.mean()))
scores = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=0).fit_predict(X)) for k in range(3, 7)}
best_k = max(scores, key=scores.get)
labels = KMeans(best_k, n_init=20, random_state=0).fit_predict(X)
clim["cluster"] = [f"C{c + 1}" for c in labels]
parks = parks.merge(clim[["cluster"]], left_on="loc_id", right_index=True)
print("silhouette by k:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)
print(pd.crosstab(parks.koppen, parks.cluster))

out = parks[["park_id", "park_name", "state", "grid_region", "loc_id", "koppen", "koppen_name", "cluster"]]
out.to_csv(config.INTERIM / "park_climate.csv", index=False)
# fill the master table's climate_zone placeholder
m = pd.read_csv(config.MASTER_TABLE, encoding="utf-8-sig")
kz = out.set_index("park_id").apply(lambda r: f"{r.koppen} ({r.koppen_name}); weather cluster {r.cluster}", axis=1)
m["climate_zone"] = m.park_id.map(kz).fillna("Not assigned (no verified location / excluded)")
m.to_csv(config.MASTER_TABLE, index=False, encoding="utf-8-sig")

# E3b: one model per climate group vs the global model
res = []
for h in config.HORIZONS_H:
    df = pd.read_parquet(config.PROCESSED / f"model_h{h}.parquet").merge(out[["park_id", "koppen", "cluster"]], on="park_id")
    pr = pd.read_parquet(config.PROCESSED / f"preds_h{h}.parquet").set_index(["park_id", "time_utc"])
    for col in ("koppen", "cluster"):
        for gname, d in df.groupby(col):
            tr, va, te = d[d.split == "train"], d[d.split == "calib"], d[d.split == "test"]
            mdl = t6.fit_lgbm(tr, va, FEATURES, dict(t6.LGB_PARAMS, num_leaves=63))
            te = te[te.elevation > 5]
            local = t6.clip(mdl.predict(te[FEATURES], num_iteration=mdl.best_iteration), te)
            glob = pr.loc[pd.MultiIndex.from_frame(te[["park_id", "time_utc"]]), "lightgbm"].to_numpy()
            rmse = lambda p: 100 * float(np.sqrt(np.mean((p - te.y.to_numpy()) ** 2)))
            res.append({"h": h, "grouping": "Koppen-Geiger" if col == "koppen" else f"weather clusters (k={best_k})", "group": gname,
                        "parks": d.park_id.nunique(), "group_model_nRMSE_%": rmse(local), "global_nRMSE_%": rmse(glob), "n": len(te)})
    print(f"h={h} done", flush=True)
r = pd.DataFrame(res)
r.to_csv(RES / "e3b_climate_groups_detail.csv", index=False)
w = r.assign(gw=r["group_model_nRMSE_%"] * r.n, ww=r["global_nRMSE_%"] * r.n)
summary = w.groupby(["h", "grouping"]).agg(group_models=("gw", "sum"), global_model=("ww", "sum"), n=("n", "sum"),
                                         groups=("group", "count"), global_wins=("group", lambda s: 0)).reset_index()
summary["group_models"] /= summary.n; summary["global_model"] /= summary.n
summary["global_wins"] = r.assign(win=r["global_nRMSE_%"] <= r["group_model_nRMSE_%"]).groupby(["h", "grouping"]).win.sum().to_numpy()
summary = summary.drop(columns="n").rename(columns={"group_models": "group_models_nRMSE_%", "global_model": "global_nRMSE_%"})
summary.to_csv(RES / "e3b_climate_groups.csv", index=False)
print(summary.round(3).to_string(index=False))
print(r.round(3).to_string(index=False))
