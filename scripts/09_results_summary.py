"""Step 9 - collect experiment CSVs into reports/results/RESULTS.md with figures."""
import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from solarpark import config

R = config.REPORTS / "results"
INK2, GRID = "#52514e", "#e1e0d9"
COL = {"bl_smart": "#a3a29b", "bl_nwp_physics": "#c3c2b7", "ridge": "#eda100", "random_forest": "#1baf7a", "xgboost_gpu": "#4a3aa7",
       "lightgbm": "#2a78d6", "lstm": "#e87ba4", "gru": "#eb6834", "ensemble_lgbm_gru": "#008300"}
NAME = {"bl_persistence": "Persistence", "bl_smart": "Smart persistence", "bl_climatology": "Climatology", "bl_nwp_physics": "ECMWF + physics",
        "ridge": "Ridge", "random_forest": "Random Forest", "xgboost_gpu": "XGBoost (GPU)", "lightgbm": "LightGBM",
        "lstm": "LSTM (GPU)", "gru": "GRU (GPU)", "ensemble_lgbm_gru": "LightGBM + GRU average"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2})


def read(name):
    p = R / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def md(df):
    df = df.copy()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


parts = ["# Forecasting Results (test period Jan-Sep 2026, daylight hours)\n",
         "All errors are **% of park capacity** (normalised power). Skill = improvement in RMSE over smart persistence. "
         "Targets are pvlib-simulated from ERA5; model inputs are day-old ECMWF forecasts (see docs/DECISIONS.md).\n"]

e1 = read("e1_models")
dl = read("e1b_deep_learning")
if e1 is not None:
    a = e1[e1.subset == "all 43 parks"]
    if dl is not None:
        a = pd.concat([a, dl[dl.model.isin(["lstm", "gru", "ensemble_lgbm_gru"])]])
    t = a.pivot_table(index="model", columns="h", values="nRMSE_%").rename(columns=lambda h: f"nRMSE {h}h")
    s = a.pivot_table(index="model", columns="h", values="skill_vs_smart").rename(columns=lambda h: f"skill {h}h")
    t = t.join(s).sort_values("nRMSE 1h")
    t.index = [NAME.get(i, i) for i in t.index]
    parts += ["## E1 - Model comparison (all 43 parks)\n", md(t.reset_index(names="model")), "\n![E1](fig_e1_models.png)\n"]
    keep = [m for m in COL if m in a.model.unique()]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    hs = sorted(a.h.unique()); w = 0.8 / len(keep)
    for i, m in enumerate(keep):
        v = a[a.model == m].set_index("h").reindex(hs)["nRMSE_%"]
        ax.bar([x + i * w for x in range(len(hs))], v, w, color=COL[m], label=NAME[m])
    ax.set_xticks([x + w * (len(keep) - 1) / 2 for x in range(len(hs))]); ax.set_xticklabels([f"{h} h ahead" for h in hs])
    ax.set_ylabel("nRMSE (% of capacity)"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    fig.savefig(R / "fig_e1_models.png", dpi=180, bbox_inches="tight"); plt.close(fig)

for name, title in [("e2_per_park_vs_global", "E2 - One global model vs one model per park (mean park nRMSE)"),
                    ("e3_regional_vs_global", "E3 - Regional models vs global model"),
                    ("e3b_climate_groups", "E3b - Climate-zone models (Koppen-Geiger, weather clusters) vs global model"),
                    ("e4_ablation", "E4 - Which inputs matter (nRMSE by horizon)"),
                    ("e4_shap_share", "E4 - Share of model attribution by input group (SHAP, %)"),
                    ("e5_aggregation", "E5 - Error when parks are added together (capacity-normalised nRMSE)"),
                    ("e5_error_corr_vs_distance", "E5 - Correlation of 1-h forecast errors vs distance between parks"),
                    ("e6_ramps_regimes", "E6 - Ramps and weather regimes"),
                    ("e7_unseen_locations", "E7 - Forecasting a location never seen in training"),
                    ("e8_intervals", "E8 - 80% prediction intervals"),
                    ("e8_cqr_monthly_coverage", "E8 - CQR coverage by month (%)"),
                    ("e9_oracle_vs_nwp", "E9 - Realistic NWP inputs vs perfect (actual) weather"),
                    ("e10_catboost_tuning", "E10 - CatBoost (GPU) and LightGBM tuning (chosen on calib, scored on test)"),
                    ("e11_real_physics_check", "E11 - Real plants: pvlib from measured irradiance vs measured power (15-min)"),
                    ("e11_real_forecasting", "E11 - Real plants: hourly forecasts, last 10 days (history + time features only)")]:
    d = read(name)
    if d is None:
        continue
    if name == "e3_regional_vs_global":
        d = d.groupby("h")[["regional_nRMSE_%", "global_nRMSE_%"]].mean().reset_index()
    parts += [f"## {title}\n", md(d), ""]
    if name == "e5_aggregation":
        x = d[(d.model == "lightgbm") & (d.subset == "all 43 parks")]
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        lv = ["park", "state", "region", "all parks"]
        for h, c in zip(sorted(x.h.unique()), ["#2a78d6", "#1baf7a", "#eb6834"]):
            ax.plot(lv, x[x.h == h][lv].iloc[0].to_numpy(), marker="o", color=c, lw=2, label=f"{h} h ahead")
        ax.set_ylabel("nRMSE (% of capacity)"); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False)
        fig.savefig(R / "fig_e5_aggregation.png", dpi=180, bbox_inches="tight"); plt.close(fig)
        parts.append("![E5](fig_e5_aggregation.png)\n")
    if name == "e11_real_physics_check":
        parts.append("![E11](fig_e11_real.png)\n")

# example forecast plot
p1 = config.PROCESSED / "preds_h1.parquet"
if p1.exists():
    pr = pd.read_parquet(p1)
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharey=True)
    for ax, (pid, a, b, ttl) in zip(axs, [("P35", "2026-03-02", "2026-03-09", "Bhadla-II, March 2026 (mostly clear)"),
                                          ("P17", "2026-07-06", "2026-07-13", "Pavagada, July 2026 (monsoon)")]):
        g = pr[(pr.park_id == pid) & (pr.time_utc >= pd.Timestamp(a, tz="UTC")) & (pr.time_utc < pd.Timestamp(b, tz="UTC"))]
        g = g.set_index("time_utc").asfreq("h").reset_index()   # night hours -> gaps, not straight lines
        ax.plot(g.time_utc, g.y, color="#0b0b0b", lw=1.6, label="actual (simulated)")
        ax.plot(g.time_utc, g.lightgbm, color="#2a78d6", lw=1.2, label="LightGBM, 1 h ahead")
        if "gru" in g:
            ax.plot(g.time_utc, g.gru, color="#eb6834", lw=1.0, ls="--", label="GRU, 1 h ahead")
        ax.set_title(ttl, fontsize=9, loc="left"); ax.set_ylabel("p_norm"); ax.grid(color=GRID, lw=0.5)
    axs[0].legend(frameon=False, fontsize=7.5, ncol=3)
    fig.tight_layout(); fig.savefig(R / "fig_example_forecasts.png", dpi=180); plt.close(fig)
    parts += ["## Example forecasts\n", "![examples](fig_example_forecasts.png)\n"]

(R / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
print("wrote", R / "RESULTS.md")
