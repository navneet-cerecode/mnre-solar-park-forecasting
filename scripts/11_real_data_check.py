"""Step 11 / E11 - simulation-to-reality check on measured data (Kaggle "Solar Power Generation Data").

Two Indian plants, 34 days (15 May - 17 Jun 2020), 15-min inverter AC power + on-site irradiance and temperatures.
The plants' location is not published, so ECMWF forecasts cannot be matched. Two checks are possible:
  A. Physics: does our pvlib chain, fed the measured irradiance and module temperature, reproduce measured AC power?
  B. Forecasting: does a LightGBM trained only on the simulated MNRE parks (history + time features) forecast the
     real plants 1 and 3 h ahead? Compared with baselines, a model trained on the plant itself, and the
     simulated model fine-tuned on the plant.
Data: data/raw/kaggle_solar/ (public GitHub mirror of the Kaggle files; row counts match the original).
Output: reports/results/e11_real_*.csv, fig_e11_real.png
"""
import _bootstrap  # noqa: F401
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

from solarpark import config

RAW = config.RAW / "kaggle_solar"
RES = config.REPORTS / "results"
HIST = ["p_lag0", "p_lag1", "p_lag2", "p_lag3", "p_lag6", "p_lag24v", "p_mean3", "p_mean6", "p_mean24", "p_std3", "p_std6",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
IST = pd.Timedelta(hours=5, minutes=30)


def load_plant(n):
    g = pd.read_csv(RAW / f"Plant_{n}_Generation_Data.csv")
    w = pd.read_csv(RAW / f"Plant_{n}_Weather_Sensor_Data.csv")
    # Plant 1 generation uses dd-mm-yyyy; the other files are ISO. Guessing (dayfirst) misreads ISO dates.
    g["t"] = pd.to_datetime(g.DATE_TIME, format="%d-%m-%Y %H:%M" if g.DATE_TIME.iat[0][2] == "-" else "ISO8601")
    w["t"] = pd.to_datetime(w.DATE_TIME, format="ISO8601")
    assert g.t.max() - g.t.min() < pd.Timedelta(days=40) and w.t.max() - w.t.min() < pd.Timedelta(days=40), "date parsing"
    n_inv = g.SOURCE_KEY.nunique()
    q = g.groupby("t").agg(ac=("AC_POWER", "sum"), inverters=("SOURCE_KEY", "nunique"))
    # missing inverter rows would look like a power drop: rescale small gaps, drop large ones
    q["ac"] = (q.ac * n_inv / q.inverters).where(q.inverters >= 0.8 * n_inv)
    q = q.join(w.set_index("t")[["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]], how="outer")
    q = q.reindex(pd.date_range(q.index.min().floor("D"), q.index.max().ceil("D") - pd.Timedelta(minutes=15), freq="15min"))
    hourly = q.resample("h", label="right", closed="left").mean()       # interval-ending, like the main dataset
    hourly = hourly[hourly.notna().sum(axis=1) > 0]
    cap = hourly.ac.quantile(0.995)                                      # effective AC capacity (not published)
    hourly["p"] = (hourly.ac / cap).clip(0, 1.05)
    hourly["plant"] = f"Plant {n}"
    return hourly, q, cap, n_inv


def physics_check(q, cap, n):
    """A. measured irradiance (kW/m2, on the module plane) + module temperature -> PVWatts DC -> losses -> inverter."""
    d = q.dropna(subset=["ac", "IRRADIATION", "MODULE_TEMPERATURE"])
    poa = d.IRRADIATION * 1000
    pdc = pvlib.pvsystem.pvwatts_dc(poa, d.MODULE_TEMPERATURE, pdc0=1.25, gamma_pdc=config.PV["gamma_pdc"])
    pac = pvlib.inverter.pvwatts(pdc * (1 - pvlib.pvsystem.pvwatts_losses() / 100), pdc0=1 / 0.96, eta_inv_nom=0.96).clip(0, 1)
    meas = d.ac / d.ac.quantile(0.995)
    day = poa > 50
    scale = float(np.sum(meas[day] * pac[day]) / np.sum(pac[day] ** 2))   # single performance-ratio calibration
    e_raw, e_cal = (pac - meas)[day], (scale * pac - meas)[day]
    return {"plant": f"Plant {n}", "corr_15min": float(np.corrcoef(pac[day], meas[day])[0, 1]),
            "nRMSE_uncalibrated_%": 100 * float(np.sqrt(np.mean(e_raw ** 2))),
            "calibration_factor": scale, "nRMSE_calibrated_%": 100 * float(np.sqrt(np.mean(e_cal ** 2))),
            "nMAE_calibrated_%": 100 * float(np.mean(np.abs(e_cal)))}, d.index, pac, meas


def features(s, h):
    """Same history/time features as solarpark.features (subset), on a complete hourly series."""
    p = s.p
    f = pd.DataFrame(index=s.index)
    f["y"] = p
    for lag in (0, 1, 2, 3, 6):
        f[f"p_lag{lag}"] = p.shift(h + lag)
    f["p_lag24v"] = p.shift(24)
    for w in (3, 6, 24):
        f[f"p_mean{w}"] = p.rolling(w).mean().shift(h)
    for w in (3, 6):
        f[f"p_std{w}"] = p.rolling(w).std().shift(h)
    utc = f.index - IST                                 # plant clocks are local (IST); the simulated data are UTC
    hr = utc.hour + 0.5
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    doy = utc.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return f


def nrmse(y, p):
    return 100 * float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


if __name__ == "__main__":
    phys, fc_rows, plots = [], [], {}
    sim_models = {}
    for h in (1, 3):
        sim = pd.read_parquet(config.PROCESSED / f"model_h{h}.parquet")
        tr, va = sim[sim.split == "train"], sim[sim.split == "calib"]
        params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, verbose=-1, seed=42)
        sim_models[h] = lgb.train(params, lgb.Dataset(tr[HIST], tr.y), 2000, valid_sets=[lgb.Dataset(va[HIST], va.y)],
                                  callbacks=[lgb.early_stopping(100, verbose=False)])

    for n in (1, 2):
        s, q, cap, n_inv = load_plant(n)
        r, idx, pac, meas = physics_check(q, cap, n)
        r["inverters"] = n_inv
        r["effective_capacity_kW"] = round(cap)
        phys.append(r)
        plots[n] = (idx, pac * r["calibration_factor"], meas)

        s = s.asfreq("h")
        days = pd.Series(s.index.normalize().unique())
        tr_end, va_end = days.iloc[19], days.iloc[23]                     # 20 days train, 4 validation, rest test
        envelope = s.p[s.index < va_end].groupby(s.index[s.index < va_end].hour).quantile(0.9)   # clear-day shape proxy
        for h in (1, 3):
            f = features(s, h)
            f["env_v"] = envelope.reindex(f.index.hour).to_numpy()
            f["env_t"] = envelope.reindex((f.index - pd.Timedelta(hours=h)).hour).to_numpy()
            f = f.dropna(subset=["y", "p_lag0", "p_lag24v"])
            tr = f[f.index < tr_end]; va = f[(f.index >= tr_end) & (f.index < va_end)]; te = f[f.index >= va_end]
            te = te[te.env_v > 0.05]                                    # daylight
            k = (te.p_lag0 / te.env_t).where(te.env_t > 0.05).clip(0, 1.2).fillna(1)
            sim_m = sim_models[h]
            local = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=10, verbose=-1, seed=42),
                              lgb.Dataset(tr[HIST], tr.y), 1000, valid_sets=[lgb.Dataset(va[HIST], va.y)],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            tuned = lgb.train(dict(objective="regression", learning_rate=0.02, num_leaves=63, min_data_in_leaf=10, verbose=-1, seed=42),
                              lgb.Dataset(tr[HIST], tr.y), 300, valid_sets=[lgb.Dataset(va[HIST], va.y)], init_model=sim_m,
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = {"Persistence": te.p_lag0, "Smart persistence (clear-day shape)": (k * te.env_v).clip(0, 1),
                     "Same hour yesterday": te.p_lag24v,
                     "LightGBM trained on simulated parks only (zero-shot)": sim_m.predict(te[HIST], num_iteration=sim_m.best_iteration).clip(0, 1),
                     "LightGBM trained on this plant (20 days)": local.predict(te[HIST], num_iteration=local.best_iteration).clip(0, 1),
                     "Simulated model fine-tuned on this plant": tuned.predict(te[HIST], num_iteration=tuned.best_iteration).clip(0, 1)}
            for name, pr in preds.items():
                fc_rows.append({"plant": f"Plant {n}", "h": h, "model": name, "nRMSE_%": nrmse(te.y, pr),
                                "nMAE_%": 100 * float(np.mean(np.abs(np.asarray(pr) - te.y))), "test_hours": len(te)})

    phys = pd.DataFrame(phys)
    fcr = pd.DataFrame(fc_rows)
    phys.to_csv(RES / "e11_real_physics_check.csv", index=False)
    fcr.to_csv(RES / "e11_real_forecasting.csv", index=False)
    print(phys.round(3).to_string(index=False))
    print(fcr.pivot_table(index="model", columns=["plant", "h"], values="nRMSE_%").round(2).to_string())

    fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharey=True)
    for ax, n in zip(axs, (1, 2)):
        idx, pac, meas = plots[n]
        m = (idx >= pd.Timestamp("2020-05-28")) & (idx < pd.Timestamp("2020-06-03"))
        ax.plot(idx[m], meas[m], color="#0b0b0b", lw=1.2, label="measured AC (normalised)")
        ax.plot(idx[m], np.asarray(pac)[m], color="#eb6834", lw=1.0, ls="--", label="pvlib from measured irradiance")
        r = phys[phys.plant == f"Plant {n}"].iloc[0]
        ax.set_title(f"Plant {n}: correlation {r.corr_15min:.3f}, calibrated nRMSE {r['nRMSE_calibrated_%']:.1f}%", fontsize=9, loc="left")
        ax.grid(color="#e1e0d9", lw=0.5)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_e11_real.png", dpi=180); plt.close(fig)
