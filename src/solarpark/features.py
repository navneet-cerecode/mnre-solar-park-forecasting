"""Model-ready datasets: one row per (park, valid time v), forecast issued at t = v - h.

Leakage rule: anything from the target series uses values at or before t (shifted by h);
anything at the valid time v is either deterministic (sun geometry, clear-sky) or an NWP
forecast issued about a day earlier (previous_day1). See docs/DECISIONS.md D2/D3.
"""
import numpy as np
import pandas as pd

from . import config, openmeteo, pvsim, registry

NWP_VARS = ["ghi", "dni", "dhi", "t2m", "rh", "cc", "ws10", "sp", "tp"]
REGIONS = ["Northern", "Western", "Southern", "Eastern", "North-Eastern"]


def lead_days(h: int) -> int:
    """D2: previous_dayN was issued ~N*24 h before valid time; usable if N*24 - h >= 6 h of NWP latency."""
    return next(n for n in config.FORECAST_LEADS_DAYS if 24 * n - h >= 6)


def _hourly_base(p: pd.Series, targets: pd.DataFrame, lead: int = 1) -> pd.DataFrame:
    """Per-park hourly frame: target, clear-sky, geometry, day-1 NWP, NWP physics power, ERA5 (oracle)."""
    loc = p.loc_id
    fc, meta = openmeteo.load_dataset("fc_ecmwf", loc)
    fc = openmeteo.rename_short(fc)
    era, _ = openmeteo.load_dataset("era5", loc)
    era = openmeteo.rename_short(era)

    t = targets.set_index("time_utc").sort_index()
    base = t[["p_norm", "p_cs_norm", "elevation", "zenith", "azimuth"]].rename(columns={"p_norm": "p", "p_cs_norm": "pcs"})
    for v in NWP_VARS:
        base[f"nwp_{v}"] = fc[f"{v}_d{lead}"].reindex(base.index)
        base[f"era_{v}"] = era[v].reindex(base.index)

    # "physical forecast": pvlib applied to the day-1 forecast weather
    cfg = pvsim.config_for_park(p.latitude, p.state, p.mounting)
    wx = pd.DataFrame({v: fc[f"{v}_d{lead}"] for v in ["ghi", "dni", "dhi", "t2m", "ws10"]}).reindex(base.index)
    base["p_nwp"] = pvsim.simulate(wx, p.latitude, p.longitude, meta["elevation_m"] or 0.0, cfg)["p_norm"]
    return base


def build_horizon(h: int, reg: pd.DataFrame | None = None) -> pd.DataFrame:
    reg = registry.load_registry() if reg is None else reg
    parks = reg[reg.tier != "excluded"].reset_index(drop=True)
    targets = pd.read_parquet(config.TARGETS_FILE)
    out = []
    for code, p in parks.iterrows():
        b = _hourly_base(p, targets[targets.park_id == p.park_id], lead_days(h))
        k = (b.p / b.pcs).where(b.pcs > 0.05).clip(0, 1.5)
        k_ff = k.ffill(limit=18)                       # last daylight clear-sky index
        f = pd.DataFrame(index=b.index)
        f["y"] = b.p
        # past (known at t = v - h)
        for lag in (0, 1, 2, 3, 6):
            f[f"p_lag{lag}"] = b.p.shift(h + lag)
        f["p_lag24v"] = b.p.shift(24)                  # same hour yesterday (v-24 <= t for h <= 24)
        for w in (3, 6, 24):
            f[f"p_mean{w}"] = b.p.rolling(w).mean().shift(h)
        for w in (3, 6):
            f[f"p_std{w}"] = b.p.rolling(w).std().shift(h)
        f["k_t"] = k_ff.shift(h)
        f["dk_t"] = (k - k.shift(1)).shift(h)
        # valid-time (deterministic or day-old forecast)
        f["pcs"] = b.pcs
        f["elevation"] = b.elevation
        f["zenith"] = b.zenith
        f["azimuth"] = b.azimuth
        for v in NWP_VARS:
            f[f"nwp_{v}"] = b[f"nwp_{v}"]
            f[f"era_{v}"] = b[f"era_{v}"]               # oracle-only features (experiment E9)
        f["p_nwp"] = b.p_nwp
        hr = f.index.hour + 0.5                        # interval-ending stamps: hour mean centred at -30 min
        doy = f.index.dayofyear
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        # metadata
        f["lat"], f["lon"] = p.latitude, p.longitude
        f["log_cap"] = np.log(p.capacity_mw_model)
        f["floating"] = int(p.mounting == "floating")
        f["park_code"] = code
        f["region_code"] = REGIONS.index(p.grid_region)
        f["park_id"] = p.park_id
        f["loc_id"] = p.loc_id
        f["state"] = p.state
        f["region"] = p.grid_region
        f["tier"] = p.tier
        f["capacity_mw"] = p.capacity_mw_model
        f["h"] = h
        # baselines
        f["bl_persistence"] = f.p_lag0
        f["bl_smart"] = (f.k_t * f.pcs).clip(0, 1)
        f["bl_nwp_physics"] = f.p_nwp
        out.append(f[(f.elevation > 0) & f.y.notna()])
    df = pd.concat(out).rename_axis("time_utc").reset_index()
    df["split"] = "none"
    for name, (a, b) in config.SPLITS.items():
        m = (df.time_utc >= pd.Timestamp(a, tz="UTC")) & (df.time_utc <= pd.Timestamp(b + " 23:00", tz="UTC"))
        df.loc[m, "split"] = name
    df = df[df.split != "none"]
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    return df


# Feature groups (used by models and the ablation experiment E4)
PAST = ["p_lag0", "p_lag1", "p_lag2", "p_lag3", "p_lag6", "p_lag24v", "p_mean3", "p_mean6", "p_mean24", "p_std3", "p_std6", "k_t", "dk_t"]
GEOM = ["pcs", "elevation", "zenith", "azimuth", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
NWP = [f"nwp_{v}" for v in NWP_VARS] + ["p_nwp"]
ORACLE = [f"era_{v}" for v in NWP_VARS]
META = ["lat", "lon", "log_cap", "floating", "region_code", "park_code"]
FEATURES = PAST + GEOM + NWP + META
