"""Read-side access to saved forecasts (test period) for the dashboard and the API."""
from functools import lru_cache

import pandas as pd

from . import config, registry

MODELS = {"lightgbm": "LightGBM", "gru": "GRU (GPU)", "ensemble_lgbm_gru": "LightGBM + GRU", "bl_smart": "Smart persistence"}


@lru_cache
def parks() -> pd.DataFrame:
    m = pd.read_csv(config.MASTER_TABLE, encoding="utf-8-sig")
    r = registry.load_registry()[["park_id", "tier", "loc_id", "capacity_mw_model", "mounting"]]
    return m.merge(r, on="park_id", how="left")


@lru_cache
def forecasts(h: int) -> pd.DataFrame:
    """Test-period rows: actual (simulated) and forecast as p_norm and MW, with 80% CQR range for LightGBM."""
    pr = pd.read_parquet(config.PROCESSED / f"preds_h{h}.parquet")
    pr = pr[pr.split == "test"]
    iv = config.PROCESSED / f"intervals_h{h}.parquet"
    if iv.exists():
        pr = pr.merge(pd.read_parquet(iv), on=["park_id", "time_utc"], how="left")
    for c in ["y", *MODELS, "lo", "hi"]:
        if c in pr:
            pr[f"{c}_mw"] = pr[c] * pr.capacity_mw
    return pr.reset_index(drop=True)


@lru_cache
def total_halfwidth_mw(h: int) -> float | None:
    """Calibrated +/- half-width (MW) of the 80% range for the all-parks total (experiment E8)."""
    p = config.REPORTS / "results" / "e8_intervals.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[(d.h == h) & d.method.str.startswith("All-parks")]
    return float(d.halfwidth_mw.iat[0]) if "halfwidth_mw" in d and len(d) and pd.notna(d.halfwidth_mw.iat[0]) else None


def aggregate(h: int, level: str | None = None, model: str = "lightgbm") -> pd.DataFrame:
    """Hourly MW sums of actual and forecast: level None = all parks, else 'state' or 'region'."""
    f = forecasts(h)
    keys = ["time_utc"] + ([level] if level else [])
    a = f.groupby(keys)[["y_mw", f"{model}_mw"]].sum().reset_index().rename(columns={f"{model}_mw": "forecast_mw", "y_mw": "actual_mw"})
    if level is None and (hw := total_halfwidth_mw(h)) is not None:
        a["lo_mw"] = (a.forecast_mw - hw).clip(lower=0)
        a["hi_mw"] = a.forecast_mw + hw
    return a


def to_ist(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("Asia/Kolkata")
