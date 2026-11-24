"""Forecast API.  Run:  uvicorn app.api:app --reload   then open http://127.0.0.1:8000/docs

ponytail: serves the saved test-period forecasts (Jan-Sep 2026). A live mode needs metered park output
for the lag features; add it when an SLDC/SPPD data feed is available.
"""
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from solarpark import config, forecasts as F

app = FastAPI(title="MNRE solar park forecasts", version="0.1.0",
              description="Hourly 1/3/6-hour-ahead forecasts (MW) with calibrated 80% ranges. Times are IST. "
                          "'actual' is pvlib-simulated from ERA5 weather.")


def check(h):
    if h not in config.HORIZONS_H:
        raise HTTPException(422, f"h must be one of {config.HORIZONS_H}")


def window(df, start, end):
    t = F.to_ist(df.time_utc)
    m = pd.Series(True, index=df.index)
    if start:
        m &= t >= pd.Timestamp(start, tz="Asia/Kolkata")
    if end:
        m &= t < pd.Timestamp(end, tz="Asia/Kolkata") + pd.Timedelta(days=1)
    out = df[m].copy()
    out.insert(0, "time_ist", F.to_ist(out.time_utc).dt.strftime("%Y-%m-%d %H:%M"))
    return out.drop(columns="time_utc").round(2)


@app.get("/parks")
def parks():
    p = F.parks()
    cols = ["park_id", "park_name", "state", "grid_region", "latitude", "longitude", "sanctioned_capacity_mw",
            "commissioned_capacity_mw", "status", "sppd", "tier", "capacity_mw_model"]
    return p[cols].astype(object).where(p[cols].notna(), None).to_dict("records")


@app.get("/forecast/park/{park_id}")
def park_forecast(park_id: str, h: int = Query(1, description='hours ahead: 1, 3 or 6'), start: str | None = Query(None, examples=["2026-07-06"]),
                  end: str | None = Query(None, examples=["2026-07-07"])):
    check(h)
    f = F.forecasts(h)
    g = f[f.park_id == park_id]
    if g.empty:
        raise HTTPException(404, f"no forecasts for park {park_id}; see /parks (excluded parks have none)")
    cols = ["time_utc", "y_mw", "lightgbm_mw", "lo_mw", "hi_mw", "gru_mw", "bl_smart_mw"]
    return window(g[[c for c in cols if c in g]], start, end).rename(
        columns={"y_mw": "actual_mw", "lightgbm_mw": "forecast_mw", "lo_mw": "range80_lo_mw", "hi_mw": "range80_hi_mw"}).to_dict("records")


@app.get("/forecast/total")
def total_forecast(h: int = Query(1, description='hours ahead: 1, 3 or 6'), level: Literal["all", "state", "region"] = "all", name: str | None = None,
                   start: str | None = Query(None, examples=["2026-07-06"]), end: str | None = None):
    check(h)
    a = F.aggregate(h, None if level == "all" else level)
    if level != "all":
        if not name:
            raise HTTPException(422, f"'name' is required for level={level}")
        a = a[a[level] == name]
        if a.empty:
            raise HTTPException(404, f"unknown {level} '{name}'")
    return window(a, start, end).to_dict("records")


@app.get("/metrics")
def metrics():
    p = config.REPORTS / "results" / "e1_models.csv"
    if not p.exists():
        raise HTTPException(404, "run scripts/06_train_models.py first")
    return pd.read_csv(p).round(3).to_dict("records")
