import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarpark import pvsim  # noqa: E402

LAT, LON = 27.54, 71.92   # Bhadla


def clear_day(date="2025-03-21"):
    idx = pd.date_range(f"{date} 01:00", periods=24, freq="h", tz="UTC")   # interval-end stamps
    geo = pvsim.solar_geometry(idx, LAT, LON, 170)
    return pd.DataFrame({"ghi": geo.ghi_cs, "dni": pvsim.pvlib.clearsky.ineichen(
        geo.zenith, geo.airmass, 3.0, dni_extra=geo.dni_extra)["dni"].fillna(0), "dhi": 0.0,
        "t2m": 30.0, "ws10": 2.0}, index=idx).assign(dhi=lambda d: (d.ghi - d.dni * np.cos(np.radians(geo.zenith))).clip(lower=0))


def test_clear_day_shape():
    wx = clear_day()
    sim = pvsim.simulate(wx, LAT, LON, 170, pvsim.config_for_park(LAT, "Rajasthan", "fixed_ground"))
    assert sim.p_norm.max() <= 1.0
    assert 0.7 < sim.p_norm.max()                      # near-rated output at clear-sky noon
    assert (sim.loc[sim.elevation < -1, "p_norm"] == 0).all()   # no output at night
    peak_hour_ist = (sim.p_norm.idxmax() + pd.Timedelta(hours=5.5) - pd.Timedelta(minutes=30)).hour
    assert 11 <= peak_hour_ist <= 13


def test_missing_weather_gives_nan_not_zero():
    wx = clear_day()
    wx.iloc[10, wx.columns.get_loc("ghi")] = np.nan
    sim = pvsim.simulate(wx, LAT, LON, 170, pvsim.config_for_park(LAT, "Rajasthan", "fixed_ground"))
    assert np.isnan(sim.p_norm.iloc[10])


def test_park_configs():
    floating = pvsim.config_for_park(22.2, "Madhya Pradesh", "floating")
    assert floating.tilt == 10.0
    desert = pvsim.config_for_park(27.5, "Rajasthan", "fixed_ground")
    assert desert.losses["soiling"] == 4 and desert.albedo == 0.25
    south = pvsim.config_for_park(14.2, "Karnataka", "fixed_ground")
    assert south.tilt == pytest.approx(14.2) and south.losses["soiling"] == 2


def test_hotter_cells_reduce_output():
    wx = clear_day()
    cfg = pvsim.config_for_park(LAT, "Rajasthan", "fixed_ground")
    cool = pvsim.simulate(wx.assign(t2m=15.0), LAT, LON, 170, cfg).p_norm.sum()
    hot = pvsim.simulate(wx.assign(t2m=45.0), LAT, LON, 170, cfg).p_norm.sum()
    assert hot < cool
