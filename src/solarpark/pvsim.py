"""Physical PV simulation with pvlib (scope document, Section 21).

Output is AC power normalised by AC capacity (p_norm, 0..1). MW = p_norm x capacity.
Input weather is hourly with timestamps marking the END of each averaging hour, so
solar geometry is evaluated at the interval midpoint (t - 30 min).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib

from . import config


@dataclass
class PVConfig:
    tilt: float
    azimuth: float = 180.0
    albedo: float = config.PV["albedo_default"]
    dc_ac: float = config.PV["dc_ac_ratio"]
    gamma_pdc: float = config.PV["gamma_pdc"]
    inverter_eta: float = config.PV["inverter_eta"]
    losses: dict = field(default_factory=lambda: dict(config.PV["losses_default"]))
    mounting: str = "fixed_ground"

    @property
    def loss_fraction(self) -> float:
        return pvlib.pvsystem.pvwatts_losses(**self.losses) / 100.0


def config_for_park(lat: float, state: str, mounting: str) -> PVConfig:
    cfg = PVConfig(tilt=round(abs(lat), 1), mounting=mounting)
    if mounting == "floating":
        cfg.tilt = config.PV["floating_tilt_deg"]
    if state in config.DESERT_STATES and mounting != "floating":
        cfg.albedo = config.PV["albedo_desert"]
        cfg.losses["soiling"] = config.PV["soiling_desert"]
    return cfg


def solar_geometry(index_end: pd.DatetimeIndex, lat: float, lon: float, altitude: float) -> pd.DataFrame:
    mid = index_end - pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
    sp = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen")
    out = pd.DataFrame({
        "zenith": sp["apparent_zenith"].to_numpy(), "elevation": sp["apparent_elevation"].to_numpy(),
        "azimuth": sp["azimuth"].to_numpy(),
        "ghi_cs": cs["ghi"].to_numpy(), "dni_cs": cs["dni"].to_numpy(), "dhi_cs": cs["dhi"].to_numpy(),
    }, index=index_end)
    out["dni_extra"] = pvlib.irradiance.get_extra_radiation(mid).to_numpy()
    out["airmass"] = pvlib.atmosphere.get_relative_airmass(out["zenith"]).to_numpy()
    return out


def _ac_chain(ghi, dni, dhi, temp_air, wind, geo: pd.DataFrame, cfg: PVConfig) -> pd.DataFrame:
    zen = geo["zenith"]
    up = zen < 90
    dni = dni.where(up, 0.0).clip(lower=0)
    poa = pvlib.irradiance.get_total_irradiance(
        cfg.tilt, cfg.azimuth, zen, geo["azimuth"], dni, ghi.clip(lower=0), dhi.clip(lower=0),
        dni_extra=geo["dni_extra"], airmass=geo["airmass"], albedo=cfg.albedo, model="perez")
    aoi = pvlib.irradiance.aoi(cfg.tilt, cfg.azimuth, zen, geo["azimuth"])
    iam = pvlib.iam.ashrae(aoi, b=config.PV["iam_b"]).fillna(0.0)
    poa_direct = poa["poa_direct"].fillna(0.0).clip(lower=0)
    poa_diffuse = poa["poa_diffuse"].fillna(0.0).clip(lower=0)
    poa_eff = poa_direct * iam + poa_diffuse
    poa_global = (poa_direct + poa_diffuse)

    if cfg.mounting == "floating":
        tcell = pvlib.temperature.faiman(poa_global, temp_air, wind, **config.PV["faiman_floating"])
    else:
        p = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
        tcell = pvlib.temperature.sapm_cell(poa_global, temp_air, wind, **p)

    pdc = pvlib.pvsystem.pvwatts_dc(poa_eff, tcell, pdc0=cfg.dc_ac, gamma_pdc=cfg.gamma_pdc)
    pdc_net = (pdc * (1 - cfg.loss_fraction)).clip(lower=0)
    pac = pvlib.inverter.pvwatts(pdc_net, pdc0=1.0 / cfg.inverter_eta, eta_inv_nom=cfg.inverter_eta)
    pac = pd.Series(np.clip(np.nan_to_num(np.asarray(pac, dtype=float)), 0.0, 1.0), index=ghi.index)
    return pd.DataFrame({"poa_global": poa_global, "t_cell": tcell, "p_norm": pac})


def simulate(weather: pd.DataFrame, lat: float, lon: float, altitude: float, cfg: PVConfig) -> pd.DataFrame:
    """weather: columns ghi, dni, dhi, t2m, ws10 (short names), UTC interval-end index."""
    geo = solar_geometry(weather.index, lat, lon, altitude)
    allsky = _ac_chain(weather["ghi"], weather["dni"], weather["dhi"], weather["t2m"], weather["ws10"], geo, cfg)
    clear = _ac_chain(geo["ghi_cs"], geo["dni_cs"], geo["dhi_cs"], weather["t2m"], weather["ws10"], geo, cfg)
    out = pd.concat([geo[["zenith", "elevation", "azimuth", "ghi_cs"]], allsky], axis=1)
    out["p_cs_norm"] = clear["p_norm"]
    # Missing weather must give missing (not zero) power.
    missing = weather[["ghi", "dni", "dhi", "t2m", "ws10"]].isna().any(axis=1)
    out.loc[missing, ["poa_global", "t_cell", "p_norm"]] = np.nan
    return out
