"""Step 3 - simulate hourly generation targets for every modelled park with pvlib.

Targets are built from ERA5 "observed" weather (never from the NWP forecasts that
will later be model inputs) - see docs/DECISIONS.md, D3.
Output: data/processed/targets_simulated.parquet (long format, one row per park-hour).
"""
import _bootstrap  # noqa: F401
import pandas as pd

from solarpark import config, openmeteo, pvsim, registry

reg = registry.load_registry()
parks = reg[reg.tier != "excluded"]
frames, cfg_rows = [], []
weather_cache = {}
for _, p in parks.iterrows():
    if p.loc_id not in weather_cache:
        wx, meta = openmeteo.load_dataset("era5", p.loc_id)
        weather_cache[p.loc_id] = (openmeteo.rename_short(wx), meta)
    wx, meta = weather_cache[p.loc_id]
    cfg = pvsim.config_for_park(p.latitude, p.state, p.mounting)
    sim = pvsim.simulate(wx, p.latitude, p.longitude, meta["elevation_m"] or 0.0, cfg)
    sim = sim.reset_index().rename(columns={"index": "time_utc"})
    sim.insert(0, "park_id", p.park_id)
    sim["p_mw"] = sim.p_norm * p.capacity_mw_model
    sim["capacity_mw"] = p.capacity_mw_model
    sim["tier"] = p.tier
    sim["target_source"] = "simulated_pvlib_era5"
    frames.append(sim)
    cfg_rows.append(dict(park_id=p.park_id, loc_id=p.loc_id, mounting=cfg.mounting, tilt=cfg.tilt,
                         azimuth=cfg.azimuth, albedo=cfg.albedo, dc_ac=cfg.dc_ac, gamma_pdc=cfg.gamma_pdc,
                         soiling_pct=cfg.losses["soiling"], total_loss_pct=round(cfg.loss_fraction * 100, 2),
                         grid_lat=meta["grid_lat"], grid_lon=meta["grid_lon"], elevation_m=meta["elevation_m"]))
    print(f"{p.park_id} {p.park_name[:38]:38s} {p.tier:10s} mean p_norm={sim.p_norm.mean():.3f}", flush=True)

targets = pd.concat(frames, ignore_index=True)
targets = targets.rename(columns={"time_utc": "time_utc"})
config.PROCESSED.mkdir(parents=True, exist_ok=True)
targets.to_parquet(config.TARGETS_FILE, index=False)
config.QC_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(cfg_rows).to_csv(config.QC_DIR / "pv_simulation_config.csv", index=False)
print(f"\n{len(targets):,} rows for {targets.park_id.nunique()} parks -> {config.TARGETS_FILE.relative_to(config.ROOT)}")
