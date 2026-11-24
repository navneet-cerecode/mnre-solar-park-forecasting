"""Park registry: turns the verified master table into modelling-ready metadata.

Tiers (scope document, Section 8.4):
  U2 core    - commissioned capacity > 0; forecast in MW at commissioned capacity
  U3 virtual - under implementation; simulated at sanctioned capacity, flagged virtual
  excluded   - dropped/revised by 2026, or no verified location
"""
import pandas as pd

from . import config


def load_master() -> pd.DataFrame:
    return pd.read_csv(config.MASTER_TABLE, encoding="utf-8-sig")


def assign_tier(row) -> str:
    if row["status"].startswith("Dropped"):
        return "excluded"
    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        return "excluded"
    if row["commissioned_capacity_mw"] > 0:
        return "U2_core"
    return "U3_virtual"


def mounting_for(name: str) -> str:
    return "floating" if "floating" in name.lower() else "fixed_ground"


def build_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = load_master()
    reg = m[["park_id", "park_name", "state", "grid_region", "district", "latitude", "longitude",
             "coordinate_method", "sanctioned_capacity_mw", "commissioned_capacity_mw", "status",
             "generation_data_availability"]].copy()
    reg["tier"] = reg.apply(assign_tier, axis=1)
    reg["exclusion_reason"] = ""
    reg.loc[reg.status.str.startswith("Dropped"), "exclusion_reason"] = "dropped/revised by 2026"
    reg.loc[reg.latitude.isna(), "exclusion_reason"] = "location not publicly verified"

    reg["mounting"] = reg.park_name.map(mounting_for)
    # Capacity used to convert normalised power to MW. Park-wise commissioning after
    # Oct-2023 is not published, so core parks use the Oct-2023 figure (a lower bound).
    reg["capacity_mw_model"] = reg.commissioned_capacity_mw.where(reg.tier == "U2_core", reg.sanctioned_capacity_mw)
    reg["capacity_basis"] = reg.tier.map({
        "U2_core": "commissioned (Aug-Oct 2023; later additions not published park-wise)",
        "U3_virtual": "sanctioned (virtual site - no commissioned capacity)",
        "excluded": "-",
    })

    # Parks at identical coordinates share one weather location (e.g. Bhadla II-IV, Khavda).
    inc = reg[reg.tier != "excluded"].copy()
    keys = inc[["latitude", "longitude"]].round(4).apply(tuple, axis=1)
    order = {k: i + 1 for i, k in enumerate(dict.fromkeys(keys))}
    inc["loc_id"] = [f"L{order[k]:02d}" for k in keys]
    reg = reg.merge(inc[["park_id", "loc_id"]], on="park_id", how="left")

    locs = (inc.groupby("loc_id")
              .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                   state=("state", "first"), parks=("park_id", lambda s: ",".join(s)))
              .reset_index())
    return reg, locs


def save_registry():
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    reg, locs = build_registry()
    reg.to_csv(config.REGISTRY_FILE, index=False)
    locs.to_csv(config.LOCATIONS_FILE, index=False)
    return reg, locs


def load_registry() -> pd.DataFrame:
    return pd.read_csv(config.REGISTRY_FILE)


def load_locations() -> pd.DataFrame:
    return pd.read_csv(config.LOCATIONS_FILE)
