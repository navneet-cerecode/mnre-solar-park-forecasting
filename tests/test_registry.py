import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarpark import registry  # noqa: E402


def test_registry_reconciles_with_official_totals():
    reg, locs = registry.build_registry()
    assert len(reg) == 50
    assert reg.sanctioned_capacity_mw.sum() == 37990          # LS Q.3633
    assert reg.commissioned_capacity_mw.sum() == 10237        # PIB 1941144
    core = reg[reg.tier == "U2_core"]
    assert len(core) == 18 and core.capacity_mw_model.sum() == 10237


def test_tiers_and_locations():
    reg, locs = registry.build_registry()
    assert (reg.tier != "excluded").sum() == 43
    assert reg.loc[reg.tier == "excluded", "loc_id"].isna().all()
    # co-located parks share a weather location
    bhadla = reg[reg.park_id.isin(["P35", "P36", "P37"])].loc_id.unique()
    assert len(bhadla) == 1
    assert locs.loc_id.is_unique
