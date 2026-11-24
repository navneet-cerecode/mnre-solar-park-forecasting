"""Step 1 - build the park registry and weather-location table from the master table."""
import _bootstrap  # noqa: F401
from solarpark import config, registry

reg, locs = registry.save_registry()
print(reg.tier.value_counts().to_string())
print(f"\nmodelled parks: {(reg.tier != 'excluded').sum()}  |  unique weather locations: {len(locs)}")
print(reg.loc[reg.tier == "excluded", ["park_id", "park_name", "exclusion_reason"]].to_string(index=False))
print(f"\nwrote {config.REGISTRY_FILE.relative_to(config.ROOT)} and {config.LOCATIONS_FILE.relative_to(config.ROOT)}")
