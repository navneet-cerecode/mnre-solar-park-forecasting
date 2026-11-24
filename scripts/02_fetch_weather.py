"""Step 2 - download reanalysis and archived NWP forecasts for every weather location.

Resumable: cached chunks are skipped. Examples:
    python scripts/02_fetch_weather.py --dry-run
    python scripts/02_fetch_weather.py --locations L01           # quick test
    python scripts/02_fetch_weather.py                            # era5 + fc_ecmwf, all locations
    python scripts/02_fetch_weather.py --datasets fc_gfs          # optional second NWP model
"""
import argparse

import _bootstrap  # noqa: F401
from solarpark import config, openmeteo, registry

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=config.DEFAULT_DATASETS, choices=list(config.DATASETS))
ap.add_argument("--locations", nargs="*", help="subset of loc_ids, e.g. L01 L05")
ap.add_argument("--dry-run", action="store_true", help="print the estimated API cost and exit")
ap.add_argument("--force", action="store_true", help="re-download cached chunks")
a = ap.parse_args()

locs = registry.load_locations()
if a.locations:
    locs = locs[locs.loc_id.isin(a.locations)]
cost = openmeteo.estimated_cost(a.datasets, len(locs))
print(f"{len(locs)} locations x {a.datasets}: ~{cost:,.0f} weighted API calls "
      f"(budget/hour {config.RATE_BUDGET['hour']}, /day {config.RATE_BUDGET['day']})")
if a.dry_run:
    raise SystemExit
for ds in a.datasets:
    openmeteo.fetch_dataset(ds, locs, force=a.force)
print("done")
