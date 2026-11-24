"""Central configuration for the MNRE solar-park forecasting pipeline.

All paths are relative to the project root so the pipeline is portable.
Design decisions behind these values are logged in docs/DECISIONS.md.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- paths
MASTER_TABLE = ROOT / "MNRE_Solar_Parks_Master_Table.csv"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
QC_DIR = REPORTS / "qc"

REGISTRY_FILE = INTERIM / "park_registry.csv"
LOCATIONS_FILE = INTERIM / "locations.csv"
TARGETS_FILE = PROCESSED / "targets_simulated.parquet"
API_USAGE_LOG = RAW / "openmeteo_usage.json"

# ---------------------------------------------------------------- study period (UTC)
# Archived NWP forecasts that are free of look-ahead (Open-Meteo Previous Runs API,
# ECMWF IFS 0.25) start in March 2024, so the modelling window is Apr-2024 .. Sep-2026.
# Reanalysis starts one month earlier to provide history for lag/rolling features.
REANALYSIS_START = "2024-03-01"
FORECAST_START = "2024-03-15"
STUDY_START = "2024-04-01"
STUDY_END = "2026-09-24"          # ERA5 has ~5 days latency; keep a safety margin

SPLITS = {                         # chronological, never shuffled
    "train": ("2024-04-01", "2025-06-30"),
    "calib": ("2025-07-01", "2025-12-31"),
    "test": ("2026-01-01", STUDY_END),
}

HORIZONS_H = [1, 3, 6]             # core intraday horizons
DAY_AHEAD_H = 24                   # extension


def horizons_from_argv(argv, default=None):
    """'h=24' or 'h=1,24' in argv selects horizons; otherwise the core horizons."""
    for a in argv:
        if a.startswith("h="):
            return [int(x) for x in a[2:].split(",")]
    return list(default or HORIZONS_H)

# ---------------------------------------------------------------- weather variables
WEATHER_VARS = [
    "shortwave_radiation",         # GHI, W/m2, mean of preceding hour
    "direct_normal_irradiance",    # DNI
    "diffuse_radiation",           # DHI
    "temperature_2m",              # deg C
    "relative_humidity_2m",        # %
    "cloud_cover",                 # %
    "wind_speed_10m",              # m/s (wind_speed_unit=ms)
    "surface_pressure",            # hPa
    "precipitation",               # mm, sum of preceding hour
]
SHORT_NAMES = {
    "shortwave_radiation": "ghi", "direct_normal_irradiance": "dni", "diffuse_radiation": "dhi",
    "temperature_2m": "t2m", "relative_humidity_2m": "rh", "cloud_cover": "cc",
    "wind_speed_10m": "ws10", "surface_pressure": "sp", "precipitation": "tp",
}

# Lead-time rule (look-ahead safety): a forecast for valid time v taken from the
# "_previous_dayN" series was issued about N*24 h before v. For issue time t and
# horizon h it is usable when N*24 - h >= ~6 h of NWP latency.
#   previous_day1 -> horizons up to 18 h (covers 1, 3, 6 h)
#   previous_day2 -> horizons up to 42 h (covers day-ahead)
FORECAST_LEADS_DAYS = [1, 2]

DATASETS = {
    "era5": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "params": {"models": "era5"},
        "variables": WEATHER_VARS,
        "start": REANALYSIS_START,
        "end": STUDY_END,
        "role": "observed weather -> simulated targets and evaluation",
    },
    "fc_ecmwf": {
        "url": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "params": {"models": "ecmwf_ifs025"},
        "variables": [f"{v}_previous_day{d}" for d in FORECAST_LEADS_DAYS for v in WEATHER_VARS],
        "start": FORECAST_START,
        "end": STUDY_END,
        "role": "archived NWP forecasts -> model features",
    },
    # Optional second NWP model (multi-model features); fetched only on request.
    "fc_gfs": {
        "url": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "params": {"models": "gfs_seamless"},
        "variables": [f"{v}_previous_day{d}" for d in FORECAST_LEADS_DAYS for v in WEATHER_VARS],
        "start": FORECAST_START,
        "end": STUDY_END,
        "role": "second NWP model (optional)",
    },
}
DEFAULT_DATASETS = ["era5", "fc_ecmwf"]

# Open-Meteo free tier: 600 calls/min, 5,000/hour, 10,000/day, where requests with
# >10 variables or >2 weeks of data count as several calls. Stay well below.
RATE_BUDGET = {"minute": 300, "hour": 3500, "day": 8000}
HTTP_TIMEOUT_S = 120
USER_AGENT = "BTech-MNRE-SolarPark-Forecasting/0.1 (academic project)"

# ---------------------------------------------------------------- PV simulation assumptions
# Every value is an assumption to be reported in the thesis (scope document, Table 16).
PV = {
    "dc_ac_ratio": 1.25,
    "gamma_pdc": -0.0037,          # 1/deg C, crystalline Si
    "inverter_eta": 0.96,
    "albedo_default": 0.20,
    "albedo_desert": 0.25,
    "floating_tilt_deg": 10.0,
    "iam_b": 0.05,                 # ASHRAE incidence-angle modifier
    "losses_default": dict(soiling=2, shading=3, snow=0, mismatch=2, wiring=2,
                           connections=0.5, lid=1.5, nameplate_rating=1, age=0, availability=3),
    "soiling_desert": 4,           # Thar / Kutch parks
    "faiman_floating": {"u0": 35.0, "u1": 8.0},   # stronger cooling over water (assumption)
}
DESERT_STATES = {"Rajasthan", "Gujarat"}
