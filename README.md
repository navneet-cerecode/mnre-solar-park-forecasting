# Multi-Site Solar Power Forecasting - MNRE-Sanctioned Solar Parks

B.Tech research project (NIT Patna, Department of Electrical Engineering). Short-term (1-6 h) solar generation forecasting across India's
MNRE-sanctioned solar parks. Background, research questions and methodology are in
`MNRE_42_Solar_Parks_Forecasting_Project_Scope.pdf`. Implementation choices that differ from or
refine the scope are logged in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Status

| Phase | Content | State |
|---|---|---|
| P0 | Scope, park verification, master table | done |
| P1 | Data pipeline: registry → weather → pvlib targets → QC | done |
| P2–P6 | Features, baselines, ML (GPU XGBoost), LSTM/GRU (GPU), experiments E1–E9, intervals | done — see `reports/results/RESULTS.md` |
| P7 | API + India map dashboard | done (serves saved test-period forecasts) |

## Layout

```
MNRE_Solar_Parks_Master_Table.csv/.xlsx   verified park metadata (input to everything)
src/solarpark/
    config.py      paths, study window, splits, weather variables, PV assumptions
    registry.py    tiers (U2 core / U3 virtual / excluded), weather locations
    openmeteo.py   resumable, rate-limited Open-Meteo downloader (cached Parquet chunks)
    pvsim.py       pvlib chain: solar position → Perez POA → cell temp → PVWatts DC → losses → AC
scripts/
    01_build_registry.py    → data/interim/park_registry.csv, locations.csv
    02_fetch_weather.py     → data/raw/openmeteo/{era5,fc_ecmwf}/<loc>/<year>.parquet
    03_simulate_targets.py  → data/processed/targets_simulated.parquet
    04_qc_report.py         → reports/qc/QC_REPORT.md (+ CSV tables, figures)
tests/             pytest unit tests (registry totals, PV physics sanity)
docs/DECISIONS.md  design-decision log
```

## Run

```bash
pip install -r requirements.txt
python scripts/01_build_registry.py
python scripts/02_fetch_weather.py --dry-run     # shows estimated API cost
python scripts/02_fetch_weather.py               # ~2 h first time (rate-limited); resumable
python scripts/03_simulate_targets.py
python scripts/04_qc_report.py
python scripts/05_build_features.py      # model datasets for 1, 3, 6 h ahead
python scripts/06_train_models.py        # E1: baselines, Ridge, RF, XGBoost (GPU), LightGBM
python scripts/07_experiments.py         # E2-E9 (or pass e.g. E5 E8)
python scripts/08_train_deep.py          # LSTM / GRU on the GPU (CUDA PyTorch)
python scripts/09_results_summary.py     # -> reports/results/RESULTS.md
python scripts/10_catboost_tuning.py     # E10: CatBoost (GPU) + LightGBM tuning grid
python scripts/11_real_data_check.py     # E11: real measured plants (Kaggle data in data/raw/kaggle_solar/)
python -m pytest -q tests
```

## Website

`site/index.html` + `site/data/*.json` is a static website (map, park explorer, totals, results). Rebuild the data with
`python scripts/12_export_site.py`; preview locally with `python -m http.server 8601 --directory site` and open
`http://localhost:8601/_preview.html` (the published copy is the hosted Artifact).

**Public site:** https://navneet-cerecode.github.io/mnre-solar-forecast/ (GitHub Pages, repo `navneet-cerecode/mnre-solar-forecast`,
website files only). To update it: `python scripts/12_export_site.py`, `python scripts/14_build_public_site.py`, then commit and
push inside `site_public/`.

## Dashboard and API

```bash
streamlit run app/dashboard.py           # http://localhost:8501 - India map, park explorer, totals, results
uvicorn app.api:app                      # http://127.0.0.1:8000/docs - /parks, /forecast/park/{id}, /forecast/total, /metrics
```

## Data sources

- Park metadata: Lok Sabha Q.3633 (10-08-2023), PIB releases, SECI (see the master table's `source` column).
- Observed weather (targets): ERA5 reanalysis via the Open-Meteo Historical Weather API.
- Forecast weather (features): ECMWF IFS 0.25° archived forecasts via the Open-Meteo Previous Runs API
  (`previous_day1`, `previous_day2`). Open-Meteo data are free for non-commercial use (CC BY 4.0); cite Open-Meteo and ECMWF/Copernicus.

## Key caveat

Targets are **simulated** (pvlib + ERA5) because no MNRE park publishes hourly metered generation.
Results describe the forecastability of weather-driven output; label them as such and replace with
measured data wherever it becomes available (`target_source` column).
