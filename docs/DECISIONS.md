# Design decisions log

Each entry records a choice made during implementation, why it was made, and its effect on the
scope document (v1.0, Oct-2026). Cite these in the thesis methodology chapter.

## D1 - Modelling window is Apr-2024 to Sep-2026 (not 2022-2025)

- **Finding:** archived NWP forecasts with a known lead time (Open-Meteo *Previous Runs API*)
  start in **January 2024 for GFS** and **mid-March 2024 for ECMWF IFS 0.25°** at Indian locations
  (probed 2026-10-03). The *Historical Forecast API* goes back to 2022, but it stitches the first hours
  of the latest run for each valid time, so it can contain information issued *after* the forecast
  issue time (look-ahead).
- **Decision:** use the Previous Runs API and a window of 2024-04-01 to 2026-09-24.
- **Splits (chronological):** train 2024-04-01 to 2025-06-30 · calibration/validation 2025-07-01 to 2025-12-31 ·
  test 2026-01-01 to 2026-09-24. Training includes one full monsoon (2024), calibration covers monsoon and
  post-monsoon 2025, and the test set covers winter, pre-monsoon and monsoon 2026.
- **Effect on scope:** replaces "train 2022-23 / validation 2024 / test 2025" (scope, Sec. 33).

## D2 - Lead-time rule for forecast features

A value from `*_previous_dayN` for valid time *v* was issued about N×24 h before *v*. With about 6 h of
NWP availability latency, it is usable at issue time *t = v − h* only if N×24 − h ≥ 6.

| Horizon h | Forecast series used | Margin |
|---|---|---|
| 1, 3, 6 h | `previous_day1` | 18-23 h |
| 24 h (day-ahead) | `previous_day2` | 24 h |

This is conservative (the freshest run is not used). Intraday skill must come from recent
(lagged) generation, as in operational practice when only day-old NWP is at hand.

## D3 - Targets from ERA5, features from NWP (no circular targets)

Simulated targets use **ERA5** weather at the valid time. Model features use **ECMWF IFS archived
forecasts**. A model therefore cannot reproduce the target simply by inverting pvlib, and its
error reflects real NWP error. An "oracle" run with ERA5 inputs is allowed only as an upper bound (E9).

## D4 - Time convention

All timestamps are **UTC, interval-ending**: the row at 07:00 UTC holds the mean of 06:00-07:00
(Open-Meteo "preceding hour" convention). Solar geometry is evaluated at the interval midpoint
(t − 30 min). IST = UTC + 5:30 is used only for display and IMD seasons.

## D5 - Weather locations

Parks with identical coordinates share one weather location: Bhadla II-IV → L25, the five Khavda parks → L09,
Pugal I-II → L29. That gives 43 modelled parks at 36 locations. Generalisation tests must hold out
co-located parks together (leave-one-location-out).

## D6 - Capacity used for MW conversion

Park-wise commissioning after Oct-2023 is not published, so core parks (U2) use their Oct-2023
commissioned capacity, a lower bound, constant over the window. Virtual parks (U3) use
sanctioned capacity. The normalised target `p_norm` is unaffected; only MW totals are.

## D7 - PV assumptions

Fixed tilt equal to latitude, south-facing; floating parks at 10° tilt with Faiman (u0=35, u1=8) cooling;
DC/AC 1.25; γ = −0.37 %/°C; PVWatts losses about 14 %, with soiling raised to 4 % and albedo to 0.25
in Rajasthan and Gujarat; ASHRAE IAM (b = 0.05); no degradation within the 2.5-year window. All are
assumptions, to be sensitivity-tested and calibrated if measured data become available.

## D8 - Excluded parks (7 of 50)

Dropped or revised by 2026: P15, P18. No verified location: P20, P32, P33, P34, P43. They stay in
the registry with an `exclusion_reason`.

## D9 - Real-data check (E11)

The Kaggle "Solar Power Generation Data" (two Indian plants, 22 inverters each, 15 May - 17 Jun 2020) was
taken from a public GitHub mirror (MainakRepositor/Datasets) because no Kaggle login is configured here.
Row counts match the original (68,778 / 3,182 / 67,698 / 3,259); three files are byte-identical to a second
mirror, and the fourth differs only because that mirror re-saved it through Excel. Plant locations are not
published, so ECMWF forecasts cannot be matched: E11 tests (A) the pvlib chain against measured power using
the on-site irradiance sensor, and (B) history + time-of-day forecasting, including zero-shot transfer of a
model trained only on the simulated MNRE parks. Plant capacity is not published either; the 99.5th percentile
of hourly AC output is used as effective capacity.

## D10 - Day-ahead horizon (24 h)

Day-ahead forecasts use the `previous_day2` ECMWF series: by rule D2 a forecast issued ~24 h before the valid
time is not yet available at a 24-h issue time. `features.lead_days(h)` applies the rule for every horizon.
Recent-power lags are still those known at t = v - 24 h.

## D11 - Climate grouping (E3b)

Koppen-Geiger classes come from the `kgcpy` lookup (kgc package: Rubel et al. 2017 map, 1986-2010). Data-driven
clusters use k-means on each location's monthly clear-sky index, cloud cover, temperature and rain from the
training period only; k is chosen by silhouette score among 3-6. Results are compared with the global model as in E3.
