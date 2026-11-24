"""Step 4 - data-quality report for weather inputs and simulated targets.

Writes reports/qc/QC_REPORT.md plus CSV tables and PNG figures.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solarpark import config, openmeteo, pvsim, registry

OUT = config.QC_DIR
OUT.mkdir(parents=True, exist_ok=True)
S0, S1 = pd.Timestamp(config.STUDY_START, tz="UTC"), pd.Timestamp(config.STUDY_END + " 23:00", tz="UTC")
INK2, GRID, C1, C2, C3 = "#52514e", "#e1e0d9", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

reg = registry.load_registry()
locs = registry.load_locations()
expected_hours = int((S1 - S0) / pd.Timedelta(hours=1)) + 1

# ------------------------------------------------------------------ weather QC
wrows, frows = [], []
fc_pairs = {}
for _, L in locs.iterrows():
    era, meta = openmeteo.load_dataset("era5", L.loc_id)
    era = openmeteo.rename_short(era).loc[S0:S1]
    fc, _ = openmeteo.load_dataset("fc_ecmwf", L.loc_id)
    fc = openmeteo.rename_short(fc).loc[S0:S1]
    geo = pvsim.solar_geometry(era.index, L.latitude, L.longitude, meta["elevation_m"] or 0)
    toa_h = geo.dni_extra * np.cos(np.radians(geo.zenith)).clip(lower=0)
    night = geo.elevation < -2
    for name, df, g in (("era5", era, "ghi"), ("fc_ecmwf_d1", fc, "ghi_d1")):
        d = "dhi" if name == "era5" else "dhi_d1"
        wrows.append(dict(
            loc_id=L.loc_id, dataset=name, hours=len(df), expected=expected_hours,
            completeness_pct=round(100 * df[g].notna().sum() / expected_hours, 2),
            any_var_missing_pct=round(100 * df.isna().any(axis=1).mean(), 2),
            night_ghi_gt5=int((df[g][night] > 5).sum()),
            ghi_gt_1p1_toa=int((df[g] > 1.1 * toa_h + 50).sum()),
            dhi_gt_ghi=int((df[d] > df[g] + 10).sum()),
            clear_sky_index_gt_1p1_pct=round(100 * ((df[g] / geo.ghi_cs)[geo.elevation > 15] > 1.1).mean(), 2),
            grid_lat=meta["grid_lat"], grid_lon=meta["grid_lon"], elevation_m=meta["elevation_m"]))
    # NWP day-1 forecast vs ERA5 (daylight, elevation > 5 deg)
    j = pd.concat([era.ghi, fc.ghi_d1, fc.ghi_d2, geo.elevation], axis=1).dropna()
    j = j[j.elevation > 5]
    for lead in ("ghi_d1", "ghi_d2"):
        e = j[lead] - j.ghi
        frows.append(dict(loc_id=L.loc_id, state=L.state, lead=lead[-2:], n=len(j),
                          bias_wm2=round(e.mean(), 1), mae_wm2=round(e.abs().mean(), 1),
                          rmse_wm2=round(np.sqrt((e ** 2).mean()), 1), corr=round(j[lead].corr(j.ghi), 3),
                          nrmse_pct_of_mean=round(100 * np.sqrt((e ** 2).mean()) / j.ghi.mean(), 1)))
    fc_pairs[L.loc_id] = j
wq = pd.DataFrame(wrows); fq = pd.DataFrame(frows)
wq.to_csv(OUT / "weather_qc.csv", index=False); fq.to_csv(OUT / "nwp_vs_era5_ghi.csv", index=False)

# ------------------------------------------------------------------ target QC
t = pd.read_parquet(config.TARGETS_FILE)
t = t[(t.time_utc >= S0) & (t.time_utc <= S1)]
t = t.merge(reg[["park_id", "park_name", "state", "grid_region"]], on="park_id")
t["month"] = t.time_utc.dt.tz_convert("Asia/Kolkata").dt.month
day = t[t.elevation > 0]
k = (t.p_norm / t.p_cs_norm).where(t.p_cs_norm > 0.1)
t["k"] = k
yr = t[(t.time_utc >= pd.Timestamp("2025-01-01", tz="UTC")) & (t.time_utc < pd.Timestamp("2026-01-01", tz="UTC"))]
trows = []
for pid, g in t.groupby("park_id"):
    gy = yr[yr.park_id == pid]
    trows.append(dict(
        park_id=pid, park_name=g.park_name.iat[0][:40], state=g.state.iat[0], tier=g.tier.iat[0],
        capacity_mw=g.capacity_mw.iat[0], hours=len(g), missing_pct=round(100 * g.p_norm.isna().mean(), 3),
        cf_2025_pct=round(100 * gy.p_norm.mean(), 2), cf_clearsky_2025_pct=round(100 * gy.p_cs_norm.mean(), 2),
        cf_monsoon_jjas_pct=round(100 * gy[gy.month.isin([6, 7, 8, 9])].p_norm.mean(), 2),
        cf_premonsoon_mam_pct=round(100 * gy[gy.month.isin([3, 4, 5])].p_norm.mean(), 2),
        energy_2025_gwh=round(gy.p_mw.sum() / 1000, 1),
        max_p_norm=round(g.p_norm.max(), 4), clipped_daylight_pct=round(100 * (g.p_norm > 0.99).sum() / max(1, (g.elevation > 0).sum()), 2),
        night_nonzero=int(((g.elevation < -2) & (g.p_norm > 0)).sum()),
        k_gt_1p05_pct=round(100 * (g.k > 1.05).mean(), 2),
        ramp_hours_dk_gt_0p3=int((g.k.diff().abs() > 0.3).sum())))
tq = pd.DataFrame(trows).sort_values("cf_2025_pct", ascending=False)
tq.to_csv(OUT / "target_qc.csv", index=False)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(9, 6.2))
tqs = tq.sort_values("cf_2025_pct")
cols = [C1 if x == "U2_core" else C3 for x in tqs.tier]
ax.barh(range(len(tqs)), tqs.cf_2025_pct, color=cols, height=0.7)
ax.set_yticks(range(len(tqs))); ax.set_yticklabels([f"{a} {b[:26]}" for a, b in zip(tqs.park_id, tqs.park_name)], fontsize=6.3)
ax.set_xlabel("Simulated capacity factor 2025 (% of AC capacity)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C1, label="U2 core (commissioned)"), Patch(color=C3, label="U3 virtual (under implementation)")],
          frameon=False, loc="lower right", fontsize=7.5)
fig.savefig(OUT / "fig_cf_by_park.png", dpi=170, bbox_inches="tight"); plt.close(fig)

piv = yr.groupby(["park_id", "month"]).p_norm.mean().unstack() * 100
piv = piv.loc[tq.park_id]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6.3)
fig.colorbar(im, ax=ax, label="monthly CF (%)", shrink=0.6)
ax.set_title("Simulated monthly capacity factor, 2025 (parks sorted by annual CF)", fontsize=9, loc="left")
fig.savefig(OUT / "fig_monthly_cf_heatmap.png", dpi=170, bbox_inches="tight"); plt.close(fig)

fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharey=True)
for ax, (a, b, ttl) in zip(axs, [("2025-01-13", "2025-01-20", "January (winter)"), ("2025-07-14", "2025-07-21", "July (monsoon)")]):
    g = t[(t.park_id == "P35") & (t.time_utc >= pd.Timestamp(a, tz="UTC")) & (t.time_utc < pd.Timestamp(b, tz="UTC"))]
    ax.plot(g.time_utc, g.p_cs_norm, color=INK2, lw=1, ls="--", label="clear-sky")
    ax.plot(g.time_utc, g.p_norm, color=C1, lw=1.6, label="simulated (ERA5)")
    ax.set_title(f"Bhadla-II (P35), {ttl}", fontsize=9, loc="left"); ax.set_ylabel("p_norm"); ax.grid(color=GRID, lw=0.5)
axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig_example_weeks_bhadla.png", dpi=170); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, lid, ttl in zip(axs, ["L25", "L13"], ["Bhadla (L25, desert)", "Kasaragod (L13, humid coast)"]):
    j = fc_pairs[lid]
    ax.hexbin(j.ghi, j.ghi_d1, gridsize=45, cmap="Blues", mincnt=1, bins="log")
    ax.plot([0, 1100], [0, 1100], color=C2, lw=1)
    r = fq[(fq.loc_id == lid) & (fq.lead == "d1")].iloc[0]
    ax.set_title(f"{ttl}\nRMSE {r.rmse_wm2} W/m2, bias {r.bias_wm2}, r = {r['corr']}", fontsize=8.5, loc="left")
    ax.set_xlabel("ERA5 GHI (W/m2)"); ax.set_ylabel("ECMWF day-1 forecast GHI (W/m2)")
fig.tight_layout(); fig.savefig(OUT / "fig_nwp_vs_era5.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------ markdown report
def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)

core = tq[tq.tier == "U2_core"]
fq1 = fq[fq.lead == "d1"]
issues = []
if (wq.completeness_pct < 99.5).any():
    issues.append(f"Weather completeness below 99.5% at: {', '.join(wq.loc[wq.completeness_pct < 99.5, 'loc_id'] + '/' + wq.loc[wq.completeness_pct < 99.5, 'dataset'])}")
if (tq.night_nonzero > 0).any():
    issues.append("Night-time non-zero simulated output found (should be impossible).")
if tq.max_p_norm.max() > 1.0:
    issues.append("p_norm above 1.0 found.")
report = f"""# Phase-1 Data Quality Report

Generated by `scripts/04_qc_report.py`. Study window {config.STUDY_START} to {config.STUDY_END} (UTC, hourly, interval-ending), {expected_hours:,} hours.

## Summary

- Weather locations: **{len(locs)}**; modelled parks: **{t.park_id.nunique()}** ({(reg.tier == 'U2_core').sum()} core, {(reg.tier == 'U3_virtual').sum()} virtual).
- ERA5 completeness: min **{wq[wq.dataset == 'era5'].completeness_pct.min()}%**; ECMWF day-1 forecast completeness: min **{wq[wq.dataset == 'fc_ecmwf_d1'].completeness_pct.min()}%**.
- Physical-consistency flags (all locations): night GHI > 5 W/m2: **{int(wq.night_ghi_gt5.sum())}** h; GHI > 1.1 x top-of-atmosphere + 50: **{int(wq.ghi_gt_1p1_toa.sum())}** h; DHI > GHI: **{int(wq.dhi_gt_ghi.sum())}** h.
- Simulated targets: max p_norm **{tq.max_p_norm.max()}**, night-time non-zero hours **{int(tq.night_nonzero.sum())}**, missing **{tq.missing_pct.max()}%** (worst park).
- 2025 simulated capacity factor (AC basis, DC/AC 1.25): core parks **{core.cf_2025_pct.min()}-{core.cf_2025_pct.max()}%** (median {core.cf_2025_pct.median():.2f}%); all parks {tq.cf_2025_pct.min()}-{tq.cf_2025_pct.max()}%.
- ECMWF day-1 GHI forecast vs ERA5 (daylight): RMSE **{fq1.rmse_wm2.min()}-{fq1.rmse_wm2.max()} W/m2** (median {fq1.rmse_wm2.median()}), correlation {fq1['corr'].min()}-{fq1['corr'].max()}.
- Hours where ERA5 GHI exceeds the Ineichen clear-sky GHI by >10% (sun > 15 deg): median **{wq[wq.dataset == 'era5'].clear_sky_index_gt_1p1_pct.median()}%** of hours - the clear-sky index therefore needs capping or a data-driven clear-sky reference in Phase 2.

**Issues requiring action:** {'; '.join(issues) if issues else 'none blocking.'}

## Interpretation and caveats

1. These are **simulated** targets (pvlib + ERA5) and describe weather-driven potential output, not metered generation: no curtailment, outages or grid limits.
2. Capacity factors are on an **AC basis** with an assumed DC/AC ratio of 1.25; published Indian CUFs are often quoted on a different basis, so compare shapes, not levels, until measured data are available.
3. The NWP-vs-ERA5 comparison measures forecast error against reanalysis, not against ground measurements. It sets the realistic floor for forecast error when targets are simulated from ERA5.

## Target QC by park

{md(tq)}

## NWP forecast vs ERA5 (GHI, daylight hours)

{md(fq.sort_values(['lead', 'rmse_wm2']))}

## Weather QC by location

{md(wq)}

## Figures

![CF by park](fig_cf_by_park.png)

![Monthly CF](fig_monthly_cf_heatmap.png)

![Example weeks](fig_example_weeks_bhadla.png)

![NWP vs ERA5](fig_nwp_vs_era5.png)
"""
(OUT / "QC_REPORT.md").write_text(report, encoding="utf-8")
print(report[:2500])
