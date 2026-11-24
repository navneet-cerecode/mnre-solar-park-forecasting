"""Step 12 - export everything the static website needs into site/data/.

  meta.json   parks, KPIs, experiment results, simplified India state outlines (SVG paths)
  h{h}.json   test-period hourly series per park, packed as base64 uint8 (value = round(p_norm * 250))
Boundaries: data/raw/boundaries/Admin2.* (datameet, Survey-of-India-based).
"""
import base64
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import shapefile

from solarpark import config, forecasts as F

SITE = config.ROOT / "site" / "data"
SITE.mkdir(parents=True, exist_ok=True)
RES = config.REPORTS / "results"
SERIES = ["y", "lightgbm", "gru", "bl_smart", "lo", "hi"]
LAT0 = 22.0
KX = np.cos(np.radians(LAT0))


def proj(lon, lat):
    """Equirectangular projection to SVG units (x right, y down)."""
    return (lon - 68.0) * KX * 20, (37.5 - lat) * 20


def rdp(pts, eps):
    """Ramer-Douglas-Peucker line simplification."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab) or 1e-12
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])


def outlines():
    r = shapefile.Reader(str(config.RAW / "boundaries" / "Admin2"))
    out = []
    for shp, rec in zip(r.shapes(), r.records()):
        parts = list(shp.parts) + [len(shp.points)]
        d = []
        for a, b in zip(parts[:-1], parts[1:]):
            ring = np.array(shp.points[a:b])
            if len(ring) < 4:
                continue
            mid = len(ring) // 2          # closed ring: first == last point, so simplify each half
            ring = np.vstack([rdp(ring[: mid + 1], 0.03)[:-1], rdp(ring[mid:], 0.03)])
            if len(ring) < 4:
                continue
            xy = [proj(x, y) for x, y in ring]
            d.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in xy) + "Z")
        if d:
            out.append({"name": rec[0], "d": "".join(d)})
    return out


def table(name):
    p = RES / f"{name}.csv"
    return json.loads(pd.read_csv(p).round(3).to_json(orient="records")) if p.exists() else []


parks = F.parks()
clim_f = config.INTERIM / "park_climate.csv"
if clim_f.exists():
    parks = parks.merge(pd.read_csv(clim_f)[["park_id", "koppen", "koppen_name", "cluster"]], on="park_id", how="left")
HS = [h for h in (*config.HORIZONS_H, config.DAY_AHEAD_H) if (config.PROCESSED / f"preds_h{h}.parquet").exists()]
f1 = F.forecasts(1)
modelled = sorted(f1.park_id.unique())
park_rows = []
for _, p in parks.iterrows():
    x, y = proj(p.longitude, p.latitude) if pd.notna(p.latitude) else (None, None)
    park_rows.append({"id": p.park_id, "name": p.park_name, "state": p.state, "region": p.grid_region, "lat": p.latitude if pd.notna(p.latitude) else None,
                      "lon": p.longitude if pd.notna(p.longitude) else None, "x": x, "y": y,
                      "sanctioned": int(p.sanctioned_capacity_mw), "commissioned": int(p.commissioned_capacity_mw),
                      "cap": float(p.capacity_mw_model) if pd.notna(p.capacity_mw_model) else None, "status": p.status,
                      "tier": p.tier, "sppd": str(p.sppd).split(" (")[0], "data": p.generation_data_availability,
                      "modelled": p.park_id in modelled,
                      "koppen": p.get("koppen") if pd.notna(p.get("koppen")) else None,
                      "koppen_name": p.get("koppen_name") if pd.notna(p.get("koppen_name")) else None,
                      "cluster": p.get("cluster") if pd.notna(p.get("cluster")) else None})

hours = pd.date_range(f1.time_utc.min().floor("D"), f1.time_utc.max().ceil("D"), freq="h", tz="UTC")
meta = {"parks": park_rows, "modelled": modelled, "t0_utc": hours[0].isoformat(), "n_hours": len(hours), "horizons": HS,
        "series": SERIES, "scale": 250, "halfwidth_mw": {h: F.total_halfwidth_mw(h) for h in HS},
        "outlines": outlines(),
        "results": {n: table(n) for n in ["e1_models", "e1b_deep_learning", "e2_per_park_vs_global", "e3_regional_vs_global", "e4_ablation",
                                          "e4_shap_share", "e5_aggregation", "e5_error_corr_vs_distance", "e6_ramps_regimes",
                                          "e7_unseen_locations_detail", "e8_intervals", "e8_cqr_monthly_coverage", "e9_oracle_vs_nwp",
                                          "e10_catboost_tuning", "e11_real_physics_check", "e11_real_forecasting",
                                          "e3b_climate_groups", "e3b_climate_groups_detail"]}}
(SITE / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")

for h in HS:
    f = F.forecasts(h)
    grid = np.zeros((len(modelled), len(SERIES), len(hours)), dtype=np.uint8)   # night hours stay 0
    pos = hours.get_indexer(f.time_utc)
    assert (pos >= 0).all()
    pi = pd.Index(modelled).get_indexer(f.park_id)
    for s, col in enumerate(SERIES):
        grid[pi, s, pos] = np.round(np.clip(f[col].fillna(0).to_numpy(), 0, 1) * 250).astype(np.uint8)
    # self-check: decoding reproduces the totals within the 0.4%-of-capacity quantisation
    cap = np.array([parks.set_index("park_id").capacity_mw_model[p] for p in modelled])
    dec_total = (grid[:, 0, :].astype(float) / 250 * cap[:, None]).sum()
    assert abs(dec_total - f.y_mw.sum()) / f.y_mw.sum() < 0.005, "packing error"
    (SITE / f"h{h}.json").write_text(json.dumps({"h": h, "data": base64.b64encode(grid.tobytes()).decode()}), encoding="utf-8")
    print(f"h={h}: {grid.shape}, {(SITE / f'h{h}.json').stat().st_size / 1e6:.1f} MB")
print(f"meta.json {(SITE / 'meta.json').stat().st_size / 1e6:.2f} MB, {len(meta['outlines'])} state outlines")
