"""Forecast metrics on daylight hours. Targets are normalised power, so errors x100 = % of capacity."""
import numpy as np
import pandas as pd

DAYLIGHT_ELEV = 5.0


def metrics(y, yhat, ref=None) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = yhat - y
    out = {"nMAE_%": 100 * np.mean(np.abs(e)), "nRMSE_%": 100 * np.sqrt(np.mean(e ** 2)), "MBE_%": 100 * np.mean(e),
           "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), "n": len(y)}
    if ref is not None:
        out["skill_vs_smart"] = 1 - np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean((np.asarray(ref, float) - y) ** 2))
    return out


def table(df: pd.DataFrame, models: list[str], ref: str = "bl_smart", by: list[str] | None = None) -> pd.DataFrame:
    d = df[df.elevation > DAYLIGHT_ELEV]
    rows = []
    groups = d.groupby(by) if by else [((), d)]
    for key, g in groups:
        for m in models:
            ok = g[m].notna() & g[ref].notna()
            r = metrics(g.y[ok], g[m][ok], g[ref][ok])
            r["model"] = m
            if by:
                r.update(dict(zip(by, key if isinstance(key, tuple) else (key,))))
            rows.append(r)
    return pd.DataFrame(rows)
