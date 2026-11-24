"""Polite, resumable Open-Meteo downloader.

* One request per (location, dataset, calendar-year chunk); each chunk is cached as
  Parquet, so an interrupted run resumes where it stopped.
* A persistent, weighted rate limiter keeps usage under the free-tier limits.
* Timestamps are UTC and mark the END of the hour that radiation/precipitation
  values were averaged/summed over (Open-Meteo "preceding hour" convention).
"""
import json
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import config


# ------------------------------------------------------------------ rate limiting
class RateLimiter:
    WINDOWS = {"minute": 60, "hour": 3600, "day": 86400}

    def __init__(self, log_path: Path = config.API_USAGE_LOG, budget: dict = config.RATE_BUDGET):
        self.log_path, self.budget = Path(log_path), budget
        self.events = []
        if self.log_path.exists():
            self.events = [tuple(e) for e in json.loads(self.log_path.read_text())]
        self._prune()

    def _prune(self):
        cutoff = time.time() - self.WINDOWS["day"]
        self.events = [e for e in self.events if e[0] >= cutoff]

    def used(self, window: str) -> float:
        cutoff = time.time() - self.WINDOWS[window]
        return sum(w for t, w in self.events if t >= cutoff)

    def wait_for(self, weight: float):
        while True:
            self._prune()
            waits = []
            for win, limit in self.budget.items():
                if self.used(win) + weight > limit:
                    cutoff = time.time() - self.WINDOWS[win]
                    oldest = min((t for t, _ in self.events if t >= cutoff), default=time.time())
                    waits.append(oldest + self.WINDOWS[win] - time.time() + 1)
            if not waits:
                return
            pause = max(1.0, min(max(waits), 900))
            print(f"    rate budget reached - sleeping {pause:.0f}s", flush=True)
            time.sleep(pause)

    def record(self, weight: float):
        self.events.append((time.time(), weight))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.events))


def call_weight(n_vars: int, n_days: int) -> float:
    """Open-Meteo counts requests with >10 variables or >14 days as several calls."""
    return max(1.0, n_vars / 10) * max(1.0, n_days / 14)


# ------------------------------------------------------------------ HTTP
_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT


def _get(url: str, params: dict, max_tries: int = 8) -> dict:
    delay = 10
    for attempt in range(1, max_tries + 1):
        try:
            r = _session.get(url, params=params, timeout=config.HTTP_TIMEOUT_S)
        except requests.RequestException as e:
            print(f"    network error ({e.__class__.__name__}); retry in {delay}s", flush=True)
            time.sleep(delay); delay = min(delay * 2, 600); continue
        if r.status_code == 200:
            return r.json()
        reason = ""
        try:
            reason = r.json().get("reason", "")
        except ValueError:
            pass
        if r.status_code == 429 or "limit" in reason.lower():
            pause = 3700 if "hour" in reason.lower() else (60 if "minute" in reason.lower() else 900)
            print(f"    HTTP 429 '{reason}' - sleeping {pause}s", flush=True)
            time.sleep(pause); continue
        if r.status_code >= 500:
            print(f"    HTTP {r.status_code}; retry in {delay}s", flush=True)
            time.sleep(delay); delay = min(delay * 2, 600); continue
        raise RuntimeError(f"Open-Meteo error {r.status_code}: {reason or r.text[:300]}")
    raise RuntimeError(f"Open-Meteo request failed after {max_tries} attempts")


# ------------------------------------------------------------------ chunked download
def year_chunks(start: str, end: str):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    for y in range(s.year, e.year + 1):
        cs, ce = max(s, date(y, 1, 1)), min(e, date(y, 12, 31))
        yield cs.isoformat(), ce.isoformat()


def chunk_path(dataset: str, loc_id: str, start: str, end: str) -> Path:
    return config.RAW / "openmeteo" / dataset / loc_id / f"{start}_{end}.parquet"


def fetch_chunk(dataset: str, loc_id: str, lat: float, lon: float, start: str, end: str,
                limiter: RateLimiter, force: bool = False) -> Path:
    path = chunk_path(dataset, loc_id, start, end)
    if path.exists() and not force:
        return path
    spec = config.DATASETS[dataset]
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": start, "end_date": end,
        "hourly": ",".join(spec["variables"]),
        "timezone": "GMT", "wind_speed_unit": "ms",
        **spec["params"],
    }
    n_days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    weight = call_weight(len(spec["variables"]), n_days)
    limiter.wait_for(weight)
    js = _get(spec["url"], params)
    limiter.record(weight)

    h = js["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df.insert(0, "time_utc", pd.to_datetime(h["time"]).tz_localize("UTC"))
    meta = {"dataset": dataset, "loc_id": loc_id, "req_lat": lat, "req_lon": lon,
            "grid_lat": js.get("latitude"), "grid_lon": js.get("longitude"),
            "elevation_m": js.get("elevation"), "units": json.dumps(js.get("hourly_units", {})),
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_url": spec["url"], "model": spec["params"].get("models", "")}
    for k, v in meta.items():
        df.attrs[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path


def fetch_dataset(dataset: str, locations: pd.DataFrame, force: bool = False):
    spec = config.DATASETS[dataset]
    limiter = RateLimiter()
    chunks = list(year_chunks(spec["start"], spec["end"]))
    total = len(chunks) * len(locations)
    done = 0
    for _, loc in locations.iterrows():
        for cs, ce in chunks:
            done += 1
            cached = chunk_path(dataset, loc.loc_id, cs, ce).exists() and not force
            print(f"[{dataset}] {done}/{total} {loc.loc_id} {cs}..{ce} {'(cached)' if cached else ''}", flush=True)
            fetch_chunk(dataset, loc.loc_id, loc.latitude, loc.longitude, cs, ce, limiter, force)


def load_dataset(dataset: str, loc_id: str) -> tuple[pd.DataFrame, dict]:
    """Concatenate cached chunks for one location; returns (frame indexed by UTC, metadata)."""
    d = config.RAW / "openmeteo" / dataset / loc_id
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached data for {dataset}/{loc_id}; run scripts/02_fetch_weather.py")
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("time_utc").set_index("time_utc").sort_index()
    meta = json.loads(files[-1].with_suffix(".json").read_text())
    return df, meta


def rename_short(df: pd.DataFrame) -> pd.DataFrame:
    """era5: ghi, dni, ... ; forecasts: ghi_d1, dni_d1, ..., ghi_d2, ..."""
    out = {}
    for c in df.columns:
        base, _, lead = c.partition("_previous_day")
        short = config.SHORT_NAMES.get(base, base)
        out[c] = f"{short}_d{lead}" if lead else short
    return df.rename(columns=out)


def estimated_cost(datasets, n_locations: int) -> float:
    total = 0.0
    for ds in datasets:
        spec = config.DATASETS[ds]
        for cs, ce in year_chunks(spec["start"], spec["end"]):
            n_days = (date.fromisoformat(ce) - date.fromisoformat(cs)).days + 1
            total += call_weight(len(spec["variables"]), n_days)
    return total * n_locations
