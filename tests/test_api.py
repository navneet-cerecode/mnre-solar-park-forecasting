import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
pytest.importorskip("fastapi")
if not (ROOT / "data" / "processed" / "preds_h1.parquet").exists():
    pytest.skip("run the training scripts first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402
from solarpark import forecasts as F  # noqa: E402

client = TestClient(app)


def test_total_equals_sum_of_parks():
    f = F.forecasts(1)
    a = F.aggregate(1)
    assert abs(a.actual_mw.sum() - f.y_mw.sum()) < 1e-3 * f.y_mw.sum()
    assert (a.lo_mw <= a.forecast_mw).all() and (a.forecast_mw <= a.hi_mw).all()


def test_park_endpoint_and_ranges():
    rows = client.get("/forecast/park/P17", params={"h": 1, "start": "2026-07-06", "end": "2026-07-06"}).json()
    assert rows and all(r["range80_lo_mw"] <= r["range80_hi_mw"] for r in rows)
    assert all(0 <= r["forecast_mw"] <= 2000.01 for r in rows)          # Pavagada is 2,000 MW


def test_errors_and_lists():
    assert client.get("/forecast/park/P43").status_code == 404           # excluded park
    assert client.get("/forecast/park/P17", params={"h": 2}).status_code == 422
    assert client.get("/forecast/total", params={"level": "state"}).status_code == 422
    assert len(client.get("/parks").json()) == 50
    assert client.get("/forecast/total", params={"level": "state", "name": "Rajasthan", "start": "2026-03-02", "end": "2026-03-02"}).status_code == 200
