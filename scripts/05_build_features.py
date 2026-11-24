"""Step 5 - build model datasets for each horizon -> data/processed/model_h{h}.parquet"""
import _bootstrap  # noqa: F401
import sys

from solarpark import config, features

for h in config.horizons_from_argv(sys.argv):
    df = features.build_horizon(h)
    # self-check: lagged power must come from before the forecast issue time
    s = df[df.park_id == df.park_id.iat[0]].set_index("time_utc").sort_index()
    probe = s.index[1000]
    assert s.loc[probe, "p_lag0"] == s.y.get(probe - __import__("pandas").Timedelta(hours=h), s.loc[probe, "p_lag0"]), "lag misaligned"
    path = config.PROCESSED / f"model_h{h}.parquet"
    df.to_parquet(path, index=False)
    print(f"h={h}: {len(df):,} rows, {df.park_id.nunique()} parks, splits {df.split.value_counts().to_dict()} -> {path.name}", flush=True)
