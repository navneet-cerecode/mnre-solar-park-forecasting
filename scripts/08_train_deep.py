"""Step 8 - deep learning (LSTM, GRU) on the GPU.

Each sample: the last 24 hours of power up to the issue time t (sequence) + the same valid-time
inputs the tree models use (NWP forecast, sun position, metadata) + a learned park embedding.
Train on train split, early-stop on calib, evaluate on test. Adds columns to preds_h{h}.parquet.
"""
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch import nn

from solarpark import config, evaluate
from solarpark.features import FEATURES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 24
TAB = [f for f in FEATURES if f != "park_code"]
torch.manual_seed(42); np.random.seed(42)


class Net(nn.Module):
    def __init__(self, cell, n_tab, n_parks):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(3, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.emb = nn.Embedding(n_parks, 8)
        self.head = nn.Sequential(nn.Linear(64 + n_tab + 8, 128), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, seq, tab, park):
        _, hid = self.rnn(seq)
        last = (hid[0] if isinstance(hid, tuple) else hid)[-1]
        return self.head(torch.cat([last, tab, self.emb(park)], 1)).squeeze(1)


def sequences(df, h):
    """(N, 24, 3) tensor of [p, clear-sky p, clear-sky index] for hours t-23..t, t = v - h."""
    tg = pd.read_parquet(config.TARGETS_FILE, columns=["park_id", "time_utc", "p_norm", "p_cs_norm"])
    p = tg.pivot(index="time_utc", columns="park_id", values="p_norm").sort_index()
    pcs = tg.pivot(index="time_utc", columns="park_id", values="p_cs_norm").reindex(index=p.index, columns=p.columns)
    k = (p / pcs).where(pcs > 0.05).clip(0, 1.5).fillna(0)
    cube = np.stack([p.fillna(0).to_numpy(), pcs.fillna(0).to_numpy(), k.to_numpy()], -1).astype("float32")  # (T, P, 3)
    t_idx = p.index.get_indexer(df.time_utc) - h
    p_idx = p.columns.get_indexer(df.park_id)
    win = t_idx[:, None] + np.arange(-SEQ + 1, 1)[None, :]
    assert (win >= 0).all(), "sequence starts before data"   # reanalysis starts a month before the study window
    return cube[win, p_idx[:, None], :]


def train_eval(cell, h, df, seq):
    tr, va = (df.split == "train").to_numpy(), (df.split == "calib").to_numpy()
    X = df[TAB].astype("float32")
    mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
    X = ((X - mu) / sd).fillna(0).to_numpy("float32")
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    S, Xt, P, Y = T(seq), T(X), T(df.park_code.to_numpy("int64")), T(df.y.to_numpy("float32"))
    net = Net(cell, len(TAB), int(df.park_code.max()) + 1).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tr_i, va_i = torch.nonzero(T(tr)).squeeze(1), torch.nonzero(T(va)).squeeze(1)
    best, best_state, bad = 1e9, None, 0
    for epoch in range(40):
        net.train()
        for b in tr_i[torch.randperm(len(tr_i), device=DEV)].split(2048):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(S[b], Xt[b], P[b]), Y[b])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.cat([net(S[b], Xt[b], P[b]) for b in va_i.split(8192)])
            vl = float(nn.functional.mse_loss(v, Y[va_i]))
        if vl < best - 1e-6:
            best, best_state, bad = vl, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        idx = torch.arange(len(df), device=DEV)
        pred = torch.cat([net(S[b], Xt[b], P[b]) for b in idx.split(8192)]).cpu().numpy()
    return np.where(df.pcs > 0, pred, 0.0), epoch + 1


if __name__ == "__main__":
    print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    rows = []
    import sys
    HS = config.horizons_from_argv(sys.argv)
    for h in HS:
        df = pd.read_parquet(config.PROCESSED / f"model_h{h}.parquet")
        seq = sequences(df, h)
        pr = pd.read_parquet(config.PROCESSED / f"preds_h{h}.parquet")
        key = pd.MultiIndex.from_frame(df[["park_id", "time_utc"]])
        for cell in ("lstm", "gru"):
            t0 = time.time()
            p, epochs = train_eval(cell, h, df, seq)
            s = pd.Series(p, index=key)
            pr[cell] = s.reindex(pd.MultiIndex.from_frame(pr[["park_id", "time_utc"]])).to_numpy()
            print(f"h={h} {cell}: {epochs} epochs, {time.time() - t0:.0f}s", flush=True)
        pr["ensemble_lgbm_gru"] = (pr.lightgbm + pr.gru) / 2
        pr.to_parquet(config.PROCESSED / f"preds_h{h}.parquet", index=False)
        test = pr[pr.split == "test"]
        r = evaluate.table(test, ["bl_smart", "lightgbm", "xgboost_gpu", "random_forest", "lstm", "gru", "ensemble_lgbm_gru"])
        r["h"] = h
        rows.append(r)
        print(r[["model", "nMAE_%", "nRMSE_%", "skill_vs_smart"]].round(3).to_string(index=False), flush=True)
    res, old = pd.concat(rows), config.REPORTS / "results" / "e1b_deep_learning.csv"
    if old.exists():
        prev = pd.read_csv(old)
        res = pd.concat([prev[~prev.h.isin(HS)], res])
    res.sort_values("h", kind="stable").to_csv(old, index=False)
