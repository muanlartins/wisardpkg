"""Targeted DWN re-run on phoneme + higgs-small with epochs=50 and batch=64.

Goal: close the phoneme -5.7pp and higgs-small -8.9pp gaps (caveat #2).
Hypothesis (per paper-drift analysis): under-training. The original grid runs
10-20 epochs; for cells with num_luts ≥ 1000 + a learnable first-layer mapping,
that's insufficient — EFD gradient is approximate so the mapping needs more
updates to converge.

Re-runs the same 6-cell DWN grid but with `epochs=50` (overrides each cell's
original epochs) and `batch_size=64` (was 32). Cache key (hp_id) differs from
the original because epochs is part of hp dict — so this is additive.

2 datasets × 6 cells × 3 seeds = 36 runs. Estimated wall-clock: ~1 hr on M2.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

REPO = Path("/Users/muanlartins/repos/masters")
sys.path.insert(0, str(REPO / "notebooks"))

from notebook_lib import fit_predict_dwn, grid_dwn  # noqa: E402

CACHE_FILE = REPO / "notebooks/cache/section4_priorwisard_points.pkl"
RANDOM_SEED = 42
N_SEEDS = 3
SYSTEM = "DWN"
MAX_TRAIN_SAMPLES = 20_000

LONG_EPOCH_SPECS = [
    ("phoneme", 1489),
    ("higgs-small", 23512),
]


def make_long_epoch_grid():
    """Take the standard grid, override epochs=50 and batch_size=64 in each cell."""
    grid = []
    for cell in grid_dwn():
        new_cell = dict(cell)
        new_cell["epochs"] = 50
        new_cell["batch_size"] = 64
        grid.append(new_cell)
    return grid


def load_dataset(ident):
    d = fetch_openml(data_id=ident, as_frame=False, parser="auto")
    X, y = d.data, d.target
    if hasattr(X, "toarray"):
        X = X.toarray()
    try:
        X = np.asarray(X, dtype=float)
    except (ValueError, TypeError):
        df = pd.DataFrame(X)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(df.mean(numeric_only=True))
        X = df.to_numpy(dtype=float)
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], np.asarray(y)[mask].astype(str)


def load_cache():
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def save_cache(df):
    tmp = CACHE_FILE.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(df, f)
    tmp.replace(CACHE_FILE)


def already_done(df, dataset, hp_id, seed):
    return bool(((df["system"] == SYSTEM) & (df["dataset"] == dataset)
                  & (df["hp_id"] == hp_id) & (df["seed"] == seed)).any())


def main():
    t_start = time.perf_counter()
    df = load_cache()
    grid = make_long_epoch_grid()
    print(f"[{time.strftime('%H:%M:%S')}] Long-epoch DWN grid: {len(grid)} cells, epochs=50, batch=64.")
    print(f"Cache: {len(df)} rows; {(df['system']==SYSTEM).sum()} DWN existing.")

    new_rows = []
    appended = 0
    for ds_name, ident in LONG_EPOCH_SPECS:
        try:
            X, y = load_dataset(ident)
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Skipping {ds_name}: load failed ({exc})")
            continue
        n, nf = X.shape
        nc = len(set(y))
        print(f"[{time.strftime('%H:%M:%S')}] {ds_name}: n={n}, features={nf}, classes={nc}")

        for hp in grid:
            hp_id = json.dumps(hp, sort_keys=True, default=str)
            for s in range(N_SEEDS):
                seed = RANDOM_SEED + s
                if already_done(df, ds_name, hp_id, seed):
                    continue
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed,
                )
                subsample_note = ""
                if len(Xtr) > MAX_TRAIN_SAMPLES:
                    Xtr_sub, _, ytr_sub, _ = train_test_split(
                        Xtr, ytr, train_size=MAX_TRAIN_SAMPLES, stratify=ytr,
                        random_state=seed,
                    )
                    Xtr, ytr = Xtr_sub, ytr_sub
                    subsample_note = f"DWN train subsampled to {MAX_TRAIN_SAMPLES}"
                sc = MinMaxScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xte_s = sc.transform(Xte)
                res = fit_predict_dwn(Xtr_s, ytr, Xte_s, yte, hp, seed)
                row = {
                    "system": SYSTEM, "dataset": ds_name, "hp_id": hp_id, "seed": seed,
                    "test_acc": res["test_acc"], "mem_bytes": res["mem_bytes"],
                    "mem_method": res["mem_method"], "train_s": res["train_s"],
                    "infer_us_per_sample": res["infer_us_per_sample"],
                    "infeasible": res["infeasible"], "infeasible_reason": res["infeasible_reason"],
                    "note": subsample_note + " | epochs=50 batch=64",
                    "hp_a": float("nan"),
                    "hp_t": hp.get("bits_per_input", float("nan")),
                    "hp_bleach": float("nan"), "hp_K": float("nan"),
                    "hp_numBits": hp.get("num_luts_l1", float("nan")),
                    "hp_numHashes": hp.get("n", float("nan")),
                }
                new_rows.append(row)
                appended += 1
                if len(new_rows) >= 9:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    save_cache(df); new_rows = []
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(df); new_rows = []
        elapsed = time.perf_counter() - t_start
        print(f"[{time.strftime('%H:%M:%S')}]   ...done {ds_name}: appended {appended}; "
              f"elapsed {elapsed/60:.1f} min")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(df)
    print(f"[{time.strftime('%H:%M:%S')}] Long-epoch sweep complete: {appended} new rows; "
          f"wall {((time.perf_counter()-t_start)/60):.1f} min")


if __name__ == "__main__":
    main()
