"""Targeted BTHOWeN re-sweep on the 6 paper-Table-3 datasets with an expanded grid.

Goal: close the Vehicle -8.1pp gap (caveat #1). The original 48-cell grid
sweeps bpi ∈ {3, 9} only, but BTHOWeN paper sweeps bpi ∈ {2,4,6,8,12,16}.
Vehicle's optimal cell (per the paper's reported memory of 2.25 KiB) likely
falls in the bpi ∈ {4, 6, 8} band our grid skips.

Expanded grid: 192 cells = bpi ∈ {2,4,6,8} × addr ∈ {4,8,16,24}
                          × nbits ∈ {256,1024} × hashes ∈ {2,3}.
Targets: Iris, Wine, Ecoli, Vehicle, SatImage, Letter Recognition.
Seeds: 5. Cache key includes hp_id so this is additive (no collision with
existing 48-cell rows).

Estimated wall-clock: ~1 hr on M2 single CPU.
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
from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

REPO = Path("/Users/muanlartins/repos/masters")
sys.path.insert(0, str(REPO / "notebooks"))

from notebook_lib import fit_predict_bthowen  # noqa: E402

CACHE_FILE = REPO / "notebooks/cache/section4_priorwisard_points.pkl"
RANDOM_SEED = 42
N_SEEDS = 5
SYSTEM = "BTHOWeN"
MAX_TRAIN_SAMPLES = 50_000

# Expanded grid — 192 cells covering bpi ∈ {2,4,6,8} (paper sweeps these)
EXPANDED_GRID = []
for addr in (4, 8, 16, 24):
    for bpi in (2, 4, 6, 8):
        for nbits in (256, 1024):
            for nhash in (2, 3):
                EXPANDED_GRID.append(dict(
                    addressSize=addr, numBits=nbits,
                    numHashes=nhash, bitsPerInput=bpi,
                ))

TABLE3_SPECS = [
    ("Iris", "sklearn", load_iris),
    ("Wine", "sklearn", load_wine),
    ("Ecoli", "openml", 39),
    ("Vehicle", "openml", 54),
    ("SatImage", "openml", 182),
    ("Letter Recognition", "openml", 6),
]


def load_dataset(name, source, ident):
    if source == "sklearn":
        d = ident()
        return d.data.astype(float), d.target.astype(str)
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
    print(f"[{time.strftime('%H:%M:%S')}] Cache: {len(df)} rows; "
          f"{(df['system']==SYSTEM).sum()} BTHOWeN existing.")
    print(f"Expanded grid: {len(EXPANDED_GRID)} cells × {len(TABLE3_SPECS)} datasets × {N_SEEDS} seeds = "
          f"{len(EXPANDED_GRID)*len(TABLE3_SPECS)*N_SEEDS} max evals (skip already-done).")

    new_rows = []
    appended = 0
    for ds_name, source, ident in TABLE3_SPECS:
        try:
            X, y = load_dataset(ds_name, source, ident)
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Skipping {ds_name}: load failed ({exc})")
            continue
        n, nf = X.shape
        nc = len(set(y))
        print(f"[{time.strftime('%H:%M:%S')}] {ds_name}: n={n}, features={nf}, classes={nc}")

        for hp in EXPANDED_GRID:
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
                    Xtr, _, ytr, _ = train_test_split(
                        Xtr, ytr, train_size=MAX_TRAIN_SAMPLES, stratify=ytr,
                        random_state=seed,
                    )
                    subsample_note = f"BTHOWeN train subsampled to {MAX_TRAIN_SAMPLES}"
                sc = MinMaxScaler().fit(Xtr)
                Xtr = sc.transform(Xtr)
                Xte = sc.transform(Xte)
                res = fit_predict_bthowen(Xtr, ytr, Xte, yte, hp, seed)
                row = {
                    "system": SYSTEM, "dataset": ds_name, "hp_id": hp_id, "seed": seed,
                    "test_acc": res["test_acc"], "mem_bytes": res["mem_bytes"],
                    "mem_method": res["mem_method"], "train_s": res["train_s"],
                    "infer_us_per_sample": res["infer_us_per_sample"],
                    "infeasible": res["infeasible"], "infeasible_reason": res["infeasible_reason"],
                    "note": subsample_note + " | expanded_table3_grid",
                    "hp_a": hp["addressSize"], "hp_t": hp["bitsPerInput"],
                    "hp_bleach": float("nan"), "hp_K": float("nan"),
                    "hp_numBits": hp["numBits"], "hp_numHashes": hp["numHashes"],
                }
                new_rows.append(row)
                appended += 1
                if len(new_rows) >= 32:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    save_cache(df); new_rows = []
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(df); new_rows = []
        elapsed = time.perf_counter() - t_start
        print(f"[{time.strftime('%H:%M:%S')}]   ...done {ds_name}: total {(df['system']==SYSTEM).sum()} BTHOWeN rows; "
              f"appended {appended}; elapsed {elapsed/60:.1f} min")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(df)
    print(f"[{time.strftime('%H:%M:%S')}] Expanded sweep complete: {appended} new rows; "
          f"wall {((time.perf_counter()-t_start)/60):.1f} min")


if __name__ == "__main__":
    main()
