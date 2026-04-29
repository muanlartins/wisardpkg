"""Extend the BTHOWeN sweep to the 10 DWN-paper datasets for coverage parity.

The original BTHOWeN sweep covered the 21-dataset F4RM suite. This script adds
phoneme, australian, nomao, miniboone, christine, jasmine, sylvine,
blood-transfusion, higgs-small, skin-segmentation. Same hyperparameter grid,
same 5 seeds, same memory accounting — appended to the same Pareto cache.

Already-done cells are skipped (cache key: system, dataset, hp_id, seed), so
this is safe to re-run. For datasets larger than 50 k samples we subsample to
50 k for tractability (BTHOWeN's training is fast but classify+bleach-search
on the full test set still scales linearly with both n_test and num_RAMs).
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

from notebook_lib import fit_predict_bthowen, grid_bthowen  # noqa: E402

CACHE_FILE = REPO / "notebooks/cache/section4_priorwisard_points.pkl"
RANDOM_SEED = 42
N_SEEDS = 5
SYSTEM = "BTHOWeN"
MAX_TRAIN_SAMPLES = 50_000

DWN_PAPER_SPECS = [
    ("phoneme", 1489),
    ("australian", 40981),
    ("nomao", 1486),
    ("miniboone", 41150),
    ("christine", 41142),
    ("jasmine", 41143),
    ("sylvine", 41146),
    ("blood-transfusion", 1464),
    ("higgs-small", 23512),
    ("skin-segmentation", 1502),
]


def load_dataset(name, ident):
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
    X = X[mask]
    y = np.asarray(y)[mask]
    return X, np.asarray(y).astype(str)


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
          f"{(df['system']==SYSTEM).sum()} BTHOWeN already; extending to "
          f"{len(DWN_PAPER_SPECS)} new datasets.")

    grid = grid_bthowen()
    new_rows = []
    appended = 0
    for ds_name, ident in DWN_PAPER_SPECS:
        try:
            X, y = load_dataset(ds_name, ident)
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Skipping {ds_name}: load failed ({exc})")
            continue
        n, nf = X.shape
        nc = len(set(y))
        print(f"[{time.strftime('%H:%M:%S')}] {ds_name}: n={n}, features={nf}, classes={nc}")

        for cell_idx, hp in enumerate(grid):
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
                    "system": SYSTEM,
                    "dataset": ds_name,
                    "hp_id": hp_id,
                    "seed": seed,
                    "test_acc": res["test_acc"],
                    "mem_bytes": res["mem_bytes"],
                    "mem_method": res["mem_method"],
                    "train_s": res["train_s"],
                    "infer_us_per_sample": res["infer_us_per_sample"],
                    "infeasible": res["infeasible"],
                    "infeasible_reason": res["infeasible_reason"],
                    "note": subsample_note,
                    "hp_a": hp["addressSize"],
                    "hp_t": hp["bitsPerInput"],
                    "hp_bleach": float("nan"),
                    "hp_K": float("nan"),
                    "hp_numBits": hp["numBits"],
                    "hp_numHashes": hp["numHashes"],
                }
                new_rows.append(row)
                appended += 1
                if len(new_rows) >= 24:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    save_cache(df)
                    new_rows = []
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(df)
            new_rows = []
        elapsed = time.perf_counter() - t_start
        bt_rows = (df["system"] == SYSTEM).sum()
        print(f"[{time.strftime('%H:%M:%S')}]   ...done {ds_name}: total {bt_rows} BTHOWeN rows; "
              f"appended {appended}; elapsed {elapsed/60:.1f} min")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(df)
    print(f"[{time.strftime('%H:%M:%S')}] Extension complete: {appended} new rows; "
          f"total {(df['system']==SYSTEM).sum()} BTHOWeN rows; "
          f"wall {((time.perf_counter()-t_start)/60):.1f} min")


if __name__ == "__main__":
    main()
