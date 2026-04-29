"""Sweep BTHOWeN across the F4RM 21-dataset suite × 48 hyperparameter cells × 5 seeds.

Appends rows to ``notebooks/cache/section4_pareto_points.pkl`` with
``system="BTHOWeN"`` and ``mem_method="bthowen_analytical_bloom_post_binarize"``,
matching the long-format schema the F4RM notebook already uses.

Cache key per row: ``(system, dataset, hp_id, seed)``. Already-completed cells
are skipped on re-run, so the sweep is fully restartable. Progress is written
incrementally after every cell.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_openml, load_breast_cancer, load_iris, load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

REPO = Path("/Users/muanlartins/repos/masters")
sys.path.insert(0, str(REPO / "notebooks"))

from notebook_lib import fit_predict_bthowen, grid_bthowen  # noqa: E402

CACHE_FILE = REPO / "notebooks/cache/section4_priorwisard_points.pkl"
RANDOM_SEED = 42
N_SEEDS = 5
SYSTEM = "BTHOWeN"
MEM_METHOD = "bthowen_analytical_bloom_post_binarize"

# 21-dataset suite — must match notebook DATASET_SPECS order
DATASET_SPECS = [
    ("Iris", "sklearn", load_iris),
    ("Wine", "sklearn", load_wine),
    ("Breast Cancer", "sklearn", load_breast_cancer),
    ("Seeds", "openml", 1499),
    ("Glass", "openml", 41),
    ("Haberman", "openml", 43),
    ("Ecoli", "openml", 39),
    ("Ionosphere", "openml", 59),
    ("Balance Scale", "openml", 11),
    ("Pima Diabetes", "openml", 37),
    ("Vehicle", "openml", 54),
    ("Banknote", "openml", 1462),
    ("Yeast", "openml", 181),
    ("Steel Plates", "openml", 1504),
    ("Segment", "openml", 36),
    ("Waveform", "openml", 60),
    ("SatImage", "openml", 182),
    ("Pendigits", "openml", 32),
    ("EEG Eye State", "openml", 1471),
    ("Magic Gamma", "openml", 1120),
    ("Letter Recognition", "openml", 6),
]


def load_dataset(name: str, source: str, ident):
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
    X = X[mask]
    y = np.asarray(y)[mask]
    return X, np.asarray(y).astype(str)


def load_cache() -> pd.DataFrame:
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def save_cache(df: pd.DataFrame) -> None:
    tmp = CACHE_FILE.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(df, f)
    tmp.replace(CACHE_FILE)


def already_done(df: pd.DataFrame, dataset: str, hp_id: str, seed: int) -> bool:
    return bool(((df["system"] == SYSTEM) & (df["dataset"] == dataset)
                  & (df["hp_id"] == hp_id) & (df["seed"] == seed)).any())


def main():
    t_start = time.perf_counter()
    df = load_cache()
    print(f"[{time.strftime('%H:%M:%S')}] Loaded cache with {len(df)} rows; "
          f"{(df['system'] == SYSTEM).sum()} BTHOWeN rows already.")

    grid = grid_bthowen()
    print(f"Sweep: {len(DATASET_SPECS)} datasets × {len(grid)} cells × "
          f"{N_SEEDS} seeds = {len(DATASET_SPECS) * len(grid) * N_SEEDS} measurements.")

    new_rows = []
    appended = 0
    for ds_name, source, ident in DATASET_SPECS:
        try:
            X, y = load_dataset(ds_name, source, ident)
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
                # Match notebook preprocessing: stratified 80/20, MinMax fit on train
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed,
                )
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
                    "note": "",
                    "hp_a": hp["addressSize"],
                    "hp_t": hp["bitsPerInput"],
                    "hp_bleach": float("nan"),
                    "hp_K": float("nan"),
                    "hp_numBits": hp["numBits"],
                    "hp_numHashes": hp["numHashes"],
                }
                new_rows.append(row)
                appended += 1
                # Save every 24 rows (1 cell × 5 seeds is 5; 24 ≈ 5 cells)
                if len(new_rows) >= 24:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    save_cache(df)
                    new_rows = []
            # Per-cell progress — quiet (one line per dataset section instead)
        # Flush after each dataset
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(df)
            new_rows = []
        elapsed = time.perf_counter() - t_start
        bthowen_rows = (df["system"] == SYSTEM).sum()
        print(f"[{time.strftime('%H:%M:%S')}]   ...done {ds_name}: total {bthowen_rows} BTHOWeN rows; "
              f"appended {appended} this run; elapsed {elapsed/60:.1f} min")

    # Final flush (defensive)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(df)
    print(f"[{time.strftime('%H:%M:%S')}] Sweep complete: {appended} new rows; "
          f"total {(df['system'] == SYSTEM).sum()} BTHOWeN rows in cache; "
          f"wall {((time.perf_counter()-t_start)/60):.1f} min")


if __name__ == "__main__":
    main()
