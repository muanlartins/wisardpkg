"""Sweep ULEEN across the F4RM 21-dataset suite + the 10 datasets DWN paper Table 5
evaluates on (where they don't already overlap).

Same runner pattern as ``run_dwn_sweep.py``: writes to the prior_wisard
sibling cache (``section4_priorwisard_points.pkl``) so it doesn't race with
the F4RM session, and the F4RM notebook's cell 75 picks it up automatically.

ULEEN is gradient-trained like DWN; per-cell wall clock is comparable. Total
budget for 31 datasets × 6 cells × 3 seeds is ~30–60 minutes on CPU.
Datasets larger than 20 k training samples are subsampled to 20 k.
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
from sklearn.datasets import (
    fetch_openml, load_breast_cancer, load_iris, load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

REPO = Path("/Users/muanlartins/repos/masters")
sys.path.insert(0, str(REPO / "notebooks"))

from notebook_lib import fit_predict_uleen, grid_uleen  # noqa: E402

CACHE_FILE = REPO / "notebooks/cache/section4_priorwisard_points.pkl"
RANDOM_SEED = 42
N_SEEDS = 3
SYSTEM = "ULEEN"
MAX_TRAIN_SAMPLES = 20_000

F4RM_SPECS = [
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
DWN_PAPER_SPECS = [
    ("phoneme", "openml", 1489),
    ("australian", "openml", 40981),
    ("nomao", "openml", 1486),
    ("miniboone", "openml", 41150),
    ("christine", "openml", 41142),
    ("jasmine", "openml", 41143),
    ("sylvine", "openml", 41146),
    ("blood-transfusion", "openml", 1464),
    ("higgs-small", "openml", 23512),
    ("skin-segmentation", "openml", 1502),
]
ALL_SPECS = F4RM_SPECS + DWN_PAPER_SPECS


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
    X = X[mask]
    y = np.asarray(y)[mask]
    return X, np.asarray(y).astype(str)


def load_cache():
    if not CACHE_FILE.exists():
        return pd.DataFrame()
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def save_cache(df):
    tmp = CACHE_FILE.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(df, f)
    tmp.replace(CACHE_FILE)


def already_done(df, dataset, hp_id, seed):
    if len(df) == 0:
        return False
    return bool(((df["system"] == SYSTEM) & (df["dataset"] == dataset)
                  & (df["hp_id"] == hp_id) & (df["seed"] == seed)).any())


def main():
    t_start = time.perf_counter()
    df = load_cache()
    print(f"[{time.strftime('%H:%M:%S')}] Cache: {len(df)} rows; "
          f"{(df['system']==SYSTEM).sum() if 'system' in df.columns else 0} ULEEN measured already.")

    grid = grid_uleen()
    print(f"Grid: {len(grid)} cells; {N_SEEDS} seeds; {len(ALL_SPECS)} datasets.")

    new_rows = []
    appended = 0
    for ds_name, source, ident in ALL_SPECS:
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
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed,
                )
                subsample_note = ""
                if len(Xtr) > MAX_TRAIN_SAMPLES:
                    Xtr, _, ytr, _ = train_test_split(
                        Xtr, ytr, train_size=MAX_TRAIN_SAMPLES, stratify=ytr,
                        random_state=seed,
                    )
                    subsample_note = f"ULEEN train subsampled to {MAX_TRAIN_SAMPLES} (CPU budget)"
                sc = MinMaxScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xte_s = sc.transform(Xte)
                res = fit_predict_uleen(Xtr_s, ytr, Xte_s, yte, hp, seed)
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
                    "hp_a": float("nan"),
                    "hp_t": hp.get("bits_per_input", float("nan")),
                    "hp_bleach": float("nan"),
                    "hp_K": hp.get("n_submodels", float("nan")),
                    "hp_numBits": hp.get("filter_entries", float("nan")),
                    "hp_numHashes": hp.get("filter_hash_functions", float("nan")),
                }
                new_rows.append(row)
                appended += 1
                if len(new_rows) >= 18:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    save_cache(df)
                    new_rows = []
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(df)
            new_rows = []
        elapsed = time.perf_counter() - t_start
        uleen_rows = (df["system"] == SYSTEM).sum() if "system" in df.columns else 0
        print(f"[{time.strftime('%H:%M:%S')}]   ...done {ds_name}: total {uleen_rows} ULEEN rows; "
              f"appended {appended}; elapsed {elapsed/60:.1f} min")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(df)
    print(f"[{time.strftime('%H:%M:%S')}] Sweep complete: {appended} new rows; "
          f"total {(df['system']==SYSTEM).sum() if 'system' in df.columns else 0} ULEEN rows; "
          f"wall {((time.perf_counter()-t_start)/60):.1f} min")


if __name__ == "__main__":
    main()
