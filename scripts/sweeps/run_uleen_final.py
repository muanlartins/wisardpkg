"""ULEEN final sweep — Gaussian thermometer + post-train pruning + paper LR.

Combines the audit's three actionable ULEEN improvements:
  1. Gaussian thermometer (paper default; +0.6pp/MNIST per paper, +6pp Iris our smoke)
  2. Post-train pruning at ratio=0.27 (~25% size reduction, ≤1% acc change)
  3. Paper LR schedule: lr=1e-3, epochs=80

Same 31-dataset suite (21 F4RM + 10 DWN-paper). 6 cells × 3 seeds = 558 max evals.
Cache key (hp_id) differs from prior runs because of `thermometer="gaussian"` and
`pruning_ratio=0.27` flags, so this is additive.

Estimated wall-clock: ~2-4 hr on M2.
"""
from __future__ import annotations
import json, pickle, sys, time, warnings
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
PRUNING_RATIO = 0.27

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


def make_final_grid():
    """6-cell ULEEN grid. Add thermometer=gaussian + pruning_ratio=0.27 + paper-LR to each cell."""
    grid = []
    for cell in grid_uleen():
        new_cell = dict(cell)
        new_cell["thermometer"] = "gaussian"
        new_cell["pruning_ratio"] = PRUNING_RATIO
        new_cell["lr"] = 1e-3
        new_cell["epochs"] = 80
        grid.append(new_cell)
    return grid


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
    grid = make_final_grid()
    print(f"[{time.strftime('%H:%M:%S')}] Final ULEEN grid: {len(grid)} cells "
          f"(gaussian thermometer + pruning {PRUNING_RATIO} + paper LR).", flush=True)
    print(f"Cache: {len(df)} rows; {(df['system']==SYSTEM).sum()} ULEEN existing.", flush=True)

    new_rows = []
    appended = 0
    for ds_name, source, ident in ALL_SPECS:
        try:
            X, y = load_dataset(ds_name, source, ident)
        except Exception as exc:
            print(f"  Skipping {ds_name}: load failed ({exc})", flush=True)
            continue
        n, nf = X.shape
        nc = len(set(y))
        print(f"[{time.strftime('%H:%M:%S')}] {ds_name}: n={n}, features={nf}, classes={nc}", flush=True)

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
                    subsample_note = f"ULEEN train subsampled to {MAX_TRAIN_SAMPLES}"
                sc = MinMaxScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)
                r = fit_predict_uleen(Xtr_s, ytr, Xte_s, yte, hp, seed)
                row = {
                    "system": SYSTEM, "dataset": ds_name, "hp_id": hp_id, "seed": seed,
                    "test_acc": r["test_acc"], "mem_bytes": r["mem_bytes"],
                    "mem_method": r["mem_method"], "train_s": r["train_s"],
                    "infer_us_per_sample": r["infer_us_per_sample"],
                    "infeasible": r["infeasible"], "infeasible_reason": r["infeasible_reason"],
                    "note": subsample_note + " | gaussian+pruned+paperlr",
                    "hp_a": float("nan"),
                    "hp_t": hp.get("bits_per_input", float("nan")),
                    "hp_bleach": float("nan"), "hp_K": float("nan"),
                    "hp_numBits": hp.get("filter_entries", float("nan")),
                    "hp_numHashes": hp.get("hash_functions", float("nan")),
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
        print(f"  ...{ds_name} done; appended {appended}; elapsed {elapsed/60:.1f} min", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Final ULEEN sweep complete: {appended} new rows; "
          f"wall {((time.perf_counter()-t_start)/60):.1f} min", flush=True)


if __name__ == "__main__":
    main()
