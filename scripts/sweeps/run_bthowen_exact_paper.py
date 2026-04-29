"""BTHOWeN exact-config sweep — paper Table III hyperparameters, 10 seeds.

For each of the 6 BTHOWeN-paper-Table-III datasets, run the EXACT (bpi, n,
entries, hashes) the paper reports as their selected config. With 10 seeds
this should hit paper numbers to within ±1-2pp seed-noise.

Datasets (paper Table III):
  Iris       — bpi=3,  n=2,  entries=128,  h=1  → paper 0.980
  Wine       — bpi=9,  n=13, entries=128,  h=3  → paper 0.983
  Ecoli      — bpi=10, n=10, entries=128,  h=2  → paper 0.875
  Vehicle    — bpi=16, n=16, entries=256,  h=3  → paper 0.762
  SatImage   — bpi=8,  n=12, entries=512,  h=4  → paper 0.880
  Letter     — bpi=15, n=20, entries=2048, h=4  → paper 0.900

Writes to section4_priorwisard_points.pkl with hp_id including these exact
configs (won't collide with prior runs since none used these exact tuples).
"""
from __future__ import annotations
import json, pickle, sys, time, warnings
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
N_SEEDS = 10
SYSTEM = "BTHOWeN"

# Paper Table III exact configs
PAPER_CONFIGS = [
    ("Iris",                "sklearn", load_iris,
     dict(addressSize=2,  bitsPerInput=3,  numBits=128,  numHashes=1)),
    ("Wine",                "sklearn", load_wine,
     dict(addressSize=13, bitsPerInput=9,  numBits=128,  numHashes=3)),
    ("Ecoli",               "openml", 39,
     dict(addressSize=10, bitsPerInput=10, numBits=128,  numHashes=2)),
    ("Vehicle",             "openml", 54,
     dict(addressSize=16, bitsPerInput=16, numBits=256,  numHashes=3)),
    ("SatImage",            "openml", 182,
     dict(addressSize=12, bitsPerInput=8,  numBits=512,  numHashes=4)),
    ("Letter Recognition",  "openml", 6,
     dict(addressSize=20, bitsPerInput=15, numBits=2048, numHashes=4)),
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
    print(f"Paper Table III sweep: {len(PAPER_CONFIGS)} datasets × {N_SEEDS} seeds.")

    new_rows = []
    appended = 0
    summary = []  # for end report
    for ds_name, source, ident, hp in PAPER_CONFIGS:
        try:
            X, y = load_dataset(ds_name, source, ident)
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Skipping {ds_name}: load failed ({exc})")
            continue
        n, nf = X.shape
        nc = len(set(y))
        hp_id = json.dumps(hp, sort_keys=True, default=str)
        print(f"[{time.strftime('%H:%M:%S')}] {ds_name}: n={n}, features={nf}, classes={nc} "
              f"a={hp['addressSize']} bpi={hp['bitsPerInput']} entries={hp['numBits']} h={hp['numHashes']}")
        accs = []
        for s in range(N_SEEDS):
            seed = RANDOM_SEED + s
            if already_done(df, ds_name, hp_id, seed):
                # Could read from cache, but we'll just count it
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed,
            )
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
                "note": "exact_paper_table3",
                "hp_a": hp["addressSize"], "hp_t": hp["bitsPerInput"],
                "hp_bleach": float("nan"), "hp_K": float("nan"),
                "hp_numBits": hp["numBits"], "hp_numHashes": hp["numHashes"],
            }
            new_rows.append(row)
            accs.append(res["test_acc"])
            appended += 1
            print(f"    {ds_name} seed={seed} → acc={res['test_acc']:.4f} "
                  f"mem={res['mem_bytes']/1024:.3f} KiB")
            if len(new_rows) >= 5:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                save_cache(df); new_rows = []
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(df); new_rows = []
        # Read all rows for this (dataset, hp_id) for stats including any cached
        sub = df[(df["system"] == SYSTEM) & (df["dataset"] == ds_name) & (df["hp_id"] == hp_id)]
        if len(sub):
            mean_acc = float(sub["test_acc"].mean())
            std_acc = float(sub["test_acc"].std()) if len(sub) > 1 else 0.0
            mem_kib = float(sub["mem_bytes"].iloc[0]) / 1024
            summary.append((ds_name, mean_acc, std_acc, mem_kib, len(sub)))
        elapsed = time.perf_counter() - t_start
        print(f"[{time.strftime('%H:%M:%S')}]   ...done {ds_name} (appended {appended}; elapsed {elapsed/60:.1f}m)")

    # Final summary table
    print(f"\n=== FINAL: BTHOWeN exact-paper-config (n_seeds={N_SEEDS}) ===")
    PAPER_TARGETS = {
        "Iris": (0.980, 0.281), "Wine": (0.983, 0.422), "Ecoli": (0.875, 0.875),
        "Vehicle": (0.762, 2.25), "SatImage": (0.880, 9.00),
        "Letter Recognition": (0.900, 78.0),
    }
    print(f"{'dataset':<22s} {'paper':>7s} {'ours':>7s} {'std':>6s} {'Δpp':>6s} "
          f"{'paper KiB':>10s} {'ours KiB':>9s} {'n_seeds':>7s}")
    for ds, our_acc, std, mem_kib, n in summary:
        p_acc, p_kib = PAPER_TARGETS.get(ds, (None, None))
        delta = (our_acc - p_acc) * 100 if p_acc else None
        flag = " ⚠️" if delta and abs(delta) > 2 else ""
        print(f"{ds:<22s} {p_acc:>7.3f} {our_acc:>7.3f} {std:>6.4f} {delta:>+6.1f}{flag} "
              f"{p_kib:>10.3f} {mem_kib:>9.3f} {n:>7d}")


if __name__ == "__main__":
    main()
