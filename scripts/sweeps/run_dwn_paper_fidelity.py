"""DWN paper-fidelity sweep — Table 17 configs on the 4 datasets where paper-config is feasible.

Drops higgs-small (5x22000, full 98k samples, 200 ep ≈ 30+ hrs/seed — too expensive).

Per Table 17:
  phoneme:    3x21000, z=200, tau=1/0.03, multi-stage LR, ep=200
  australian: 3x12000, z=200, tau=1/0.03, multi-stage LR, ep=200
  christine:  3x26000, z=20, tau=1/0.03, multi-stage LR, ep=200
  jasmine:    3x20000, z=200, tau=1/0.03, multi-stage LR, ep=200

We use a SHORTENED schedule (75 ep total, [(1e-2, 30), (1e-3, 30), (1e-4, 15)])
since the smoke test on phoneme hit -0.9pp at 75 ep already.

3 seeds. Writes to section4_priorwisard_points.pkl. New hp_id (paper_fidelity).
"""
from __future__ import annotations
import json, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

REPO = Path("/Users/muanlartins/repos/masters")
sys.path.insert(0, str(REPO / "notebooks"))

from notebook_lib import fit_predict_dwn  # noqa: E402

CACHE_FILE = REPO / "notebooks/cache/section4_priorwisard_points.pkl"
RANDOM_SEED = 42
N_SEEDS = 3
SYSTEM = "DWN"

# Paper Table 17 configurations (shortened LR schedule).
# tau=1/0.001 reserved for skin-seg/higgs/miniboone — too expensive for now.
PAPER_FIDELITY_CONFIGS = [
    ("phoneme",    1489, dict(bits_per_input=200, n=6, layer_sizes=[21000, 21000, 21000],
                               tau=1/0.03,
                               lr_schedule=[(1e-2, 30), (1e-3, 30), (1e-4, 15)],
                               batch_size=32)),
    ("australian", 40981, dict(bits_per_input=200, n=6, layer_sizes=[12000, 12000, 12000],
                               tau=1/0.03,
                               lr_schedule=[(1e-2, 30), (1e-3, 30), (1e-4, 15)],
                               batch_size=32)),
    ("jasmine",    41143, dict(bits_per_input=200, n=6, layer_sizes=[20000, 20000, 20000],
                               tau=1/0.03,
                               lr_schedule=[(1e-2, 30), (1e-3, 30), (1e-4, 15)],
                               batch_size=32)),
    ("christine",  41142, dict(bits_per_input=20, n=6, layer_sizes=[26000, 26000, 26000],
                               tau=1/0.03,
                               lr_schedule=[(1e-2, 30), (1e-3, 30), (1e-4, 15)],
                               batch_size=32)),
]


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
    print(f"[{time.strftime('%H:%M:%S')}] Cache: {len(df)} rows; {(df['system']==SYSTEM).sum()} DWN existing.", flush=True)

    new_rows = []
    appended = 0
    summary = []
    for ds_name, ident, hp in PAPER_FIDELITY_CONFIGS:
        try:
            X, y = load_dataset(ident)
        except Exception as exc:
            print(f"  Skipping {ds_name}: load failed ({exc})", flush=True)
            continue
        n, nf = X.shape
        nc = len(set(y))
        hp_id = json.dumps(hp, sort_keys=True, default=str)
        print(f"\n[{time.strftime('%H:%M:%S')}] {ds_name}: n={n}, features={nf}, classes={nc}", flush=True)
        print(f"  config: bpi={hp['bits_per_input']}, layers={hp['layer_sizes']}, tau={hp['tau']:.2f}", flush=True)

        accs = []
        for s in range(N_SEEDS):
            seed = RANDOM_SEED + s
            if already_done(df, ds_name, hp_id, seed):
                print(f"    seed={seed} cached, skipping", flush=True)
                continue
            t_seed = time.perf_counter()
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed,
            )
            sc = MinMaxScaler().fit(Xtr)
            Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)
            r = fit_predict_dwn(Xtr_s, ytr, Xte_s, yte, hp, seed)
            elapsed = time.perf_counter() - t_seed
            print(f"    seed={seed} → acc={r['test_acc']:.4f} mem={r['mem_bytes']/1024:.1f} KiB ({elapsed/60:.1f} min)",
                  flush=True)
            row = {
                "system": SYSTEM, "dataset": ds_name, "hp_id": hp_id, "seed": seed,
                "test_acc": r["test_acc"], "mem_bytes": r["mem_bytes"],
                "mem_method": r["mem_method"], "train_s": r["train_s"],
                "infer_us_per_sample": r["infer_us_per_sample"],
                "infeasible": r["infeasible"], "infeasible_reason": r["infeasible_reason"],
                "note": "dwn_paper_fidelity_table17",
                "hp_a": float("nan"), "hp_t": hp["bits_per_input"],
                "hp_bleach": float("nan"), "hp_K": float("nan"),
                "hp_numBits": hp["layer_sizes"][0], "hp_numHashes": hp["n"],
            }
            new_rows.append(row)
            accs.append(r["test_acc"])
            appended += 1
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_cache(df); new_rows = []  # flush after each seed (long-running)

        if accs:
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            summary.append((ds_name, mean_acc, std_acc))
        elapsed = (time.perf_counter() - t_start) / 60
        print(f"  ...{ds_name} done; total appended={appended}; cum elapsed={elapsed:.1f}m", flush=True)

    # Final summary
    print("\n=== FINAL: DWN paper-fidelity (Table 17 configs, shortened 75-ep schedule) ===")
    PAPER_TARGETS = {
        "phoneme": 0.895, "australian": 0.901, "christine": 0.736, "jasmine": 0.816,
    }
    print(f"{'dataset':<14s} {'paper':>7s} {'ours':>7s} {'std':>6s} {'Δpp':>6s}")
    for ds, our_acc, std in summary:
        p_acc = PAPER_TARGETS.get(ds)
        delta = (our_acc - p_acc) * 100 if p_acc else None
        flag = " ⚠️" if delta and abs(delta) > 2 else ""
        print(f"{ds:<14s} {p_acc:>7.3f} {our_acc:>7.3f} {std:>6.4f} {delta:>+6.1f}{flag}")
    print(f"\nTotal wall-clock: {(time.perf_counter()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
