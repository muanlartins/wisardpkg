"""Compare our DWN/BTHOWeN measurements against the original papers.

For each (system, dataset) where we have a published number, find the best
configuration in our sweep (max test_acc averaged over seeds) and report
the delta vs the paper. Per spec §8: deltas > 5 pp warrant investigation.

Outputs a markdown comparison table to stdout.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/Users/muanlartins/repos/masters/notebooks/cache/section4_priorwisard_points.pkl")

# DWN paper Table 5 — accuracy only (no model size given for tabular)
DWN_TABLE5 = {
    "phoneme":            0.895,
    "skin-segmentation":  1.000,
    "higgs-small":        0.727,
    "australian":         0.901,
    "nomao":              0.966,
    "Segment":            0.998,
    "miniboone":          0.946,
    "christine":          0.736,
    "jasmine":            0.816,
    "sylvine":            0.952,
    "blood-transfusion":  0.780,
}

# BTHOWeN paper Table 3 — (model_size_KiB, test_acc)
BTHOWEN_TABLE3 = {
    "Iris":               (0.281, 0.980),
    "Wine":               (0.422, 0.983),
    "Ecoli":              (0.875, 0.875),
    "Vehicle":            (2.250, 0.762),
    "SatImage":           (9.000, 0.880),
    "Letter Recognition": (78.00, 0.900),
}


def best_per_dataset(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Return per-(dataset, hp_id) means; pick best test_acc per dataset."""
    sub = df[(df["system"] == system) & (df["seed"] >= 0)]
    if len(sub) == 0:
        return pd.DataFrame()
    means = sub.groupby(["dataset", "hp_id"]).agg(
        test_acc=("test_acc", "mean"),
        mem_bytes=("mem_bytes", "mean"),
        train_s=("train_s", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    best = means.loc[means.groupby("dataset")["test_acc"].idxmax()]
    return best.set_index("dataset")


def main():
    with open(CACHE, "rb") as f:
        df = pickle.load(f)

    bthowen_best = best_per_dataset(df, "BTHOWeN")
    dwn_best = best_per_dataset(df, "DWN")

    print("# Verification vs Original Papers")
    print(f"\nCache state: {df['system'].value_counts().to_dict()}\n")

    # ----- BTHOWeN vs Table 3 -----
    print("\n## BTHOWeN — paper Table 3 vs our best-cell measurement\n")
    print("| Dataset | Paper acc | Our acc | Δ (pp) | Paper KiB | Our best-cell KiB | Δ KiB |")
    print("|---------|----------:|--------:|-------:|----------:|------------------:|------:|")
    for ds, (paper_kib, paper_acc) in BTHOWEN_TABLE3.items():
        if ds not in bthowen_best.index:
            print(f"| {ds} | {paper_acc:.3f} | — | — | {paper_kib:.3f} | — | — |")
            continue
        row = bthowen_best.loc[ds]
        our_acc = row["test_acc"]
        our_kib = row["mem_bytes"] / 1024
        d_acc = (our_acc - paper_acc) * 100
        d_kib = our_kib - paper_kib
        flag = " ⚠️" if abs(d_acc) > 5 else ""
        print(f"| {ds} | {paper_acc:.3f} | {our_acc:.3f}{flag} | {d_acc:+.1f} | {paper_kib:.3f} | {our_kib:.3f} | {d_kib:+.3f} |")

    # ----- DWN vs Table 5 -----
    print("\n## DWN — paper Table 5 vs our best-cell measurement\n")
    print("| Dataset | Paper acc | Our acc | Δ (pp) | Our best-cell KiB |")
    print("|---------|----------:|--------:|-------:|------------------:|")
    for ds, paper_acc in DWN_TABLE5.items():
        if ds not in dwn_best.index:
            print(f"| {ds} | {paper_acc:.3f} | (sweep not done yet) | — | — |")
            continue
        row = dwn_best.loc[ds]
        our_acc = row["test_acc"]
        our_kib = row["mem_bytes"] / 1024
        d_acc = (our_acc - paper_acc) * 100
        flag = " ⚠️" if abs(d_acc) > 5 else ""
        print(f"| {ds} | {paper_acc:.3f} | {our_acc:.3f}{flag} | {d_acc:+.1f} | {our_kib:.3f} |")

    # ----- Per-system coverage summary -----
    print("\n## Coverage summary\n")
    for sys_name in ("BTHOWeN", "DWN"):
        sub = df[df["system"] == sys_name]
        n_ds = sub["dataset"].nunique()
        n_rows = len(sub)
        n_cells = sub["hp_id"].nunique()
        print(f"- **{sys_name}**: {n_rows} rows across {n_ds} datasets, {n_cells} unique hyperparameter cells")


if __name__ == "__main__":
    main()
