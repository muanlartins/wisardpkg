# Python Package Layer

The C++ extension is wrapped by a thin Python package that provides:

1. A re-export shim so existing user code (`import wisardpkg as wp`) continues to work unchanged after the extension was renamed to a submodule.
2. A `wisardpkg.models` subpackage with pure-Python ports of three recent weightless architectures: **BTHOWeN**, **DWN**, **ULEEN**.

**Source files:**
- `wisardpkg/__init__.py` — re-exports the C++ core, exposes `wisardpkg.models` lazily
- `wisardpkg/models/__init__.py` — top-level `BTHOWeN` plus lazy `DWN` / `ULEEN` (torch-gated)
- `wisardpkg/models/bthowen.py` — BTHOWeN port
- `wisardpkg/models/dwn.py` — DWN port (PyTorch CPU)
- `wisardpkg/models/uleen.py` — ULEEN port (PyTorch CPU)

## Package Layout

```
wisardpkg/
├── __init__.py              # re-exports wisardpkg._native.*, lazily attaches `models`
├── _native.cpython-…so      # the C++ extension (built from src/wisard_bind.cc)
└── models/
    ├── __init__.py          # exports BTHOWeN; DWN/ULEEN via PEP 562 lazy import
    ├── bthowen.py           # uses wp.BloomWisard + wp.GaussianThermometer
    ├── dwn.py               # pure PyTorch (no C++ dependency)
    └── uleen.py             # pure PyTorch (no C++ dependency)
```

The C++ extension was renamed from a top-level `wisardpkg.so` to `wisardpkg/_native.cpython-…so` so that it can sit inside a real Python package alongside the model ports. The user-facing API is preserved by `from ._native import *` at the top of `wisardpkg/__init__.py`:

```python
import wisardpkg as wp
wp.Wisard(addressSize=4)             # C++ class, unchanged
wp.GaussianThermometer(8)            # C++ class; ctor arg is thermometerSize (positional), src/wisard_bind.cc:121
wp.BloomWisard(addressSize=8, numBits=1024, numHashes=3, hashMode="h3")
```

## Optional Dependencies

`setup.py` declares an optional `torch` extra:

```bash
pip install .                  # core only — BTHOWeN works
pip install ".[torch]"         # adds torch>=2.0, numpy>=1.21 — enables DWN, ULEEN
```

`BTHOWeN` is always importable because it uses only the C++ core. `DWN` and `ULEEN` are imported lazily via PEP 562 (`__getattr__` in `wisardpkg/models/__init__.py`); attempting to access them without the `torch` extra raises `ImportError` at access time, not at `import wisardpkg`.

## Quick Reference

| Class | Paper | Backend | Optional dependency |
|-------|-------|---------|---------------------|
| `BTHOWeN` | Susskind et al., **PACT 2022** (arXiv:2203.01479) | C++ core (`BloomWisard` + H3 + `GaussianThermometer`) | none |
| `DWN` (alias `DWNClassifier`) | Bacellar et al., **ICML 2024** (arXiv:2410.11112) | PyTorch (CPU port of paper's CUDA EFD kernels) | `torch` |
| `ULEEN` (alias `ULEENClassifier`) | Susskind et al., **ACM TACO 2023** (doi:10.1145/3629522) | PyTorch (CPU port of paper's libtorch H3 extension) | `torch` |

Common API surface across all three:

```python
clf = SomeModel(...hyperparams...)
clf.fit(X, y, seed=0)              # X: (n, d) np.ndarray of floats; y: labels
y_pred = clf.predict(X_test)
size_bytes = clf.model_size_bytes()  # post-binarisation footprint, paper-style
```

---

## BTHOWeN

**Reference:** Susskind et al., *Weightless Neural Networks for Efficient Edge Inference*, PACT 2022 — [arXiv:2203.01479](https://arxiv.org/abs/2203.01479). Reference implementation: [github.com/ZSusskind/BTHOWeN](https://github.com/ZSusskind/BTHOWeN).

**Architecture:** Counting-Bloom-filter WiSARD with three pieces:

1. **Gaussian thermometer encoding** — per-feature percentile thresholds at the inverse Gaussian CDF (`wp.GaussianThermometer`).
2. **H3 universal hashing** (Carter & Wegman, 1979) for the Bloom filter address generation. This is exposed natively via `BloomWisard(hashMode="h3")` (see `classification-models.md`).
3. **Binary-search bleach tuning** on a held-out validation slice of the training set, then a single fixed bleach value applied to all subsequent predictions. Driven from Python via `BloomWisard.getRawVotes`.

**Training procedure** (mirrors the upstream `software_model/train_swept_models.py`):

1. Train all samples (Bloom counters increment).
2. Find the maximum counter value across all filters (sets the bleach search ceiling).
3. Binary-search the bleach threshold β maximising validation accuracy.
4. At inference, vote = number of RAMs whose Bloom min-count ≥ β, summed per discriminator.

The 90/10 train/val split is gated on `n > 10000` to match the upstream heuristic — small datasets fall back to evaluating the bleach search on the training set itself.

### Python API

```python
from wisardpkg.models import BTHOWeN

clf = BTHOWeN(
    addressSize=8,     # bits per Bloom filter (= upstream filter_inputs)
    numBits=1024,      # Bloom table slots (power of two; = filter_entries)
    numHashes=3,       # H3 hashes per filter (= filter_hashes)
    bitsPerInput=8,    # thermometer bits per input feature (= bits_per_input)
    valSplit=0.1,      # held-out validation fraction (only used when n > 10000)
)
clf.fit(X_train, y_train, seed=0)
y_pred = clf.predict(X_test)
size = clf.model_size_bytes()    # post-binarisation, matches paper's calc_model_size.py
```

### Memory accounting

Mirrors the BTHOWeN paper's `software_model/calc_model_size.py`:

```
filters_per_disc = ceil(entrySize / addressSize)
bits_in_model    = num_classes × filters_per_disc × numBits
mem_bytes        = ceil(bits_in_model / 8)
```

This is the **post-binarisation** footprint (one bit per Bloom slot after counters are collapsed at the chosen bleach threshold), which is what BTHOWeN paper Table III reports.

### Reproducibility caveat

H3 random constants are drawn from the C++ `rand()` PRNG inside `bloomfilter.cc:initH3` and there is currently no `srand(seed)` hook in the wrapper. `BTHOWeN.fit` seeds Python's `random` and `numpy` before constructing the model, but the C++ `rand()` state still depends on whatever else has called it in the process. For run-to-run determinism on the same Python process you can usually rely on `seed=0`; across processes, expect small accuracy deltas.

---

## DWN

**Reference:** Bacellar et al., *Differentiable Weightless Neural Networks*, ICML 2024 — [arXiv:2410.11112](https://arxiv.org/abs/2410.11112). Reference implementation: [github.com/alanbacellar/DWN](https://github.com/alanbacellar/DWN).

**Architecture:** A stack of LUT layers connected by learnable / random / arange mappings, trained with backprop via the **Extended Finite Difference (EFD)** trick to approximate gradients through the non-differentiable LUT lookup. The reference ships custom CUDA kernels for the EFD forward/backward (`custom_operators/cuda/efd_cuda_kernel.cu`); this port re-implements them as **vectorised PyTorch ops on CPU**, mathematically identical to the CUDA reference (verified by `torch.autograd.gradcheck` and bit-exact comparison against the closed-form weight derivation, see `test/test_dwn.py`).

### Components (matches upstream `torch_dwn.*`)

| Class / function | Purpose |
|---|---|
| `EFDFunction` (autograd Function) | Differentiable LUT lookup: forward = `luts[j, addr(x[b], mapping[j])]`; backward = scatter-add into `luts_grad` plus per-input gradient via the precomputed EFD decay tensor `D[k, a, a2]`. |
| `LUTLayer` | A layer of `output_size` independent LUTs each reading `n` input bits. Supports `mapping ∈ {"random", "arange", "learnable", torch.Tensor}`. Owns LUT params, the precomputed `decay_D` buffer, and the spectral-regularization `C` buffer. |
| `LearnableMapping` | Differentiable input-to-LUT mapping: forward picks one input per output via `argmax`; backward uses a temperature-softmax surrogate. |
| `GroupSum` | Reduction head: groups the last dim into `k` slices and sums each, dividing by `tau`. |
| `DistributiveThermometer` | Per-feature percentile-threshold encoder, identical to the reference. |
| `DWNClassifier` (alias `DWN`) | Convenience wrapper used by sweeps and the F4RM notebook. Builds `LUTLayer → ... → LUTLayer → GroupSum`, trains via Adam + StepLR. |

### Python API

```python
from wisardpkg.models import DWN

clf = DWN(
    layer_sizes=[1000, 1000, 200],   # one int per LUT layer (output_size)
    n=6,                              # bits per LUT (paper's sweet spot)
    bits_per_input=8,                 # distributive thermometer bits per feature
    mapping="random",                 # "random" | "arange" | "learnable" | torch.Tensor
    epochs=20, lr=1e-2, batch_size=32,
    spectral_lambda=0.0,              # paper §3.4 regularization weight (off by default)
)
clf.fit(X_train, y_train, seed=0)
y_pred = clf.predict(X_test)
size = clf.model_size_bytes()
```

### EFD details

Forward:
```
output[b, j] = luts[j, addr(x[b], mapping[j])]
```

Backward (per the CUDA reference, vectorised):
```
luts_grad[j, addr(x[b], mapping[j])]      += output_grad[b, j]
input_grad[b, mapping[j, k]]              += output_grad[b, j] · w[b, j, k]
    where w[b, j, k] = Σ_{a2} luts[j, a2] · D[k, addr[b,j], a2]
```

The decay tensor `D[k, a, a2] = sign_k(a2) · α · β^{dist(a & mask_k, a2 & mask_k)}` is precomputed once per `LUTLayer` and stored as a buffer. With the paper defaults `α = 0.5 · 0.75^(n-1)`, `β = 0.25/0.75 ≈ 0.333`, and `n = 6`, this is a 6 × 64 × 64 float32 tensor (96 KiB).

### Spectral regularization

Implemented (DWN paper §3.4). `LUTLayer.spectral_norm_squared()` returns `||L · C||_F^2` where `C[i, j] := (1/2^n) · Π_{a ∈ S(i)} (2·j_a − 1)` is the Fourier-coefficient matrix. The `DWNClassifier` adds `spectral_lambda · Σ_layers spectral_norm_squared` to the loss when `spectral_lambda > 0`.

### Memory accounting

`mem_method = "dwn_binarized_lut"` — post-binarisation footprint matching DWN paper Table 1:

```
bits_in_luts    = num_luts × 2^n
bytes_in_mapping= num_luts × n × 4   (int32 indices)
mem_bytes       = ceil(bits_in_luts / 8) + bytes_in_mapping
                  summed across all LUT layers.
```

### Caveats vs paper

- **Learnable Reduction (§3.3)** — not implemented. Affects ultra-low-NAND2-cost chip targets only; software-side accuracy on the Pareto plots is unaffected.
- **Default training schedule** — sweep-budget tradeoff. The paper Table 14 schedule is 100 ep, multi-stage `1e-2 → 1e-3 → 1e-4 → 1e-5`. The default in `DWNClassifier` is shorter for sweeping; pass `epochs`/`lr_schedule` to reproduce paper Table 5 exactly.

---

## ULEEN

**Reference:** Susskind et al., *Pruned Lightweight Encoders for Computer Vision*, ACM TACO 2023 — [doi:10.1145/3629522](https://doi.org/10.1145/3629522). Reference implementation: [github.com/ZSusskind/ULEEN](https://github.com/ZSusskind/ULEEN).

**Architecture:** Builds on BTHOWeN by replacing the counting Bloom filters with **continuous-valued Bloom filters** trained end-to-end via backprop with the straight-through estimator (STE), and by combining multiple **independently trained submodels** as an ensemble.

### Components (matches upstream `software_model/*`)

| Class / function | Purpose |
|---|---|
| `_h3_hash` | Vectorised PyTorch port of `software_model/cpp/h3_hash.cpp`. Bit-exact vs the libtorch C++ extension (verified by `test/test_uleen.py`). |
| `_STEBinarize` (autograd Function) | Forward `sign(x)`, backward identity. The straight-through estimator from §3.1.1. |
| `BackpropWiSARD` | One ULEEN submodel — `n_classes` continuous-Bloom-filter discriminators in a single `nn.Parameter` of shape `(classes, filters_per_disc, filter_entries)`. |
| `BackpropMultiWiSARD` | `nn.ModuleList` ensemble — `len(configs)` independently trained `BackpropWiSARD`s. |
| `_LinearThermometer` / `_GaussianThermometer` | Per-feature thermometer encoders (linear: min/max per feature; gaussian: percentile thresholds at the inverse Gaussian CDF). |
| `ULEENClassifier` (alias `ULEEN`) | Convenience wrapper used by sweeps and the F4RM notebook. Per-submodel CE summation, Adam optimiser, optional StepLR. |

### Python API

```python
from wisardpkg.models import ULEEN

clf = ULEEN(
    bits_per_input=3,                # thermometer bits per feature
    filter_inputs=12,                # bits per Bloom filter (the "n" in n-tuple)
    filter_entries=64,               # Bloom table slots (power of 2)
    filter_hash_functions=2,         # H3 hashes per filter
    n_submodels=3,                   # ensemble size
    dropout_p=0.0,                   # Bernoulli dropout on per-RAM responses
    epochs=30, lr=1e-2, batch_size=32, decay_lr=True,
    thermometer="linear",            # "linear" (default) or "gaussian"
)
clf.fit(X_train, y_train, seed=0)
y_pred = clf.predict(X_test)
size = clf.model_size_bytes()
```

### Memory accounting

`mem_method = "uleen_binarized_table"` — post-binarisation footprint, matching the paper's `compute_model_size` (`software_model/train_model.py:163-173`):

```
bits_per_submodel  = classes × filters_per_disc × filter_entries
mem_bytes          = ceil(bits_per_submodel / 8)
                      summed across all submodels.
```

`BackpropWiSARD.deployed_size_bytes(include_metadata=True)` will additionally count the H3 hash constants (8 bytes per `(filter_hash_functions × filter_inputs)`) and the input-bit ordering (4 bytes per input bit). This `include_metadata=True` mode reports the deploy-time footprint; the default mirrors the paper's reporting style for Pareto-plot consistency.

### Caveats vs paper

- **Default `lr=1e-2, epochs=30`** vs reference `lr=1e-3, epochs=100`. The shorter schedule converges in sweep-friendly time; pass `lr=1e-3, epochs=100` to reproduce paper Table 2 numbers.
- **Linear thermometer default**, not Gaussian. The reference's `create_model` defaults to Gaussian; paper Fig. 11 shows Gaussian gives ~0.6 pp on MNIST. Pass `thermometer="gaussian"` to match the paper.
- **Pruning (§3.1.3) not implemented end-to-end.** The structural support exists (`BackpropWiSARD.prune_filters`, `mask`, `bias`), but there is no automated prune-during-training step. Paper reports ~27% size reduction with <1% accuracy drop.

---

## Hyperparameter Sweep Scripts

`scripts/sweeps/` contains 10 sweep drivers used to reproduce the F4RM paper's prior-WiSARD comparison numbers. Each script is a restartable hyperparameter search that writes to a per-driver pickle cache (so a Ctrl-C mid-sweep is safe).

| Script | Purpose |
|---|---|
| `run_bthowen_exact_paper.py` | Reproduce the cell-by-cell BTHOWeN paper Table III configurations on the same datasets. |
| `run_bthowen_extension.py` | Extend the BTHOWeN cell grid to the F4RM dataset suite. |
| `run_bthowen_sweep.py` | Full hyperparameter grid for BTHOWeN (`addressSize × numBits × numHashes × bitsPerInput`). |
| `run_bthowen_table3_grid.py` | Restricted Table III grid only. |
| `run_dwn_long_epochs.py` | DWN with the paper's 100-epoch multi-stage LR schedule. |
| `run_dwn_paper_fidelity.py` | DWN with the paper Table 5 hyperparameters per dataset. |
| `run_dwn_sweep.py` | Full hyperparameter grid for DWN. |
| `run_uleen_final.py` | Best ULEEN configuration found during sweeping. |
| `run_uleen_paper_lr.py` | ULEEN with `lr=1e-3, epochs=100` (paper schedule). |
| `run_uleen_sweep.py` | Full hyperparameter grid for ULEEN. |

`scripts/verify_against_papers.py` reads the sweep caches and prints a per-dataset best-cell comparison against BTHOWeN paper Table III and DWN paper Table 5.

These scripts are **not** part of the installed package — they live under `scripts/` for reproducibility of the F4RM paper experiments. They depend on the F4RM dataset loaders that live in the parent `masters/` repo (`notebooks/notebook_lib.py`); they will not run in isolation against an arbitrary checkout of `wisardpkg/`.

---

## Tests

| File | Backend | What it covers |
|---|---|---|
| `test/test_bloomwisard.py` | C++ | Build/train/classify/rank/reset, both `hashMode="murmur"` and `hashMode="simhash"` paths. |
| `test/test_dwn.py` | PyTorch | EFD forward/backward, `gradcheck`, end-to-end fit. Skipped if `torch` is not installed. |
| `test/test_uleen.py` | PyTorch | H3 hash bit-exactness, end-to-end fit, deterministic seeding. Skipped if `torch` is not installed. |

`test_dwn.py` and `test_uleen.py` are **not** registered in `test/testset.py` (which is the C++-only suite); run them directly with `python3 test/test_dwn.py` / `python3 test/test_uleen.py` after `pip install ".[torch]"`.
