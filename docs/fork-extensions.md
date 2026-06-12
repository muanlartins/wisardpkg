# Fork Extensions vs Upstream

This is a fork of [IAZero/wisardpkg](https://github.com/IAZero/wisardpkg). It
adds binarization techniques, classification hooks, large-address RAMs, a 2D
local mapping, multi-resolution tuples, and a `wisardpkg.models` Python
subpackage that ports three recent weightless architectures.

| | Upstream | This fork |
|---|---|---|
| Remote | `git@github.com:IAZero/wisardpkg` | `git@github.com:muanlartins/wisardpkg` |
| Working branch | `develop` | `muanlartins` |
| Distribution | top-level `wisardpkg` extension | Python package `wisardpkg` + `wisardpkg._native` C++ submodule |

The user-facing API is unchanged: `import wisardpkg as wp; wp.Wisard(...)`
still works. `wisardpkg/__init__.py` re-exports the C++ core via
`from ._native import *`
([`wisardpkg/__init__.py:24`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/wisardpkg/__init__.py#L24)).

The deeper engineering reference lives under `claude-docs/`. This page is the
single index of *what the fork adds and why*. The `iazero.github.io` class
pages linked from the original `docs/` tree only cover the upstream surface
(`Wisard`, `ClusWisard`, `Discriminator`, `KernelCanvas`, `BinInput`,
`DataSet`, `Synthesizer`, `RAMDataHandle`) — every entry in the tables below
is fork-only.

## Why these extensions exist

The fork is the experimental engine for the **F4RM paper** (the WiSARD
prior-art comparison in `masters/articles/f4rms/`). Section 4 of that paper
benchmarks recent weightless architectures (BTHOWeN, DWN, ULEEN) against the
F4RM dataset suite. The ports under `wisardpkg/models/` and the sweep drivers
under `scripts/sweeps/` are how those numbers are produced — see
[Reproducing the F4RM numbers](#reproducing-the-f4rm-numbers).

The custom binarizers (`Circular`, `Logarithmic`, `Supervised`,
`Stochastic`, fitted thermometers) and the `ColorMaskBinarization` palette
encoder came out of the same research line (`ColorMaskBinarization` was
originally built for the Where's Waldo detector in
`masters/notebooks/waldo/`).

## New binarization techniques

All are PyBind-exposed in
[`src/wisard_bind.cc`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc).
Full reference: [`claude-docs/binarization.md`](../claude-docs/binarization.md).

| Class | Source | Constructor | Notes |
|---|---|---|---|
| `DistributiveThermometer` | `src/binarization/distributivethermometer.cc` | `(thermometerSize=32)` | Per-feature percentile thresholds; `fit()` + `getThresholds`/`setThresholds` ([bind 112](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L112)) |
| `GaussianThermometer` | `src/binarization/gaussianthermometer.cc` | `(thermometerSize=32)` | Thresholds at the inverse Gaussian CDF; `fit()` ([bind 120](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L120)) |
| `ExponentialThermometer` | `src/binarization/exponentialthermometer.cc` | `(thermometerSize=32)` | Thresholds at the Exponential CDF; `fit()` ([bind 128](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L128)) |
| `LogarithmicThermometer` | `src/binarization/logarithmicthermometer.cc` | `(thermometerSize=32)` | Log-spaced thresholds over the fitted per-feature `[min,max]`; `fit()` ([bind 136](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L136)) |
| `CircularThermometer` | `src/binarization/circularthermometer.cc` | `(thermometerSize=32, minimum=0, maximum=2π)` | Periodic values (lat/lon, hour); each bit fires within a circular-distance window ([source 1](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/binarization/circularthermometer.cc#L1), [bind 144](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L144)) |
| `SupervisedThermometer` | `src/binarization/supervisedthermometer.cc` | `(thermometerSize=32, method="class_conditional", minBitsPerFeature=2)` | Label-aware; `method ∈ {class_conditional, mi_allocation, entropy_weighted}`; variable bits per feature via `getSizes()` ([bind 154](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L154)) |
| `StochasticThermometer` | `src/binarization/stochasticthermometer.cc` | `(thermometerSize=32)` | `fit()` then `optimize(data, labels, addressSize, ...)` — coordinate descent on thresholds against a Wisard validation slice ([bind 166](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L166)) |
| `ColorMaskBinarization` | `src/binarization/colormaskbinarization.cc` | `(redMin=0.5, ...)` | Palette-targeted 3-bits-per-pixel RGB encoder; input is a flat row-major `H*W*3` vector, output `3*pixel_count` bits ([source 1](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/binarization/colormaskbinarization.cc#L1), [bind 90](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L90)) |

All fitted thermometers expose `getThresholds()` / `setThresholds()` so a
fitted encoder can be saved and restored without re-fitting.

## New classification model

| Class | Source | Notes |
|---|---|---|
| `BloomWisard` | `src/models/bloomwisard/` | Counting-Bloom-filter RAMs; fixed memory regardless of distinct-address count. `hashMode ∈ {murmur, simhash, h3}` ([`bloomfilter.cc`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/bloomwisard/bloomfilter.cc), wrapper [`bloomwisardwrapper.cc:24`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/bloomwisardwrapper.cc#L24)) |

`BloomWisard(addressSize, numBits=1024, numHashes=3, hashMode=...)` adds
`getRawVotes()` for custom bleaching from Python — used by the BTHOWeN port's
binary-search bleach tuning
([bind 331](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L331)).
H3 universal hashing (`hashMode="h3"`) is the address-generation scheme the
BTHOWeN and ULEEN ports rely on
([`bloomfilter.cc:22` `initH3`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/bloomwisard/bloomfilter.cc#L22)).
Full reference: [`claude-docs/classification-models.md`](../claude-docs/classification-models.md).

## Wisard classification hooks

The upstream `Wisard` constructor is extended with opt-in kwargs parsed in
[`src/wrappers/wisardwrapper.cc`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc).
Each flag defaults to off
([`wisard.cc:56`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L56)),
so a plain `Wisard(addressSize=N)` behaves exactly like upstream.

| Kwarg | Type | Parsed at | Effect |
|---|---|---|---|
| `negativeEvidence` (+ `negativeAlpha`, `negativeMode`) | bool, double, str | [wrapper 57](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L57) | Subtract competing-class evidence; `negativeMode ∈ {uniform, max_competitor, normalized}` ([wisard.cc:338](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L338)) |
| `sharedDiscriminator` (+ `sharedBeta`) | bool, double | [wrapper 67](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L67) | Cross-class shared discriminator term ([wisard.cc:326](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L326)) |
| `attentionWeighting` | bool | [wrapper 74](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L74) | Attention-like per-RAM weighting ([wisard.cc:246](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L246)) |
| `softBleaching` | bool | [wrapper 50](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L50) | Soft per-class totals instead of a hard bleach threshold ([wisard.cc:288](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L288)) |
| `crossClassScoring` | bool | [wrapper 53](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L53) | Cross-class score adjustment ([wisard.cc:265](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L265)) |

Per-RAM weights are computed/applied separately: `computeRAMWeights(dataset,
metric="entropy")` with `metric ∈ {entropy, information_gain, purity}`, plus
`getRAMWeights` / `setRAMWeights` / `pruneRAMs`
([bind 324](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L324),
[wisard.cc:378](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L378)).
The α=0 / β=0 equivalence checks in `test/test_wisard_hooks.py` prove every
hook is a no-op when disabled.

## Large-address RAMs (`addressSize > 64`)

Upstream caps an `addressSize` of 64 because an address is packed into a
64-bit `addr_t`. The fork adds a parallel string-key path so a single RAM can
read more than 64 input bits. A RAM switches to the large path when `base ==
2 && addressSize > 64`
([`ram.cc:19`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/ram.cc#L19),
also lines [9](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/ram.cc#L9)
and [27](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/ram.cc#L27))
and keys its table on `large_addr_t = std::string` instead of `addr_t`
([`definetypes.cc:8`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/common/definetypes.cc#L8)).

Limitation: large-address RAMs do **not** deserialize from JSON — the
constructor that reads a saved config only restores the standard ≤64-bit path
([`ram.cc:16`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/ram.cc#L16)).
Covered by the `addressSize > 64` case in `test/test_wisard_hooks.py`. Full
reference: [`claude-docs/rams-and-discriminators.md`](../claude-docs/rams-and-discriminators.md).

## New mappings

| Class | Source | Notes |
|---|---|---|
| `Local2DMapping` | `src/mapping/local2dmapping.cc` | Windowed 2D receptive fields over an `imageHeight × imageWidth × bitsPerPixel` input; `stride`, `ramsPerWindow` ([bind 198](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wisard_bind.cc#L198)) |
| Multi-resolution `RandomMapping` | `src/mapping/randommapping.cc` | Same RAM count as standard, but variable tuple sizes linearly spaced in `[max(2, tupleSize/2), tupleSize·3/2]` and shuffled ([randommapping.cc:75](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/mapping/randommapping.cc#L75)) |

Multi-resolution is enabled by passing `multiResolution=True` to the `Wisard`
constructor, not on the mapping itself — the wrapper sets
`mappingGenerator->multiResolution`
([`wisardwrapper.cc:47`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L47);
consumed at [`wisard.cc:221`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/models/wisard/wisard.cc#L221)).
The wrapper re-applies `setTupleSize(addressSize)` after kwargs processing
([`wisardwrapper.cc:82`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/src/wrappers/wisardwrapper.cc#L82))
so a user-supplied mapping always inherits the Wisard's `addressSize`. Full
reference: [`claude-docs/mapping.md`](../claude-docs/mapping.md).

## Python package layer: `wisardpkg.models`

Pure-Python ports of three recent weightless architectures, added alongside
the C++ core. Full reference:
[`claude-docs/python-models.md`](../claude-docs/python-models.md).

| Class | Paper | Backend | Optional dep | Source |
|---|---|---|---|---|
| `BTHOWeN` | Susskind et al., PACT 2022 (arXiv:2203.01479) | C++ core (`BloomWisard` + H3 + `GaussianThermometer`) | none | `wisardpkg/models/bthowen.py` |
| `DWN` / `DWNClassifier` | Bacellar et al., ICML 2024 (arXiv:2410.11112) | PyTorch CPU port of the CUDA EFD kernels | `torch` | `wisardpkg/models/dwn.py` |
| `ULEEN` / `ULEENClassifier` | Susskind et al., ACM TACO 2023 (doi:10.1145/3629522) | PyTorch CPU port of the libtorch H3 extension | `torch` | `wisardpkg/models/uleen.py` |

`BTHOWeN` is always importable (uses only the C++ core). `DWN` and `ULEEN`
are lazy-imported via PEP 562 and require the `torch` extra; accessing them
without it raises `ImportError` at access time, not at `import wisardpkg`
([`wisardpkg/models/__init__.py:26`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/wisardpkg/models/__init__.py#L26)).

```bash
pip install .            # core + BTHOWeN
pip install ".[torch]"   # adds torch + numpy → DWN, ULEEN
```

Common API across all three: `fit(X, y, seed=0)`, `predict(X_test)`,
`model_size_bytes()` (post-binarisation footprint, paper-style).

## Reproducing the F4RM numbers

`scripts/sweeps/` holds 10 restartable sweep drivers that produce the F4RM
paper's section-4 prior-WiSARD comparison. Each caches to a per-driver pickle
so a Ctrl-C mid-sweep is safe to resume. Full reference:
[`claude-docs/python-models.md`](../claude-docs/python-models.md#hyperparameter-sweep-scripts).

| Script | Purpose |
|---|---|
| `run_bthowen_exact_paper.py` | Reproduce the BTHOWeN paper Table III configs (per-dataset bpi/n/entries/hashes) at 10 seeds |
| `run_bthowen_extension.py` | Extend the BTHOWeN grid to the F4RM dataset suite |
| `run_bthowen_sweep.py` | Full BTHOWeN hyperparameter grid |
| `run_bthowen_table3_grid.py` | Restricted Table III grid |
| `run_dwn_long_epochs.py` | DWN, paper's 100-epoch multi-stage LR schedule |
| `run_dwn_paper_fidelity.py` | DWN, paper Table 5 hyperparameters |
| `run_dwn_sweep.py` | Full DWN hyperparameter grid |
| `run_uleen_final.py` | Best ULEEN config found during sweeping |
| `run_uleen_paper_lr.py` | ULEEN, `lr=1e-3, epochs=100` (paper schedule) |
| `run_uleen_sweep.py` | Full ULEEN hyperparameter grid |

`scripts/verify_against_papers.py` reads the caches and prints a per-dataset
best-cell comparison against the published numbers.

These scripts are **not** part of the installed package. They depend on the
F4RM dataset loaders in the parent `masters/` repo
(`notebooks/notebook_lib.py`, imported at
[`run_bthowen_exact_paper.py:30`](https://github.com/muanlartins/wisardpkg/blob/muanlartins/scripts/sweeps/run_bthowen_exact_paper.py#L30))
and will not run against an isolated checkout of `wisardpkg/`.

## See also

- [`../CLAUDE.md`](../CLAUDE.md) — top-level codebase guide and quick reference
- [`../claude-docs/architecture.md`](../claude-docs/architecture.md) — class hierarchy, data flow, Python package layer
- [`../claude-docs/build-and-testing.md`](../claude-docs/build-and-testing.md) — build, the `[torch]` extra, the full test list
