# Architecture

## Overview

wisardpkg is a C++ Python extension (via PyBind11) implementing the WiSARD (Wilkie, Stonham and Aleksander's Recognition Device) weightless neural network and its variants. The library supports classification, regression, clustering, and data synthesis — all built on the core concept of RAM (Random Access Memory) nodes that learn binary patterns.

**Version:** 2.0.0a7
**License:** MIT
**Source language:** C++ with Python bindings
**Build system:** setuptools + pybind11

## Core Concept: Weightless Neural Networks

Unlike traditional neural networks that learn continuous weights, WiSARD uses RAM lookup tables. Each RAM node:
1. Receives a subset of the binary input bits (determined by a mapping)
2. Computes an address from those bits (base-N weighted sum)
3. Looks up (or stores) a value at that address

This makes training a single write operation and inference a single read — extremely fast.

## Class Hierarchy

```
Model (abstract)
├── ClassificationModel
│   ├── Wisard → WisardWrapper (Python-exposed)
│   ├── ClusWisard → ClusWisardWrapper (Python-exposed)
│   └── BloomWisard → BloomWisardWrapper (Python-exposed)
│         └── BloomDiscriminator → BloomRAM → BloomFilter (counting Bloom filter
│             with MurmurHash3 / SimHash LSH / H3 universal-hash modes)
└── RegressionModel
    ├── RegressionWisard → RegressionWisardWrapper (Python-exposed)
    └── ClusRegressionWisard → ClusRegressionWisardWrapper (Python-exposed)

Discriminator → DiscriminatorWrapper (Python-exposed, standalone use)
RAM (standard ≤64-bit address path AND large-address >64-bit string-key path)

ClassificationBase (strategy pattern)
├── Bleaching
├── BestBleaching
├── BBleaching
└── Weighted

MappingGenerator (abstract)
├── RandomMapping (uniform OR multi-resolution tuple sizes)
└── Local2DMapping (tiled 2D windows for image-shaped inputs)

BinBase (abstract binarization — 13 Python-bound subclasses, src/wisard_bind.cc:90–179)
├── Thresholding
├── MeanThresholding
├── SimpleThermometer
├── DynamicThermometer
├── CircularThermometer           (static, periodic values; ctor (size, min, max))
├── DistributiveThermometer      (fit-based, per-feature percentile thresholds)
├── GaussianThermometer          (fit-based, per-feature Gaussian CDF thresholds)
├── ExponentialThermometer       (fit-based, per-feature Exponential CDF thresholds)
├── LogarithmicThermometer       (fit-based, per-feature log-spaced thresholds)
├── StochasticThermometer        (fit+optimize, coordinate descent on thresholds)
├── SupervisedThermometer        (fit-based, label-aware: class_conditional /
│                                  mi_allocation / entropy_weighted)
├── ColorMaskBinarization        (static, palette predicates — 3 bits/pixel)
└── KernelCanvas → KernelCanvasWrapper (Python-exposed)

Mean (abstract regression aggregation)
├── SimpleMean
├── PowerMean
├── Median
├── HarmonicMean
├── HarmonicPowerMean
├── GeometricMean
├── ExponentialMean
└── LogisticMean

# Python-side ports (wisardpkg.models — pure Python, optional torch dep)
BTHOWeN          (Susskind PACT 2022 — built on BloomWisard + GaussianThermometer + H3)
DWNClassifier    (Bacellar ICML 2024 — pure PyTorch, EFD CPU port)
ULEENClassifier  (Susskind ACM TACO 2023 — pure PyTorch, continuous Bloom filters)
```

## Data Flow

```
vector<double> (continuous input)
    │
    ▼ [Binarization: Thresholding / Thermometer / KernelCanvas / MeanThresholding]
    │
BinInput (compact binary vector, 8 bits per byte)
    │
    ▼ [Organize into labeled/unlabeled collections]
    │
DataSet (collection with optional labels or y-values)
    │
    ▼ [Model.train()]
    │
WiSARD Model (internal RAM structure: address → vote/value)
    │
    ▼ [Model.classify() / Model.predict()]
    │
string (classification label) OR double (regression value)
```

## Module Organization

All source is in `src/` as `.cc` files included from the master header `wisardpkg.h`:

| Directory | Purpose |
|-----------|---------|
| `src/common/` | Type definitions (`definetypes.cc`), utilities (`utils.cc`), exceptions (`exceptions.cc`) |
| `src/binarization/` | Binary encoding techniques (`binbase.cc`, `thresholding.cc`, `meanthresholding.cc`, `thermometer.cc`, `distributivethermometer.cc`, `gaussianthermometer.cc`, `exponentialthermometer.cc`, `logarithmicthermometer.cc`, `circularthermometer.cc`, `stochasticthermometer.cc`, `supervisedthermometer.cc`, `colormaskbinarization.cc`, `kernelcanvas.cc`) |
| `src/classification_methods/` | Vote aggregation strategies (`bleaching.cc`, `bestbleaching.cc`, `bbleaching.cc`, `weighted.cc`, `register.cc`). `bbleaching.cc` is C++-internal only — not bound in `register.cc` or `wisard_bind.cc`, so unreachable from Python. |
| `src/mapping/` | Input-to-RAM bit assignment (`mappinggenerator.cc`, `randommapping.cc`, `local2dmapping.cc`, `mappinggeneratorhelper.cc`) |
| `src/data/` | Data containers (`bininput.cc`, `dataset.cc`) |
| `src/synthetic_data/` | Data generation (`synthesizers.cc`) |
| `src/models/base/` | Abstract model interfaces (`model.cc`, `classificationmodel.cc`, `regressionmodel.cc`) |
| `src/models/wisard/` | Core WiSARD (`ram.cc`, `discriminator.cc`, `wisard.cc`, `ramdatahandle.cc`) |
| `src/models/cluswisard/` | Clustering WiSARD (`cluster.cc`, `cluswisard.cc`) |
| `src/models/bloomwisard/` | Counting-Bloom-filter WiSARD (`bloomfilter.cc`, `bloomram.cc`, `bloomdiscriminator.cc`, `bloomwisard.cc`, `lsh.h`, `murmur3.h`) |
| `src/models/regressionwisard/` | Regression variant (`regressionram.cc`, `regressionwisard.cc`, `meanfunctions.cc`, `regressionramdatahandle.cc`) |
| `src/models/clusregressionwisard/` | Clustering regression (`clusregressionwisard.cc`) |
| `src/wrappers/` | PyBind11 wrappers that handle `py::kwargs` → C++ member assignment |
| `wisardpkg/` | Python package layer: re-exports the C++ core as `wisardpkg._native`, adds `wisardpkg.models` subpackage with BTHOWeN / DWN / ULEEN ports |
| `scripts/sweeps/` | Hyperparameter sweep drivers for reproducing the F4RM paper's prior-WiSARD comparison numbers |

## Key Design Patterns

### Header-Only Compilation
All `.cc` files are `#include`'d into `wisardpkg.h`, which is the single compilation unit. The `generate_include.py` script can produce a standalone `include/wisardpkg.hpp` for C++-only usage.

### Wrapper Pattern
Each Python-exposed model has a `*Wrapper` class that:
1. Inherits from the C++ implementation class
2. Accepts `py::kwargs` in its constructor
3. Iterates kwargs, type-casts values, and assigns to parent members
4. Clones polymorphic objects (ClassificationBase*, MappingGenerator*, Mean*) to manage ownership

### Strategy Pattern
- **Classification methods**: Pluggable via `ClassificationBase*` — swap Bleaching for BestBleaching without changing the model
- **Mean functions**: Pluggable via `Mean*` — swap SimpleMean for PowerMean in regression models
- **Mapping generators**: Pluggable via `MappingGenerator*` — `RandomMapping` (default) and `Local2DMapping` (image windows)

### Factory/Registry
`ClassificationMethods::load()` and `MappingGeneratorHelper::load()` deserialize from JSON by dispatching on a `className` field.

## Core Type Definitions

```cpp
addr_t               = unsigned long long    // RAM addresses
index_size_t         = unsigned long long    // Index sizes
bin_t                = char                  // Binary values (stored in BinInput)
content_t            = int                   // RAM content for classification (vote counts)
ram_t                = unordered_map<addr_t, content_t>    // Classification RAM
regression_content_t = vector<double>        // RAM content for regression [count, sum_y, fit]
regression_ram_t     = unordered_map<addr_t, regression_content_t>  // Regression RAM
```

## File Suffixes (Persistence)

| Suffix | Purpose |
|--------|---------|
| `.wdpkg` | RAM data files |
| `.json` | Model configuration |
| `.wpkds` | Dataset files |

## Serialization

Models serialize to JSON via `json()` methods. RAMDataHandle and RegressionRAMDataHandle use Base64-encoded binary blocks for efficient RAM storage. DataSet has its own text-based format with prefixes: `R` (regression), `C` (classification), `U` (unsupervised).

## Memory Accounting: `getsizeof` vs `deployedSizeBytes`

Every `Model` exposes two byte-size queries with different meaning. `getsizeof()` is the **in-process** footprint (full C++ structs, hash-map buckets, label strings, weight vectors). `deployedSizeBytes()` is the **minimal shippable** learned state on one yardstick comparable across model families: lossless bit-packed seen-address set for `Wisard`/`ClusWisard`, the fixed `numRAMs × numBits` bit-table for `BloomWisard`. It is `virtual` on `Model` with the default `return getsizeof()` (`src/models/base/model.cc:10`); regression models keep that default, the WiSARD families override it. This is the distinction the F4RM paper's memory comparison relies on. Full reference (per-model formulas, the lifecycle/inspection methods `trainSingle`/`untrainSingle`/`reset`/`getTupleSizes`, and the `BloomWisard` inspectors): `classification-models.md` → "Model lifecycle and inspection".

## Key Source Files

- **Entry point**: `src/wisard_bind.cc` — defines `PYBIND11_MODULE(_native, m)` with all class bindings (the module name `_native` is set via `setup.py` so the extension lands at `wisardpkg/_native.…so`)
- **Master header**: `src/wisardpkg.h` — includes everything in dependency order
- **Version**: `src/version.h` — single `__version__` constant
- **JSON library**: `src/libs/json.hpp` — nlohmann/json (vendored)
- **Python package**: `wisardpkg/__init__.py` re-exports `wisardpkg._native.*` so `import wisardpkg as wp` works unchanged; `wisardpkg/models/` adds BTHOWeN (always available) and DWN/ULEEN (require optional `torch` extra)

## Python Package Layer

The build now produces a real Python package, not a single top-level extension:

```
wisardpkg/
├── __init__.py              # re-exports wisardpkg._native.*
├── _native.cpython-…so      # the C++ extension (built from src/wisard_bind.cc)
└── models/
    ├── __init__.py          # exports BTHOWeN; DWN/ULEEN via PEP 562 lazy import
    ├── bthowen.py           # uses wp.BloomWisard + wp.GaussianThermometer
    ├── dwn.py               # pure PyTorch (no C++ dependency)
    └── uleen.py             # pure PyTorch (no C++ dependency)
```

The user-facing API is preserved: `import wisardpkg as wp; wp.Wisard(...)` works exactly as before. The new `wisardpkg.models` subpackage adds Python-side ports of three recent weightless architectures — see `python-models.md` for the full reference.

The optional `torch` extra (`pip install ".[torch]"`) gates DWN and ULEEN; BTHOWeN is always available because it only uses the C++ core.
