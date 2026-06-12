# wisardpkg

C++/Python library implementing the WiSARD (Wilkie, Stonham and Aleksander's Recognition Device) weightless neural network and its variants. Binary inputs are stored in RAM lookup tables — training is a write, inference is a read.

This is a fork with several extensions on top of upstream `IAZero/wisardpkg`: `BloomWisard` and 4 new fitted thermometers (`Distributive`, `Gaussian`, `Exponential`, `Stochastic`) plus `SupervisedThermometer`, large-address RAMs (`addressSize > 64`), Wisard classification hooks (negative evidence, RAM weights, shared discriminator, attention weighting), multi-resolution mappings, `ColorMaskBinarization` (palette-targeted 3-bits-per-pixel encoder, originally for Where's Waldo), and a `wisardpkg.models` Python subpackage with ports of three recent weightless architectures (BTHOWeN, DWN, ULEEN).

**Branch:** `main` (single canonical branch of this fork; see `CONTRIBUTING.md`); upstream `IAZero/wisardpkg` is on `develop`. The dedicated `performance` branch is the only long-lived exception, for the C++ reimplementation.
**Version:** 2.0.0a7
**Language:** C++ with PyBind11 Python bindings + Python package layer
**Build:** `pip install .` (core) or `pip install ".[torch]"` (with DWN/ULEEN); also `make install`
**Tests:** `make unittest` (compiles C++ then runs `test/testset.py`); `python3 test/test_dwn.py` and `test_uleen.py` separately for the torch-gated ports

## Documentation Index

### Core Concepts
- [Architecture](claude-docs/architecture.md) — Class hierarchy, data flow, module layout, design patterns, type definitions, Python package layer
- [RAMs and Discriminators](claude-docs/rams-and-discriminators.md) — RAM nodes (standard ≤64 and large-address >64 paths), address computation, vote storage, Discriminator aggregation, per-RAM weights and pruning, RAMDataHandle, mental images

### Binarization (continuous → binary)
- [Binarization](claude-docs/binarization.md) — All 13 techniques: Thresholding, MeanThresholding, SimpleThermometer, DynamicThermometer, CircularThermometer, DistributiveThermometer, GaussianThermometer, ExponentialThermometer, LogarithmicThermometer, StochasticThermometer, SupervisedThermometer, KernelCanvas, ColorMaskBinarization. Includes guidelines for choosing addressSize and thermometerSize.

### Classification
- [Classification Models](claude-docs/classification-models.md) — Wisard (supervised, with optional negative-evidence / RAM-weighting / shared-discriminator / attention / soft-bleaching / cross-class hooks), ClusWisard (clustering: supervised/semi-supervised/unsupervised), BloomWisard (counting Bloom filters with `murmur` / `simhash` / `h3` hash modes, `getRawVotes` for custom bleaching)
- [Classification Methods](claude-docs/classification-methods.md) — Bleaching, BestBleaching, BBleaching, Weighted (pluggable vote aggregation strategies)
- [Python Models](claude-docs/python-models.md) — `wisardpkg.models` subpackage: BTHOWeN (PACT 2022), DWN (ICML 2024), ULEEN (ACM TACO 2023). Hyperparameter sweep scripts under `scripts/sweeps/`.

### Regression
- [Regression Models](claude-docs/regression-models.md) — RegressionWisard, ClusRegressionWisard, all Mean functions (SimpleMean, PowerMean, Median, HarmonicMean, GeometricMean, ExponentialMean, LogisticMean)

### Data and Mapping
- [Data Structures](claude-docs/data-structures.md) — BinInput, DataSet, RAMDataHandle, RegressionRAMDataHandle, Synthesizer
- [Mapping](claude-docs/mapping.md) — MappingGenerator, RandomMapping (uniform OR multi-resolution tuple sizes)

### Build and Testing
- [Build and Testing](claude-docs/build-and-testing.md) — setup.py (with `[torch]` extra), Makefile, generate_include.py, C++ test list, Python-port test list, `scripts/sweeps/` reproducibility drivers, deployment

### Performance / Benchmarking
- [Performance Design](claude-docs/performance-design.md) — standalone optimized C++ standard-WiSARD reimplementation (the `performance` branch): bit-packed inputs, direct count-array / open-addressing RAM backends, bleaching parity, `deployedSizeBytes` yardstick, `wisard_bench` CLI + `*.wbin` shared format
- [Benchmark Methodology](claude-docs/benchmark-methodology.md) — fair C++-vs-pybind11-vs-pure-Python comparison on MNIST: shared binarized+mapped artifact, accuracy gate before timing, metrics, hygiene, visualizers, MNIST acquisition

## Quick Reference

### Pipeline
```
vector<double> → [Binarization] → BinInput → [DataSet] → Model.train() → Model.classify()/predict()
```

### Models at a Glance
| Model | Task | Key Feature |
|-------|------|-------------|
| `Wisard` | Classification | One discriminator per class; optional hooks for negative evidence, RAM weighting, shared discriminator, attention weighting, soft bleaching, cross-class scoring |
| `ClusWisard` | Classification | Multiple discriminators per class (clustering) + unsupervised mode |
| `BloomWisard` | Classification | Counting-Bloom-filter RAMs (`murmur` / `simhash` / `h3` hash modes); fixed memory regardless of distinct-address count |
| `RegressionWisard` | Regression | RAM nodes store target values, aggregated by Mean function |
| `ClusRegressionWisard` | Regression | Multiple RegressionWisards with adaptive clustering |
| `Discriminator` | Low-level | Standalone RAM group, usable independently |
| `BTHOWeN` (Python) | Classification | BloomWisard + Gaussian thermometer + H3 + binary-search bleach tuning (Susskind, PACT 2022) |
| `DWN` / `DWNClassifier` (Python, torch) | Classification | Differentiable Weightless Networks via EFD trick (Bacellar, ICML 2024) |
| `ULEEN` / `ULEENClassifier` (Python, torch) | Classification | Continuous-Bloom-filter ensemble trained via STE (Susskind, ACM TACO 2023) |

### Key Source Locations
| Area | Path |
|------|------|
| Python bindings | `src/wisard_bind.cc` (module name `_native`) |
| Master header | `src/wisardpkg.h` |
| Core types | `src/common/definetypes.cc` |
| RAM implementation | `src/models/wisard/ram.cc` (standard + large-address paths) |
| WiSARD model | `src/models/wisard/wisard.cc` |
| ClusWiSARD model | `src/models/cluswisard/cluswisard.cc` |
| BloomWiSARD model | `src/models/bloomwisard/` |
| Regression model | `src/models/regressionwisard/regressionwisard.cc` |
| Binarization | `src/binarization/` (incl. `colormaskbinarization.cc`) |
| Mapping | `src/mapping/` |
| Python package | `wisardpkg/__init__.py`, `wisardpkg/models/{bthowen,dwn,uleen}.py` |
| Sweep drivers | `scripts/sweeps/run_*.py`, `scripts/verify_against_papers.py` |
| Tests | `test/` |
| Existing docs | `docs/` |

### Build Commands
```bash
make install                       # pip install (core only)
pip install ".[torch]"             # adds torch + numpy → enables DWN, ULEEN
make unittest                      # compile + run C++-port tests
python3 test/test_dwn.py           # DWN tests (need torch)
python3 test/test_uleen.py         # ULEEN tests (need torch)
make geninclude                    # generate standalone C++ header
make clean                         # remove compiled .so files from test/
```

### Codebase Notes
- `WisardWrapper(addressSize, mappingGenerator=...)` re-applies `setTupleSize(addressSize)` *after* kwargs processing — so a user-supplied mapping always picks up the Wisard's addressSize, even when the mapping was constructed without one. (Earlier versions silently lost the addressSize when a mapping was passed via kwargs and the user hadn't set tupleSize on the mapping themselves.)
- All `.cc` files are `#include`'d into a single compilation unit via `wisardpkg.h` — they are not independently compiled
- The C++ extension is built as `wisardpkg._native` (a submodule), and `wisardpkg/__init__.py` re-exports its symbols so `import wisardpkg as wp` is unchanged
- PyBind11 wrappers in `src/wrappers/` handle `py::kwargs` → C++ member conversion
- Polymorphic objects (ClassificationBase*, MappingGenerator*, Mean*) are always cloned in wrappers to manage ownership
- JSON serialization uses nlohmann/json (vendored at `src/libs/json.hpp`)
- Factory pattern in `register.cc` and `mappinggeneratorhelper.cc` for deserialization dispatch
- `stochasticthermometer.cc` is included AFTER model definitions (it uses `Wisard` and `DataSet` in its `optimize()` method)
- Fitted thermometers (Distributive, Gaussian, Exponential, Stochastic, Supervised) all support `setThresholds()` for restoring saved threshold states
- Build with `--no-build-isolation` flag: `pip install --no-build-isolation .` (pybind11 must be pre-installed in the venv)
- The `wisardpkg.models.DWN` and `wisardpkg.models.ULEEN` ports are pure-Python PyTorch implementations with no C++ dependency — they can be imported and used independently of the C++ extension once `torch` is installed
