# wisardpkg

C++/Python library implementing the WiSARD (Wilkie, Stonham and Aleksander's Recognition Device) weightless neural network and its variants. Binary inputs are stored in RAM lookup tables — training is a write, inference is a read.

**Branch:** `develop` (most active)
**Version:** 2.0.0a7
**Language:** C++ with PyBind11 Python bindings
**Build:** `pip install .` or `make install`
**Tests:** `make unittest` (compiles C++ then runs `test/testset.py`)

## Documentation Index

### Core Concepts
- [Architecture](claude-docs/architecture.md) — Class hierarchy, data flow, module layout, design patterns, type definitions
- [RAMs and Discriminators](claude-docs/rams-and-discriminators.md) — RAM nodes, address computation, vote storage, Discriminator aggregation, RAMDataHandle, RegressionRAMDataHandle, mental images

### Binarization (continuous → binary)
- [Binarization](claude-docs/binarization.md) — All 9 techniques: Thresholding, MeanThresholding, SimpleThermometer, DynamicThermometer, DistributiveThermometer, GaussianThermometer, ExponentialThermometer, StochasticThermometer, KernelCanvas. Includes guidelines for choosing addressSize and thermoSize.

### Classification
- [Classification Models](claude-docs/classification-models.md) — Wisard (supervised), ClusWisard (clustering: supervised/semi-supervised/unsupervised)
- [Classification Methods](claude-docs/classification-methods.md) — Bleaching, BestBleaching, BBleaching, Weighted (pluggable vote aggregation strategies)

### Regression
- [Regression Models](claude-docs/regression-models.md) — RegressionWisard, ClusRegressionWisard, all Mean functions (SimpleMean, PowerMean, Median, HarmonicMean, GeometricMean, ExponentialMean, LogisticMean)

### Data and Mapping
- [Data Structures](claude-docs/data-structures.md) — BinInput, DataSet, RAMDataHandle, RegressionRAMDataHandle, Synthesizer
- [Mapping](claude-docs/mapping.md) — MappingGenerator, RandomMapping (how input bits are assigned to RAMs)

### Build and Testing
- [Build and Testing](claude-docs/build-and-testing.md) — setup.py, Makefile, generate_include.py, test structure, deployment

## Quick Reference

### Pipeline
```
vector<double> → [Binarization] → BinInput → [DataSet] → Model.train() → Model.classify()/predict()
```

### Models at a Glance
| Model | Task | Key Feature |
|-------|------|-------------|
| `Wisard` | Classification | One discriminator per class |
| `ClusWisard` | Classification | Multiple discriminators per class (clustering) + unsupervised mode |
| `RegressionWisard` | Regression | RAM nodes store target values, aggregated by Mean function |
| `ClusRegressionWisard` | Regression | Multiple RegressionWisards with adaptive clustering |
| `Discriminator` | Low-level | Standalone RAM group, usable independently |

### Key Source Locations
| Area | Path |
|------|------|
| Python bindings | `src/wisard_bind.cc` |
| Master header | `src/wisardpkg.h` |
| Core types | `src/common/definetypes.cc` |
| RAM implementation | `src/models/wisard/ram.cc` |
| WiSARD model | `src/models/wisard/wisard.cc` |
| ClusWiSARD model | `src/models/cluswisard/cluswisard.cc` |
| Regression model | `src/models/regressionwisard/regressionwisard.cc` |
| Binarization | `src/binarization/` |
| Tests | `test/` |
| Existing docs | `docs/` |

### Build Commands
```bash
make install      # pip install
make unittest     # compile + run tests
make geninclude   # generate standalone C++ header
make clean        # remove compiled .so files from test/
```

### Codebase Notes
- All `.cc` files are `#include`'d into a single compilation unit via `wisardpkg.h` — they are not independently compiled
- PyBind11 wrappers in `src/wrappers/` handle `py::kwargs` → C++ member conversion
- Polymorphic objects (ClassificationBase*, MappingGenerator*, Mean*) are always cloned in wrappers to manage ownership
- JSON serialization uses nlohmann/json (vendored at `src/libs/json.hpp`)
- Factory pattern in `register.cc` and `mappinggeneratorhelper.cc` for deserialization dispatch
- `stochasticthermometer.cc` is included AFTER model definitions (it uses `Wisard` and `DataSet` in its `optimize()` method)
- Fitted thermometers (Distributive, Gaussian, Exponential, Stochastic) all support `setThresholds()` for restoring saved threshold states
- Build with `--no-build-isolation` flag: `pip install --no-build-isolation .` (pybind11 must be pre-installed in the venv)
