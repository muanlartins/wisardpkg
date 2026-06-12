# wisardpkg

A C++/Python library of WiSARD (Wilkie, Stonham and Aleksander's Recognition
Device) weightless neural networks and their variants. Binary inputs are stored
in RAM lookup tables — training is a write, inference is a read — so models are
fast to train and cheap to run.

This is a research fork of [`IAZero/wisardpkg`](https://github.com/IAZero/wisardpkg)
maintained at [`github.com/muanlartins/wisardpkg`](https://github.com/muanlartins/wisardpkg).
The C++ core is exposed to Python through [pybind11](https://github.com/pybind/pybind11);
JSON serialization uses [nlohmann/json](https://github.com/nlohmann/json).

**Version:** 2.0.0a7 — **License:** MIT (see [`LICENSE.txt`](LICENSE.txt))

## What this fork adds

On top of upstream, this fork contributes:

- **BloomWisard** — a WiSARD classifier whose RAM nodes are counting Bloom
  filters, giving fixed memory regardless of how many distinct addresses are
  seen. Selectable hash family (`murmur` double-hashing, `simhash` LSH, or `h3`
  universal hashing), bleaching by min-count, and `getRawVotes` for driving a
  custom bleach-threshold search from Python.
- **Fitted thermometers** — `DistributiveThermometer` (empirical percentiles),
  `GaussianThermometer` (inverse-normal-CDF), `ExponentialThermometer`
  (exponential inverse-CDF), `LogarithmicThermometer` (log-spaced),
  `CircularThermometer` (periodic quantities), `StochasticThermometer`
  (coordinate-descent threshold optimizer), and `SupervisedThermometer`
  (label-aware, three allocation strategies). All support
  `fit` / `getThresholds` / `setThresholds`.
- **Large-address RAMs** (`addressSize > 64`) — a byte-packed `std::string`-keyed
  path so tuple sizes are no longer capped at 64 bits.
- **Wisard classification hooks** — optional negative-evidence scoring, learned
  per-RAM weighting + pruning, a shared background discriminator,
  attention-like RAM reweighting, soft bleaching, and cross-class vote scoring.
- **ColorMaskBinarization** — a palette-targeted 3-bits-per-pixel RGB encoder
  (originally built for a Where's Waldo detector), pairing with `Local2DMapping`.
- **Spatially-aware and multi-resolution mappings** — `Local2DMapping` (windowed,
  preserving 2D co-occurrence) and a multi-resolution flag on `RandomMapping`
  (variable tuple sizes with vote reweighting).
- **`deployedSizeBytes`** — a cross-model deployed-footprint metric for fair
  memory comparison across model families and against the Python ports.
- **`wisardpkg.models` Python subpackage** — pure-Python ports of three recent
  weightless architectures: **BTHOWeN** (Susskind, PACT 2022), **DWN**
  (Bacellar, ICML 2024), and **ULEEN** (Susskind, ACM TACO 2023). BTHOWeN runs
  on the C++ core and is always importable; DWN and ULEEN are pure-PyTorch and
  torch-gated.

## Install

Python (from a checkout of this repo):

```bash
pip install .            # C++ core only
pip install ".[torch]"   # also pulls torch + numpy → enables DWN and ULEEN
```

If pybind11 is already installed in your environment, you can skip the
build-isolation step:

```bash
pip install --no-build-isolation .   # requires pybind11 pre-installed in the venv
```

C++ only — copy the standalone header into your project and include it:

```cpp
#include "wisardpkg.hpp"
namespace wp = wisardpkg;
```

Regenerate that header from the source tree with `make geninclude`.

## Quickstart

```python
import wisardpkg as wp

X = [[1,1,1,0,0,0,0,0],
     [1,1,1,1,0,0,0,0],
     [0,0,0,0,1,1,1,1],
     [0,0,0,0,0,1,1,1]]
y = ["cold", "cold", "hot", "hot"]

wsd = wp.Wisard(3)            # addressSize = 3 bits per RAM
wsd.train(X, y)
print(wsd.classify(X))       # ['cold', 'cold', 'hot', 'hot']
```

The torch-gated ports are reached through the `wisardpkg.models` subpackage,
which is imported lazily so `torch` is only required when you use DWN or ULEEN:

```python
from wisardpkg.models import BTHOWeN       # always available
from wisardpkg.models import DWN, ULEEN    # needs pip install ".[torch]"
```

## Models at a glance

| Model | Task | Key feature |
|-------|------|-------------|
| `Wisard` | Classification | One discriminator per class; optional negative-evidence / RAM-weighting / shared-discriminator / attention / soft-bleaching / cross-class hooks |
| `ClusWisard` | Classification | Multiple discriminators per class (clustering) + unsupervised mode |
| `BloomWisard` | Classification | Counting-Bloom-filter RAMs (`murmur` / `simhash` / `h3`); fixed memory regardless of distinct-address count |
| `RegressionWisard` | Regression | RAM nodes store target values, aggregated by a Mean function |
| `ClusRegressionWisard` | Regression | Multiple RegressionWisards with adaptive clustering |
| `Discriminator` | Low-level | Standalone RAM group, usable independently |
| `BTHOWeN` (Python) | Classification | BloomWisard + Gaussian thermometer + H3 + binary-search bleach tuning (PACT 2022) |
| `DWN` / `DWNClassifier` (Python, torch) | Classification | Differentiable Weightless Networks via the EFD trick (ICML 2024) |
| `ULEEN` / `ULEENClassifier` (Python, torch) | Classification | Continuous-Bloom-filter ensemble trained via STE (ACM TACO 2023) |

## Documentation

- **`docs/`** — the upstream documentation site
  ([iazero.github.io/wisardpkg](https://iazero.github.io/wisardpkg/)).
- **`claude-docs/`** — this fork's in-depth developer reference:
  [architecture](claude-docs/architecture.md),
  [RAMs and discriminators](claude-docs/rams-and-discriminators.md),
  [binarization](claude-docs/binarization.md),
  [classification models](claude-docs/classification-models.md) and
  [methods](claude-docs/classification-methods.md),
  [Python models](claude-docs/python-models.md),
  [regression](claude-docs/regression-models.md),
  [data structures](claude-docs/data-structures.md),
  [mapping](claude-docs/mapping.md),
  [build and testing](claude-docs/build-and-testing.md),
  [performance design](claude-docs/performance-design.md), and
  [benchmark methodology](claude-docs/benchmark-methodology.md).
- **`CONTRIBUTING.md`** — branch policy, build/test commands, and doc
  conventions for this fork.

## Contributing

This fork follows a single-branch model — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Attribution

Forked from [`IAZero/wisardpkg`](https://github.com/IAZero/wisardpkg) by the
IAZero group (COPPE/UFRJ). Built on [pybind11](https://github.com/pybind/pybind11)
and [nlohmann/json](https://github.com/nlohmann/json). Released under the MIT
License.
