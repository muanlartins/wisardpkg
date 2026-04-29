# Build System and Testing

## Build System

### setup.py (Primary Build)

The package uses setuptools with a custom `BuildExt` class for C++ compilation via PyBind11.

**Extension:** Single compilation unit built as `wisardpkg._native` from `src/wisard_bind.cc` (which includes `wisardpkg.h`, which includes everything else). The extension lives **inside** the Python package as a `_native` submodule; `wisardpkg/__init__.py` re-exports its symbols so `import wisardpkg as wp` keeps working.

**Python package:** `setup.py` declares `packages=['wisardpkg', 'wisardpkg.models']`, so both `wisardpkg/__init__.py` and `wisardpkg/models/` ship alongside the compiled extension.

**Platform-specific flags:**
- **Unix/Linux:** C++11 detection (`-std=c++11` or `-std=c++0x`), `-fvisibility=hidden`
- **macOS (Darwin):** `-stdlib=libc++`, `-mmacosx-version-min=10.7`
- **MSVC (Windows):** `/EHsc` exception handling

**Dependencies:**
- `pybind11 >= 2.2` (build requirement, must be pre-installed in the venv if using `--no-build-isolation`)
- `setuptools` (build requirement)
- C++11-compatible compiler

**Optional extras** (`extras_require` in `setup.py`):
- `torch` → installs `torch>=2.0`, `numpy>=1.21`. Required for `wisardpkg.models.DWN` and `wisardpkg.models.ULEEN`. `wisardpkg.models.BTHOWeN` works without it.

**Install:**
```bash
pip install wisardpkg                          # from PyPI (core only)
pip install .                                  # from local source (core only — BTHOWeN works)
pip install ".[torch]"                         # adds torch + numpy → enables DWN, ULEEN
pip install --no-build-isolation .             # if pybind11 is already in the venv
```

### Makefile

| Target | Command | Description |
|--------|---------|-------------|
| `compile` | `cd src && make` | Compiles C++ source |
| `clean` | `cd test && rm -f *.so` | Removes compiled extension from test dir |
| `install` | `pip3 install .` | Installs via pip |
| `uninstall` | `pip3 uninstall wisardpkg` | Uninstalls |
| `geninclude` | `python3 generate_include.py` | Generates standalone C++ header |
| `cpptest` | `g++ test.cc ... && ./a.out` | Compiles and runs C++ test |
| `unittest` | `make compile && python3 test/testset.py` | Compiles then runs Python tests |
| `compile2` | `cd src && make test-build` | Alternative compile target |

### generate_include.py

Generates `include/wisardpkg.hpp` — a single-file C++ header that wraps all source files in a `wisardpkg` namespace. Useful for C++ projects that want to use the library without Python:

```cpp
#include "wisardpkg.hpp"
using namespace wisardpkg;
```

The script:
1. Reads 55 source files in dependency order
2. Strips `#include` lines referencing internal files
3. Wraps everything in `namespace wisardpkg { ... }`
4. Adds header guards and version info
5. Includes nlohmann/json inline

### Version Management

Version is defined in `src/version.h` as `__version__ = "2.0.0a7"` and is parsed by both `setup.py` and `generate_include.py`.

---

## Testing

### Test Structure

All tests in `test/` as Python unittest files. `test/testset.py` is the test runner that aggregates all test suites.

### Running Tests

```bash
make unittest                      # compile + run all tests
python3 test/testset.py            # run tests (assumes already compiled)
python3 -m pytest test/            # also works with pytest
```

### Test Files

C++ core tests (registered in `test/testset.py`, run by `make unittest`):

| File | What It Covers |
|------|----------------|
| `test_wisard.py` | Train, classify, score, json, mental images, leave-one-out, getsizeof |
| `test_wisard_hooks.py` | `negativeEvidence` (uniform / max_competitor / normalized), `sharedDiscriminator`, `attentionWeighting`, RAM weights (entropy / information_gain / purity + `setRAMWeights`/`pruneRAMs` round-trip), large-address RAM path (`addressSize > 64`). Includes α=0/β=0 equivalence checks proving the hooks are no-ops when disabled. |
| `test_bloomwisard.py` | BloomWisard build/train/classify/rank/reset, both `hashMode="murmur"` and `hashMode="simhash"` paths. |
| `test_discriminator.py` | Train, classify, json, custom mapping |
| `test_cluswisard.py` | Supervised/semi-supervised/unsupervised train+classify, json, mental images |
| `test_regressionwisard.py` | Train, predict, json, mean functions |
| `test_clusregressionwisard.py` | Train, predict, json |
| `test_bininput.py` | Constructors, indexing, serialization, extend |
| `test_dataset.py` | All constructor variants, add methods, save/load, labels/Ys |
| `test_kernel_canvas.py` | Transform with/without direction |
| `test_thermometer.py` | SimpleThermometer and DynamicThermometer |
| `test_fitted_thermometers.py` | `Distributive`, `Gaussian`, `Exponential`, `Stochastic` thermometers — fit/transform plus `getThresholds`/`setThresholds` round-trip. |
| `test_thresholding.py` | Basic threshold binarization |
| `test_mean_thresholding.py` | Mean-based threshold binarization |
| `test_ramdatahandle.py` | Get, set, serialization |
| `test_regressionramdatahandle.py` | Get, set, serialization |
| `test_synthesizer.py` | Synthetic data generation from mental images |

Python-side ports (NOT registered in `test/testset.py`; require the `torch` extra; run them directly):

| File | What It Covers |
|------|----------------|
| `test_dwn.py` | EFD forward/backward (bit-exact vs CUDA reference formula), `gradcheck`, end-to-end fit. |
| `test_uleen.py` | H3 hash bit-exactness vs the libtorch C++ reference, end-to-end fit, deterministic seeding. |

```bash
python3 test/test_dwn.py     # after `pip install ".[torch]"`
python3 test/test_uleen.py
```

### Test Data Pattern

Most tests use a common pattern:
```python
def setUp(self):
    self.X = [
        [1,0,1,0,1,0,1,0,1],
        [1,0,1,1,1,0,1,0,1],
        [0,0,0,0,1,0,0,0,1],
        [0,0,0,0,0,0,0,0,0],
    ]
    self.y = ["cold","cold","hot","hot"]
```

9-element binary vectors with "cold"/"hot" labels. Regression tests use float targets (0.1–0.8).

### Assertions Used

- `assertIsInstance(result, expected_type)` — type checking
- `assertSequenceEqual(result, expected)` — list/sequence equality
- `assertEqual(a, b)` — exact equality
- `assertAlmostEqual(a, b, places=2)` — float tolerance
- `assertDictEqual(a, b)` — dictionary equality
- `assertIn(item, collection)` — membership check

---

## Deployment

### PyPI

Package name: `wisardpkg`
```bash
pip install wisardpkg
```

### Publishing (maintainer)

```bash
./publish.sh     # uploads to PyPI
./release.sh     # creates a release
```

### C++ Usage

After running `make geninclude`:
```cpp
#include "include/wisardpkg.hpp"
using namespace wisardpkg;

// Use classes directly (no Python)
Wisard w;
// ...
```

### Continuous Integration

`.travis.yml` configured for Travis CI (Python testing on Linux).

---

## Hyperparameter Sweep Scripts

`scripts/sweeps/` contains 10 restartable sweep drivers used to reproduce the F4RM paper's prior-WiSARD comparison numbers (BTHOWeN, DWN, ULEEN). Each script writes results to a per-driver pickle cache so that a Ctrl-C mid-sweep is safe to resume from.

| Script | Purpose |
|---|---|
| `run_bthowen_exact_paper.py` | Reproduce the cell-by-cell BTHOWeN paper Table III configurations. |
| `run_bthowen_extension.py` | Extend the BTHOWeN cell grid to the F4RM dataset suite. |
| `run_bthowen_sweep.py` | Full hyperparameter grid for BTHOWeN. |
| `run_bthowen_table3_grid.py` | Restricted Table III grid only. |
| `run_dwn_long_epochs.py` | DWN with the paper's 100-epoch multi-stage LR schedule. |
| `run_dwn_paper_fidelity.py` | DWN with the paper Table 5 hyperparameters. |
| `run_dwn_sweep.py` | Full hyperparameter grid for DWN. |
| `run_uleen_final.py` | Best ULEEN configuration found during sweeping. |
| `run_uleen_paper_lr.py` | ULEEN with `lr=1e-3, epochs=100` (paper schedule). |
| `run_uleen_sweep.py` | Full hyperparameter grid for ULEEN. |

`scripts/verify_against_papers.py` reads the sweep caches and prints a per-dataset best-cell comparison against the published paper numbers.

These scripts are **not** part of the installed package — they live under `scripts/` and depend on the F4RM dataset loaders in the parent `masters/` repo (`notebooks/notebook_lib.py`). They will not run in isolation against an arbitrary checkout of `wisardpkg/`. See `python-models.md` for the model API surface they exercise.
