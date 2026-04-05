# Build System and Testing

## Build System

### setup.py (Primary Build)

The package uses setuptools with a custom `BuildExt` class for C++ compilation via PyBind11.

**Extension:** Single compilation unit `wisardpkg` from `src/wisard_bind.cc` (which includes `wisardpkg.h`, which includes everything else).

**Platform-specific flags:**
- **Unix/Linux:** C++11 detection (`-std=c++11` or `-std=c++0x`), `-fvisibility=hidden`
- **macOS (Darwin):** `-stdlib=libc++`, `-mmacosx-version-min=10.7`
- **MSVC (Windows):** `/EHsc` exception handling

**Dependencies:**
- `pybind11 >= 2.2` (build requirement)
- `setuptools` (build requirement)
- C++11-compatible compiler

**Install:**
```bash
pip install wisardpkg          # from PyPI
pip install .                   # from local source
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

| File | Tests | What It Covers |
|------|-------|----------------|
| `test_wisard.py` | 8 | Train, classify, score, json, mental images, leave-one-out, getsizeof |
| `test_discriminator.py` | 5 | Train, classify, json, custom mapping |
| `test_cluswisard.py` | 8 | Supervised/semi-supervised/unsupervised train+classify, json, mental images |
| `test_regressionwisard.py` | 4 | Train, predict, json, mean functions |
| `test_clusregressionwisard.py` | 4 | Train, predict, json |
| `test_bininput.py` | 6 | Constructors, indexing, serialization, extend |
| `test_dataset.py` | 14 | All constructor variants, add methods, save/load, labels/Ys |
| `test_kernel_canvas.py` | 2 | Transform with/without direction |
| `test_thermometer.py` | 4 | Simple and dynamic thermometer encoding |
| `test_thresholding.py` | 1 | Basic threshold binarization |
| `test_mean_thresholding.py` | 1 | Mean-based threshold binarization |
| `test_ramdatahandle.py` | 3 | Get, set, serialization |
| `test_regressionramdatahandle.py` | 3 | Get, set, serialization |
| `test_synthesizer.py` | 2 | Synthetic data generation from mental images |

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
