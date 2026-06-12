# performance — optimized standard WiSARD (PoC)

Standalone C++17 reimplementation of the **standard WiSARD only** (one
discriminator per class, bleaching inference), built as an optimization
proof-of-concept to benchmark head-to-head against the pybind11 library's
`Wisard`. No ClusWisard / Bloom / Regression / hooks. See
`../claude-docs/performance-design.md` for the full rationale.

## Build

```bash
cd performance
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

C++17, `-O3 -march=native`. **OpenMP is optional**: `find_package(OpenMP)`
is used; if found it is linked and `WP_OPENMP` is defined, otherwise the
build still succeeds and runs single-threaded. On macOS Apple-clang the
CMake script auto-points `OpenMP_ROOT` at Homebrew `libomp`
(`/opt/homebrew/opt/libomp` or `/usr/local/opt/libomp`); override with
`-DOpenMP_ROOT=...` if libomp lives elsewhere, or pass
`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` to force single-threaded.

## Run

```bash
# Self-test: fabricates a 3-class linearly-separable dataset in memory,
# trains + classifies, asserts accuracy > 0.90. No external data needed.
./build/wisard_bench --selftest [--threads N]

# Benchmark a shared *.wbin file (see format below):
./build/wisard_bench --input mnist.wbin [--threads N] [--reps R] [--out result.json]
```

`--reps R` re-runs train+infer and reports the **median** train/infer time;
accuracy / model_bytes / peak_rss are from the final rep. `--threads N`
caps the OpenMP thread count (ignored without OpenMP; the JSON records the
thread count actually used).

### Results JSON

```json
{
  "train_ms": 0.0,
  "infer_ms_total": 0.0,
  "infer_ms_per_sample": 0.0,
  "throughput_sps": 0.0,
  "peak_rss_bytes": 0,
  "model_bytes": 0,
  "accuracy": 0.0,
  "address_size": 0,
  "num_rams": 0,
  "reps": 1,
  "threads": 1
}
```

`peak_rss_bytes` is `getrusage(RUSAGE_SELF).ru_maxrss`, normalized to bytes
(bytes on macOS, `×1024` on Linux). `model_bytes` is the deployed-footprint
yardstick (see Parity).

## `*.wbin` byte layout

Self-describing, **little-endian** (the format guard is the `WBIN` magic;
the bench is x86-64 / arm64 little-endian only — the Python exporter must
write LE). All multi-byte integers are stored in native LE order.

| Offset order | Field | Type | Notes |
|---|---|---|---|
| 1 | magic | `char[4]` = `WBIN` | format guard |
| 2 | version | `u32` = 1 | |
| 3 | entrySize | `u32` | bits per sample (post-binarization) |
| 4 | addressSize | `u32` | tuple size |
| 5 | numRAMs | `u32` | number of address tuples = `ceil(entrySize/addressSize)` (after completeAddressing padding) |
| 6 | numClasses | `u32` | label cardinality |
| 7 | nTrain | `u64` | train sample count |
| 8 | nTest | `u64` | test sample count |
| 9 | mapping | `u32[numRAMs * addressSize]` | row-major `[r][k]` input-bit index per RAM slot — identical to the lib's mapping |
| 10 | trainLabels | `u32[nTrain]` | class id per train sample |
| 11 | testLabels | `u32[nTest]` | class id per test sample |
| 12 | trainBits | `u64[nTrain * W]` | LSB-first packed, `W = ceil(entrySize/64)`, row-major per sample |
| 13 | testBits | `u64[nTest * W]` | same packing |

Bit `i` of sample `s` lives at `words[s*W + (i>>6)] >> (i&63) & 1` (LSB-first
within each `u64`). This matches the address weighting (`a |= bit << k`) so
no per-bit reshuffle happens at load.

Notes on the layout (where the design doc left it unpinned):
- The header carries `entrySize`/`addressSize`/`numRAMs`/`numClasses` as the
  fixed-order block above; `nTrain`/`nTest` are `u64` and follow the four
  `u32` descriptors. Class ids are dense `u32` in `[0, numClasses)`.
- `numRAMs` is carried explicitly (not recomputed) so any
  completeAddressing padding the exporter applied is honored verbatim.
- The mapping is stored flat row-major (`[numRAMs][addressSize]`), the same
  shape the kernel indexes with `idx + r*addressSize`.

`src/io.cpp` ships a minimal writer (used only by an internal round-trip
check) and the reader the bench consumes.

## Producing a `*.wbin` from the Python library

The exporter is **outside** this C++ tree (it is the single source of
parity and runs the library's thermometer + mapping once, off the timed
path). A Python script that:

1. Loads the dataset, binarizes train/test with the library's chosen
   thermometer (e.g. `DistributiveThermometer`) → `entrySize` bits/sample.
2. Builds a `wp.Wisard(addressSize, ...)`, trains once to materialize the
   mapping, reads `getMappingJson()` (`src/models/wisard/wisard.cc:164`) to
   recover the per-RAM bit-index tuples; flattens them row-major.
3. Computes `numRAMs = len(mapping)`, maps labels to dense `u32` class ids.
4. Repacks each binarized sample **LSB-first** into `u64` words
   (`W = ceil(entrySize/64)`), MSB-first→LSB-first transpose if needed.
5. Writes the header + mapping + label arrays + packed bit blocks in the
   exact field order above (little-endian).

Both this PoC and a thin pybind reader then consume the identical file, so
mapping, binarization, split, and `addressSize` are byte-identical on both
sides and only the train/infer kernels differ.

## Parity checklist (vs. the pybind `Wisard`)

- **numRAMs** = `ceil(entrySize/addressSize)` with completeAddressing
  padding (carried in the file header, not recomputed).
- **Address build**: LSB-first base-2, `a |= bit(s, idx[r][k]) << k` — the
  library's `index += bit*p; p*=2` (`ram.cc:211`) with no hash, no bounds
  check, no multiply chain.
- **Vote storage**: per-address counts (`content_t = uint32_t`), so
  bleaching vote magnitudes match the library. Direct count-array when
  `addressSize ≤ 20`, `unordered_map` fallback above.
- **Bleaching** reproduces `Bleaching::run` (`bleaching.cc:24`):
  `score(class) = #RAMs with vote > bleaching`; raise `bleaching` to the
  smallest positive vote seen this round; loop while ambiguous and
  `biggest > 1`. `confidence = 1` (the lib default) — ambiguity when a
  non-max class is within `<1` of the max.
- **Tie-break** matches `getBiggestCandidate` (`classificationbase.cc:9`):
  `>=` scan, so the **last** class in iteration order wins ties (here:
  highest class id; see Deviations).
- **ignoreZero** = `false` (standard default); address 0 is not special-cased.
- **modelBytes** mirrors `deployedSizeBytes()` (`ram.cc:196`):
  `Σ_class Σ_ram distinctSeenAddresses(ram) · ceil(addressSize/8)`. For the
  direct backend `distinctSeenAddresses` is the first-touch counter; for the
  hash backend it is the occupied-slot count.
- **Parallelism**: train over classes (disjoint arenas, no atomics), infer
  over test samples (read-only). OpenMP optional; results are identical
  serial vs. parallel.

## Deviations from `performance-design.md`

- The doc's JSON sketch uses `addressSize`/`backend` keys; the orchestrator
  spec asked for `address_size`/`num_rams`/`throughput_sps`. This build
  emits the orchestrator's snake_case key set (`address_size`, `num_rams`,
  `throughput_sps`) and omits the `backend` string. Backend is still chosen
  per-RAM internally (`addressSize ≤ 20` → direct).
- Class iteration order: the library keys discriminators on **label
  strings** in a `std::map`, so its tie-break (`>=`, last-wins) follows
  sorted-string order. This PoC consumes dense `u32` class ids and iterates
  in **numeric id order**; ties go to the highest id. To be byte-identical
  with the library's string ordering, the exporter must assign class ids in
  the library's sorted-label order (document this in the exporter). Affects
  only tie-break order on score ties, not accuracy on separable data.
- The `template<unsigned A>` gather specialization mentioned in the design
  is left as a runtime loop (`addressSize` is a runtime field). The gather
  is already branch-free; the template unroll is a future lever.
