# Performance Branch — Optimized Standard WiSARD

A standalone, single-class C++17 reimplementation of the **standard WiSARD only** (one discriminator per class, bleaching inference), built as an optimization proof-of-concept and benchmarked head-to-head against the library's `Wisard`.

This is **not a Python→C++ port** — the library is already C++ behind pybind11 (`src/wisard_bind.cc`). What this branch removes is the cost the library pays for being a general, serializable, polymorphic *zoo*: per-bit `BinInput::get` with bounds checks (`src/data/bininput.cc:26`), `std::unordered_map` vote storage (`src/common/definetypes.cc:7`), virtual dispatch through `ClassificationBase*` / `MappingGenerator*` (`src/models/wisard/wisard.cc:534`), and `std::map<std::string,…>` keyed on label strings in the inner loops (`src/models/wisard/wisard.cc:206`). The reimplementation is a cache- and SIMD-friendly reference freed of the pybind boundary and runtime polymorphism.

See sibling docs for what is being beaten: [rams-and-discriminators.md](rams-and-discriminators.md), [data-structures.md](data-structures.md), [mapping.md](mapping.md), [classification-methods.md](classification-methods.md), [architecture.md](architecture.md).

## What the optimized version is beating

| Hot-path cost in the library | Source | Why it is slow |
|---|---|---|
| Per-bit address build: `index += image[addresses[i]]*p; p*=base` | `ram.cc:208`–`218` | One `BinInput::operator[]` call per tuple bit; each does `index/8`, `7-(index%8)`, shift, mask, **plus an `index>=size()` bounds check that throws** (`bininput.cc:26`–`33`). `addressSize` dependent loads, no vectorization. |
| Vote storage = `unordered_map<addr_t,int>` per RAM | `definetypes.cc:7`, `ram.cc:235` | Pointer-chasing hash node per distinct address; cache-hostile; `find`/`insert` on every train and classify. |
| Per-RAM `getVote` virtual-free but `RAM` is heap `std::vector<RAM>` | `discriminator.cc:338` | Each `RAM` carries its own `std::vector<int> addresses`, two maps, flags — no contiguous per-discriminator address table. |
| `rank()` builds `std::map<std::string,std::vector<int>> allvotes` | `wisard.cc:206`–`217` | Heap vector + red-black-tree node per class per sample; string-keyed. |
| `classificationMethod->run(allvotes)` virtual call + re-scan | `wisard.cc:322`, `bleaching.cc:24` | Bleaching re-iterates the whole `allvotes` map each threshold round through a vtable. |
| Hook pipeline (`multiResolution`, `useRAMWeights`, `attention`, `crossClass`, `negativeEvidence`, `sharedDiscriminator`) | `wisard.cc:220`–`362` | Branches and extra passes on every sample even when disabled. The PoC drops all of these — standard WiSARD has none. |

Parity target for memory is the library's existing **`deployedSizeBytes()`** (`src/models/base/model.cc:10`, `ram.cc:196`–`199`): per RAM = `numDistinctSeenAddresses × ⌈tupleSize/8⌉` bytes, summed over RAMs and discriminators. The PoC reports the identical metric so memory is comparable (§Memory metric).

## Design

### Bit-packed input

The library stores one `std::string` of bytes per sample, MSB-first within each byte, and reads it one bit at a time with a bounds-checked accessor (`bininput.cc:26`). The PoC stores each sample as `uint64_t words[(entrySize+63)/64]`, all samples in one contiguous `std::vector<uint64_t>` (row-major: sample `s` occupies `words[s*W .. s*W+W)`). Bit order within a word is **LSB-first** (bit `i` of the sample → `words[i>>6] >> (i&63) & 1`); the bench file is written in this same convention so no per-bit reshuffle is needed at load.

```
PackedInputs { uint64_t* data; size_t W /*words per sample*/; size_t n; }
bit(s,i) = (data[s*W + (i>>6)] >> (i&63)) & 1ULL
```

### Branchless tuple-address extraction

The mapping fixes, per RAM `r`, the `addressSize` input-bit indices it reads (`bininput` indices `mapping[r][0..addressSize)`). The address is the integer whose bit `k` is `bit(s, mapping[r][k])` — i.e. the LSB-first base-2 weighted sum the library computes in `ram.cc:211`, but assembled without a hash, without bounds checks, and without a `*p` multiply chain:

```
addr_t a = 0;
for (k = 0; k < addressSize; ++k)
    a |= addr_t(bit(s, idx[r][k])) << k;
```

Two layout choices remove the per-bit branch and improve locality:

- **Address index table** `idx` is stored contiguously per discriminator as `addr_t* idx` of shape `[numRAMs][addressSize]` (one flat allocation), so the gather walks sequential memory.
- The `bit()` extraction is branchless (shift+mask, no `if`); the `<< k` accumulation is a single OR. For `addressSize ≤ 64` the address always fits one `addr_t` (the library's large-address string path, `ram.cc:220`, is **out of scope** — the PoC is the standard model).

An optional fast path precomputes, per RAM, a packed bit-gather when the selected indices are word-aligned, but the scalar gather above is the reference and is already branch-free.

### RAM storage — flat vs. open-addressing

Two storage backends, chosen per-RAM at construction from `addressSize`:

| Condition | Backend | Layout | Train | Classify |
|---|---|---|---|---|
| `2^addressSize` small (`addressSize ≤ DIRECT_BITS`, default 20 → ≤1 M entries) | **Direct array** | `std::vector<content_t>` of length `2^addressSize`, contiguous per RAM | `counts[a]++` | `counts[a]` |
| `addressSize > DIRECT_BITS` | **Open-addressing hash** | flat robin-hood / `flat_hash` of `addr_t → content_t`, single buffer, no node allocation | linear-probe insert/inc | linear-probe load |

Both store **per-address counts** (`content_t = uint32_t`), so bleaching has the same vote magnitudes as the library. The direct backend is the common MNIST case (`addressSize` 8–20) and turns train/classify into a single indexed load/store — no hashing, no probing. The hash backend mirrors the library's sparse map but with open addressing and one contiguous buffer instead of per-node allocation, and is only reached for large tuples.

Per-discriminator memory is contiguous: all RAM count-arrays (or hash buffers) for one class live in one arena, indexed by `ramOffset[r]`, so a classify pass over one discriminator streams linearly.

### No virtual calls in train/classify

The library reaches votes through `ClassificationBase*` (`wisard.cc:534`) and `MappingGenerator*` (`wisard.cc:535`). The PoC has exactly one model (standard WiSARD) and one inference rule (bleaching), so both are plain functions — no abstract bases, no `clone()`, no factory dispatch. `addressSize` is a runtime field but can be lifted to a template parameter (`template<unsigned A>`) for the direct-array path to let the compiler unroll the gather and fix the array stride; the bench instantiates the `A` actually requested and falls back to a runtime loop otherwise.

### Bleaching parity

The library's `Bleaching::run` (`src/classification_methods/bleaching.cc:24`–`48`) computes, per class, the **number of RAMs whose vote exceeds the current threshold**, then raises the threshold to the running minimum positive vote and repeats until no two classes are ambiguous within `confidence` (`classificationbase.cc:isThereAmbiguity`). The PoC reproduces this exactly:

1. Per sample, gather the per-RAM vote vector for every class (one `content_t` per RAM).
2. Score(class, t) = count of RAMs with vote `> t`.
3. Start `t = 0`; raise `t` to the smallest positive vote seen across all classes; repeat while ambiguous and `biggest > 1`.
4. Winner = argmax score (ties broken `>=`, matching `getBiggestCandidate`, `classificationbase.cc:9`).

The vote vectors are dense per-class arrays (length `numRAMs`), not a `std::map`, and the threshold loop scans those arrays directly. `confidence` defaults to 1 to match `Bleaching()` (`bleaching.cc:5`).

### Parallelism

Embarrassingly parallel along two axes; the bench picks one to keep the comparison clean:

- **Train**: parallelize over discriminators (one per class). Each class owns a disjoint arena → no contention, no atomics. `#pragma omp parallel for` over the class list, each thread trains its discriminator over all its samples.
- **Inference**: parallelize over **test samples** (`#pragma omp parallel for` over the test set). Each sample reads all discriminators (read-only after train) and runs bleaching independently into `labels[s]`. This is the axis the bench times for `infer_ms_total`.

OpenMP is optional (`find_package(OpenMP)`); without it the loops run serially and the bench still produces identical results, only slower. A `--threads` flag and `OMP_NUM_THREADS` honor the cap; the results JSON records the thread count used.

### Memory metric

`modelBytes()` mirrors `Wisard::deployedSizeBytes()` (`wisard.cc:511`) bit-for-bit so the two libraries report the **same yardstick**, independent of backend:

```
modelBytes = Σ_class Σ_ram  distinctSeenAddresses(ram) · ⌈addressSize/8⌉
```

For the **direct-array** backend `distinctSeenAddresses` = count of nonzero slots (a single pass, or maintained as a running counter on first-touch during train). For the **hash** backend it is the number of occupied slots. This deliberately reports the *deployable* footprint (the seen-address set, bit-packed), not the working `2^addressSize` array — identical definition to `ram.cc:196`, so memory numbers are directly comparable even though the PoC's *runtime* allocation for the direct backend is larger. Reporting both keeps the comparison honest: `model_bytes` (parity yardstick) and, optionally, `runtime_bytes` (actual allocation).

## Benchmark CLI — `wisard_bench`

Reads **pre-binarized, pre-mapped** inputs from a shared binary file so the comparison isolates the train/infer kernels (no thermometer, no RNG, no mapping cost inside the timed region — both libraries consume the same bytes and the same mapping). The Python lib's side of the comparison loads the identical file via a thin reader.

### Shared input file format (`*.wbin`)

A self-describing little-endian binary, written once by a Python exporter that runs the library's thermometer + mapping, then consumed by both benches:

| Field | Type | Notes |
|---|---|---|
| magic | `char[4]` = `WBIN` | format guard |
| version | `u32` | = 1 |
| entrySize | `u32` | bits per sample (post-binarization) |
| addressSize | `u32` | tuple size |
| numRAMs | `u32` | = number of address tuples |
| numClasses | `u32` | label cardinality |
| nTrain, nTest | `u64` each | sample counts |
| mapping | `u32[numRAMs][addressSize]` | input-bit index per RAM slot — **identical to the lib's mapping** |
| trainLabels | `u32[nTrain]` | class id per train sample |
| testLabels | `u32[nTest]` | class id per test sample |
| trainBits | `u64[nTrain][W]` | LSB-first packed, `W = ⌈entrySize/64⌉` |
| testBits | `u64[nTest][W]` | same packing |

The mapping is **carried in the file**, exported from the Python lib's `getMappingJson()` (`wisard.cc:164`), so both sides read the exact same RAM-to-bit assignment. No RNG is consulted at bench time.

### Run

```
wisard_bench --input mnist.wbin [--threads N] [--confidence 1] [--out results.json]
```

Steps, timed with `std::chrono::steady_clock`:
1. mmap/read the file (untimed).
2. Build discriminators (allocate arenas) — untimed setup.
3. **Train** all classes over `trainBits` → `train_ms`.
4. **Infer** (bleaching) over `testBits` → `infer_ms_total`; `infer_ms_per_sample = infer_ms_total / nTest`.
5. Score accuracy against `testLabels`.
6. Sample `peak_rss_bytes` via `getrusage(RUSAGE_SELF).ru_maxrss` (× 1024 on Linux, bytes on macOS — normalized).
7. Compute `model_bytes` (§Memory metric).

### Results JSON

```json
{
  "train_ms": 0.0,
  "infer_ms_total": 0.0,
  "infer_ms_per_sample": 0.0,
  "peak_rss_bytes": 0,
  "model_bytes": 0,
  "accuracy": 0.0,
  "addressSize": 0,
  "reps": 1,
  "threads": 1,
  "backend": "direct|hash"
}
```

`reps` re-runs train+infer and reports the median (warm-cache) timing; `peak_rss_bytes` and `model_bytes` are from the final rep.

## Parity requirements (fair benchmark)

For the head-to-head to measure *kernel* speed and not incidental differences, the following must be **identical** on both sides. Items are enforced by consuming the same `*.wbin` file, except where noted.

| Requirement | How |
|---|---|
| **Identical mapping** | Both consume `mapping` from the `*.wbin` file, exported from the lib's `getMappingJson()` (`wisard.cc:164`). No bench-time RNG. (Alternative: identical seeded `std::mt19937` + same `entrySize`/`addressSize`/`completeAddressing` — but file-carried mapping is the default to remove all doubt.) |
| **Identical thermometer / binarization** | Done once in the Python exporter using the lib's binarization; the PoC never binarizes. Bits in the file are final. |
| **Identical addressSize** | Single field in the file header; both build `numRAMs = ⌈entrySize/addressSize⌉` RAMs over the same tuples. |
| **Identical train/test split** | `trainBits`/`testBits` and their label arrays are fixed in the file; both sides train and test on the exact same rows in the exact same order. |
| **Identical completeAddressing / padding** | The exporter applies the lib's `completeMapping` padding (`randommapping.cc:137`) before writing tuples, so `numRAMs` and any duplicated bit indices match. |
| **Identical bleaching rule + confidence** | PoC reproduces `Bleaching::run` (`bleaching.cc:24`) with `confidence=1` (the lib default). Tie-break matches `getBiggestCandidate` (`classificationbase.cc:9`). |
| **Identical ignoreZero** | Standard WiSARD default is `false`; the PoC does not special-case address 0 unless the flag is set in the header. |
| **Same memory yardstick** | `model_bytes` uses the library's `deployedSizeBytes` definition (`model.cc:10`, `ram.cc:196`). |
| **Same accuracy** | A correctness gate: PoC accuracy must equal the lib's on the same file (within tie-break-order noise). A mismatch means a parity bug, not a win. |

## Optimization levers (summary)

1. Bit-packed `uint64_t` inputs, contiguous all-samples buffer; LSB-first to match address weighting with zero reshuffle.
2. Branchless shift+OR tuple-address gather over a flat per-discriminator index table; no `BinInput` bounds checks.
3. Direct count-array RAM backend for small `addressSize` (single indexed load/store); open-addressing hash only for large tuples.
4. Contiguous per-discriminator arena (all RAM tables in one allocation) for linear-streaming classify.
5. Zero virtual dispatch in train/classify; optional `template<unsigned A>` specialization to unroll the gather.
6. Dense per-class vote vectors + direct-scan bleaching instead of `std::map<std::string,vector<int>>` + vtable.
7. OpenMP over classes (train) and over test samples (infer); no atomics (disjoint arenas / read-only inference).
8. Drop the entire standard-WiSARD-irrelevant hook pipeline (`wisard.cc:220`–`362`).

## Scaffold (files the performance branch will contain)

| File | Purpose |
|---|---|
| `CMakeLists.txt` | C++17, `-O3 -march=native`, optional `find_package(OpenMP)`, builds `wisard_bench`. |
| `include/wisard.hpp` | Public API: `PackedInputs`, `Ram` backends, `Discriminator`, `Wisard` (train/classify/bleaching), `modelBytes()`. Header-only-friendly. |
| `src/wisard.cpp` | Implementation of train/classify kernels, address gather, bleaching, memory metric. |
| `src/io.cpp` | `*.wbin` reader/writer: header parse, mapping + label + packed-bit loaders. |
| `src/bench_main.cpp` | `wisard_bench` CLI: arg parse, timed train/infer loop, accuracy, RSS, results-JSON emit. |
| `README.md` | Build + run instructions, the `*.wbin` exporter command, parity checklist, how to reproduce the comparison against the pybind `Wisard`. |

The `*.wbin` exporter itself is a small Python script (lives next to the bench in the branch or under `scripts/`) that drives the existing library's thermometer + `getMappingJson()` and writes the file — it is the single source of parity and is intentionally outside the timed C++ region.
