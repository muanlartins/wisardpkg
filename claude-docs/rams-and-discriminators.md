# RAMs and Discriminators

The fundamental building blocks of WiSARD. RAMs store learned patterns as address-value pairs; Discriminators aggregate multiple RAMs to represent a single class or cluster.

## RAM (Random Access Memory Node)

**Source:** `src/models/wisard/ram.cc`

A RAM node is a lookup table that maps a subset of input bits to a vote count.

### How It Works

1. **Address computation**: The RAM receives a `BinInput` and a list of bit `addresses` (indices into the input). It computes an address using a base-N weighted sum:
   ```
   index = sum(input[addresses[i]] * base^i)  for i in 0..len(addresses)
   ```
   Default base is 2 (binary), so the address is the binary number formed by the selected bits.

2. **Training** (`train(BinInput)`): Increments the vote counter at the computed address:
   ```
   positions[index] += 1
   ```
   If `ignoreZero=true`, address 0 is skipped (prevents all-zero patterns from voting).

3. **Classification** (`getVote(BinInput)`): Returns the stored vote count at the computed address. Unseen addresses return 0.

4. **Untraining** (`untrain(BinInput)`): Decrements the vote counter (for leave-one-out evaluation).

### Storage

```cpp
map<addr_t, content_t> positions;       // sparse integer-keyed map (standard path)
map<large_addr_t, content_t> largePositions;  // sparse string-keyed map (large-address path)
vector<int> addresses;                   // which input bit indices this RAM reads
bool useLargeAddr;                       // switch between the two paths
```

**Standard path (`addressSize ≤ 64` with `base == 2`)**: the address is computed as a `uint64_t` via a base-N weighted sum. `positions` is a sparse `unordered_map<uint64_t, int>`.

**Large-address path (`addressSize > 64` with `base == 2`)**: `uint64_t` can no longer hold a single address, so the RAM switches to a byte-packed string key. Each address becomes a `std::string` of `⌈addressSize / 8⌉` bytes with one bit per input bit:

```cpp
large_addr_t getLargeIndex(const BinInput& image) const {
    int n = addresses.size();
    int nbytes = (n + 7) / 8;
    large_addr_t key(nbytes, '\0');
    for (int i = 0; i < n; i++) {
        if (image[addresses[i]])
            key[i / 8] |= (char)(1 << (i % 8));
    }
    return key;
}
```

`train`, `getVote`, `untrain`, `reset`, and `getsizeof` all dispatch on `useLargeAddr` and use `largePositions` when true. The mental-image reconstruction (`getMentalImage`) also has a large-address branch that decodes the per-byte bit pattern back into per-position vote accumulators.

JSON serialization and deserialization are only supported on the standard path; a RAM trained with `addressSize > 64` serializes to an empty RAM block (deliberate: the experiments that need large addresses don't round-trip through JSON). Mixing the two paths in one model is safe — `useLargeAddr` is per-RAM.

**Why this matters:** the original wisardpkg hard-capped addressSize at 64 via `checkLimitAddressSize`. That cap is dropped for `base == 2` specifically (other bases still enforce it), so experiments that need very large tuple sizes (e.g., full-image WiSARD on MNIST-scale inputs) can run without changing the model surface.

### Mental Image

`getMentalImage()` reconstructs which input positions contributed to stored votes. For each address in the positions map, it decomposes the address back into bit positions and accumulates counts. This shows which input bits the RAM "cares about."

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addresses` | `vector<int>` | (from mapping) | Input bit indices this RAM reads |
| `ignoreZero` | `bool` | false | Skip address 0 from voting |
| `base` | `int` | 2 | Numeral base for address computation |

---

## RegressionRAM

**Source:** `src/models/regressionwisard/regressionram.cc`

A regression variant where each address stores `[count, sum_y, fit_adjustment]` instead of a single vote count.

### Key Differences from RAM

- **Storage**: `unordered_map<addr_t, regression_content_t>` where `regression_content_t = vector<double>` of length 3: `[count, accumulated_y, fit_adjustment]`
- **Training**: `train(BinInput, y)` stores both the training count and the accumulated target value
- **Vote**: `getVote()` returns `{count, sum_y}` — the prediction is `sum_y / count` (computed by the mean function)
- **Refinement**: `calculateFit()` and `applyFit()` implement iterative correction:
  1. `calculateFit()`: Compute per-sample prediction errors
  2. `applyFit()`: Distribute corrections across RAM addresses

### Gating Parameters

- `minZero`: Minimum number of 0-bits in the address required to produce a vote. If the input has fewer zeros than this threshold, the RAM returns a zero vote.
- `minOne`: Same but for 1-bits.

These prevent predictions when the input is too sparse or too dense for the RAM's learned patterns.

---

## Discriminator

**Source:** `src/models/wisard/discriminator.cc`

A Discriminator groups multiple RAMs to represent one class (or cluster). Each RAM reads a different subset of the input bits.

### Structure

```cpp
vector<RAM> rams;     // the RAM nodes
int count;            // total training samples seen
int entrySize;        // input binary vector size
```

### RAM Initialization

Two methods for assigning input bits to RAMs:

**`setRAMShuffle(addressSize, ignoreZero, completeAddressing, base)`**:
1. Create index list `[0, 1, 2, ..., entrySize-1]`
2. If `completeAddressing=true` and `entrySize % addressSize != 0`: pad with random duplicate indices to make it evenly divisible
3. Shuffle all indices randomly
4. Partition into groups of `addressSize` — each group becomes one RAM's address set
5. Number of RAMs = `ceil(entrySize / addressSize)`

**`setRAMByMapping(mapping, ignoreZero, base)`**:
- Directly assigns pre-computed index groups to RAMs

### Classification

`classify(BinInput, totalTrained=0)` returns a vector of votes, one per RAM:
- Each RAM computes its vote via `getVote(input)`
- If `balanced=true`: each vote is scaled by `count / totalTrained` to normalize classes with different training set sizes

### Training

`train(BinInput)`: Trains every RAM with the input and increments `count`.

`untrain(BinInput)`: Untrains every RAM and decrements `count`.

### Mental Image

`getMentalImage()`: Aggregates mental images from all RAMs into a single vector of length `entrySize`, showing which input bits are most important for this class.

### Per-RAM Weights and Pruning

The `Wisard` model exposes a per-class, per-RAM weight vector that participates in `rank()` between the raw `Discriminator::classify` step and the bleaching/method aggregation. Weights are computed from a labelled dataset using one of three information-theoretic metrics, all normalised per-discriminator to `[0, 1]`:

| Metric | Definition |
|--------|------------|
| `"entropy"` (default) | `1 − H(class | RAM fires) / H_max`. RAMs whose firings concentrate in a single class get high weight; RAMs firing uniformly across classes get near-zero weight. |
| `"information_gain"` | Mutual information between binary RAM output (fires / doesn't) and the class label. |
| `"purity"` | Fraction of RAM firings that came from the correct class (the discriminator's own class). |

```python
import wisardpkg as wp

w = wp.Wisard(addressSize=4)
w.train(X, y)

# Compute and apply weights
w.computeRAMWeights(X, "information_gain")     # populates and activates ramWeights

# Inspect / round-trip
weights = w.getRAMWeights()                      # {label: [w_0, w_1, ..., w_{n_rams-1}]}
w.setRAMWeights(weights)                         # restore from a saved dict

# Prune low-weight RAMs by zeroing their weight (RAMs aren't deleted, just silenced)
w.pruneRAMs(0.1)                                 # any weight < 0.1 → 0
```

Once `computeRAMWeights` or `setRAMWeights` is called, the `useRAMWeights` flag is set internally and `rank()` will multiply each RAM's vote by its weight on the way through the hook pipeline (see `classification-models.md` for the hook order). To deactivate, call `setRAMWeights({})` or set all weights to 1.

`pruneRAMs(threshold)` sets every weight below `threshold` to 0; high-weight RAMs are unchanged. This is a soft prune — the underlying RAM storage is preserved, only the contribution to votes is suppressed.

**Source:** `src/models/wisard/wisard.cc:378–530` (computation, get/set, prune); `src/wrappers/wisardwrapper.cc` exposes the four entry points to Python.

### Serialization

`json()` outputs the full discriminator state including all RAM positions, entry size, count, and configuration. Supports optional `RAMDataHandle` for external RAM data storage.

### Python API

```python
disc = wp.Discriminator(
    addressSize=3,
    entrySize=9,
    ignoreZero=False,
    completeAddressing=True,
    base=2
)
disc.train(wp.BinInput([1,0,1,0,1,0,1,0,1]))
votes = disc.classify(wp.BinInput([1,0,1,0,1,0,1,0,1]))
# votes is a dict: {ram_index: vote_count}
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addressSize` | `int` | required | Bits per RAM (tuple size) |
| `entrySize` | `int` | required | Total input binary vector size |
| `ignoreZero` | `bool` | false | RAMs skip address 0 |
| `completeAddressing` | `bool` | true | Pad to fill all RAM slots |
| `base` | `int` | 2 | Numeral base for address computation |
| `indexes` | `vector<int>` | — | Custom input bit indices |
| `mapping` | `vector<vector<int>>` | — | Pre-computed RAM-to-bit mapping |

---

## RAMDataHandle

**Source:** `src/models/wisard/ramdatahandle.cc`

Handles serialization and persistence of classification RAM data.

### Storage Format

Base64-encoded binary blocks where each entry is `(addr_t address, content_t value)`. Multiple RAMs are separated by `"."` in the serialized string.

### Python API

```python
# Create from discriminator data
handle = wp.RAMDataHandle(discriminator_data_string)

# Access RAM data
ram_data = handle.get(ram_index)          # full RAM dict
value = handle.get(ram_index, address)    # single value

# Modify
handle.set(ram_index, address, new_value)

# Serialize
data_string = handle.data()              # all RAMs
ram_string = handle.data(ram_index)       # single RAM
```

---

## RegressionRAMDataHandle

**Source:** `src/models/regressionwisard/regressionramdatahandle.cc`

Same concept as RAMDataHandle but for regression RAMs. Each entry stores `(address, [count, sum_y, fit])` as 4 doubles (32 bytes per entry).

### Python API

```python
handle = wp.RegressionRAMDataHandle(data_string)
# or from raw dict: wp.RegressionRAMDataHandle({addr: [count, sum_y, fit], ...})

ram_dict = handle.get(ram_index)
handle.set(ram_index, {addr: [count, sum_y, fit], ...})
serialized = handle.data()
```
