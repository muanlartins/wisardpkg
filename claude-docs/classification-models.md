# Classification Models

Classification models: **Wisard** (standard supervised), **ClusWisard** (clustering-based, supports supervised/semi-supervised/unsupervised), and **BloomWisard** (counting-Bloom-filter-backed RAMs for memory-efficient inference).

For Python-side ports of recent weightless architectures that build on top of these models — **BTHOWeN** (PACT 2022), **DWN** (ICML 2024), **ULEEN** (ACM TACO 2023) — see `python-models.md`.

## Wisard

**Source:** `src/models/wisard/wisard.cc`, `src/wrappers/wisardwrapper.cc`

The core WiSARD classifier. Maintains one Discriminator per class. Training stores patterns; classification aggregates votes across all discriminators and applies a classification method (bleaching).

### How It Works

**Training:**
1. For each `(BinInput, label)` pair in the dataset:
   - If no discriminator exists for `label`, create one with a random mapping (via `mappingGenerator`)
   - Train the discriminator (each of its RAMs stores the address from the input)

**Classification:**
1. `rank(BinInput)` — present the input to every discriminator, collect vote vectors
   - Each discriminator returns a vector of RAM votes
   - If `balanced=true`: votes are scaled by `discriminator.count / totalTrained`
2. `classificationMethod->run(allvotes)` — the pluggable strategy (e.g., Bleaching) picks the winning class

### Python API

```python
import wisardpkg as wp

# Create
wisard = wp.Wisard(
    addressSize=3,
    bleachingActivated=True,   # shorthand: sets Bleaching(True, confidence)
    confidence=1,
    ignoreZero=False,
    completeAddressing=True,
    base=2,
    balanced=False,
    verbose=False
)

# Or with explicit classification method
wisard = wp.Wisard(addressSize=3, classificationMethod=wp.BestBleaching())

# Or with custom mapping
wisard = wp.Wisard(addressSize=3, mappingGenerator=wp.RandomMapping(monoMapping=True))

# Train
X = [[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]]
y = ["cold", "hot"]
wisard.train(X, y)

# Classify
predictions = wisard.classify(X)  # ["cold", "hot"]

# Score (accuracy)
accuracy = wisard.score(X, y)  # 1.0

# Get activation rankings per class
ranks = wisard.rank(wp.BinInput([1,0,1,0,1,0,1,0,1]))
# {"cold": [vote1, vote2, vote3], "hot": [vote1, vote2, vote3]}

# Mental images (which bits matter per class)
images = wisard.getMentalImages()
# {"cold": [count_bit0, count_bit1, ...], "hot": [...]}

# Leave-one-out: untrain a specific sample, classify, retrain
wisard.leaveOneOut(input, label)
wisard.leaveMoreOut(inputs, labels)

# Serialization
json_str = wisard.json()
wisard2 = wp.Wisard(json_str)  # reconstruct from JSON

# Memory footprint
size = wisard.getsizeof()

# Get tuple sizes per discriminator's RAMs
tuple_sizes = wisard.getTupleSizes()
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addressSize` | `int` | required | Bits per RAM (tuple size) |
| `bleachingActivated` | `bool` | true | Enable bleaching (creates Bleaching method) |
| `confidence` | `int` | 1 | Bleaching confidence threshold |
| `classificationMethod` | `ClassificationBase*` | Bleaching | Pluggable classification strategy |
| `ignoreZero` | `bool` | false | RAMs skip address 0 |
| `completeAddressing` | `bool` | true | Pad mappings to fill RAM slots |
| `base` | `int` | 2 | Numeral base for RAM addressing |
| `balanced` | `bool` | false | Weight votes by class training frequency |
| `verbose` | `bool` | false | Print training/classification info |
| `monoMapping` | `bool` | false | All classes share the same bit-to-RAM mapping |
| `mappingGenerator` | `MappingGenerator*` | RandomMapping | Custom mapping generator |
| `indexes` | `vector<int>` | — | Custom input bit indices |
| `mapping` | `map<str, vector<vector<int>>>` | — | Pre-computed per-class mappings |

### Optional Classification Hooks

These kwargs modify how per-class scores are computed at `classify()`/`rank()` time. All default off; existing callers are unaffected. They can be combined, and their order of application inside `wisard.cc::rank()` is:

1. base RAM votes per class (`Discriminator::classify`)
2. **multi-resolution reweight** — multiply each RAM's vote by its tuple size (only when `mappingGenerator->multiResolution == true`)
3. **`useRAMWeights`** — multiply each RAM's vote by its learned per-RAM weight
4. **`attentionWeighting`** — multiply each RAM's vote by `1 − |vote − mean| / max`
5. **`crossClassScoring`** — normalise each RAM's vote by the cross-class total
6. **`softBleaching`** vs `classificationMethod->run()` — aggregate per-RAM votes into per-class scalars
7. **`sharedDiscriminator`** — subtract the `__shared__` discriminator's response (scaled by `sharedBeta`)
8. **`negativeEvidence`** — penalise each class by some function of competing classes' scores

| Kwarg | Type | Default | Effect |
|-------|------|---------|--------|
| `negativeEvidence` | `bool` | false | Penalize each class's score by others' activations |
| `negativeAlpha` | `float` | 0.0 | Penalty strength (0 = identity) |
| `negativeMode` | `str` | `"uniform"` | One of `"uniform"`, `"max_competitor"`, `"normalized"` |
| `sharedDiscriminator` | `bool` | false | Train a `__shared__` discriminator on all samples; subtract its response from every class score |
| `sharedBeta` | `float` | 0.0 | Subtraction weight for the shared response |
| `attentionWeighting` | `bool` | false | Down-weight per-RAM votes that disagree with the sample's own vote consensus |
| `softBleaching` | `bool` | false | Replace `classificationMethod` with an iterative soft-bleaching loop that progressively lowers the threshold until ambiguity resolves |
| `crossClassScoring` | `bool` | false | Per-RAM index `j`: rescale each class's vote by `n_classes / (1 + total_j)` so RAMs that fire for many classes contribute less |
| `useRAMWeights` | (implicit) | false | Multiply per-RAM votes by a learned per-RAM weight before aggregation. Enabled by calling `computeRAMWeights(dataset, metric)` or `setRAMWeights(dict)` |

Negative-evidence modes:

- `"uniform"`: `score_c' = score_c - α · Σ_{c' ≠ c} score_{c'}` — every other class contributes equally.
- `"max_competitor"`: `score_c' = score_c - α · max_{c' ≠ c} score_{c'}` — only the closest rival penalizes.
- `"normalized"`: `score_c' = 1000 · score_c / (1 + α · Σ_{c' ≠ c} score_{c'})` — multiplicative normalization; integer-safe.

RAM-weight metrics (via `computeRAMWeights(dataset, metric)`):

- `"entropy"`: weight = `1 − H(class | RAM fires) / H_max`. RAMs whose firings concentrate in a single class get high weight; RAMs firing uniformly get near-zero weight.
- `"information_gain"`: mutual information between binary RAM output (fires / doesn't) and class label.
- `"purity"`: fraction of RAM firings that came from the correct class (the discriminator's own class).

All metrics are normalized per-discriminator to `[0, 1]`. `pruneRAMs(threshold)` zeros every weight below the threshold; `getRAMWeights()` / `setRAMWeights(dict)` round-trip the weight map.

```python
# Negative evidence (three modes)
w = wp.Wisard(3, negativeEvidence=True, negativeAlpha=0.3,
              negativeMode="max_competitor")

# RAM weighting
w = wp.Wisard(3)
w.train(X)
w.computeRAMWeights(X, "information_gain")   # populates and activates weights
w.pruneRAMs(0.1)                              # zero low-MI RAMs

# Shared discriminator
w = wp.Wisard(3, sharedDiscriminator=True, sharedBeta=0.1)

# Attention weighting
w = wp.Wisard(3, attentionWeighting=True)
```

**Intended use.** These hooks are experiment-oriented — they let you reproduce the techniques from `TechniquesExploration.ipynb` (negative evidence, weighted RAM voting, shared background discriminator, attention-like weighting). For the F4RM paper's canonical experiments, all hooks are disabled.

---

## ClusWisard

**Source:** `src/models/cluswisard/cluswisard.cc`, `src/models/cluswisard/cluster.cc`, `src/wrappers/cluswisardwrapper.cc`

A clustering extension of WiSARD that creates multiple discriminators per class (or in unsupervised mode, discovers clusters automatically). Each class has a `Cluster` that adaptively creates new discriminators when existing ones don't match well enough.

### How Clustering Works

Each `Cluster` holds multiple discriminators. When a new sample arrives:

1. Compute a similarity score against each discriminator:
   ```
   score = sum(votes) / (max(votes) * num_rams)
   ```
   This ranges 0–1; high when votes are consistent across RAMs.

2. Compute the adaptive threshold:
   ```
   limit = min(1.0, minScore + count / threshold)
   ```
   This rises as more samples are seen, becoming more selective over time.

3. If any discriminator scores >= `limit`: train that discriminator
4. If no match AND discriminators < `discriminatorsLimit`: create a new discriminator
5. Otherwise: train the best-scoring discriminator (forced assignment)

### Learning Modes

**Supervised** (`train(X, y)`): Each class label gets its own Cluster. Labeled samples train the corresponding cluster.

**Semi-supervised** (`train(X, y_dict)`): `y_dict` is a dict mapping indices to labels. Labeled samples train class clusters; unlabeled samples train the shared `unsupervisedCluster`.

**Unsupervised** (`trainUnsupervised(X)`): All samples train a single shared cluster. Classes are discovered as cluster indices.

### Python API

```python
clus = wp.ClusWisard(
    addressSize=3,
    minScore=0.1,
    threshold=10,
    discriminatorsLimit=5,
    classificationMethod=wp.Bleaching(True, 1),
    verbose=False,
    ignoreZero=False,
    completeAddressing=True,
    base=2
)

# Supervised
clus.train(X, y)
predictions = clus.classify(X)

# Semi-supervised (dict labels: {index: label})
clus.train(X, {0: "cold", 3: "hot"})

# Unsupervised
clus.trainUnsupervised(X)
cluster_ids = clus.classifyUnsupervised(X)

# Score
accuracy = clus.score(X, y)

# Rankings
ranks = clus.rank(X)                    # supervised
ranks = clus.rank(wp.BinInput([...]))   # single sample
ranks = clus.rankUnsupervised(X)        # unsupervised

# Mental images
images = clus.getMentalImages()          # all clusters
image = clus.getMentalImage("cold")      # single class

# Detailed scores (debugging)
scores = clus.getAllScores(X)

# Adjust parameters after creation
clus.setMinScore(0.2)
clus.setThreshold(20)
clus.setDiscriminatorsLimit(10)

# Serialization
json_str = clus.json()
clus2 = wp.ClusWisard(json_str)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addressSize` | `int` | required | Bits per RAM |
| `minScore` | `float` | required | Baseline cluster acceptance threshold |
| `threshold` | `int` | required | Controls adaptive threshold growth rate |
| `discriminatorsLimit` | `int` | required | Max discriminators per cluster |
| `classificationMethod` | `ClassificationBase*` | Bleaching | Vote aggregation strategy |
| `verbose` | `bool` | false | Print info |
| `ignoreZero` | `bool` | false | RAMs skip address 0 |
| `completeAddressing` | `bool` | true | Pad mappings |
| `base` | `int` | 2 | Numeral base |

### Cluster Score Semantics

The score `sum(votes) / (max(votes) * num_rams)` measures how uniformly the input activates the discriminator's RAMs:
- **1.0**: Every RAM voted the maximum — perfect match
- **0.0**: No RAM responded — completely unknown pattern
- Intermediate values indicate partial recognition

The adaptive limit `minScore + count/threshold` starts low (accepting most patterns into existing discriminators) and grows over time (becoming more selective, forcing creation of new discriminators for novel patterns). Capped at 1.0.

### Label Format

In supervised classification, ClusWisard labels output as `"class::cluster_index"` internally but returns the class portion to the user. In unsupervised mode, labels are plain cluster indices.

---

## BloomWisard

**Source:** `src/models/bloomwisard/{bloomfilter,bloomram,bloomdiscriminator,bloomwisard}.cc`, `src/wrappers/bloomwisardwrapper.cc`

A WiSARD variant where each RAM is replaced by a **counting Bloom filter**. Instead of storing one counter per seen address, each address is hashed into `numHashes` positions in a fixed-size `numBits`-slot counter array. Training increments all hashed slots; classification queries the minimum across them (the counting-Bloom-filter approximation of "how many times has this address been seen").

The result: bounded memory per RAM (`numBits` slots regardless of how many distinct addresses are seen) at the cost of a controlled false-positive rate, while still supporting bleaching. BloomWisard is the standalone implementation of the Bloom-filter RAM backend used by ULEEN and related ultra-low-energy edge variants; here it is exposed directly so it can be swapped in wherever a `Wisard` would be used.

### How It Works

Each RAM holds a `BloomFilter(numBits, numHashes, hashMode)`:
- **`add(key)`**: compute `numHashes` positions from `key`, increment each slot.
- **`count(key)`**: return the minimum across the hashed slots — a lower bound on the number of times `key` was added.
- **`query(key)`**: equivalent to `count(key) > 0`.

At classification time, the discriminator's vote for a class is the sum of `count(address)` over all its RAMs, which then flows through the standard classification method (bleaching by default).

### Hash Modes

| `hashMode` | Implementation |
|-----------|----------------|
| `"murmur"` (default) | MurmurHash3 double-hashing: `h_i(x) = h1(x) + i · h2(x) mod numBits` |
| `"simhash"` | LSH via random hyperplane projections — similar binary inputs map to similar hash positions |
| `"h3"` | H3 universal hashing (Carter & Wegman, 1979): for each hash `i`, XOR the precomputed random constants `H[i, j]` over the bits where `key[j] == 1`, then reduce mod `numBits`. This is the hash family used by BTHOWeN. |

Use `"murmur"` for standard Bloom behavior; use `"simhash"` when you want similar patterns (in Hamming distance) to alias intentionally; use `"h3"` to reproduce BTHOWeN-style training (the `wisardpkg.models.BTHOWeN` wrapper sets this for you). H3 constants are initialised per-RAM via `initH3(keyLength)` once `addressSize` (and thus the per-RAM key length) is known; this happens automatically when training begins.

### Python API

```python
import wisardpkg as wp

bw = wp.BloomWisard(addressSize=3, numBits=1024, numHashes=3)
bw.train(X)
preds = bw.classify(X)
scores = bw.rank(X[0])  # {"cold": int, "hot": int, ...}
bw.reset()              # clear all Bloom filters in place

# SimHash LSH variant
bw = wp.BloomWisard(addressSize=3, numBits=1024, numHashes=3, hashMode="simhash")

# H3 universal hashing (BTHOWeN-style)
bw = wp.BloomWisard(addressSize=8, numBits=1024, numHashes=3, hashMode="h3")
bw.train(ds)
raw = bw.getRawVotes(some_bin_input)  # {label: [min_count_per_RAM, ...]}
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addressSize` | `int` | required | Bits per RAM |
| `numBits` | `int` | 1024 | Bloom-filter counter array size |
| `numHashes` | `int` | 3 | Number of hash functions per key |
| `hashMode` | `str` | `"murmur"` | `"murmur"`, `"simhash"`, or `"h3"` |
| `classificationMethod` | `ClassificationBase*` | Bleaching | Same plug-in model as `Wisard` |
| `ignoreZero` | `bool` | false | Skip the all-zero address |
| `completeAddressing` | `bool` | true | Pad mappings |
| `monoMapping` | `bool` | false | All classes share the same mapping |
| `verbose` | `bool` | false | Print info |

### Inspecting raw votes (`getRawVotes`)

`rank()` returns one aggregated integer per class — the standard Bloom-WiSARD vote summed over all RAMs. `getRawVotes()` returns the **per-RAM Bloom min-counts** before any threshold is applied, so external code can implement custom bleaching strategies on top of it:

```cpp
std::map<std::string, std::vector<int>>             getRawVotes(const BinInput&);
std::vector<std::map<std::string, std::vector<int>>> getRawVotes(const DataSet&);
```

```python
raw = bw.getRawVotes(bin_input)
# {"cold": [min_count_RAM0, min_count_RAM1, ...], "hot": [...]}

# Apply a custom bleach β: vote = number of RAMs whose min-count ≥ β
def vote(raw, label, beta):
    return sum(1 for c in raw[label] if c >= beta)
```

This is what `wisardpkg.models.BTHOWeN` uses to drive its binary-search bleach tuning loop — see `python-models.md`.

### When to Use BloomWisard vs Wisard

| Situation | Use |
|-----------|-----|
| Small address sizes, plenty of memory | `Wisard` (exact counts) |
| Large address sizes where `2^a` is unreasonable | `BloomWisard` (fixed memory, tunable FP rate) |
| Memory-constrained deployment (edge, MCU) | `BloomWisard` |
| Need similarity-based aliasing of near-patterns | `BloomWisard` with `hashMode="simhash"` |
| Reproducing paper results that assume exact counts | `Wisard` |

### Trade-off

BloomWisard's false-positive rate rises with fill ratio. Rough guide: for target FP rate `p`, pick `numBits ≈ -n · ln(p) / (ln 2)^2` where `n` is the expected number of distinct addresses per RAM. `numHashes ≈ (numBits / n) · ln 2` minimizes FPR at that capacity. The defaults (`numBits=1024, numHashes=3`) are a reasonable starting point for moderate datasets.
