# Classification Models

Two classification models: **Wisard** (standard supervised) and **ClusWisard** (clustering-based, supports supervised/semi-supervised/unsupervised).

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
