# Classification Methods

Classification methods are pluggable strategies (Strategy pattern) that decide the final class label from the raw vote vectors produced by discriminators. All inherit from `ClassificationBase`.

**Source files:** `src/classification_methods/classificationbase.cc`, `bleaching.cc`, `bestbleaching.cc`, `bbleaching.cc`, `weighted.cc`, `register.cc`

## Base Class: ClassificationBase

```cpp
class ClassificationBase {
    virtual string run(map<string, vector<int>>& allvotes) = 0;
    virtual ClassificationBase* clone() = 0;
    virtual string className() = 0;
    virtual string json() = 0;
protected:
    string getBiggestCandidate(map<string, int>& candidates);
    int getConfidence(map<string, int>& candidates, int biggest);
    static tuple<bool, int> isThereAmbiguity(map<string, int>& candidates, int confidence);
};
```

**Input**: `allvotes` — a map from class label to a vector of RAM votes (one vote per RAM in that class's discriminator).

**Ambiguity detection**: Compares the top two candidates' total votes. If the gap is smaller than `confidence`, there is ambiguity.

**Confidence formula**: `(biggest - secondBiggest) / biggest` (not used directly — the integer `confidence` threshold is compared against the vote gap).

---

## Bleaching

The standard WiSARD classification method. Iteratively raises a vote threshold to disambiguate tied classes.

**Source:** `src/classification_methods/bleaching.cc`

### Algorithm

1. Start with threshold = 0
2. For each class, count how many RAMs have votes **above** the current threshold
3. Find the class with the highest count
4. Check for ambiguity (is the gap between top two classes < `confidence`?)
5. If ambiguous and `bleachingActivated=true`: raise threshold to the minimum vote seen in this round, go to step 2
6. If unambiguous or bleaching disabled: return the winning class

### Intuition

Bleaching progressively filters out "weak" votes (low counts), keeping only the RAMs that have strong recognition of the input. Classes that match well on many RAMs will survive higher thresholds.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bleachingActivated` | `bool` | true | Enable iterative threshold raising |
| `confidence` | `int` | 1 | Minimum vote gap to declare a winner |

### Python Usage

```python
wisard = wp.Wisard(addressSize=3, classificationMethod=wp.Bleaching(True, 1))
```

---

## BestBleaching

Like Bleaching, but doesn't stop at the first unambiguous result — it tries all threshold levels and returns the one with the **best confidence**.

**Source:** `src/classification_methods/bestbleaching.cc`

### Algorithm

1. Iterate through all bleaching levels (increasing threshold until no votes remain)
2. At each level, compute the winning class and its confidence margin
3. Track which level produced the maximum confidence without ambiguity
4. Return the winner from the best level (or fall back to the last iteration)

### When to Use

When standard Bleaching picks a winner too early at a low threshold with poor confidence. BestBleaching explores the full range and picks the most decisive point.

### Parameters

None — no constructor arguments.

### Python Usage

```python
wisard = wp.Wisard(addressSize=3, classificationMethod=wp.BestBleaching())
```

---

## BBleaching (Binary Bleaching)

Uses binary search instead of linear iteration to find the optimal threshold.

**Source:** `src/classification_methods/bbleaching.cc`

### Algorithm

1. Find the maximum vote value across all classes/RAMs
2. Start with threshold = `maxVote / 2`
3. Binary search: if ambiguous, increase threshold; if unambiguous, decrease
4. Step size halves each iteration (power-of-2 increments)
5. Stop when step size reaches 0

### Advantage

Faster convergence for large vote ranges — O(log N) iterations instead of O(N).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bleachingActivated` | `bool` | true | Enable binary search bleaching |

---

## Weighted

Applies per-class, per-RAM weights to the votes before the bleaching decision.

**Source:** `src/classification_methods/weighted.cc`

### Algorithm

Same iterative bleaching as `Bleaching`, but instead of counting RAMs above threshold, it sums `weights[class][ram_index]` for RAMs above threshold.

This allows certain RAMs or certain classes to have more influence on the final decision.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `map<string, vector<double>>` | required | Weight per class per RAM |
| `bleachingActivated` | `bool` | true | Enable iterative bleaching |
| `confidence` | `int` | 1 | Minimum weighted gap |

### Python Usage

```python
weights = {"cold": [1.0, 1.5, 0.8], "hot": [0.8, 1.0, 1.2]}
wisard = wp.Wisard(addressSize=3, classificationMethod=wp.Weighted(weights))

# Update weights later
wisard.classificationMethod.setWeights(new_weights)
```

---

## Registry (Serialization)

**Source:** `src/classification_methods/register.cc`

`ClassificationMethods` is a static factory class for JSON serialization/deserialization:

```json
{
  "className": "Bleaching",
  "params": {"bleachingActivated": true, "confidence": 1}
}
```

`ClassificationMethods::load(config)` dispatches on `className` to instantiate the right subclass. Supported: `"Bleaching"`, `"BestBleaching"`, `"Weighted"`. Unknown types default to `Bleaching()`.

---

## Comparison

| Method | Strategy | Speed | Best For |
|--------|----------|-------|----------|
| Bleaching | Linear threshold scan, stops at first clear winner | Fast (usually few iterations) | General use, default |
| BestBleaching | Full threshold scan, picks best confidence | Slower (scans all levels) | When early stopping picks poorly |
| BBleaching | Binary search for threshold | Fast (log N iterations) | Large vote ranges |
| Weighted | Weighted votes + bleaching | Same as Bleaching | Domain knowledge about feature importance |
