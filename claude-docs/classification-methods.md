# Classification Methods

Classification methods are pluggable strategies (Strategy pattern) that decide the final class label from the raw vote vectors produced by discriminators. All inherit from `ClassificationBase`.

**Compiled methods:** `Bleaching`, `BestBleaching`, `Weighted` (all three bound to Python and registered for JSON). `BBleaching` ships as source but is orphaned — never compiled, bound, or registered (see its section below).

**Source files:** `src/classification_methods/classificationbase.cc`, `bleaching.cc`, `bestbleaching.cc`, `weighted.cc`, `register.cc`. The orphaned `bbleaching.cc` is **not** `#include`'d by `src/wisardpkg.h`.

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

## BBleaching (Binary Bleaching) — NOT available

> **`BBleaching` cannot be used from Python or C++ in this build.** The source file exists but is never compiled, never bound to Python, and never registered for deserialization. Do not pass `wp.BBleaching(...)` as a `classificationMethod` — the symbol does not exist on the `wisardpkg` module. Use `BestBleaching` instead (it performs the full threshold scan that BBleaching was meant to accelerate).

**Source:** `src/classification_methods/bbleaching.cc` (orphaned — see below).

### Intended algorithm

The file describes a binary search over the bleaching threshold:

1. Find the maximum vote value across all classes/RAMs (`getBiggestValue`, bbleaching.cc:49)
2. Start with threshold = `biggest / 2` (bbleaching.cc:14–15)
3. If ambiguous, raise the threshold by the current step; otherwise lower it (bbleaching.cc:33–38)
4. Step size halves each iteration (`piece = biggest / 2^steps`, bbleaching.cc:30)
5. Stop when `piece == 0` (bbleaching.cc:31)

This would give O(log N) iterations instead of `BestBleaching`'s O(N) linear scan.

### Why it does not work

It is wired into nothing and would not even compile against the current `ClassificationBase` interface:

| Gap | Evidence |
|-----|----------|
| Not included in the compilation unit | `src/wisardpkg.h:17–23` includes `classificationbase`, `bleaching`, `bestbleaching`, `weighted`, `register` — **not** `bbleaching.cc`. The whole library is one translation unit (`.cc` files are `#include`'d), so an un-included file is dead. |
| Not bound to Python | `src/wisard_bind.cc:298–309` binds only `Bleaching`, `BestBleaching`, and `Weighted`. There is no `py::class_<BBleaching, ...>`. |
| Not in the JSON registry | `src/classification_methods/register.cc:16–25` dispatches only `"Bleaching"`, `"BestBleaching"`, `"Weighted"`; any other `className` (including `"BBleaching"`) falls through to `new Bleaching()` (register.cc:25). A serialized BBleaching would silently load back as plain Bleaching. |
| Signature incompatible with the base class | `ClassificationBase` (classificationbase.cc:2–7) requires `run` to return `std::map<std::string,int>` **by value**, plus pure-virtual `clone() const`, `className() const`, `json() const`. `BBleaching` returns `std::map<std::string,int>&` (a reference to a heap-leaked map, bbleaching.cc:8–41), implements **neither** `clone` nor `className`, and its `json()` (bbleaching.cc:44) has no `const` and no return statement. It also calls `isThereAmbiguity(*labels)` with one argument (bbleaching.cc:28) while the base declares it with two (classificationbase.cc:32). As written it is an abstract class and could not be instantiated even if it were included. |

To get binary-search-style threshold tuning today, use the Python `BTHOWeN` port (`wisardpkg/models/bthowen.py`), which does its own binary-search bleach tuning on top of `BloomWisard` — see [Python Models](python-models.md).

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

`ClassificationMethods::load(config)` dispatches on `className` to instantiate the right subclass (register.cc:16–25). Supported: `"Bleaching"`, `"BestBleaching"`, `"Weighted"`. Any other type — including `"BBleaching"` — falls through to `new Bleaching()` (register.cc:25), so it loads back as plain Bleaching, not as the named method.

---

## Comparison

Only the first three rows are usable. `BBleaching` is listed for completeness — it is not available in this build (see its section above).

| Method | Available | Strategy | Speed | Best For |
|--------|-----------|----------|-------|----------|
| Bleaching | yes | Linear threshold scan, stops at first clear winner | Fast (usually few iterations) | General use, default |
| BestBleaching | yes | Full threshold scan, picks best confidence | Slower (scans all levels) | When early stopping picks poorly |
| Weighted | yes | Weighted votes + bleaching | Same as Bleaching | Domain knowledge about feature importance |
| BBleaching | **no** | Binary search for threshold (intended) | — | — (orphaned source; use BestBleaching or the BTHOWeN port) |
