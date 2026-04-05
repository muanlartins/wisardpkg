# Regression Models

Two regression models: **RegressionWisard** (standard) and **ClusRegressionWisard** (clustering-based). Both predict continuous values using `RegressionRAM` nodes and pluggable `Mean` aggregation functions.

## RegressionWisard

**Source:** `src/models/regressionwisard/regressionwisard.cc`, `src/wrappers/regressionwisardwrapper.cc`

A single-model regression variant. Instead of one discriminator per class, it has a flat list of `RegressionRAM` nodes that collectively predict a continuous value.

### How It Works

**Training:**
1. On first `train()` call, creates RAMs via `setRAMShuffle()` (same mapping logic as Discriminator)
2. Each RAM receives the full input and its target value `y`
3. Each RAM stores `[count, accumulated_y, fit_adjustment]` at the computed address
4. If `steps > 0`: runs refinement iterations:
   - For each training sample: compute the per-RAM prediction error
   - Distribute corrections via `applyFit()` across all RAMs

**Prediction:**
1. Each RAM computes its address from the input and returns `{count, sum_y}`
2. The pluggable `Mean` function aggregates all RAM outputs into a single prediction
3. Default (`SimpleMean`): `prediction = sum(all_sum_y) / sum(all_counts)`

### Mean Functions

All inherit from `Mean` base class and implement `double calculate(vector<vector<double>>& votes)`.

| Function | Formula | Constructor | Notes |
|----------|---------|-------------|-------|
| `SimpleMean` | `sum(y) / sum(count)` | `()` | Default. Weighted average across RAMs |
| `PowerMean` | `(sum(y^p) / sum(count))^(1/p)` | `(int p)` | Generalized mean; p=1 is arithmetic, p=2 is quadratic |
| `Median` | Middle value of per-RAM means | `()` | Robust to outlier RAMs |
| `HarmonicMean` | `N / sum(1/y)` | `()` | Biased toward smaller values |
| `HarmonicPowerMean` | Harmonic mean of p-th powers | `(int p)` | Combines harmonic and power |
| `GeometricMean` | `(product(y))^(1/N)` | `()` | Multiplicative averaging |
| `ExponentialMean` | `log(mean(exp(y)))` | `()` | Sensitive to large values |
| `LogisticMean` | `gamma / (1 + exp(-(beta1 * mean + beta2)))` | `(double beta1, double beta2, double gamma)` | Sigmoid-shaped output |

### Refinement (steps parameter)

When `steps > 0`, the model performs iterative correction after initial training:
1. For each training sample, predict with current RAM state
2. Compute per-RAM error (difference between RAM's contribution and the target)
3. `calculateFit()`: Store the error in the RAM's fit_adjustment slot
4. `applyFit()`: Distribute smoothed corrections back to RAM addresses
5. Repeat for `steps` iterations

This is analogous to boosting — each iteration reduces the residual error.

### Python API

```python
import wisardpkg as wp

# Create
rw = wp.RegressionWisard(
    addressSize=3,
    orderedMapping=False,
    completeAddressing=True,
    ignoreZero=False,
    minZero=0,
    minOne=0,
    mean=wp.SimpleMean(),
    steps=0
)

# Train
X = [[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]]
y = [0.3, 0.8]
rw.train(X, y)

# Or train sample by sample
rw.train(wp.BinInput([1,0,1,0,1,0,1,0,1]), 0.3)

# Predict
predictions = rw.predict(X)       # [0.3, 0.8] (approx)
value = rw.predict(wp.BinInput([1,0,1,0,1,0,1,0,1]))  # single

# Change mean function
rw.setMeanFunc(wp.PowerMean(2))
rw.setMeanFunc(wp.ExponentialMean())

# Serialization
json_str = rw.json()
rw2 = wp.RegressionWisard(json_str)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addressSize` | `int` | required | Bits per RAM |
| `orderedMapping` | `bool` | false | Sequential indices (true) vs shuffled (false) |
| `completeAddressing` | `bool` | true | Pad mappings |
| `ignoreZero` | `bool` | false | Shorthand for `minZero=1` |
| `minZero` | `int` | 0 | Min 0-bits in address to produce a vote |
| `minOne` | `int` | 0 | Min 1-bits in address to produce a vote |
| `mean` | `Mean*` | `SimpleMean()` | Aggregation function |
| `steps` | `int` | 0 | Refinement iterations |
| `mapping` | `vector<int>` | — | Custom bit mapping |

---

## ClusRegressionWisard

**Source:** `src/models/clusregressionwisard/clusregressionwisard.cc`, `src/wrappers/clusregressionwisardwrapper.cc`

A clustering extension of RegressionWisard. Maintains multiple `RegressionWisard` instances (called `rews`) and adaptively assigns samples to clusters during training.

### How Clustering Works

Same adaptive logic as ClusWisard but for regression:

1. For each training sample, compute a similarity score against each existing RegressionWisard:
   ```
   score = getSimilarityScore(votes)
   ```
2. Compute adaptive threshold: `limit = min(1.0, minScore + count / threshold)`
3. If any RegressionWisard scores >= `limit`: train it
4. If no match AND count < `limit`: create a new RegressionWisard
5. Otherwise: train the best-scoring one

**Prediction**: Present the input to all RegressionWisards, pick the one with the highest similarity score, and return its prediction.

### Python API

```python
crw = wp.ClusRegressionWisard(
    addressSize=3,
    minScore=0.1,
    threshold=10,
    limit=5,
    orderedMapping=False,
    completeAddressing=True,
    ignoreZero=False,
    minZero=0,
    minOne=0,
    mean=wp.SimpleMean(),
    steps=0
)

# Train
crw.train(X, y)
# Or sample by sample
crw.train(wp.BinInput([1,0,1,0,1,0,1,0,1]), 0.3)

# Predict
predictions = crw.predict(X)

# Change mean for all clusters
crw.setMeanFunc(wp.PowerMean(2))

# Serialization
json_str = crw.json()
crw2 = wp.ClusRegressionWisard(json_str)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `addressSize` | `int` | required | Bits per RAM |
| `minScore` | `double` | 0.1 | Baseline cluster acceptance threshold |
| `threshold` | `int` | 10 | Adaptive threshold growth control |
| `limit` | `int` | 5 | Max number of RegressionWisard clusters |
| `orderedMapping` | `bool` | false | Sequential vs shuffled bit indices |
| `completeAddressing` | `bool` | true | Pad mappings |
| `ignoreZero` | `bool` | false | Shorthand for minZero=1 |
| `minZero` | `int` | 0 | Min 0-bits gate |
| `minOne` | `int` | 0 | Min 1-bits gate |
| `mean` | `Mean*` | `SimpleMean()` | Aggregation function |
| `steps` | `int` | 0 | Refinement iterations |
| `mapping` | `vector<int>` | — | Custom bit mapping |

---

## When to Use Which

| Scenario | Model |
|----------|-------|
| Simple regression, single distribution | `RegressionWisard` |
| Multi-modal data, different patterns need different models | `ClusRegressionWisard` |
| Need robust aggregation, outlier resistance | Use `Median` mean function |
| Need non-linear output shaping | Use `LogisticMean` |
| Improving accuracy with iterative correction | Set `steps > 0` |
