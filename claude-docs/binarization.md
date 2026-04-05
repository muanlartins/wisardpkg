# Binarization

WiSARD operates on binary inputs. Binarization techniques convert real-valued (continuous) data into binary vectors suitable for the model. All binarizers inherit from `BinBase` and implement `transform()`.

**Source files:** `src/binarization/` — `binbase.cc`, `thresholding.cc`, `meanthresholding.cc`, `thermometer.cc`, `distributivethermometer.cc`, `gaussianthermometer.cc`, `exponentialthermometer.cc`, `stochasticthermometer.cc`, `kernelcanvas.cc`

There are two categories of thermometers:
- **Static** (thresholds set at construction): SimpleThermometer, DynamicThermometer
- **Fitted** (thresholds learned from training data via `fit()`): Distributive, Gaussian, Exponential, Stochastic

## Base Class: BinBase

```cpp
class BinBase {
    virtual BinInput transform(const vector<double>& data) = 0;  // 1D input
    BinInput transform(const vector<vector<double>>& data);       // 2D input (delegates to 1D)
};
```

All binarizers return a `BinInput` (compact bitpacked binary vector).

---

## Thresholding

The simplest binarizer. Compares each value against a fixed threshold.

**Source:** `src/binarization/thresholding.cc`

**Python API:**
```python
binarizer = wp.Thresholding(threshold=0.5)
binary = binarizer.transform([0.2, 0.7, 0.5, 0.1])
# Result: [0, 1, 1, 0]
```

**Algorithm:**
```
output[i] = 0  if data[i] < threshold
output[i] = 1  if data[i] >= threshold
```

**Properties:**
- Input/output ratio: 1:1 (no expansion)
- Not adaptive — threshold is fixed at construction
- Best for data with a known decision boundary

**Constructor parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `tValue` | `double` | The threshold boundary |

---

## MeanThresholding

Adaptive thresholding where the threshold is the mean of the input vector. Computed per-transform call.

**Source:** `src/binarization/meanthresholding.cc`

**Python API:**
```python
binarizer = wp.MeanThresholding()
binary = binarizer.transform([0.2, 0.7, 0.5, 0.1])
# mean = 0.375; Result: [0, 1, 1, 0]
```

**Algorithm:**
```
threshold = mean(data)
output[i] = 0  if data[i] < threshold
output[i] = 1  if data[i] >= threshold
```

**Properties:**
- Input/output ratio: 1:1 (no expansion)
- Adaptive — threshold changes per input sample
- Always produces roughly 50/50 split of 0s and 1s
- No constructor parameters

---

## SimpleThermometer

Thermometer (unary) encoding — each input value is expanded into multiple bits by comparing against evenly-spaced thresholds. Captures magnitude information, not just above/below.

**Source:** `src/binarization/thermometer.cc`

**Python API:**
```python
binarizer = wp.SimpleThermometer(size=4, minimum=0.0, maximum=1.0)
binary = binarizer.transform([0.3, 0.8])
# With 4 thresholds in [0, 1]: e.g., [0.25, 0.5, 0.75, 1.0]
# 0.3 → [1, 0, 0, 0]  (only > 0.25)
# 0.8 → [1, 1, 1, 0]  (> 0.25, > 0.5, > 0.75, not > 1.0)
# Result: [1, 0, 0, 0, 1, 1, 1, 0]
print(binarizer.getSize())  # 4
```

**Algorithm:**
1. Generate `thermometerSize` evenly-spaced thresholds between `minimum` and `maximum` using `math::ranges()`
2. For each input value, compare against each threshold:
   - If `data[i] > threshold[j]` → bit = 1, else bit = 0
3. Output size = `input_size * thermometerSize`

**Properties:**
- Input/output ratio: 1:N (expands by `thermometerSize` factor)
- Same encoding parameters for all input dimensions
- Preserves ordinal magnitude information

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `size_t` | 2 | Number of bits per input value |
| `minimum` | `double` | 0.0 | Lower bound of value range |
| `maximum` | `double` | 0.0 | Upper bound of value range |

---

## DynamicThermometer

Per-dimension variable thermometer encoding — each input dimension can have a different number of bits and a different value range.

**Source:** `src/binarization/thermometer.cc`

**Python API:**
```python
binarizer = wp.DynamicThermometer(
    sizes=[3, 5],          # 3 bits for dim 0, 5 bits for dim 1
    minimum=[0.0, -1.0],   # per-dimension min
    maximum=[1.0, 1.0]     # per-dimension max
)
binary = binarizer.transform([0.5, 0.2])
print(binarizer.getSize())  # 8 (3 + 5)
```

**Algorithm:**
Same as SimpleThermometer but with per-dimension threshold vectors:
1. For each dimension `i`, generate `sizes[i]` thresholds in `[minimum[i], maximum[i]]`
2. Each dimension independently encoded
3. Output size = `sum(sizes)`

**Properties:**
- Input/output ratio: varies per dimension
- Flexible — different resolution per feature
- Throws `Exception` if `minimum`/`maximum` vector sizes don't match `sizes`

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sizes` | `vector<size_t>` | required | Bits per dimension |
| `minimum` | `vector<double>` | `[]` (defaults to 0.0 each) | Per-dimension lower bound |
| `maximum` | `vector<double>` | `[]` (defaults to 1.0 each) | Per-dimension upper bound |

---

## KernelCanvas

Advanced binarization for sequential/temporal data. Projects normalized input through random kernel vectors and activates the nearest kernels. Designed for spatial or time-series data.

**Source:** `src/binarization/kernelcanvas.cc`
**Wrapper:** `src/wrappers/kernelcanvaswrapper.cc`

**Python API:**
```python
kc = wp.KernelCanvas(
    dim=2,                  # input dimensionality per time step
    numberOfKernels=64,     # number of random kernel vectors
    bitsByKernel=3,         # output bits per activated kernel
    activationDegree=0.07,  # fraction of kernels to activate per point
    useDirection=False       # include temporal direction vectors
)
# input: sequence of 2D points [[x1,y1], [x2,y2], ...]
binary = kc.transform([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 3 points of dim=2
```

**Algorithm:**
1. **Kernel initialization**: `numberOfKernels` random vectors in [-1, 1]^dim (or [-1,1]^amplifyDim if useDirection=true)
2. **Per-transform**:
   a. Compute per-dimension mean and std of the input sequence
   b. For each time step:
      - Normalize: `point[j] = tanh((mean[j] - data[j]) / std[j])`
      - If `useDirection`: compute temporal gradient (sine/cosine of direction between consecutive points), expanding point to `dim + 2*(dim-1)` dimensions
      - Compute Euclidean distance from normalized point to each kernel
      - Sort kernels by distance
      - Activate nearest `round(activationDegree * numberOfKernels)` kernels
      - Set `bitsByKernel` consecutive output bits to 1 for each activated kernel
3. **Output size**: `numberOfKernels * bitsByKernel`

**Properties:**
- Input: flattened sequence of `dim`-dimensional points
- Output: fixed-size binary vector of `numberOfKernels * bitsByKernel` bits
- Temporal accumulation — multiple time steps contribute to the same output
- Non-adaptive kernels (random, fixed at construction)
- Distance-based activation (K-nearest-neighbors style)

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | required | Input dimensionality per time step |
| `numberOfKernels` | `int` | required | Number of random kernel vectors |
| `bitsByKernel` | `int` | 3 | Output bits per kernel |
| `activationDegree` | `float` | 0.07 | Fraction of kernels activated per point |
| `useDirection` | `bool` | false | Include temporal direction encoding |

**Direction encoding (useDirection=true):**
- Expands each point from `dim` to `dim + 2*(dim-1)` dimensions
- Extra dimensions encode sine and cosine of the gradient angle between consecutive normalized points
- Kernels are also generated in this expanded space

---

## DistributiveThermometer

Data-driven thermometer that splits based on the **empirical distribution** of the training data (equal percentiles), rather than assuming a distribution shape. Each region contains approximately the same number of data points.

Based on [Bacellar et al., ESANN 2022](https://www.esann.org/sites/default/files/proceedings/2022/ES2022-94.pdf).

**Source:** `src/binarization/distributivethermometer.cc`

**Python API:**
```python
dt = wp.DistributiveThermometer(32)
dt.fit(X_train.tolist())              # learn per-feature percentile thresholds
binary = dt.transform(sample.tolist())
thresholds = dt.getThresholds()        # inspect: list of lists (n_features × thermo_size)
dt.setThresholds(saved_thresholds)     # restore thresholds
```

**Algorithm:**
1. `fit(data)`: For each feature (column), sort all training values
2. Compute B thresholds at percentile positions `k/(B+1)` for k=1..B
3. Each threshold = the value at that percentile in the sorted array
4. `transform()`: same `>` comparison as SimpleThermometer, but per-feature thresholds

**Properties:**
- Per-feature thresholds (each feature gets its own distribution)
- No distribution assumptions — purely data-driven
- High resolution where data is dense, low resolution where sparse
- Requires `fit()` before `transform()`

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thermometerSize` | `size_t` | 32 | Number of bits per feature |

---

## GaussianThermometer

Assumes each feature follows a **normal distribution**. Computes per-feature mean (μ) and standard deviation (σ), then places thresholds at equal-probability regions under the Gaussian curve.

**Source:** `src/binarization/gaussianthermometer.cc`

**Python API:**
```python
gt = wp.GaussianThermometer(32)
gt.fit(X_train.tolist())
binary = gt.transform(sample.tolist())
```

**Algorithm:**
1. `fit(data)`: For each feature, compute μ and σ from training data
2. Threshold k = μ + σ × Φ⁻¹(k/(B+1)), where Φ⁻¹ is the inverse normal CDF (probit)
3. Probit implemented via Peter Acklam's rational approximation (accurate to ~1e-9)
4. Edge cases: σ=0 (constant feature) → all thresholds = μ

**Properties:**
- Per-feature thresholds based on Gaussian assumption
- Higher resolution near the mean, lower at the tails
- Works well when data is roughly normally distributed

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thermometerSize` | `size_t` | 32 | Number of bits per feature |

---

## ExponentialThermometer

Assumes each feature follows an **exponential distribution**. Computes per-feature mean (μ) and places thresholds using the exponential inverse CDF.

**Source:** `src/binarization/exponentialthermometer.cc`

**Python API:**
```python
et = wp.ExponentialThermometer(32)
et.fit(X_train.tolist())
binary = et.transform(sample.tolist())
```

**Algorithm:**
1. `fit(data)`: For each feature, compute μ (mean)
2. Threshold k = -ln(1 - k/(B+1)) × μ (inverse CDF of exponential distribution)
3. Edge cases: μ ≤ 0 → all thresholds = 0

**Properties:**
- Per-feature thresholds based on exponential assumption
- Higher resolution near zero, lower at the tail
- Works well for right-skewed non-negative data

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thermometerSize` | `size_t` | 32 | Number of bits per feature |

---

## SupervisedThermometer

Supervised binarization with **3 interchangeable methods** selected via the `method` parameter. All require labels during `fit()`.

**Source:** `src/binarization/supervisedthermometer.cc`

**Python API:**
```python
st = wp.SupervisedThermometer(32, method='class_conditional')
st.fit(X_train.tolist(), y_train.tolist())
binary = st.transform(sample.tolist())
```

**Methods:**

| Method | Description |
|--------|-------------|
| `'class_conditional'` (default) | Equal-frequency quantiles per class, merged. Concentrates thresholds at class-overlap regions. |
| `'mi_allocation'` | Computes MI(feature, label), allocates more bits to high-MI features. Distributive thresholds within each. |
| `'entropy_weighted'` | Sliding-window class entropy as weights; thresholds at entropy-weighted quantiles. More resolution at class boundaries. |

**Properties:**
- `class_conditional` and `entropy_weighted`: uniform bits per feature (same total as Distributive)
- `mi_allocation`: variable bits per feature (use `getSizes()` to inspect). Total budget preserved.
- All require labels during `fit()` (supervised)

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thermometerSize` | `size_t` | 32 | Base bits per feature |
| `method` | `string` | `"class_conditional"` | Threshold placement method |
| `minBitsPerFeature` | `size_t` | 2 | Minimum bits per feature (only used by `mi_allocation`) |

---

## StochasticThermometer

Starts from Distributive percentile thresholds, then **optimizes each threshold individually** using stochastic coordinate descent. The `optimize()` method trains internal WiSARD models to evaluate each threshold position against a validation split. Evaluations are parallelized across CPU cores.

**Source:** `src/binarization/stochasticthermometer.cc`
**Note:** Included after model definitions in `wisardpkg.h` because `optimize()` uses `Wisard` and `DataSet` internally.

**Python API:**
```python
st = wp.StochasticThermometer(32)
st.fit(X_train.tolist())
val_acc = st.optimize(
    X_train.tolist(), y_train.tolist(),
    addressSize=28,
    validationSize=0.2,     # fraction of training data for validation
    rounds=5,               # full passes over all thresholds
    stepsPerThreshold=20,   # positions tested per threshold (half left, half right)
    numThreads=0            # 0 = auto-detect CPU cores
)
binary = st.transform(sample.tolist())
```

**Algorithm:**
1. `fit(data)`: Same as DistributiveThermometer (percentile baseline)
2. `optimize(data, labels, addressSize, ...)`:
   a. Create a fixed random mapping (shared across all evaluations for fair comparison)
   b. Create thread-local Wisard instances with the same mapping
   c. For each round (threshold processing order randomized each round):
      - For each threshold (feature f, index k):
        - Pre-generate all candidate positions (half left, half right shifts within bands)
        - Evaluate candidates in parallel: each thread copies BinInputs, modifies the bit, trains its Wisard, classifies validation
        - Commit the best threshold
   d. Return best validation accuracy

**Performance characteristics:**
- Total evaluations per optimize: `n_features × thermo_size × stepsPerThreshold × rounds`
- Parallelized across CPU cores (near-linear scaling up to ~8 threads)
- Fixed mapping ensures accuracy comparisons across candidates are fair
- Typical settings: `stepsPerThreshold=20, rounds=2`

**Constructor parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thermometerSize` | `size_t` | 32 | Number of bits per feature |

**optimize() parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `vector<vector<double>>` | required | Training data (raw features) |
| `labels` | `vector<string>` | required | Training labels |
| `addressSize` | `int` | required | WiSARD address size for internal evaluation |
| `validationSize` | `double` | 0.2 | Fraction of training data for validation |
| `rounds` | `int` | 5 | Full passes over all thresholds |
| `stepsPerThreshold` | `int` | 100 | Positions tested per threshold |
| `numThreads` | `int` | 0 | Number of threads (0 = auto-detect) |

---

## Choosing addressSize and thermoSize

**addressSize** (bits per RAM) — controls pattern specificity:
- Guideline: `addressSize ≤ log2(N_training_samples)`
- Too large → overfitting (most RAM addresses seen only once)
- Too small → underfitting (too coarse)
- Bleaching compensates for moderate overfitting

**thermoSize** (bits per feature) — controls information richness:
- More bits = finer value resolution, but larger binary input
- Diminishing returns past ~8 bits for most datasets
- Constraint: keep `n_features × thermoSize` under ~10,000 bits for reasonable speed

| Dataset size | Suggested addressSize | Suggested thermoSize |
|-------------|----------------------|---------------------|
| < 500 samples | 8-10 | 32 (if few features) |
| 500-5,000 | 14-20 | 32 (if few features) |
| 5,000-50,000 | 24-28 | 8-32 (depends on n_features) |
| 50,000+ | 28 | 4-8 (if many features like images) |

---

## Comparison Table

| Technique | Expansion | Requires fit() | Per-feature | Best For |
|-----------|-----------|---------------|-------------|----------|
| Thresholding | 1:1 | No | No | Known boundary, simple data |
| MeanThresholding | 1:1 | No | No | Unknown boundary, balanced splits |
| SimpleThermometer | 1:N fixed | No | No (global) | Uniform data, same-scale features |
| DynamicThermometer | 1:N variable | No | Yes | Heterogeneous features with known ranges |
| DistributiveThermometer | 1:N fixed | Yes (data) | Yes | General purpose, no distribution assumptions |
| GaussianThermometer | 1:N fixed | Yes (data) | Yes | Roughly normal data |
| ExponentialThermometer | 1:N fixed | Yes (data) | Yes | Right-skewed non-negative data |
| SupervisedThermometer | 1:N fixed/variable | Yes (data + labels) | Yes | 3 methods: class-conditional, MI allocation, entropy-weighted |
| StochasticThermometer | 1:N fixed | Yes + optimize() | Yes | Coordinate descent threshold optimization |
| KernelCanvas | Sequence → fixed | No | N/A | Time-series, spatial sequences |
