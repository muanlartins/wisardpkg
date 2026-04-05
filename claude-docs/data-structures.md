# Data Structures

Core data containers for binary inputs, datasets, and RAM data persistence.

## BinInput

**Source:** `src/data/bininput.cc`

A compact, bitpacked binary vector. Stores 8 bits per byte for memory efficiency (8x compression vs `vector<short>`).

### Internal Storage

- Uses a `char` string for bitpacking
- `remain` field tracks unused bits in the last byte (padding)
- Logical size = `(byte_length * 8) - remain`

### Serialization

`data()` returns `Base64(remain_byte + bitpacked_bytes)`. The string constructor deserializes this format.

### Python API

```python
import wisardpkg as wp

# From size (all zeros)
b = wp.BinInput(9)

# From vector
b = wp.BinInput([1, 0, 1, 0, 1, 0, 1, 0, 1])

# From serialized string
b = wp.BinInput(other.data())

# Access
bit = b[3]         # read
b[3] = 1           # write

# Properties
length = len(b)    # 9
as_list = b.list() # [1, 0, 1, 0, 1, 0, 1, 0, 1]

# Serialization
encoded = b.data()  # Base64 string

# Extend
b.extend(wp.BinInput([1, 1, 0]))
b.extend([1, 1, 0])
```

### Error Handling

- `get(index)` / `set(index, value)`: throw `Exception` if index is out of range

---

## DataSet

**Source:** `src/data/dataset.cc`

A flexible container supporting supervised, semi-supervised, unsupervised, and regression data. Stores `BinInput` objects with optional string labels and/or double target values.

### Data Modes

| Mode | Labels | Y values | Detection |
|------|--------|----------|-----------|
| **Supervised classification** | `labels.size() == data.size()` | empty | `isClassification()` |
| **Supervised regression** | empty | `Y.size() == data.size()` | `isRegression()` |
| **Semi-supervised** | partial labels or Y | — | `isSemiSupervised()` |
| **Unsupervised** | empty | empty | `isUnsupervised()` |

### Internal Storage

```cpp
vector<BinInput> data;                    // binary samples
unordered_map<int, string> labels;        // index → class label (sparse)
unordered_map<int, double> Y;             // index → target value (sparse)
vector<int> labelIndices;                 // which indices have labels
vector<int> unlabelIndices;               // which indices are unlabeled
```

Labels and Y use `unordered_map<int, ...>` (sparse) to support semi-supervised learning where only some samples have labels.

### Python API

```python
import wisardpkg as wp

# Supervised classification
ds = wp.DataSet(
    [[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]],
    ["cold", "hot"]
)

# Supervised regression
ds = wp.DataSet(
    [[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]],
    [0.3, 0.8]
)

# Semi-supervised (dict labels — only some indices labeled)
ds = wp.DataSet(
    [[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0], [1,1,0,0,1,1,0,0,1]],
    {0: "cold", 2: "hot"}  # index 1 has no label
)

# Unsupervised
ds = wp.DataSet([[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]])

# From BinInput objects
ds = wp.DataSet([wp.BinInput([1,0,1]), wp.BinInput([0,1,0])], ["a", "b"])

# From serialized strings
ds = wp.DataSet([b1.data(), b2.data()], ["a", "b"])

# Add samples
ds.add([1,1,1,0,0,0,1,1,1], "cold")
ds.add(wp.BinInput([1,1,1,0,0,0,1,1,1]), "cold")
ds.add([1,1,1,0,0,0,1,1,1])  # unsupervised add

# Access
sample = ds[0]            # BinInput
sample = ds.get(0)        # BinInput
label = ds.getLabel(0)    # "cold"
target = ds.getY(0)       # 0.3 (regression)
all_labels = ds.getLabels()   # {0: "cold", 1: "hot", ...}
all_ys = ds.getYs()           # {0: 0.3, 1: 0.8, ...}

# Modify
ds[0] = wp.BinInput([0,0,0,0,0,0,0,0,0])
ds.set(0, wp.BinInput([0,0,0,0,0,0,0,0,0]))

# Size
n = len(ds)

# Persistence
ds.save("my_dataset")  # saves to my_dataset.wpkds
ds2 = wp.DataSet("my_dataset.wpkds")  # load
```

### File Format (`.wpkds`)

Text-based format:
```
[R|C|U]$label1#data1.label2#data2...
```
- Prefix: `R` (regression), `C` (classification), `U` (unsupervised)
- `$` separates prefix from entries
- Each entry: `label#base64_data` separated by `.`
- Labels/targets are Base64-encoded
- Data uses BinInput's `data()` encoding

---

## RAMDataHandle

**Source:** `src/models/wisard/ramdatahandle.cc`

Handles serialization and external access to classification RAM data. Useful for inspecting, modifying, or comparing discriminator memory.

### Python API

```python
# Create from discriminator JSON data
handle = wp.RAMDataHandle(disc_data_string)

# Read
ram_dict = handle.get(ram_index)                # entire RAM as dict
value = handle.get(ram_index, address)          # single vote count

# Write
handle.set(ram_index, address, new_value)

# Serialize
all_data = handle.data()                        # all RAMs
single = handle.data(ram_index)                 # one RAM

# Compare
are_equal = handle.compare(other_handle)
```

### Storage Format

Base64-encoded binary blocks: each entry is `(addr_t, content_t)` = 8+4 bytes. Multiple RAMs separated by `"."`.

---

## RegressionRAMDataHandle

**Source:** `src/models/regressionwisard/regressionramdatahandle.cc`

Same concept for regression RAMs. Each entry stores `(address, [count, sum_y, fit_adjustment])` = 8+24 bytes.

### Python API

```python
# From serialized string
handle = wp.RegressionRAMDataHandle(data_string)

# From raw dict
handle = wp.RegressionRAMDataHandle({addr: [count, sum_y, fit], ...})

# From list of RAM dicts
handle = wp.RegressionRAMDataHandle([{...}, {...}, ...])

# Read/write
ram_dict = handle.get(ram_index)
handle.set(ram_index, {addr: [count, sum_y, fit], ...})

# Serialize
all_data = handle.data()
single = handle.data(ram_index)
```

---

## Synthesizer

**Source:** `src/synthetic_data/synthesizers.cc`

Generates synthetic binary samples from a WiSARD mental image. Useful for data augmentation or understanding what the model has learned.

### How It Works

1. Takes a mental image (vector of activation counts per input position) from a trained discriminator
2. `makeCube()`: For each position, creates a set of random indices proportional to the activation count
3. `make()`: Picks a random index, returns 1 for positions that include it, 0 otherwise

The generated samples statistically reflect the learned activation patterns — positions with high activation counts are more likely to be 1.

### Python API

```python
# Get mental image from trained model
images = wisard.getMentalImages()
mental_image = images["cold"]  # list of ints

# Create synthesizer
synth = wp.Synthesizer(mental_image)

# Generate samples
sample = synth.make()  # list of 0/1
```
