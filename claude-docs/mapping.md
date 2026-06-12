# Mapping

Mapping determines which input bits feed into which RAMs. This is a critical part of WiSARD — the random assignment of input bits to RAM groups is what gives the network its generalization ability.

**Source files:** `src/mapping/mappinggenerator.cc`, `src/mapping/randommapping.cc`, `src/mapping/mappinggeneratorhelper.cc`

## Concept

Given an input binary vector of size `entrySize` and a desired `tupleSize` (bits per RAM):

1. Create an index list: `[0, 1, 2, ..., entrySize-1]`
2. Optionally pad to make the length a multiple of `tupleSize` (complete addressing)
3. Shuffle the indices randomly
4. Partition into groups of `tupleSize` — each group defines one RAM

**Example**: entrySize=9, tupleSize=3
```
Indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]
Shuffled: [5, 2, 8, 0, 7, 3, 6, 1, 4]
RAM 0: [5, 2, 8]  — reads bits 5, 2, 8
RAM 1: [0, 7, 3]  — reads bits 0, 7, 3
RAM 2: [6, 1, 4]  — reads bits 6, 1, 4
```

## Base Class: MappingGenerator

```cpp
class MappingGenerator {
    bool completeAddressing;  // pad to fill all RAM slots
    bool monoMapping;         // all labels share the same mapping

    virtual vector<vector<int>> getMapping(string label) = 0;
    virtual MappingGenerator* clone() = 0;
    virtual string json() = 0;
    virtual string className() = 0;

    // Accessors
    map<string, vector<vector<int>>> getMappings();
    void setMappings(map<string, vector<vector<int>>>);
    void setMapping(vector<vector<int>>, string label);
    void setIndexes(vector<int>);
    void setEntrySize(unsigned int);
    void setTupleSize(unsigned int);
    unsigned int getEntrySize();

protected:
    vector<int> indexes;
    map<string, vector<vector<int>>> mapping;  // cached per label
    unsigned int tupleSize;

    static void completeMapping(int tupleSize, vector<int>& indexes);
    void entrySizeToIndexes(unsigned int);
    void checkEntrySize(unsigned int);   // must be >= 2
    void checkTupleSize(unsigned int);   // must be >= 2
};
```

## Available implementations

- **RandomMapping** — uniform random assignment of input bits to RAMs. The default; ignores any spatial or structural prior on the input.
- **Local2DMapping** — for image-shaped inputs. Each RAM only sees bits from a contiguous 2D window of the image, with windows tiled across the image at a configurable stride. Preserves spatial co-occurrence (essential for tasks where the *arrangement* of pixel features matters, not just their bag).

## RandomMapping

**Source:** `src/mapping/randommapping.cc`

The default mapping. Generates random bit-to-RAM assignments. For spatially-structured inputs, see `Local2DMapping` below.

### Key Features

**Lazy generation**: Mappings are created on first access per label (not at construction time).

**monoMapping**: When `true`, all labels share a single mapping. When `false` (default), each label gets a unique random mapping. This affects whether different classes "see" the input through the same or different bit groupings.

**completeAddressing**: When `true` (default), if `entrySize` is not evenly divisible by `tupleSize`, random duplicate indices are added to fill the last group. When `false`, the last group is simply smaller.

### Python API

```python
import wisardpkg as wp

# Default (lazy initialization)
rm = wp.RandomMapping(monoMapping=False, completeAddressing=True)

# With explicit entry size and tuple size
rm = wp.RandomMapping(entrySize=9, tupleSize=3, monoMapping=False, completeAddressing=True)

# With custom indexes
rm = wp.RandomMapping(indexes=[0,1,2,3,4,5,6,7,8], tupleSize=3)

# Use with Wisard
wisard = wp.Wisard(addressSize=3, mappingGenerator=rm)

# Inspect mappings after training
mappings = rm.getMappings()  # {"cold": [[5,2,8],[0,7,3],[6,1,4]], "hot": [[3,6,1],...]}

# Set mappings manually
rm.setMapping([[0,1,2],[3,4,5],[6,7,8]], "cold")
rm.setMappings({"cold": [[0,1,2],[3,4,5]], "hot": [[2,1,0],[5,4,3]]})
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entrySize` | `unsigned int` | — | Total input bits (auto-creates sequential indexes) |
| `indexes` | `vector<int>` | — | Custom input bit indices |
| `tupleSize` | `unsigned int` | — | Bits per RAM group |
| `monoMapping` | `bool` | false | Share one mapping across all labels |
| `completeAddressing` | `bool` | true | Pad to fill all RAM slots |
| `multiResolution` | `bool` | false | Variable-size RAMs (see below) |

### Complete Addressing Example

entrySize=10, tupleSize=3:
- Without complete addressing: 3 RAMs of 3 bits + 1 RAM of 1 bit = 4 RAMs
- With complete addressing: pad to 12 indices (add 2 random duplicates), then 4 RAMs of 3 bits each

### Multi-Resolution Mapping

When `multiResolution=true`, RAM tuple sizes are no longer uniform — they are linearly spaced from `max(2, tupleSize/2)` to `tupleSize * 3/2`, then perturbed to cover exactly `entrySize` bits, then shuffled so position doesn't determine size. The result is a mapping where some RAMs see fewer bits (`addressSize/2`-ish, more general) and others see more (`addressSize·3/2`-ish, more specific), giving the network a multi-scale view of the input.

`Wisard::rank()` knows about this mode: when `multiResolution=true` it multiplies each RAM's vote by that RAM's tuple size before bleaching, so larger RAMs get proportionally more weight. Without that reweighting, smaller RAMs would dominate purely because they fire more often. The reweight happens in `wisard.cc:221-231`.

```python
rm = wp.RandomMapping(entrySize=64, tupleSize=8, multiResolution=True)
w = wp.Wisard(addressSize=8, mappingGenerator=rm)
# OR pass it directly as a Wisard kwarg (the WisardWrapper forwards it
# onto the embedded mappingGenerator):
w = wp.Wisard(addressSize=8, multiResolution=True)
```

This is experiment-oriented; the F4RM paper's canonical experiments use uniform tuple sizes.

## Local2DMapping

**Source:** `src/mapping/local2dmapping.cc`

A spatially-aware mapping for image-shaped inputs. Standard `RandomMapping` shuffles bits uniformly across all RAMs, which destroys the 2D co-occurrence that makes structural patterns (stripes, edges, blob configurations) detectable. `Local2DMapping` confines each RAM's input bits to a contiguous `(windowHeight × windowWidth × bitsPerPixel)` window of the image, with windows tiled across the image at a configurable stride.

### Input layout convention

The flat bit vector is row-major over pixels, with the `bitsPerPixel` bits for each pixel **contiguous**. That is, the bit at `(row=h, col=w, bit=k)` lives at flat index `h * imageWidth * bitsPerPixel + w * bitsPerPixel + k`.

This matches:

- `ColorMaskBinarization.transform(flatten(image_HWC))` — 3 bits per pixel.
- Any per-channel thermometer applied to a flattened `(H, W, C)` array — set `bitsPerPixel = C * thermometerSize`.

### Python API

```python
import wisardpkg as wp

# Image of 128x128 RGB pixels encoded with ColorMaskBinarization (3 bits/pixel).
# Each RAM sees a 16x16 window (768 bits), randomly samples 12 of them.
# Windows tile the image with stride 8 (overlapping).
m = wp.Local2DMapping(
    imageHeight=128, imageWidth=128, bitsPerPixel=3,
    windowHeight=16, windowWidth=16,
    tupleSize=12,
    stride=8,
    ramsPerWindow=4,   # emit 4 different random subsets per window
)
w = wp.Wisard(addressSize=12, mappingGenerator=m, bleaching=True)

print(m.getNumberOfRAMs())   # ((128-16)/8 + 1)**2 * 4 = 15**2 * 4 = 900
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imageHeight`   | `unsigned int` | required | Image height (pixels) |
| `imageWidth`    | `unsigned int` | required | Image width (pixels) |
| `bitsPerPixel`  | `unsigned int` | required | Number of bits per pixel in the flattened input |
| `windowHeight`  | `unsigned int` | required | Window height (pixels) — must be ≤ `imageHeight` |
| `windowWidth`   | `unsigned int` | required | Window width (pixels) — must be ≤ `imageWidth` |
| `tupleSize`     | `unsigned int` | required | Bits per RAM — must be ≤ `windowHeight * windowWidth * bitsPerPixel` |
| `stride`        | `unsigned int` | 0 | Stride between window origins; 0 means `min(windowHeight, windowWidth)` (no overlap) |
| `ramsPerWindow` | `unsigned int` | 1 | Number of RAMs to emit per window (each samples a different random subset of the window's bits) |
| `monoMapping`   | `bool` | false | Share one mapping across all class labels |

### How to choose dimensions for image inputs

- **`windowHeight` / `windowWidth`** — the receptive field of each RAM. Should be at least large enough to contain the smallest discriminative feature: for *Where's Waldo* at 128 px, Waldo's striped torso is ~16-24 pixels tall, so `windowHeight=16, windowWidth=16` is a sensible lower bound. Smaller windows = faster, narrower features; larger windows = slower, more global context.
- **`stride`** — controls overlap. `stride = windowSize` gives a non-overlapping tile (cheapest). `stride = windowSize / 2` doubles the RAM count and gives 4× more windows that each see a partially-shared patch. For detection at sub-window-scale, overlapping strides matter; for classification of whole patches, non-overlapping is often enough.
- **`tupleSize`** — bits per RAM. Same guidance as for `RandomMapping`: `≤ log2(N_training_samples)` to avoid one-shot RAM addresses unless paired with bleaching or a Bloom-backed model. The new constraint is that `tupleSize ≤ windowHeight * windowWidth * bitsPerPixel`, so a small window caps it.
- **`ramsPerWindow`** — controls the redundancy at each spatial location. With `ramsPerWindow=1`, each window emits exactly one RAM that randomly samples `tupleSize` of its bits — fast but blind to ~all the bits in the window when `tupleSize << window bits`. With `ramsPerWindow=4`, the window emits 4 different random tuples — substantially more coverage.

### Number of RAMs

```
rowsOfWindows = (imageHeight - windowHeight) / stride + 1
colsOfWindows = (imageWidth  - windowWidth)  / stride + 1
numRams = rowsOfWindows * colsOfWindows * ramsPerWindow
```

`Local2DMapping::getNumberOfRAMs()` returns this value (exposed to Python). Useful as a sanity check before training (large windows / dense strides / many `ramsPerWindow` can balloon RAM counts).

### When to use this vs RandomMapping

- Use `Local2DMapping` when the input has 2D structure that matters: images, spectrograms, time-frequency representations, anything where adjacent positions are correlated.
- Use `RandomMapping` when the input has no inherent ordering, or when you want the model to remain *invariant* to spatial position (which `RandomMapping` does in a weak sense — though that invariance comes at the cost of structural sensitivity).

### Why it matters

Stock `RandomMapping` paired with a spatially-meaningful encoder (e.g., `ColorMaskBinarization`) can produce results *worse than chance* on image-shaped inputs: the encoder emits 0/1 bits at specific spatial positions to mark colour predicates, then the random mapping immediately destroys their geometric relationships. `Local2DMapping` keeps the relationships intact within each RAM's window. See the `notebooks/waldo/` series in the project root for an empirical case study.

---

## Registry (Serialization)

**Source:** `src/mapping/mappinggeneratorhelper.cc`

`MappingGeneratorHelper::load(config)` deserializes from JSON:
```json
{
  "className": "RandomMapping",
  "params": {
    "indexes": [0,1,2,...],
    "tupleSize": 3,
    "monoMapping": false,
    "completeAddressing": true,
    "mapping": {"label": [[...], [...]]}
  }
}
```

## RAM Count Calculation

The utility function `calculateNumberOfRams(entrySize, addressSize, completeAddressing)` in `utils.cc` computes how many RAMs will be created:
- With complete addressing: `ceil(entrySize / addressSize)`
- Without: `floor(entrySize / addressSize)` (+ 1 if remainder > 0, but that last RAM has fewer bits)
