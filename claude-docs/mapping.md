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

## RandomMapping

**Source:** `src/mapping/randommapping.cc`

The only mapping implementation. Generates random bit-to-RAM assignments.

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
