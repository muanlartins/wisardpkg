# DataSet

A container of `BinInput` samples with optional string labels and/or double
targets. Supports four modes, auto-detected from what you pass:

| Mode | Pass |
|------|------|
| Supervised classification | data + list of string labels |
| Supervised regression | data + list of float targets |
| Semi-supervised | data + `{index: label}` dict (only some labelled) |
| Unsupervised | data only |

```python
import wisardpkg as wp

# classification
ds = wp.DataSet([[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]], ["cold", "hot"])

# regression
ds = wp.DataSet([[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]], [0.3, 0.8])

# semi-supervised (index 1 unlabelled)
ds = wp.DataSet([[...],[...],[...]], {0: "cold", 2: "hot"})

# unsupervised
ds = wp.DataSet([[1,0,1,0,1,0,1,0,1], [0,1,0,1,0,1,0,1,0]])

ds.add([1,1,1,0,0,0,1,1,1], "cold")
ds[0]                                   # BinInput
ds.getLabel(0); ds.getY(0)
len(ds)

ds.save("my_dataset")                   # -> my_dataset.wpkds
ds2 = wp.DataSet("my_dataset.wpkds")    # load
```

For the `.wpkds` text format and source `file:line`, see
[data-structures](../../claude-docs/data-structures.md#dataset).
