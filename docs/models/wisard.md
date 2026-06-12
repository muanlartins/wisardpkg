# Wisard

The core supervised WiSARD classifier: one discriminator per class. Training stores
patterns; `classify` aggregates RAM votes and applies a classification method
(bleaching by default).

## Constructor

```python
import wisardpkg as wp

wsd = wp.Wisard(
    addressSize=3,            # required — bits per RAM
    bleachingActivated=True,  # optional — shorthand for Bleaching(True, confidence)
    confidence=1,             # optional
    ignoreZero=False,         # optional
    completeAddressing=True,  # optional
    base=2,                   # optional
    verbose=False,            # optional
)

# or from a saved config
wsd2 = wp.Wisard(wsd.json())
```

`classify()` returns a `list[str]` (one label per input) or `str` for a single
`BinInput`. The old `returnActivationDegree` / `returnConfidence` /
`returnClassesDegrees` dict-output kwargs are **not** handled by this fork's
`WisardWrapper` — use `rank()` to get raw per-class votes instead.

## Methods

```python
X = [[1,1,1,0,0,0,0,0], [0,0,0,0,1,1,1,1]]
y = ["cold", "hot"]

wsd.train(X, y)
out = wsd.classify(X)          # ["cold", "hot"]
acc = wsd.score(X, y)
ranks = wsd.getMentalImages()  # {label: [per-bit counts]}
votes = wsd.rank(wp.BinInput(X[0]))   # {label: total vote}

wsd.leaveOneOut([1,1,1,0,0,0,0,0], "cold")
wsd.leaveMoreOut(X, y)

cfg = wsd.json()               # config + RAM data
cfg = wsd.json(True, "path/")  # offload RAM data to files
```

## Details and internals

The fork adds classification hooks (negative evidence, RAM weights, shared
discriminator, attention weighting, soft bleaching, cross-class scoring,
multi-resolution mapping), large-address RAMs, and the lifecycle methods
`trainSingle` / `untrainSingle` / `reset` / `getTupleSizes` / `deployedSizeBytes`.
Full reference, the hook order, and source `file:line`:
[classification-models](../../claude-docs/classification-models.md#wisard).
