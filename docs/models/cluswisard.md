# ClusWisard

A clustering extension of WiSARD: multiple discriminators per class (and an
unsupervised mode that discovers clusters). Each class holds a `Cluster` that spawns
a new discriminator when existing ones don't match well enough.

## Constructor

```python
import wisardpkg as wp

clus = wp.ClusWisard(
    addressSize=3,            # required — bits per RAM
    minScore=0.1,             # required — baseline cluster acceptance (0..1)
    threshold=10,             # required — adaptive-threshold growth rate
    discriminatorsLimit=5,    # required — max discriminators per cluster
    bleachingActivated=True,  # optional
    ignoreZero=False,         # optional
    completeAddressing=True,  # optional
    base=2,                   # optional
    verbose=False,            # optional
)

clus2 = wp.ClusWisard(clus.json())   # from a saved config
```

The constructor keyword is `discriminatorsLimit` (not `discriminatorLimit`).
`classify()` returns a `list[str]`; the old `returnActivationDegree` /
`returnConfidence` / `returnClassesDegrees` dict-output kwargs are **not** handled by
this fork — use `rank()` / `getAllScores()` for scores.

## Methods

```python
clus.train(X, y)                  # supervised
clus.train(X, {0: "cold", 3: "hot"})   # semi-supervised
clus.trainUnsupervised(X)         # unsupervised

clus.classify(X)
clus.classifyUnsupervised(X)
clus.score(X, y)

clus.rank(X); clus.rankUnsupervised(X)
clus.getAllScores(X)
clus.getMentalImages(); clus.getMentalImage("cold")

clus.setMinScore(0.2); clus.setThreshold(20); clus.setDiscriminatorsLimit(10)
```

## Details and internals

Cluster-score semantics, the adaptive limit `minScore + count/threshold`, and source
`file:line`:
[classification-models](../../claude-docs/classification-models.md#cluswisard).
