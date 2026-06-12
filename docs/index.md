# wisardpkg (muanlartins fork)

C++/Python library for WiSARD weightless neural networks and variants. This is a
fork of [IAZero/wisardpkg](https://github.com/IAZero/wisardpkg) (`muanlartins`
branch) and the experimental engine for the WiSARD F4RM paper.

**New here?** Start with **[Fork extensions vs upstream](fork-extensions.md)** —
the single page that states what this fork adds over upstream and why:
`BloomWisard`, fitted thermometers (Distributive / Gaussian / Exponential /
Logarithmic / Stochastic / Supervised), `CircularThermometer`,
`ColorMaskBinarization`, `Local2DMapping`, multi-resolution mappings,
large-address RAMs (`addressSize > 64`), the `Wisard` classification hooks, the
`wisardpkg.models` ports (BTHOWeN, DWN, ULEEN), and the `scripts/sweeps/` F4RM
reproduction drivers.

These `docs/` pages are concise user-facing quickstarts. For the full
engineering reference (algorithms, source `file:line` citations, every
constructor parameter, the fork hooks) see the `claude-docs/` set linked from
each page and indexed in the repo [`CLAUDE.md`](../CLAUDE.md).

## Models

- [Wisard](models/wisard.md) — standard supervised classifier (fork adds opt-in
  hooks: [fork-extensions](fork-extensions.md#wisard-classification-hooks))
- [ClusWisard](models/cluswisard.md) — clustering (supervised / semi / unsupervised)
- [Discriminator](models/discriminator.md) — standalone RAM group
- BloomWisard, RegressionWisard, ClusRegressionWisard — see
  [classification-models](../claude-docs/classification-models.md) and
  [regression-models](../claude-docs/regression-models.md)

## Binarization (continuous → binary)

- [KernelCanvas](binarization/kernelcanvas.md) — sequence / spatial encoder
- All techniques (thresholding, the fitted/supervised/stochastic thermometers,
  ColorMask) with algorithms and constructor tables:
  [binarization](../claude-docs/binarization.md)

## Data

- [BinInput](data/bininput.md) — bitpacked binary vector
- [DataSet](data/dataset.md) — labelled / regression / unsupervised container

## Mapping

- RandomMapping (uniform / multi-resolution) and Local2DMapping (image windows):
  [mapping](../claude-docs/mapping.md)

## Python models & F4RM reproduction

- [Python models](../claude-docs/python-models.md) — `wisardpkg.models` BTHOWeN /
  DWN / ULEEN ports and the `scripts/sweeps/` drivers
- [Fork extensions → Reproducing the F4RM numbers](fork-extensions.md#reproducing-the-f4rm-numbers)

## Others

- [Synthesizer](others/synthesizer.md) — generate samples from a mental image
- [RAMDataHandle](others/ramdatahandle.md) — inspect / edit serialized RAM data

## Build

```bash
pip install .            # core (C++) + BTHOWeN
pip install ".[torch]"   # adds torch + numpy → enables DWN, ULEEN
```

`import wisardpkg as wp; wp.Wisard(...)` works exactly as upstream — the C++
core is re-exported from the `wisardpkg._native` submodule. See
[build-and-testing](../claude-docs/build-and-testing.md).
