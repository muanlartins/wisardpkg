"""Clean reference port of ULEEN (Susskind et al., ACM TACO 2023).

ULEEN = Ultra Low-Energy Edge Networks. Builds on BTHOWeN by replacing
the counting Bloom filters with **continuous-valued Bloom filters** trained
end-to-end via backprop with the straight-through estimator (STE), and by
combining multiple **independently trained submodels** as an ensemble.

The reference at https://github.com/ZSusskind/ULEEN ships:

* `software_model/model.py` — `BackpropWiSARD` and `BackpropMultiWiSARD`
* `software_model/cpp/h3_hash.cpp` — H3 hash as a libtorch C++ extension
* `software_model/train_model.py` — Adam + per-submodel CE loss training loop

Our port replicates the architecture in pure PyTorch:

* The H3 hash becomes a vectorised einsum + bitwise-XOR loop (matches the
  reference C++ exactly — verified via gradcheck-style equality on a small
  example).
* `BackpropWiSARD` is a `torch.nn.Module` storing the continuous Bloom
  filter table as a single `nn.Parameter` of shape
  `(classes, filters_per_disc, filter_entries)`. Training uses standard
  PyTorch autograd through `STEBinarize` and `amin`.
* `BackpropMultiWiSARD` is a `nn.ModuleList` ensemble; the reference's
  per-submodel CE-loss summation is reproduced in `ULEENClassifier.fit`.

Memory accounting (`mem_method = "uleen_binarized_table"`) follows the
paper's Table 2 reporting style — post-binarisation footprint:

```
bits_per_submodel  = classes × filters_per_disc × filter_entries
bytes_h3           = h × filter_inputs × 8     (int64 H3 constants)
bytes_input_order  = input_bits × 4            (int32 indices)
mem_bytes          = ceil(bits_per_submodel / 8) + bytes_h3 + bytes_input_order
                      summed across all submodels.
```
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


# ===========================================================================
# H3 universal hashing — vectorised PyTorch port of ULEEN's libtorch C++
# ===========================================================================

def _h3_hash(x_bits: torch.Tensor, hash_values: torch.Tensor) -> torch.Tensor:
    """Apply H3 universal hashing to binary inputs.

    Args:
        x_bits: (N, b) tensor of binary 0/1 values (any int dtype works).
        hash_values: (h, b) tensor of int64 random constants.

    Returns:
        (N, h) tensor of int64 hash outputs.

    Mirrors `software_model/cpp/h3_hash.cpp` exactly:
      selected[b, n, h] = hash_values[h, b] if x_bits[n, b] else 0
      out[n, h] = XOR_b(selected[b, n, h])
    """
    x_long = x_bits.to(torch.int64)
    # einsum "hb, nb -> nbh" matches the reference's "hb,db->bdh" up to permute
    selected = torch.einsum('hb,nb->nbh', hash_values, x_long)
    out = torch.zeros((x_bits.size(0), hash_values.size(0)),
                      dtype=torch.int64, device=x_bits.device)
    # XOR-reduce along the b dimension; note we keep selected[:, b, :] -> shape (n, h)
    for b in range(x_bits.size(1)):
        out = out ^ selected[:, b, :]
    return out


def _generate_h3_values(filter_inputs: int, filter_entries: int,
                        n_hash: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random H3 constants drawn uniformly from [0, filter_entries)."""
    assert (filter_entries & (filter_entries - 1)) == 0, "filter_entries must be a power of 2"
    if generator is None:
        return torch.from_numpy(np.random.randint(0, filter_entries,
                                                   (n_hash, filter_inputs))).long()
    return torch.randint(0, filter_entries, (n_hash, filter_inputs),
                         generator=generator, dtype=torch.int64)


# ===========================================================================
# Straight-Through Estimator — ULEEN's binarisation
# ===========================================================================

class _STEBinarize(torch.autograd.Function):
    """sign(x): forward outputs -1 or +1, backward is identity."""

    @staticmethod
    def forward(ctx, x):
        return ((x >= 0).float() * 2) - 1

    @staticmethod
    def backward(ctx, grad_outp):
        return grad_outp


# ===========================================================================
# Single ULEEN submodel
# ===========================================================================

class BackpropWiSARD(nn.Module):
    """One ULEEN submodel — `classes` continuous-Bloom-filter discriminators."""

    def __init__(self, n_inputs: int, n_classes: int,
                 filter_inputs: int, filter_entries: int,
                 filter_hash_functions: int, dropout_p: float = 0.0,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        assert filter_inputs > 0 and filter_entries > 0 and filter_hash_functions > 0

        self.n_inputs = int(n_inputs)
        self.n_classes = int(n_classes)
        self.filter_inputs = int(filter_inputs)
        self.filter_entries = int(filter_entries)
        self.filter_hash_functions = int(filter_hash_functions)

        input_bits = int(math.ceil(n_inputs / filter_inputs) * filter_inputs)
        self.null_bits = input_bits - n_inputs
        self.filters_per_discriminator = input_bits // filter_inputs

        # Continuous Bloom-filter table — the only trainable parameter
        self.table = nn.Parameter(torch.empty(self.n_classes,
                                               self.filters_per_discriminator,
                                               self.filter_entries))
        nn.init.uniform_(self.table, -1, 1)

        # Random H3 constants (frozen)
        self.register_buffer("hash_values", _generate_h3_values(
            filter_inputs, filter_entries, filter_hash_functions, generator=generator,
        ))
        # Random input ordering (frozen)
        if generator is None:
            order = torch.from_numpy(np.random.permutation(input_bits)).long()
        else:
            order = torch.randperm(input_bits, generator=generator)
        self.register_buffer("input_order", order)

        # Pruning state — bias is non-trainable by default; flips on once
        # we apply the post-pruning bias-learning step (not implemented here;
        # we keep the structure for forward-API parity with the reference).
        self.pruned = False
        self.register_buffer("mask", torch.ones(self.n_classes,
                                                  self.filters_per_discriminator))
        self.bias = nn.Parameter(torch.zeros(self.n_classes), requires_grad=False)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def get_filter_responses(self, x_b: torch.Tensor, raw: bool = False) -> torch.Tensor:
        """For each input compute the (classes × filters × hashes) lookup tensor."""
        batch_size = x_b.shape[0]
        padded = nn.functional.pad(x_b, (0, self.null_bits))
        mapped = padded[:, self.input_order]
        hash_inputs = mapped.view(batch_size * self.filters_per_discriminator,
                                  self.filter_inputs)
        hash_outputs = _h3_hash(hash_inputs, self.hash_values)

        # Reshape and gather, mirroring the reference's permute pattern
        flt = (hash_outputs
               .view(batch_size, self.filters_per_discriminator, self.filter_hash_functions)
               .permute(1, 0, 2)
               .reshape(1, self.filters_per_discriminator, -1)
               .expand(self.n_classes, -1, -1))
        flat_lookup = self.table.gather(2, flt)
        lookup = (flat_lookup
                  .view(self.n_classes, self.filters_per_discriminator,
                        batch_size, self.filter_hash_functions)
                  .permute(2, 0, 1, 3))

        if raw:
            return lookup
        # Binarise to ±1 then reduce across the hash-function dim with min
        bin_lookup = _STEBinarize.apply(lookup)
        reduced = bin_lookup.amin(axis=-1)
        return self.dropout(reduced)

    def forward(self, x_b: torch.Tensor) -> torch.Tensor:
        responses = self.get_filter_responses(x_b)
        if self.pruned:
            responses = responses * self.mask
        return responses.sum(axis=2) + self.bias

    def clamp(self) -> None:
        with torch.no_grad():
            self.table.clamp_(-1, 1)

    def prune_filters(self, keep_idxs: torch.Tensor) -> None:
        """Resize this submodel to keep only the filters at `keep_idxs` (uniform pruning).

        Mirrors the reference `prune_model.py` line 188-209 logic.
        """
        device = self.table.device
        keep_idxs = torch.as_tensor(keep_idxs, dtype=torch.long, device=device)
        new_input_order = torch.cat([
            self.input_order[self.filter_inputs * i : self.filter_inputs * (i + 1)]
            for i in keep_idxs.tolist()
        ])
        new_table_data = self.table.data[:, keep_idxs].clone()
        with torch.no_grad():
            self.input_order = new_input_order
            self.table = nn.Parameter(new_table_data, requires_grad=True)
            self.mask = torch.ones((self.n_classes, len(keep_idxs)), device=device)
            self.filters_per_discriminator = int(len(keep_idxs))

    def deployed_size_bytes(self, include_metadata: bool = False) -> int:
        """Post-binarisation footprint.

        ULEEN paper's `compute_model_size` (`train_model.py:163-173`) reports
        only the binarised Bloom-filter table, *excluding* H3 hash constants
        and the input-bit ordering. Our default mirrors the paper to keep
        our Pareto plots comparable; pass `include_metadata=True` to also
        count the H3 + input_order bytes (the deploy-time footprint with
        unfrozen metadata).
        """
        bits_table = self.n_classes * self.filters_per_discriminator * self.filter_entries
        size = math.ceil(bits_table / 8)
        if include_metadata:
            bytes_h3 = self.filter_hash_functions * self.filter_inputs * 8
            bytes_order = (self.n_inputs + self.null_bits) * 4
            size += bytes_h3 + bytes_order
        return size


# ===========================================================================
# Ensemble of submodels
# ===========================================================================

class BackpropMultiWiSARD(nn.Module):
    """ULEEN ensemble — `len(configs)` independently trained BackpropWiSARDs."""

    def __init__(self, n_inputs: int, n_classes: int,
                 configs: Sequence[tuple[int, int, int]],
                 dropout_p: float = 0.0,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.models = nn.ModuleList([
            BackpropWiSARD(n_inputs, n_classes,
                           filter_inputs=cfg[0], filter_entries=cfg[1],
                           filter_hash_functions=cfg[2],
                           dropout_p=dropout_p, generator=generator)
            for cfg in configs
        ])

    def forward(self, x_b: torch.Tensor) -> torch.Tensor:
        # Stack to (n_submodels, batch, classes)
        return torch.stack([m(x_b) for m in self.models])

    def clamp(self) -> None:
        for m in self.models:
            m.clamp()

    def deployed_size_bytes(self) -> int:
        return sum(m.deployed_size_bytes() for m in self.models)


# ===========================================================================
# Thermometer encoding (matches the simple linear encoding ULEEN uses by default)
# ===========================================================================

class _LinearThermometer:
    """Linear thermometer encoder — equally-spaced thresholds per feature.

    ULEEN's reference repo doesn't ship its preprocessing for non-shipped
    datasets; for our F4RM-style sweeps we use a per-feature linear
    thermometer (min/max from training data).
    """

    def __init__(self, num_bits: int = 3):
        assert num_bits > 0
        self.num_bits = int(num_bits)
        self.thresholds: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "_LinearThermometer":
        x = torch.as_tensor(x, dtype=torch.float32)
        mn = x.min(dim=0)[0]
        mx = x.max(dim=0)[0]
        # thresholds at (i+1)/(num_bits+1) of the (mn, mx) range
        steps = torch.arange(1, self.num_bits + 1, dtype=torch.float32)
        self.thresholds = mn.unsqueeze(-1) + steps.unsqueeze(0) * (
            (mx - mn) / (self.num_bits + 1)).unsqueeze(-1)
        return self

    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.thresholds is None:
            raise RuntimeError("LinearThermometer.binarize called before fit")
        x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(-1)
        return (x > self.thresholds).float()


class _GaussianThermometer:
    """Gaussian thermometer encoder — per-feature percentile thresholds based
    on a fitted normal distribution. Matches the ULEEN reference's
    `create_model` default (and BTHOWeN's `GaussianThermometer`).

    For an n-bit encoding, divides the Gaussian into (n+1) equal-probability
    regions; thresholds are the inter-region quantiles. This gives finer
    resolution near the data center than a linear (min/max) thermometer.
    """

    def __init__(self, num_bits: int = 3):
        assert num_bits > 0
        self.num_bits = int(num_bits)
        self.thresholds: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "_GaussianThermometer":
        x = torch.as_tensor(x, dtype=torch.float32)
        mu = x.mean(dim=0)
        sigma = x.std(dim=0).clamp_min(1e-6)
        # Inverse normal CDF at percentiles (i+1)/(num_bits+1)
        steps = torch.arange(1, self.num_bits + 1, dtype=torch.float32) / (self.num_bits + 1)
        # Φ⁻¹(p) using torch.erfinv: Φ⁻¹(p) = sqrt(2) * erfinv(2p - 1)
        zscores = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * steps - 1)
        # broadcast to (n_feat, num_bits)
        self.thresholds = mu.unsqueeze(-1) + zscores.unsqueeze(0) * sigma.unsqueeze(-1)
        return self

    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.thresholds is None:
            raise RuntimeError("GaussianThermometer.binarize called before fit")
        x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(-1)
        return (x > self.thresholds).float()


# ===========================================================================
# Public ULEEN classifier — convenience wrapper used by notebook_lib
# ===========================================================================

class ULEENClassifier:
    """ULEEN ensemble classifier with linear thermometer encoding.

    Mirrors the structure of ``train_model.train_model`` from the reference.
    Per-submodel cross-entropy losses are summed (so each submodel gets its
    own gradient signal); inference sums the per-submodel logits.

    Hyperparameters surfaced for sweeping:
        bits_per_input         thermometer bits per feature
        filter_inputs          n bits per Bloom filter (the "n" in n-tuple)
        filter_entries         Bloom filter table size in slots (power of 2)
        filter_hash_functions  number of H3 hashes per filter
        n_submodels            ensemble size
        dropout_p              Bernoulli dropout on per-RAM responses
        epochs, lr, batch_size, decay_lr   training-loop knobs
    """

    def __init__(self, bits_per_input: int = 3,
                 filter_inputs: int = 12, filter_entries: int = 64,
                 filter_hash_functions: int = 2, n_submodels: int = 3,
                 dropout_p: float = 0.0, epochs: int = 30, lr: float = 1e-2,
                 batch_size: int = 32, decay_lr: bool = True,
                 thermometer: str = "linear"):
        # Note on `lr`: the reference uses 1e-3 with ~100 epochs on MNIST
        # (`train_model.train_model`). For the F4RM Pareto sweep we use a
        # shorter 30-epoch schedule, so the default lr is bumped to 1e-2 to
        # converge in that budget. Pass `lr=1e-3, epochs=100` to reproduce
        # paper Table 2 numbers when CPU budget allows.
        self.bits_per_input = int(bits_per_input)
        self.filter_inputs = int(filter_inputs)
        self.filter_entries = int(filter_entries)
        self.filter_hash_functions = int(filter_hash_functions)
        self.n_submodels = int(n_submodels)
        self.dropout_p = float(dropout_p)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.decay_lr = bool(decay_lr)
        # Encoder choice: "linear" (default, current behavior) or "gaussian"
        # (paper Fig. 11 says +0.6pp on MNIST). Reference defaults to gaussian.
        if thermometer not in ("linear", "gaussian"):
            raise ValueError(f"thermometer must be 'linear' or 'gaussian', got {thermometer!r}")
        self.thermometer_kind = thermometer
        self.thermometer: Optional[object] = None
        self.model: Optional[BackpropMultiWiSARD] = None
        self.classes_: list[str] = []

    def _binarise(self, X) -> torch.Tensor:
        bin_x = self.thermometer.binarize(torch.as_tensor(X, dtype=torch.float32))
        return bin_x.flatten(start_dim=1)

    def fit(self, X, y, seed: int = 0, verbose: bool = False) -> "ULEENClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        torch.manual_seed(seed)
        np.random.seed(seed)

        therm_cls = _GaussianThermometer if self.thermometer_kind == "gaussian" else _LinearThermometer
        self.thermometer = therm_cls(self.bits_per_input).fit(X)
        Xb = self._binarise(X)

        self.classes_ = sorted({str(v) for v in y.tolist()})
        cls_to_i = {c: i for i, c in enumerate(self.classes_)}
        y_i = torch.tensor([cls_to_i[str(v)] for v in y.tolist()], dtype=torch.long)
        nc = len(self.classes_)

        in_size = Xb.size(1)
        configs = [(self.filter_inputs, self.filter_entries,
                    self.filter_hash_functions)] * self.n_submodels
        self.model = BackpropMultiWiSARD(in_size, nc, configs,
                                          dropout_p=self.dropout_p)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = (torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=max(1, (self.epochs * 3) // 10),
                        gamma=0.1) if self.decay_lr else None)
        loss_fn = nn.CrossEntropyLoss()

        n_samples = Xb.size(0)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                model_results = self.model(Xb[idx])  # (n_submodels, batch, classes)
                # Reference loss: sum of per-submodel cross-entropy
                loss = sum(loss_fn(o, y_i[idx]) for o in model_results)
                loss.backward()
                optimizer.step()
                self.model.clamp()
            if scheduler is not None:
                scheduler.step()
            if verbose:
                print(f"  epoch {epoch+1}/{self.epochs}: loss={float(loss):.4f}")
        return self

    def predict(self, X) -> np.ndarray:
        Xb = self._binarise(np.asarray(X, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            results = self.model(Xb)
            # Sum per-submodel logits (matches reference inference)
            outputs = results.sum(axis=0)
            preds = outputs.argmax(dim=1).cpu().numpy()
        return np.array([self.classes_[p] for p in preds])

    def prune(self, X, y, ratio: float = 0.27, finetune_epochs: int = 5,
              uniform: bool = True, seed: int = 0) -> "ULEENClassifier":
        """Post-train pruning + bias-learning + fine-tune (paper §3.1.3).

        Mirrors `prune_model.py` from the reference repo:
          1. Score every RAM by `-Σ_train (responses with correct-class boosted by -(C-1))`
             — high score means RAM is hurting, low means helping.
          2. For uniform pruning, score per-RAM = max across classes.
          3. Drop the top-`ratio` highest-scoring RAMs from each submodel.
          4. (Skipped here for tabular speed: per-submodel bias retraining; we just
             fine-tune the whole pruned model to absorb the prune step.)
          5. Fine-tune the pruned model for `finetune_epochs` at the original LR/batch.

        Paper claims ~27% size reduction (`ratio=0.27`) with <1% acc drop.
        """
        if self.model is None:
            raise RuntimeError("prune() requires a trained model — call fit() first.")
        Xb = self._binarise(np.asarray(X, dtype=np.float32))
        cls_to_i = {c: i for i, c in enumerate(self.classes_)}
        y_i = torch.tensor([cls_to_i[str(v)] for v in y.tolist()], dtype=torch.long)
        nc = len(self.classes_)

        # ---- 1-2. Score each RAM per submodel ----
        self.model.eval()
        for submodel in self.model.models:
            scores = torch.zeros((nc, submodel.filters_per_discriminator))
            with torch.no_grad():
                for i in range(0, len(Xb), self.batch_size):
                    batch_x = Xb[i : i + self.batch_size]
                    batch_y = y_i[i : i + self.batch_size]
                    responses = submodel.get_filter_responses(batch_x)  # (batch, classes, filters)
                    # Boost correct class so it contributes -(C-1) instead of +1.
                    for j, lbl in enumerate(batch_y.tolist()):
                        responses[j][lbl] = responses[j][lbl] * -(nc - 1)
                    scores -= responses.sum(axis=0).cpu()
            # ---- 3. Pick filters to keep ----
            if uniform:
                per_filter_score = scores.amax(axis=0)  # max across classes
            else:
                per_filter_score = scores  # (classes, filters) — would need per-class masking
            prune_count = int(ratio * submodel.filters_per_discriminator)
            if prune_count <= 0 or prune_count >= submodel.filters_per_discriminator:
                continue
            prune_idxs = (-per_filter_score).topk(prune_count).indices.tolist()
            keep_idxs = [k for k in range(submodel.filters_per_discriminator) if k not in set(prune_idxs)]
            submodel.prune_filters(torch.tensor(keep_idxs, dtype=torch.long))

        # ---- 5. Fine-tune the pruned model briefly ----
        if finetune_epochs > 0:
            torch.manual_seed(seed + 1000)  # different from initial fit seed
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr * 0.1)  # smaller LR
            loss_fn = nn.CrossEntropyLoss()
            n_samples = Xb.size(0)
            for epoch in range(finetune_epochs):
                self.model.train()
                perm = torch.randperm(n_samples)
                for i in range(0, n_samples, self.batch_size):
                    idx = perm[i : i + self.batch_size]
                    optimizer.zero_grad()
                    model_results = self.model(Xb[idx])
                    loss = sum(loss_fn(o, y_i[idx]) for o in model_results)
                    loss.backward()
                    optimizer.step()
                    self.model.clamp()
        return self

    def deployed_size_bytes(self) -> int:
        return self.model.deployed_size_bytes()
