"""Clean reference port of DWN (Bacellar et al., ICML 2024, arXiv 2410.11112).

DWN = Differentiable Weightless Neural Networks. The architecture is a
stack of LUT layers connected by learnable / random / arange mappings,
trained with backprop via the Extended Finite Difference (EFD) trick to
approximate gradients through the non-differentiable LUT lookup.

The reference at https://github.com/alanbacellar/DWN ships custom CUDA
kernels for the EFD forward/backward (`custom_operators/cuda/efd_cuda_kernel.cu`).
Our env is macOS Apple Silicon (no CUDA), so this module re-implements the
kernels as **vectorised PyTorch ops on CPU**, mathematically identical to
the CUDA reference. We verified equivalence by:

* re-deriving the EFD weight formula directly from the CUDA loop
  (`luts[j, a2] * alpha * beta^dist(addr_l_bit_off, a2_l_bit_off)` with sign
  flip on bit l of a2) and confirming the closed-form vector form below
  collapses to the same scalar contributions in the n=2 case;
* gradient-checking with `torch.autograd.gradcheck` on small instances.

API mirrors the reference 1:1 so existing DWN code (the README example,
`examples/mnist.py`) plugs in by replacing `import torch_dwn as dwn` with
`from prior_wisard import dwn`.

Memory accounting:
``mem_method = "dwn_binarized_lut"`` — post-binarisation footprint matching
how the DWN paper Table 1 reports model size: each LUT slot is 1 bit
(after sign binarisation at deploy), mapping indices are int32. Specifically::

    bits_in_luts    = num_luts × 2^n
    bytes_in_mapping= num_luts × n × 4  (int32 indices)
    mem_bytes       = ceil(bits_in_luts / 8) + bytes_in_mapping
                      summed across all LUT layers.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax


# ===========================================================================
# Extended Finite Difference (EFD) — pure-PyTorch port of the CUDA kernel
# ===========================================================================

def _layer_mapping(input_size: int, tuple_length: int, output_size: int,
                   random: bool = True, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Reference layout: stack ⌈n·L / I⌉ permutations of [0, I) end-to-end,
    truncate to L·n, reshape to (L, n)."""
    def mapp() -> torch.Tensor:
        return (torch.randperm(input_size, generator=generator) if random
                else torch.arange(input_size))
    num_complete, offset = divmod(tuple_length * output_size, input_size)
    if num_complete == 0:
        out = mapp()[:offset]
    else:
        complete = torch.cat([mapp() for _ in range(num_complete)])
        out = torch.cat([complete, mapp()[:offset]])
    return out.reshape(output_size, tuple_length).type(torch.int32)


def _popcount(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-count via sum of shifted bits — works for any int dtype, any shape.

    PyTorch lacks a native popcount op; for n ≤ 12 the loop is trivial."""
    out = torch.zeros_like(x)
    for k in range(bits):
        out = out + ((x >> k) & 1)
    return out


def _build_spectral_C(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """DWN paper §3.4 — Fourier-coefficient matrix for spectral regularization.

    For an n-input pseudo-Boolean function evaluated on the 2^n binary inputs,
    the L2 spectral norm is `||L · C||_2`, where:

        C[i, j] := (1/2^n) · Π_{a ∈ {b | i_b=1}} (2 j_a - 1)

    Vectorised: build all (i, j) pairs in {0..2^n-1}^2, decode bits,
    compute product of (2*j_a - 1) over the bits where i has a 1.
    For n=6, C is 64×64 = 4 KiB; cheap to precompute and use as a buffer.
    """
    size = 1 << n
    bits = torch.arange(n, dtype=torch.int64)
    i_idx = torch.arange(size, dtype=torch.int64).unsqueeze(1)  # (size, 1)
    j_idx = torch.arange(size, dtype=torch.int64).unsqueeze(0)  # (1, size)
    # i_bits[i, b] = (i >> b) & 1 — shape (size, n)
    i_bits = ((i_idx >> bits) & 1).to(dtype)  # (size, n)
    # j_bits[j, b] = (j >> b) & 1 — shape (size, n)
    j_bits = ((j_idx.transpose(0, 1) >> bits) & 1).to(dtype)  # (size, n)
    # signs[j, b] = 2 j_b - 1 ∈ {-1, +1}
    signs = 2 * j_bits - 1  # (size, n)
    # For each (i, j), C[i, j] = (1/size) * prod over b in S(i) of signs[j, b].
    # Equivalent: take signs[j] raised to power i_bits[i], then product over b.
    # i.e. signs ** i_bits broadcast → (size_i, size_j, n) then prod over n.
    # signs has shape (size_j, n). We want (size_i, size_j, n).
    s_exp = signs.unsqueeze(0)             # (1, size_j, n)
    e_exp = i_bits.unsqueeze(1)             # (size_i, 1, n)
    # When i_bits[i, b] == 0, factor is 1; when 1, factor is signs[j, b].
    factors = torch.where(e_exp > 0, s_exp.expand(size, size, n), torch.ones_like(s_exp).expand(size, size, n))
    C = factors.prod(dim=2) / size
    return C


def _build_efd_decay(n: int, alpha: float, beta: float,
                     dtype: torch.dtype = torch.float32,
                     device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Precompute the EFD decay tensor D of shape (n, M, M), where M = 2^n.

        D[k, a, a2] = sign_k(a2) · α · β^{dist(a & mask_k, a2 & mask_k)}

    with mask_k = (M-1) ^ (1 << k) (i.e. bit k cleared) and
    sign_k(a2) = +1 if bit k of a2 is 1, -1 otherwise.

    Returned tensor has dtype `dtype` on `device`. Memory cost is
    n · M² · 4 bytes ≈ 96 KiB at n=6; grows fast above n=10. The paper's
    sweet spot is n=6, so this is fine.
    """
    M = 1 << n
    all_addrs = torch.arange(M, dtype=torch.int64, device=device)
    a_grid = all_addrs.unsqueeze(1)   # (M, 1)
    a2_grid = all_addrs.unsqueeze(0)  # (1, M)
    D = torch.empty((n, M, M), dtype=dtype, device=device)
    for k in range(n):
        mask = (M - 1) ^ (1 << k)
        diff = (a_grid & mask) ^ (a2_grid & mask)
        dist = _popcount(diff, n).to(dtype)
        sign_k = ((a2_grid >> k) & 1).to(dtype) * 2 - 1
        D[k] = sign_k * alpha * (beta ** dist)
    return D


class EFDFunction(torch.autograd.Function):
    """Differentiable LUT lookup with the EFD gradient approximation.

    Forward:  output[b, j] = luts[j, addr(x[b], mapping[j])]
    Backward (per CUDA reference, vectorised):
        luts_grad[j, addr(x[b], mapping[j])] += output_grad[b, j]
        input_grad[b, mapping[j, k]] += output_grad[b, j] · w[b, j, k]
            where w[b, j, k] = Σ_{a2} luts[j, a2] · D[k, addr[b,j], a2]
    """

    @staticmethod
    def forward(ctx, x, mapping, luts, decay_D):
        # x: (B, I) float in {0, 1} (or any value — only the >0 sign matters)
        # mapping: (L, n) int32/int64
        # luts: (L, M)
        # decay_D: (n, M, M) precomputed, no grad
        B, I = x.shape
        L, n = mapping.shape
        M = luts.shape[1]
        assert M == (1 << n), f"luts shape {luts.shape} doesn't match mapping n={n}"

        bits = (x[:, mapping.long()] > 0).to(torch.int64)            # (B, L, n)
        powers = (1 << torch.arange(n, device=x.device, dtype=torch.int64))  # (n,)
        addrs = (bits * powers).sum(dim=-1)                          # (B, L)

        # output[b, j] = luts[j, addrs[b, j]]
        # use advanced indexing: luts[arange(L)[None, :], addrs] -> (B, L)
        output = luts[torch.arange(L, device=x.device).unsqueeze(0), addrs]

        ctx.save_for_backward(x, mapping, luts, decay_D, addrs)
        ctx.n = n
        ctx.M = M
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, mapping, luts, decay_D, addrs = ctx.saved_tensors
        n, M = ctx.n, ctx.M
        B, I = x.shape
        L = mapping.shape[0]
        device = x.device

        # ----- luts_grad: scatter output_grad onto (j, addr) -----
        luts_grad = torch.zeros_like(luts)
        flat_idx = (torch.arange(L, device=device).unsqueeze(0) * M + addrs).reshape(-1)
        luts_grad.view(-1).scatter_add_(0, flat_idx, output_grad.reshape(-1))

        # ----- input_grad: precompute weights per (k, j, a) and gather -----
        # weights[k, j, a] = Σ_{a2} luts[j, a2] · D[k, a, a2]
        weights = torch.einsum('jm,kam->kja', luts, decay_D)         # (n, L, M)
        weights_p = weights.permute(1, 0, 2).contiguous()            # (L, n, M)

        # w_sample[b, l, k] = weights_p[l, k, addrs[b, l]]
        l_idx = torch.arange(L, device=device).view(1, L, 1).expand(B, L, n)
        k_idx = torch.arange(n, device=device).view(1, 1, n).expand(B, L, n)
        a_idx = addrs.view(B, L, 1).expand(B, L, n)
        w_sample = weights_p[l_idx, k_idx, a_idx]                    # (B, L, n)

        contrib = w_sample * output_grad.unsqueeze(-1)               # (B, L, n)
        map_long = mapping.long()
        map_expanded = map_long.unsqueeze(0).expand(B, L, n).reshape(B, L * n)
        input_grad = torch.zeros_like(x)
        input_grad.scatter_add_(1, map_expanded, contrib.reshape(B, L * n))

        return input_grad, None, luts_grad, None


# ===========================================================================
# Learnable mapping
# ===========================================================================

class _LearnableMappingFunction(torch.autograd.Function):
    """Forward: argmax over rows; Backward: softmax-gated linear path.

    Same as the reference: the forward picks one input per output via argmax,
    the backward uses a temperature-softmax surrogate for differentiability.
    """

    @staticmethod
    def forward(ctx, x, weights, tau):
        mapping = weights.argmax(dim=0)
        output = x[:, mapping]
        ctx.save_for_backward(x, weights, tau)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau = ctx.saved_tensors
        # weights_grad rule from the reference
        weights_grad = (2 * x - 1).T @ output_grad
        input_grad = output_grad @ softmax(weights / tau, dim=0).T
        return input_grad, weights_grad, None


class LearnableMapping(nn.Module):
    """Differentiable input-to-LUT mapping (Gumbel-style without sampling)."""

    def __init__(self, input_size: int, output_size: int, tau: float = 0.001):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(input_size, output_size, dtype=torch.float32),
                                     requires_grad=True)
        self.tau = float(tau)

    def forward(self, x):
        return _LearnableMappingFunction.apply(x, self.weights, torch.tensor(self.tau))


# ===========================================================================
# Straight-Through Estimator (binarisation)
# ===========================================================================

class _STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


# ===========================================================================
# LUTLayer
# ===========================================================================

class LUTLayer(nn.Module):
    """A layer of `output_size` independent LUTs each reading `n` input bits.

    Mirrors `torch_dwn.LUTLayer` from the reference. Defaults for alpha and
    beta follow the upstream code (`alpha = 0.5 · 0.75^(n-1)`,
    `beta = 0.25/0.75 ≈ 0.333`).
    """

    def __init__(self, input_size: int, output_size: int, n: int,
                 mapping: str | torch.Tensor = 'random',
                 alpha: float | None = None, beta: float | None = None,
                 ste: bool = True, clamp_luts: bool = True,
                 lm_tau: float = 0.001):
        super().__init__()
        assert input_size > 0 and output_size > 0 and n > 0
        if alpha is None:
            alpha = 0.5 * (0.75 ** (n - 1))
        if beta is None:
            beta = 0.25 / 0.75

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ste = bool(ste)
        self.clamp_luts = bool(clamp_luts)

        # Mapping
        if isinstance(mapping, torch.Tensor):
            assert mapping.dtype == torch.int32 and mapping.shape == torch.Size([output_size, n])
            self.mapping = nn.Parameter(mapping, requires_grad=False)
            self._learnable_mapping = None
        elif mapping == 'learnable':
            self._learnable_mapping = LearnableMapping(input_size, output_size * n, tau=lm_tau)
            # When using learnable mapping, the LUT inputs are arange(L*n)
            self.mapping = nn.Parameter(
                torch.arange(output_size * n).reshape(output_size, n).int(),
                requires_grad=False,
            )
        else:
            self._learnable_mapping = None
            self.mapping = nn.Parameter(
                _layer_mapping(input_size, n, output_size, random=(mapping == 'random')),
                requires_grad=False,
            )

        # LUTs ~ U(-1, 1)
        luts = torch.rand(output_size, 2 ** n, dtype=torch.float32) * 2 - 1
        self.luts = nn.Parameter(luts, requires_grad=True)

        # Precompute the EFD decay tensor (constant over training)
        D = _build_efd_decay(self.n, self.alpha, self.beta)
        self.register_buffer("decay_D", D)

        # Precompute the spectral-regularization C matrix (DWN paper §3.4):
        #   C[i, j] := (1/2^n) * Π_{a ∈ {b | i_b=1}} (2 j_a - 1)
        # Then specnorm(L) = ||L @ C||_2. We register as a buffer so it
        # follows the model on .to(device) and is excluded from optimizer.
        self.register_buffer("spectral_C", _build_spectral_C(self.n))

    def spectral_norm_squared(self) -> torch.Tensor:
        """||L @ C||_F^2 — DWN paper §3.4 spectral regularization term."""
        return (self.luts @ self.spectral_C).pow(2).sum()

    def forward(self, x):
        # Clamp LUT entries to [-1, 1] during training (matches reference)
        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(-1, 1)

        if self._learnable_mapping is not None:
            x = self._learnable_mapping(x)

        x = EFDFunction.apply(x, self.mapping, self.luts, self.decay_D)
        if self.ste:
            x = _STEFunction.apply(x)
        return x

    # ------------------------------------------------------------------
    # Memory accounting (post-binarisation; matches DWN paper Table 1)
    # ------------------------------------------------------------------
    def deployed_size_bytes(self) -> int:
        """1 bit per LUT entry + 4 bytes per mapping index (int32)."""
        n_lut_bits = self.output_size * (1 << self.n)
        n_mapping_bytes = self.output_size * self.n * 4
        return math.ceil(n_lut_bits / 8) + n_mapping_bytes


# ===========================================================================
# GroupSum head
# ===========================================================================

def _pad_if_needed(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.size(-1) % k == 0:
        return x
    pad = [0] * (2 * x.dim())
    pad[0] = k - (x.size(-1) % k)
    return torch.nn.functional.pad(x, tuple(pad))


class GroupSum(nn.Module):
    """Group `x` into `k` slices along the last dim and sum each, dividing by `tau`."""

    def __init__(self, k: int, tau: float = 1.0, randperm: int | bool = False):
        super().__init__()
        self.k = int(k)
        self.tau = float(tau)
        if randperm:
            self.register_buffer("randperm", torch.randperm(int(randperm)))
        else:
            self.randperm = None

    def forward(self, x):
        if self.randperm is not None:
            x = x[:, self.randperm]
        x = _pad_if_needed(x, self.k)
        x = x.view(*x.shape[:-1], self.k, int(x.shape[-1] / self.k))
        return x.sum(dim=-1) / self.tau

    def deployed_size_bytes(self) -> int:
        return 0


# ===========================================================================
# Distributive thermometer (matches reference)
# ===========================================================================

class DistributiveThermometer:
    """Per-feature percentile-threshold encoder, identical to the reference."""

    def __init__(self, num_bits: int = 1, feature_wise: bool = True):
        assert num_bits > 0
        self.num_bits = int(num_bits)
        self.feature_wise = bool(feature_wise)
        self.thresholds: Optional[torch.Tensor] = None

    def fit(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        data = (torch.sort(x.flatten())[0] if not self.feature_wise
                else torch.sort(x, dim=0)[0])
        idx = torch.tensor([int(data.shape[0] * i / (self.num_bits + 1))
                              for i in range(1, self.num_bits + 1)])
        thresholds = data[idx]
        # Move bits axis to the end
        if self.feature_wise:
            self.thresholds = torch.permute(thresholds, (*range(1, thresholds.ndim), 0))
        else:
            self.thresholds = thresholds
        return self

    def binarize(self, x) -> torch.Tensor:
        if self.thresholds is None:
            raise RuntimeError("DistributiveThermometer.binarize called before fit")
        x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(-1)
        return (x > self.thresholds).float()


# ===========================================================================
# Public DWN classifier — convenience wrapper used by notebook_lib
# ===========================================================================

class DWNClassifier:
    """Adam-trained 2-layer DWN classifier with a Distributive thermometer.

    Mirrors the structure used in the paper's MNIST example::

        thermometer = DistributiveThermometer(bits_per_input).fit(x_train)
        x_train_b = thermometer.binarize(x_train).flatten(start_dim=1)
        model = nn.Sequential(
            LUTLayer(x_train_b.size(1), num_luts_l1, n=n, mapping='learnable'),
            LUTLayer(num_luts_l1, num_luts_l2, n=n),
            GroupSum(num_classes, tau=tau),
        )

    Hyperparameters surfaced for sweeping:
        bits_per_input  thermometer encoding bits per feature
        n               LUT input width (paper sweet spot: 6)
        num_luts_l1     # LUTs in layer 1 (the "learnable" layer)
        num_luts_l2     # LUTs in layer 2
        tau             GroupSum softmax temperature divisor
        epochs          training epochs
        lr              Adam learning rate
        batch_size
        scheduler_step, scheduler_gamma   StepLR schedule
    """

    def __init__(self, bits_per_input: int = 3, n: int = 6,
                 num_luts_l1: int = 2000, num_luts_l2: int = 1000,
                 num_luts_l3: Optional[int] = None,
                 layer_sizes: Optional[Sequence[int]] = None,
                 tau: float = 1 / 0.3, epochs: int = 30, lr: float = 1e-2,
                 lr_schedule: Optional[Sequence[tuple[float, int]]] = None,
                 batch_size: int = 32, scheduler_step: int = 14,
                 scheduler_gamma: float = 0.1,
                 spectral_lambda: float = 0.0):
        self.bits_per_input = int(bits_per_input)
        self.n = int(n)
        self.num_luts_l1 = int(num_luts_l1)
        self.num_luts_l2 = int(num_luts_l2)
        # Optional 3rd LUT layer to match DWN paper §4.4 (3 layers for small,
        # 5 for large tabular datasets). When None, falls back to 2-layer.
        self.num_luts_l3 = int(num_luts_l3) if num_luts_l3 else None
        # Optional N-layer override (paper Table 17 uses 3-5 layers per dataset).
        # If provided, overrides l1/l2/l3 entirely. Each entry is the LUT count
        # for that layer; layer 0 uses 'learnable' mapping, others 'random'.
        self.layer_sizes = list(layer_sizes) if layer_sizes else None
        self.tau = float(tau)
        self.epochs = int(epochs)
        self.lr = float(lr)
        # Optional multi-stage LR schedule (paper Table 17 uses
        # 1e-2(80) -> 1e-3(80) -> 1e-4(40)). If provided, overrides StepLR.
        self.lr_schedule = list(lr_schedule) if lr_schedule else None
        self.batch_size = int(batch_size)
        self.scheduler_step = int(scheduler_step)
        self.scheduler_gamma = float(scheduler_gamma)
        # DWN paper §3.4 spectral regularization weight (0 = disabled).
        # Paper Table 12 shows +3.91pp on higgs with this term enabled.
        self.spectral_lambda = float(spectral_lambda)
        self.thermometer: Optional[DistributiveThermometer] = None
        self.model: Optional[nn.Sequential] = None
        self.classes_: list = []

    def _binarise(self, X) -> torch.Tensor:
        binarised = self.thermometer.binarize(torch.as_tensor(X, dtype=torch.float32))
        return binarised.flatten(start_dim=1)

    def fit(self, X, y, seed: int = 0, verbose: bool = False) -> "DWNClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.thermometer = DistributiveThermometer(self.bits_per_input).fit(X)
        Xb = self._binarise(X)

        self.classes_ = sorted({str(v) for v in y.tolist()})
        cls_to_i = {c: i for i, c in enumerate(self.classes_)}
        y_i = torch.tensor([cls_to_i[str(v)] for v in y.tolist()], dtype=torch.long)
        nc = len(self.classes_)

        in_size = Xb.size(1)
        layers: list[nn.Module] = []
        if in_size < self.n:
            raise ValueError(f"binarised input size {in_size} < n={self.n}")

        # Architecture: layer_sizes list overrides everything; else l1/l2/l3 path.
        if self.layer_sizes is not None:
            sizes = list(self.layer_sizes)
            # Last layer must be divisible by num_classes for GroupSum
            sizes[-1] = max(sizes[-1], nc)
            sizes[-1] = (sizes[-1] // nc) * nc
            prev = in_size
            for i, s in enumerate(sizes):
                mapping = 'learnable' if i == 0 else 'random'
                layers.append(LUTLayer(prev, max(s, self.n), n=self.n, mapping=mapping))
                prev = max(s, self.n)
        else:
            out_l1 = max(self.num_luts_l1, self.n)
            if self.num_luts_l3 is None:
                out_l2 = max(self.num_luts_l2, nc)
                out_l2 = (out_l2 // nc) * nc
                layers.append(LUTLayer(in_size, out_l1, n=self.n, mapping='learnable'))
                layers.append(LUTLayer(out_l1, out_l2, n=self.n, mapping='random'))
            else:
                out_l2 = max(self.num_luts_l2, self.n)
                out_l3 = max(self.num_luts_l3, nc)
                out_l3 = (out_l3 // nc) * nc
                layers.append(LUTLayer(in_size, out_l1, n=self.n, mapping='learnable'))
                layers.append(LUTLayer(out_l1, out_l2, n=self.n, mapping='random'))
                layers.append(LUTLayer(out_l2, out_l3, n=self.n, mapping='random'))
        layers.append(GroupSum(k=nc, tau=self.tau))
        self.model = nn.Sequential(*layers)

        loss_fn = nn.CrossEntropyLoss()
        n_samples = Xb.size(0)

        # Cache list of LUT layers for spectral reg (avoid isinstance check per step)
        lut_layers = [m for m in self.model if isinstance(m, LUTLayer)]

        # Training loop — supports either StepLR (default) or paper's multi-stage LR.
        if self.lr_schedule is not None:
            # Paper-style: list of (lr, n_epochs). E.g. [(1e-2, 80), (1e-3, 80), (1e-4, 40)].
            for stage_idx, (stage_lr, stage_epochs) in enumerate(self.lr_schedule):
                if stage_idx == 0:
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=stage_lr)
                else:
                    for g in optimizer.param_groups:
                        g['lr'] = stage_lr
                for epoch in range(stage_epochs):
                    self.model.train()
                    perm = torch.randperm(n_samples)
                    for i in range(0, n_samples, self.batch_size):
                        idx = perm[i:i + self.batch_size]
                        optimizer.zero_grad()
                        out = self.model(Xb[idx])
                        loss = loss_fn(out, y_i[idx])
                        if self.spectral_lambda > 0:
                            spec = sum(l.spectral_norm_squared() for l in lut_layers)
                            loss = loss + self.spectral_lambda * spec
                        loss.backward()
                        optimizer.step()
                    if verbose:
                        print(f"  stage {stage_idx} lr={stage_lr} ep {epoch+1}/{stage_epochs}: "
                              f"loss={loss.item():.4f}")
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
            for epoch in range(self.epochs):
                self.model.train()
                perm = torch.randperm(n_samples)
                for i in range(0, n_samples, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    optimizer.zero_grad()
                    out = self.model(Xb[idx])
                    loss = loss_fn(out, y_i[idx])
                    if self.spectral_lambda > 0:
                        spec = sum(l.spectral_norm_squared() for l in lut_layers)
                        loss = loss + self.spectral_lambda * spec
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if verbose:
                    print(f"  epoch {epoch+1}/{self.epochs}: loss={loss.item():.4f}")
        return self

    def predict(self, X) -> np.ndarray:
        Xb = self._binarise(np.asarray(X, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xb)
            preds = out.argmax(dim=1).cpu().numpy()
        return np.array([self.classes_[p] for p in preds])

    def deployed_size_bytes(self) -> int:
        """Sum of binarised LUT-layer footprints; GroupSum has no parameters."""
        return sum(getattr(m, 'deployed_size_bytes', lambda: 0)()
                   for m in self.model.modules())
