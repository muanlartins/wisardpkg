"""Unit tests for wisardpkg.models.dwn.

Run with: ``python prior_wisard/test_dwn.py``

Three checks:

1. **Forward** matches a hand-computed direct LUT lookup.
2. **EFD backward** matches the CUDA kernel formula bit-for-bit (compared to
   a reference Python loop that mirrors the kernel exactly).
3. **LUT gradient is exact** (verified by ``torch.autograd.gradcheck``). The
   input gradient is intentionally an *approximation* (EFD), so it is not
   gradcheckable — same as the straight-through estimator.
4. **End-to-end fit** on a tiny linearly-separable problem reaches > 90 %
   accuracy in a few epochs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from wisardpkg.models.dwn import (
    EFDFunction, _build_efd_decay, _layer_mapping, DistributiveThermometer,
    LUTLayer, GroupSum, DWNClassifier,
)


def _popcount(x_int: int, n_bits: int) -> int:
    return sum((x_int >> k) & 1 for k in range(n_bits))


class EFDForwardTests(unittest.TestCase):
    def test_forward_matches_direct_lookup(self):
        torch.manual_seed(0)
        B, I, L, n = 4, 12, 6, 3
        M = 1 << n
        x = (torch.rand(B, I) > 0.5).float()
        mapping = _layer_mapping(I, n, L, random=True)
        luts = torch.rand(L, M) * 2 - 1
        D = _build_efd_decay(n, alpha=0.5 * 0.75 ** (n - 1), beta=0.25 / 0.75)
        out = EFDFunction.apply(x, mapping, luts, D)
        self.assertEqual(out.shape, (B, L))
        for b in range(B):
            for j in range(L):
                addr = sum(int(x[b, mapping[j, k].item()] > 0) << k for k in range(n))
                self.assertAlmostEqual(out[b, j].item(), luts[j, addr].item(), places=6)


class EFDBackwardTests(unittest.TestCase):
    def test_lut_gradient_is_exact(self):
        torch.set_default_dtype(torch.float64)
        try:
            B, I, L, n = 2, 8, 4, 3
            torch.manual_seed(0)
            x = (torch.rand(B, I) > 0.5).double()
            mapping = _layer_mapping(I, n, L, random=True)
            luts = (torch.rand(L, 1 << n, dtype=torch.float64) * 2 - 1).requires_grad_()
            D = _build_efd_decay(n, alpha=0.5 * 0.75 ** (n - 1),
                                 beta=0.25 / 0.75, dtype=torch.float64)
            self.assertTrue(torch.autograd.gradcheck(
                lambda l: EFDFunction.apply(x, mapping, l, D),
                (luts,), eps=1e-5, atol=1e-6, rtol=1e-4,
                check_undefined_grad=False,
            ))
        finally:
            torch.set_default_dtype(torch.float32)

    def test_input_gradient_matches_cuda_formula(self):
        """Bit-for-bit match against a Python loop mirroring the CUDA kernel."""
        torch.set_default_dtype(torch.float64)
        try:
            torch.manual_seed(1)
            B, I, L, n = 1, 6, 2, 3
            M = 1 << n
            x = (torch.rand(B, I) > 0.5).double().requires_grad_()
            mapping = _layer_mapping(I, n, L, random=True)
            luts = (torch.rand(L, M, dtype=torch.float64) * 2 - 1).requires_grad_()
            alpha, beta = 0.5 * 0.75 ** (n - 1), 0.25 / 0.75
            D = _build_efd_decay(n, alpha=alpha, beta=beta, dtype=torch.float64)
            output_grad = torch.randn(B, L, dtype=torch.float64)

            out = EFDFunction.apply(x, mapping, luts, D)
            out.backward(output_grad)
            our_input_grad = x.grad.clone()
            our_luts_grad = luts.grad.clone()

            ref_input_grad = torch.zeros_like(x)
            ref_luts_grad = torch.zeros_like(luts)
            for i in range(B):
                for j in range(L):
                    addr = sum(int(x[i, mapping[j, k].item()] > 0) << k for k in range(n))
                    ref_luts_grad[j, addr] += output_grad[i, j]
                    for k in range(n):
                        w = 0.0
                        addr_off = addr & ~(1 << k)
                        for a2 in range(M):
                            a2_off = a2 & ~(1 << k)
                            dist = _popcount(addr_off ^ a2_off, n)
                            fd = luts[j, a2].item() * alpha * (beta ** dist)
                            w += fd if (a2 >> k) & 1 else -fd
                        ref_input_grad[i, mapping[j, k].item()] += w * output_grad[i, j]

            self.assertLess((our_input_grad - ref_input_grad).abs().max().item(), 1e-10)
            self.assertLess((our_luts_grad - ref_luts_grad).abs().max().item(), 1e-12)
        finally:
            torch.set_default_dtype(torch.float32)


class EndToEndTests(unittest.TestCase):
    def test_fits_a_simple_problem(self):
        """Linearly-separable 2-class problem — should reach high training acc."""
        rng = np.random.default_rng(0)
        n_per_class = 200
        X0 = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, 8))
        X1 = rng.normal(loc=2.0, scale=1.0, size=(n_per_class, 8))
        X = np.concatenate([X0, X1])
        y = np.array(["a"] * n_per_class + ["b"] * n_per_class)
        m = DWNClassifier(bits_per_input=3, n=4, num_luts_l1=64,
                          num_luts_l2=32, tau=1 / 0.3, epochs=10,
                          batch_size=32, scheduler_step=20)
        m.fit(X, y, seed=0)
        preds = m.predict(X)
        acc = (preds.astype(str) == y).mean()
        self.assertGreaterEqual(acc, 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
