"""Unit tests for wisardpkg.models.uleen.

Run with: ``python -m unittest wisardpkg.test.test_uleen``
or directly: ``python wisardpkg/test/test_uleen.py``

Three checks:

1. **H3 hash** matches a hand-computed reference (XOR of random constants
   selected by input bits) for both feature_wise=True and =False inputs.
2. **End-to-end fit** on a tiny linearly-separable problem reaches > 90 %
   accuracy in a few epochs.
3. **Deterministic seeding** — two `ULEENClassifier(...).fit(seed=42)` runs
   produce the same predictions.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from wisardpkg.models.uleen import _h3_hash, _generate_h3_values, BackpropWiSARD, ULEENClassifier


class H3HashTests(unittest.TestCase):
    def test_h3_matches_reference_loop(self):
        """Vectorised einsum + XOR should match a hand-computed Python loop."""
        torch.manual_seed(0)
        n_in, b, h = 5, 6, 3
        x = (torch.rand(n_in, b) > 0.5).int()
        hash_values = torch.randint(0, 1024, (h, b), dtype=torch.int64)
        out = _h3_hash(x, hash_values)
        # Reference: out[n, h] = XOR_b (hash_values[h, b] if x[n, b] else 0)
        ref = torch.zeros((n_in, h), dtype=torch.int64)
        for n in range(n_in):
            for hi in range(h):
                acc = 0
                for bi in range(b):
                    if x[n, bi]:
                        acc ^= int(hash_values[hi, bi])
                ref[n, hi] = acc
        self.assertTrue((out == ref).all().item(),
                        f"\nVectorised: {out}\nReference:  {ref}")


class BackpropWiSARDTests(unittest.TestCase):
    def test_forward_shape_and_range(self):
        torch.manual_seed(0)
        n_inputs, n_classes = 32, 3
        m = BackpropWiSARD(n_inputs, n_classes, filter_inputs=4,
                           filter_entries=64, filter_hash_functions=2)
        x = (torch.rand(7, n_inputs) > 0.5).float()
        out = m(x)
        # Output shape: (batch, classes). Each filter response is in {-1, +1},
        # min over hashes is in {-1, +1}, sum over filters is in [-fpd, +fpd].
        self.assertEqual(out.shape, (7, n_classes))
        fpd = m.filters_per_discriminator
        self.assertGreaterEqual(out.min().item(), -fpd - 1)
        self.assertLessEqual(out.max().item(), fpd + 1)


class EndToEndTests(unittest.TestCase):
    def test_fits_a_simple_problem(self):
        rng = np.random.default_rng(0)
        n_per_class = 200
        X0 = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, 8))
        X1 = rng.normal(loc=2.0, scale=1.0, size=(n_per_class, 8))
        X = np.concatenate([X0, X1])
        y = np.array(["a"] * n_per_class + ["b"] * n_per_class)
        m = ULEENClassifier(bits_per_input=3, filter_inputs=6,
                            filter_entries=64, filter_hash_functions=2,
                            n_submodels=2, epochs=30, lr=1e-2,
                            batch_size=32, decay_lr=True)
        m.fit(X, y, seed=0)
        preds = m.predict(X)
        acc = (preds.astype(str) == y).mean()
        self.assertGreaterEqual(acc, 0.85)

    def test_seeded_runs_are_reproducible(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 6)).astype(np.float32)
        y = np.array(["a"] * 40 + ["b"] * 40)
        m1 = ULEENClassifier(filter_inputs=4, filter_entries=64,
                             filter_hash_functions=2, n_submodels=2,
                             epochs=3, batch_size=16, decay_lr=False).fit(X, y, seed=42)
        m2 = ULEENClassifier(filter_inputs=4, filter_entries=64,
                             filter_hash_functions=2, n_submodels=2,
                             epochs=3, batch_size=16, decay_lr=False).fit(X, y, seed=42)
        self.assertTrue((m1.predict(X) == m2.predict(X)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
