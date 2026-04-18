from unittest import TestCase, main
import wisardpkg as wp


def _toy_dataset():
    """Small 2-class dataset used across hook tests."""
    X = wp.DataSet()
    data = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
    ]
    y = ["cold"] * 4 + ["hot"] * 4
    for row, lbl in zip(data, y):
        X.add(row, lbl)
    return X, y


class NegativeEvidenceTestCase(TestCase):
    """negativeEvidence penalizes a class by others' response; three modes."""

    def setUp(self):
        self.X, self.y = _toy_dataset()

    def test_uniform(self):
        wsd = wp.Wisard(
            3,
            negativeEvidence=True,
            negativeAlpha=0.5,
            negativeMode="uniform",
        )
        wsd.train(self.X)
        preds = wsd.classify(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_max_competitor(self):
        wsd = wp.Wisard(
            3,
            negativeEvidence=True,
            negativeAlpha=0.3,
            negativeMode="max_competitor",
        )
        wsd.train(self.X)
        preds = wsd.classify(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_normalized(self):
        wsd = wp.Wisard(
            3,
            negativeEvidence=True,
            negativeAlpha=0.2,
            negativeMode="normalized",
        )
        wsd.train(self.X)
        preds = wsd.classify(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_alpha_zero_matches_baseline(self):
        """negativeAlpha=0 should leave classification unchanged."""
        X, y = _toy_dataset()
        baseline = wp.Wisard(3)
        baseline.train(X)
        base_preds = baseline.classify(X)

        hooked = wp.Wisard(3, negativeEvidence=True, negativeAlpha=0.0, negativeMode="uniform")
        hooked.train(X)
        hooked_preds = hooked.classify(X)
        self.assertSequenceEqual(base_preds, hooked_preds)


class SharedDiscriminatorTestCase(TestCase):

    def test_build_and_run(self):
        X, y = _toy_dataset()
        wsd = wp.Wisard(3, sharedDiscriminator=True, sharedBeta=0.1)
        wsd.train(X)
        preds = wsd.classify(X)
        self.assertEqual(len(preds), len(y))

    def test_beta_zero_matches_baseline(self):
        X, y = _toy_dataset()
        baseline = wp.Wisard(3)
        baseline.train(X)
        hooked = wp.Wisard(3, sharedDiscriminator=True, sharedBeta=0.0)
        hooked.train(X)
        self.assertSequenceEqual(baseline.classify(X), hooked.classify(X))


class AttentionWeightingTestCase(TestCase):

    def test_build_and_run(self):
        X, y = _toy_dataset()
        wsd = wp.Wisard(3, attentionWeighting=True)
        wsd.train(X)
        preds = wsd.classify(X)
        self.assertEqual(len(preds), len(y))


class RAMWeightsTestCase(TestCase):

    def setUp(self):
        self.X, self.y = _toy_dataset()

    def _fresh(self):
        return wp.Wisard(3)

    def test_compute_and_get_weights_entropy(self):
        wsd = self._fresh()
        wsd.train(self.X)
        wsd.computeRAMWeights(self.X, "entropy")
        w = wsd.getRAMWeights()
        self.assertIn("cold", w)
        self.assertIn("hot", w)
        for _, vec in w.items():
            self.assertTrue(all(0.0 <= val <= 1.0 for val in vec))

    def test_compute_information_gain(self):
        wsd = self._fresh()
        wsd.train(self.X)
        wsd.computeRAMWeights(self.X, "information_gain")
        w = wsd.getRAMWeights()
        self.assertTrue(len(w) == 2)

    def test_compute_purity(self):
        wsd = self._fresh()
        wsd.train(self.X)
        wsd.computeRAMWeights(self.X, "purity")
        w = wsd.getRAMWeights()
        for _, vec in w.items():
            self.assertTrue(all(0.0 <= val <= 1.0 for val in vec))

    def test_set_weights_roundtrip(self):
        wsd = self._fresh()
        wsd.train(self.X)
        wsd.computeRAMWeights(self.X, "entropy")
        w = wsd.getRAMWeights()

        other = self._fresh()
        other.train(self.X)
        other.setRAMWeights(w)
        self.assertEqual(other.getRAMWeights(), w)

    def test_prune_respects_threshold(self):
        wsd = self._fresh()
        wsd.train(self.X)
        wsd.computeRAMWeights(self.X, "entropy")
        wsd.pruneRAMs(0.5)
        w = wsd.getRAMWeights()
        for _, vec in w.items():
            for val in vec:
                self.assertTrue(val == 0.0 or val >= 0.5)


class LargeAddressRAMTestCase(TestCase):
    """Address sizes > 64 use a string-keyed large_ram_t path in ram.cc."""

    def test_construct_and_train_large_address(self):
        addr = 72  # > 64 triggers useLargeAddr
        n_features = 10
        # Each sample is a 720-bit vector
        n_bits = n_features * addr
        import random
        random.seed(0)
        X = wp.DataSet()
        for i in range(30):
            row = [random.randint(0, 1) for _ in range(n_bits)]
            X.add(row, "A" if i % 2 == 0 else "B")

        wsd = wp.Wisard(addr)
        wsd.train(X)
        preds = wsd.classify(X)
        self.assertEqual(len(preds), 30)


if __name__ == "__main__":
    main(verbosity=2)
