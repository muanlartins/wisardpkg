from unittest import TestCase, main
import wisardpkg as wp


class BloomWisardTestCase(TestCase):
    """Smoke tests for the counting-Bloom-filter-backed WiSARD variant."""

    def setUp(self):
        self.X = wp.DataSet()
        data = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
        ]
        self.y = ["cold"] * 5 + ["hot"] * 4
        for row, label in zip(data, self.y):
            self.X.add(row, label)

    def test_build_default(self):
        try:
            bwsd = wp.BloomWisard(addressSize=3)
            self.assertIsInstance(bwsd, wp.BloomWisard)
        except TypeError:
            self.fail("BloomWisard default construction failed")

    def test_build_with_params(self):
        try:
            bwsd = wp.BloomWisard(addressSize=3, numBits=2048, numHashes=4)
            self.assertIsInstance(bwsd, wp.BloomWisard)
        except TypeError:
            self.fail("BloomWisard parameterized construction failed")

    def test_train_and_classify(self):
        bwsd = wp.BloomWisard(addressSize=3, numBits=1024, numHashes=3)
        bwsd.train(self.X)
        preds = bwsd.classify(self.X)
        self.assertEqual(len(preds), len(self.y))
        # Training accuracy should be close to 1 on this separable toy set
        correct = sum(1 for p, t in zip(preds, self.y) if p == t)
        self.assertGreaterEqual(correct / len(self.y), 0.7)

    def test_rank_returns_score_map(self):
        bwsd = wp.BloomWisard(addressSize=3)
        bwsd.train(self.X)
        scores = bwsd.rank(self.X[0])
        self.assertIn("cold", scores)
        self.assertIn("hot", scores)
        self.assertIsInstance(scores["cold"], int)

    def test_hash_mode_simhash(self):
        # SimHash is an alternative LSH-based hash mode; it must at least build and train.
        try:
            bwsd = wp.BloomWisard(
                addressSize=3, numBits=1024, numHashes=3, hashMode="simhash"
            )
            bwsd.train(self.X)
            preds = bwsd.classify(self.X)
            self.assertEqual(len(preds), len(self.y))
        except (RuntimeError, TypeError):
            self.fail("BloomWisard with hashMode='simhash' failed")

    def test_hash_mode_h3(self):
        # H3 universal hashing (Carter & Wegman 1979) — used by BTHOWeN.
        bwsd = wp.BloomWisard(
            addressSize=3, numBits=1024, numHashes=3, hashMode="h3"
        )
        bwsd.train(self.X)
        preds = bwsd.classify(self.X)
        self.assertEqual(len(preds), len(self.y))
        # On this separable toy set H3 should also classify correctly.
        correct = sum(1 for p, t in zip(preds, self.y) if p == t)
        self.assertGreaterEqual(correct / len(self.y), 0.7)
        # New accessors should report the configured values back.
        self.assertEqual(bwsd.getHashMode(), "h3")
        self.assertEqual(bwsd.getNumBits(), 1024)
        self.assertEqual(bwsd.getNumHashes(), 3)
        self.assertGreater(bwsd.getNumberOfRAMS(), 0)

    def test_get_raw_votes(self):
        # getRawVotes exposes per-RAM Bloom-filter min-counts per class —
        # used by BTHOWeN's bleach binary search.
        bwsd = wp.BloomWisard(addressSize=3, numBits=128, numHashes=2)
        bwsd.train(self.X)
        raw = bwsd.getRawVotes(self.X[0])
        # Returned dict has one entry per class with one int per RAM.
        self.assertIn("cold", raw)
        self.assertIn("hot", raw)
        n_rams = bwsd.getNumberOfRAMS()
        self.assertEqual(len(raw["cold"]), n_rams)
        self.assertEqual(len(raw["hot"]), n_rams)

    def test_reset(self):
        bwsd = wp.BloomWisard(addressSize=3)
        bwsd.train(self.X)
        bwsd.reset()
        # After reset, classifying should still work (empty Bloom filters → uniform scores)
        preds = bwsd.classify(self.X)
        self.assertEqual(len(preds), len(self.y))


if __name__ == "__main__":
    main(verbosity=2)
