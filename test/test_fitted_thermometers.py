from unittest import TestCase, main
import wisardpkg as wp


# Shared fit data: 2 features, 10 samples each
FIT_DATA = [
    [0.0, 10.0], [0.1, 8.0], [0.2, 6.0], [0.3, 5.0], [0.4, 4.0],
    [0.5, 3.0], [0.6, 2.0], [0.7, 1.5], [0.8, 1.0], [1.0, 0.5],
]
SAMPLE = [0.5, 3.0]


class DistributiveThermometerTestCase(TestCase):

    def test_build(self):
        try:
            dt = wp.DistributiveThermometer(thermometerSize=8)
            self.assertIsInstance(dt, wp.DistributiveThermometer)
        except (RuntimeError, TypeError):
            self.fail("DistributiveThermometer build failed")

    def test_fit_and_transform(self):
        dt = wp.DistributiveThermometer(thermometerSize=8)
        dt.fit(FIT_DATA)
        out = dt.transform(SAMPLE)
        # Output length = n_features * size
        self.assertEqual(len(out.list()), len(SAMPLE) * 8)

    def test_get_thresholds_after_fit(self):
        dt = wp.DistributiveThermometer(thermometerSize=8)
        dt.fit(FIT_DATA)
        th = dt.getThresholds()
        # 2 features → 2 threshold lists; each of length size-1 or size depending on impl
        self.assertEqual(len(th), 2)
        for per_feat in th:
            self.assertTrue(len(per_feat) > 0)

    def test_set_thresholds_roundtrip(self):
        dt = wp.DistributiveThermometer(thermometerSize=8)
        dt.fit(FIT_DATA)
        th = dt.getThresholds()
        out_before = dt.transform(SAMPLE).list()

        dt2 = wp.DistributiveThermometer(thermometerSize=8)
        dt2.setThresholds(th)
        out_after = dt2.transform(SAMPLE).list()

        self.assertSequenceEqual(out_before, out_after)


class GaussianThermometerTestCase(TestCase):

    def test_build_fit_transform(self):
        gt = wp.GaussianThermometer(thermometerSize=8)
        gt.fit(FIT_DATA)
        out = gt.transform(SAMPLE)
        self.assertEqual(len(out.list()), len(SAMPLE) * 8)

    def test_set_thresholds_roundtrip(self):
        gt = wp.GaussianThermometer(thermometerSize=8)
        gt.fit(FIT_DATA)
        th = gt.getThresholds()
        gt2 = wp.GaussianThermometer(thermometerSize=8)
        gt2.setThresholds(th)
        self.assertSequenceEqual(
            gt.transform(SAMPLE).list(),
            gt2.transform(SAMPLE).list(),
        )


class ExponentialThermometerTestCase(TestCase):

    def test_build_fit_transform(self):
        et = wp.ExponentialThermometer(thermometerSize=8)
        et.fit(FIT_DATA)
        out = et.transform(SAMPLE)
        self.assertEqual(len(out.list()), len(SAMPLE) * 8)

    def test_set_thresholds_roundtrip(self):
        et = wp.ExponentialThermometer(thermometerSize=8)
        et.fit(FIT_DATA)
        th = et.getThresholds()
        et2 = wp.ExponentialThermometer(thermometerSize=8)
        et2.setThresholds(th)
        self.assertSequenceEqual(
            et.transform(SAMPLE).list(),
            et2.transform(SAMPLE).list(),
        )


class StochasticThermometerTestCase(TestCase):

    def test_build_fit_transform(self):
        st = wp.StochasticThermometer(thermometerSize=8)
        st.fit(FIT_DATA)
        out = st.transform(SAMPLE)
        self.assertEqual(len(out.list()), len(SAMPLE) * 8)

    def test_set_thresholds_roundtrip(self):
        st = wp.StochasticThermometer(thermometerSize=8)
        st.fit(FIT_DATA)
        th = st.getThresholds()
        st2 = wp.StochasticThermometer(thermometerSize=8)
        st2.setThresholds(th)
        self.assertSequenceEqual(
            st.transform(SAMPLE).list(),
            st2.transform(SAMPLE).list(),
        )


if __name__ == "__main__":
    main(verbosity=2)
