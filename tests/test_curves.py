import unittest

import numpy as np

from smartrouter.curves import (cost_quality_curve, curve_summary, metrics_at, pct_strong_for_quality,
                                pick_threshold, random_pct_strong_for_quality, threshold_for_pct_strong)


class CurveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(4000) < 0.3).astype(int)
        self.p_good = self.y * 0.8 + 0.1 + rng.random(4000) * 0.05
        self.p_rand = rng.random(4000)

    def test_endpoints(self):
        c = cost_quality_curve(self.p_good, self.y)
        self.assertEqual(c.iloc[0]["pct_strong"], 0.0)
        self.assertEqual(c.iloc[-1]["pct_strong"], 1.0)
        self.assertAlmostEqual(c.iloc[-1]["quality"], 1.0)
        self.assertAlmostEqual(c.iloc[0]["quality"], 1 - self.y.mean(), places=3)
        self.assertAlmostEqual(c.iloc[-1]["rel_cost"], 1.0)

    def test_good_router_beats_random_router(self):
        good = curve_summary(cost_quality_curve(self.p_good, self.y))
        rnd = curve_summary(cost_quality_curve(self.p_rand, self.y))
        self.assertGreater(good["apgr"], 0.8)
        self.assertAlmostEqual(rnd["apgr"], 0.5, delta=0.05)
        self.assertGreater(good["lift_area"], 0.05)
        self.assertAlmostEqual(rnd["lift_area"], 0.0, delta=0.02)

    def test_quality_is_monotone_in_pct_strong(self):
        c = cost_quality_curve(self.p_rand, self.y)
        self.assertTrue((np.diff(c["quality"].to_numpy()) >= -1e-12).all())

    def test_savings_vs_random(self):
        c = cost_quality_curve(self.p_good, self.y)
        ours = pct_strong_for_quality(c, 0.95)
        rand = random_pct_strong_for_quality(1 - self.y.mean(), 0.95)
        self.assertLess(ours, rand)

    def test_pick_threshold_meets_target_cheapest(self):
        c = cost_quality_curve(self.p_good, self.y)
        t = pick_threshold(c, 0.95)
        m = metrics_at(self.p_good, self.y, t)
        self.assertGreaterEqual(m["quality"], 0.95)
        # cheaper than sending everything to the strong tier
        self.assertLess(m["pct_strong"], 0.6)

    def test_pick_threshold_unreachable_target_falls_back_to_best_quality(self):
        c = cost_quality_curve(self.p_rand, self.y)
        t = pick_threshold(c, 1.5)
        self.assertAlmostEqual(metrics_at(self.p_rand, self.y, t)["quality"], 1.0)

    def test_threshold_for_pct_strong(self):
        t = threshold_for_pct_strong(self.p_rand, 0.25)
        self.assertAlmostEqual(float((self.p_rand > t).mean()), 0.25, delta=0.01)
        self.assertEqual(float((self.p_rand > threshold_for_pct_strong(self.p_rand, 0.0)).mean()), 0.0)
        self.assertEqual(float((self.p_rand > threshold_for_pct_strong(self.p_rand, 1.0)).mean()), 1.0)

    def test_metrics_confusion_adds_up(self):
        m = metrics_at(self.p_good, self.y, 0.5)
        self.assertEqual(m["tp"] + m["fp"] + m["fn"] + m["tn"], len(self.y))


if __name__ == "__main__":
    unittest.main()
