import unittest

from smartrouter.features import FEATURE_NAMES, describe_signals, handcrafted_features, handcrafted_matrix
from smartrouter.rules import rule_route

WEIRD = ["", "   ", "🙂" * 300, "请解释量子纠缠", "x" * 100_000, "hello\x00world", "?!?!", "\n\n\n"]


class FeatureTests(unittest.TestCase):
    def test_shape_and_finite(self):
        import numpy as np

        for t in WEIRD + ["hi", "def f(x): return x"]:
            v = handcrafted_features(t)
            self.assertEqual(v.shape, (len(FEATURE_NAMES),))
            self.assertTrue(np.isfinite(v).all())
        self.assertEqual(handcrafted_matrix([]).shape, (0, len(FEATURE_NAMES)))
        self.assertEqual(handcrafted_matrix(["a", "b"]).shape, (2, len(FEATURE_NAMES)))

    def test_signals(self):
        self.assertIn("contains code", describe_signals("```python\nprint(1)\n```"))
        self.assertIn("math/formal reasoning", describe_signals("please prove this theorem"))
        self.assertIn("greeting/small talk", describe_signals("hello"))


class RuleTests(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(rule_route("hi").tier, "low")
        self.assertEqual(rule_route("What is the capital of France?").tier, "low")
        self.assertEqual(rule_route("Implement and prove correctness ```def f(x): return x```").tier, "high")

    def test_rules_never_crash(self):
        for t in WEIRD:
            d = rule_route(t)
            self.assertIn(d.tier, ("low", "high"))
            self.assertTrue(0.0 <= d.score <= 1.0)


if __name__ == "__main__":
    unittest.main()
