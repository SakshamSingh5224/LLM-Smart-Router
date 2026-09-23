import unittest

from smartrouter.labels import HIGH, LOW, score_to_complexity, score_to_tier


class LabelTests(unittest.TestCase):
    def test_complexity_buckets(self):
        self.assertEqual(score_to_complexity(5), "simple")
        self.assertEqual(score_to_complexity(4), "medium")
        for s in (3, 2, 1):
            self.assertEqual(score_to_complexity(s), "complex")

    def test_tier_default_threshold(self):
        self.assertEqual(score_to_tier(5), LOW)
        self.assertEqual(score_to_tier(4), LOW)
        self.assertEqual(score_to_tier(3), HIGH)
        self.assertEqual(score_to_tier(1), HIGH)

    def test_tier_custom_threshold(self):
        self.assertEqual(score_to_tier(4, threshold=5), HIGH)
        self.assertEqual(score_to_tier(5, threshold=5), LOW)


if __name__ == "__main__":
    unittest.main()
