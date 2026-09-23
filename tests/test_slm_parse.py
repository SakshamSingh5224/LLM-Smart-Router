import unittest

from smartrouter.labels import HIGH, LOW
from smartrouter.slm_router import build_messages, complexity_to_tier, parse_decision


class ParseTests(unittest.TestCase):
    def test_valid_json(self):
        d = parse_decision('{"complexity": "simple", "confidence": 0.9}')
        self.assertTrue(d.parse_ok)
        self.assertEqual((d.tier, d.complexity, d.confidence), (LOW, "simple", 0.9))

    def test_complex_goes_high(self):
        self.assertEqual(parse_decision('{"complexity":"complex","confidence":1}').tier, HIGH)

    def test_medium_mapping_is_configurable(self):
        raw = '{"complexity":"medium","confidence":0.5}'
        self.assertEqual(parse_decision(raw, medium_tier=LOW).tier, LOW)
        self.assertEqual(parse_decision(raw, medium_tier=HIGH).tier, HIGH)

    def test_json_embedded_in_prose(self):
        d = parse_decision('Sure! {"complexity": "complex", "confidence": 0.7} hope that helps')
        self.assertTrue(d.parse_ok)
        self.assertEqual(d.tier, HIGH)

    def test_confidence_is_clamped(self):
        self.assertEqual(parse_decision('{"complexity":"simple","confidence":7}').confidence, 1.0)
        self.assertEqual(parse_decision('{"complexity":"simple","confidence":-3}').confidence, 0.0)

    def test_bad_confidence_defaults(self):
        self.assertEqual(parse_decision('{"complexity":"simple","confidence":"high"}').confidence, 0.5)

    def test_garbage_falls_back_to_high(self):
        for bad in ("", "not json", '{"complexity": "banana"}', "[]", None):
            d = parse_decision(bad)
            self.assertFalse(d.parse_ok, bad)
            self.assertEqual(d.tier, HIGH, bad)

    def test_long_query_is_truncated(self):
        msgs = build_messages("x" * 5000)
        self.assertLess(len(msgs[1]["content"]), 1500)
        self.assertEqual(msgs[0]["role"], "system")

    def test_complexity_to_tier(self):
        self.assertEqual(complexity_to_tier("simple"), LOW)
        self.assertEqual(complexity_to_tier("complex"), HIGH)


if __name__ == "__main__":
    unittest.main()
