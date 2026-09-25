import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

try:
    import sklearn  # noqa: F401
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


@unittest.skipUnless(HAVE_SK, "scikit-learn not installed")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from synth import make_df
        from smartrouter.classifier_router import ClassifierRouter
        from smartrouter.training import train_router

        cls.train, cls.val, cls.test = make_df(1500, 1), make_df(500, 2), make_df(500, 3)
        cls.artifact, cls.out = train_router(
            cls.train, cls.val, backends=("tfidf",), Cs=(1.0,),
            featurizer_opts={"tfidf": {"min_df": 1}}, log=lambda *_: None,
        )
        cls.router = ClassifierRouter(cls.artifact)

    def test_val_auc_is_high_on_learnable_data(self):
        self.assertGreater(self.artifact["val_auc"], 0.85)

    def test_thresholds_are_ordered_by_strictness(self):
        t = self.artifact["thresholds"]
        # stricter quality target -> lower threshold -> more strong-tier calls
        self.assertGreaterEqual(t["aggressive"], t["balanced"])
        self.assertGreaterEqual(t["balanced"], t["conservative"])

    def test_route_easy_and_hard(self):
        easy = self.router.route("hi there", mode="aggressive")
        hard = self.router.route("derive the equation for graphs and solve it step by step with a proof")
        self.assertEqual(easy.tier, "low")
        self.assertEqual(hard.tier, "high")
        self.assertEqual(hard.source, "classifier")

    def test_modes_move_cost(self):
        qs = self.test["prompt"].tolist()
        p = self.router.predict_proba(qs)
        share = {m: float((p > t).mean()) for m, t in self.router.modes.items()}
        self.assertLessEqual(share["aggressive"], share["balanced"] + 1e-9)
        self.assertLessEqual(share["balanced"], share["conservative"] + 1e-9)

    def test_arbitrary_input_never_crashes(self):
        for q in ["", "   ", "🙂" * 500, "请解释量子纠缠", "x" * 100_000, "hello\x00world", "?!?!", "\n\n\n"]:
            d = self.router.route(q, explain=True)
            self.assertIn(d.tier, ("low", "high"))
            self.assertTrue(0.0 <= d.confidence <= 1.0)
            self.assertIsNotNone(d.reasoning)

    def test_unknown_mode_raises_and_threshold_override_works(self):
        with self.assertRaises(ValueError):
            self.router.route("hi", mode="nope")
        self.assertEqual(self.router.route("hi there", threshold=1.0).tier, "low")
        self.assertEqual(self.router.route("hi there", threshold=0.0).tier, "high")

    def test_classifier_failure_falls_back_to_rules_and_opens_breaker(self):
        from smartrouter.classifier_router import CircuitBreaker, ClassifierRouter

        class Boom:
            def transform(self, texts):
                raise RuntimeError("boom")

        art = dict(self.artifact, featurizer=Boom())
        r = ClassifierRouter(art, breaker=CircuitBreaker(failure_threshold=3, reset_after_s=60))
        for _ in range(3):
            d = r.route("design a distributed architecture for payments and analyze the trade-offs "
                        "in detail with a table, then prove the consistency guarantees hold")
            self.assertEqual(d.source, "rules_fallback")
            self.assertEqual(d.tier, "high")
        self.assertTrue(r.breaker.is_open)
        self.assertEqual(r.route("hi").source, "rules_fallback")  # short-circuited

    def test_min_confidence_escalates_uncertain_low_to_high(self):
        from smartrouter.classifier_router import ClassifierRouter

        r = ClassifierRouter(self.artifact, min_confidence=1.01)  # everything "uncertain"
        self.assertEqual(r.route("hi there").tier, "high")

    def test_artifact_roundtrip(self):
        import joblib
        from smartrouter.classifier_router import ClassifierRouter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "router.joblib")
            joblib.dump(self.artifact, path, compress=3)
            r2 = ClassifierRouter.load(path)
            a = self.router.predict_proba(["design a distributed architecture for queues"])[0]
            b = r2.predict_proba(["design a distributed architecture for queues"])[0]
            self.assertAlmostEqual(float(a), float(b), places=6)


if __name__ == "__main__":
    unittest.main()
