from smartrouter.synth import generate_synthetic_data
from smartrouter.training import train_router
from smartrouter.classifier_router import ClassifierRouter

def test_end_to_end_pipeline(tmp_path):
    df_train = generate_synthetic_data(40)
    df_val = generate_synthetic_data(20)

    artifact, _ = train_router(df_train, df_val, backends=["tfidf"], Cs=[1.0])
    router = ClassifierRouter(artifact)

    decision = router.route("Compute factorial for 10")
    assert decision.tier in ("low", "high")
    assert decision.source == "classifier"
