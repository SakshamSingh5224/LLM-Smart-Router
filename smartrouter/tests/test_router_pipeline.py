from smartrouter.synth import generate_synthetic_data
from smartrouter.training import train_router
from smartrouter.classifier_router import ClassifierRouter

def test_end_to_end_pipeline():
    df_train, df_val = generate_synthetic_data(40), generate_synthetic_data(20)
    artifact, _ = train_router(df_train, df_val, backends=["tfidf"], Cs=[1.0])
    decision = ClassifierRouter(artifact).route("Compute factorial for 10")
    assert decision.source == "classifier"
