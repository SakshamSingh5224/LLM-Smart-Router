from smartrouter.curves import metrics_at, cost_quality_curve
import numpy as np

def test_metrics_calculation():
    m = metrics_at(np.array([0.8, 0.2, 0.9, 0.1]), np.array([1, 0, 1, 0]), threshold=0.5)
    assert m["accuracy"] == 1.0
