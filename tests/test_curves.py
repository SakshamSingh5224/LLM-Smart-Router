from smartrouter.curves import metrics_at, cost_quality_curve
import numpy as np

def test_metrics_calculation():
    p_strong = np.array([0.8, 0.2, 0.9, 0.1])
    y_true = np.array([1, 0, 1, 0])
    m = metrics_at(p_strong, y_true, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["cost_savings"] > 0.0

def test_cost_quality_curve_shape():
    p_strong = np.array([0.8, 0.2])
    y_true = np.array([1, 0])
    df = cost_quality_curve(p_strong, y_true)
    assert len(df) == 101
