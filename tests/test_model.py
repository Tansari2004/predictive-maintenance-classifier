import numpy as np

from src.train import metrics_at_threshold, select_threshold


def test_threshold_respects_false_positive_constraint():
    y = np.array([0, 0, 0, 0, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.20, 0.40, 0.60, 0.90])
    threshold = select_threshold(y, probabilities, max_false_positive_rate=0.25)
    metrics = metrics_at_threshold(y, probabilities, threshold)
    assert metrics["false_positive_rate"] <= 0.25
    assert metrics["recall"] == 1.0


def test_metrics_include_all_confusion_matrix_cells():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9])
    metrics = metrics_at_threshold(y, probabilities, 0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
