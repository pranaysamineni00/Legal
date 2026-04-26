import numpy as np
import pytest
from evaluation import compute_per_clause_metrics, tune_per_clause_thresholds


def test_precision_at_recall_threshold_basic():
    from evaluation import precision_at_recall_threshold
    # Perfect classifier: y_score == y_true
    y_true  = np.array([1, 1, 0, 0, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    p = precision_at_recall_threshold(y_true, y_score, recall_target=0.80)
    assert p == pytest.approx(1.0)


def test_precision_at_recall_threshold_returns_zero_when_unreachable():
    from evaluation import precision_at_recall_threshold
    # y_true has no positives at all — function must return 0.0
    y_true  = np.array([0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    p = precision_at_recall_threshold(y_true, y_score, recall_target=0.80)
    assert p == pytest.approx(0.0)


def test_tune_per_clause_thresholds_returns_dict_of_floats():
    from evaluation import tune_per_clause_thresholds
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 3))
    labels = rng.integers(0, 2, size=(100, 3)).astype(float)
    id_to_clause = {0: "A", 1: "B", 2: "C"}
    thresholds = tune_per_clause_thresholds(logits, labels, id_to_clause)
    assert set(thresholds.keys()) == {"A", "B", "C"}
    for v in thresholds.values():
        assert 0.0 < v < 1.0


def test_compute_aggregate_metrics_shapes():
    from evaluation import compute_aggregate_metrics
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 4))
    labels = rng.integers(0, 2, size=(50, 4)).astype(float)
    thresholds = {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5}
    id_to_clause = {0: "A", 1: "B", 2: "C", 3: "D"}
    metrics = compute_aggregate_metrics(logits, labels, thresholds, id_to_clause)
    for key in ("macro_f1", "micro_f1", "micro_precision", "micro_recall"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_compute_per_clause_metrics_columns():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(80, 3))
    labels = rng.integers(0, 2, size=(80, 3)).astype(float)
    # Ensure at least one positive per clause
    labels[:10, :] = 1.0
    id_to_clause = {0: "Governing Law", 1: "Non-Compete", 2: "Indemnification"}
    n_positives = {0: 20, 1: 15, 2: 10}
    thresholds = tune_per_clause_thresholds(logits, labels, id_to_clause)
    df = compute_per_clause_metrics(logits, labels, thresholds, id_to_clause, n_positives)
    required = {"clause_type", "precision", "recall", "f1",
                "precision_at_80_recall", "aupr", "n_positive_train", "threshold"}
    assert required.issubset(set(df.columns))
    assert len(df) == 3


def test_per_clause_aupr_in_range():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(60, 2))
    labels = rng.integers(0, 2, size=(60, 2)).astype(float)
    labels[:5, :] = 1.0
    id_to_clause = {0: "A", 1: "B"}
    thresholds = tune_per_clause_thresholds(logits, labels, id_to_clause)
    df = compute_per_clause_metrics(logits, labels, thresholds, id_to_clause, {0: 10, 1: 8})
    assert (df["aupr"] >= 0.0).all() and (df["aupr"] <= 1.0).all()


def test_per_clause_aupr_zero_when_no_positives():
    """Clause with no positive examples must return aupr=0.0."""
    logits = np.zeros((20, 2))
    labels = np.zeros((20, 2))  # clause 0 has no positives at all
    labels[:, 1] = 1.0          # clause 1 has all positives
    id_to_clause = {0: "Empty", 1: "Full"}
    thresholds = {"Empty": 0.5, "Full": 0.5}
    df = compute_per_clause_metrics(logits, labels, thresholds, id_to_clause, {0: 0, 1: 20})
    empty_row = df[df["clause_type"] == "Empty"]
    assert float(empty_row["aupr"].iloc[0]) == pytest.approx(0.0)
