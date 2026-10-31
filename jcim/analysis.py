"""Representation comparisons and leakage-safe linear readouts.

Similarity is descriptive: high CKA or an easily aligned representation does not
establish useful transfer. Compare trained models with random initializations
and use held-out readouts and adaptation experiments to test functional reuse.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _features(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not all(result.shape):
        raise ValueError(f"{name} must be a nonempty sample-by-feature matrix.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _paired_features(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _features(x, "x")
    y = _features(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Representations must describe the same samples in the same order.")
    return x - x.mean(axis=0), y - y.mean(axis=0)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Return centered, biased linear CKA for corresponding sample rows.

    Feature widths may differ. The statistic is invariant to orthogonal feature
    transformations and isotropic scaling, but can be high at initialization.
    A constant representation has no centered variance and returns 0.0.
    """
    x, y = _paired_features(x, y)
    # Normalization before matrix products avoids overflow at ordinary scales.
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    x = x / x_norm
    y = y / y_norm
    cross = x.T @ y
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.sum(cross * cross) / denominator, 0.0, 1.0))


def orthogonal_procrustes_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Return the optimal centered orthogonal alignment score in [0, 1].

    Each matrix is independently centered and normalized by its Frobenius norm.
    The score is the nuclear norm of X.T @ Y divided by ||X||_F * ||Y||_F.
    For equal feature widths, the minimum squared alignment residual of these
    normalized matrices is 2 - 2 * score. Rectangular cross matrices use the
    corresponding partial-isometry alignment. A constant input returns 0.0.
    Flexible alignments can score highly when feature width exceeds sample
    count, so random-initialization controls remain essential.
    """
    x, y = _paired_features(x, y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    singular_values = np.linalg.svd((x / x_norm).T @ (y / y_norm), compute_uv=False)
    return float(np.clip(singular_values.sum(), 0.0, 1.0))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average recall across classes present in the held-out ground truth."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or not y_true.size:
        raise ValueError("Labels must be nonempty one-dimensional arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must have the same shape.")
    return float(np.mean([np.mean(y_pred[y_true == label] == label) for label in np.unique(y_true)]))


def _class_labels(values: np.ndarray, rows: int, num_classes: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size != rows:
        raise ValueError(f"{name} must have one class label per feature row.")
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain integer class labels.")
    if np.any(result < 0) or np.any(result >= num_classes):
        raise ValueError(f"{name} labels must lie in [0, num_classes).")
    return result.astype(np.int64, copy=False)


def fit_ridge_readout(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_classes: int = 8,
    seed: int = 0,
    alpha: float = 1.0,
) -> dict:
    """Fit a standardized multiclass ridge readout using training data only.

    Means and scales are estimated on training features. An unpenalized bias
    and ridge-penalized weights predict one-hot class indicators; argmax gives
    the class. The positive regularization strength is fixed in advance, not
    selected on test data. The solve is deterministic; ``seed`` is accepted for
    a common interface with stochastic readouts. Test labels affect metrics
    only. The returned dictionary is JSON serializable.
    """
    del seed
    if not isinstance(num_classes, (int, np.integer)) or num_classes < 2:
        raise ValueError("num_classes must be an integer greater than one.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    train_x = _features(train_x, "train_x")
    test_x = _features(test_x, "test_x")
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and test features must have the same width.")
    train_y = _class_labels(train_y, train_x.shape[0], num_classes, "train_y")
    test_y = _class_labels(test_y, test_x.shape[0], num_classes, "test_y")

    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    targets = np.eye(num_classes, dtype=np.float64)[train_y]
    bias = targets.mean(axis=0)
    centered_targets = targets - bias

    # Choose the smaller positive-definite system for wide representations.
    if train_z.shape[1] <= train_z.shape[0]:
        covariance = train_z.T @ train_z
        covariance.flat[:: covariance.shape[0] + 1] += alpha
        weights = np.linalg.solve(covariance, train_z.T @ centered_targets)
    else:
        gram = train_z @ train_z.T
        gram.flat[:: gram.shape[0] + 1] += alpha
        weights = train_z.T @ np.linalg.solve(gram, centered_targets)
    predictions = np.argmax(test_z @ weights + bias, axis=1)
    return {
        "accuracy": float(np.mean(predictions == test_y)),
        "balanced_accuracy": balanced_accuracy(test_y, predictions),
        "predictions": predictions.tolist(),
        "alpha": float(alpha),
        "classes_seen": np.unique(train_y).tolist(),
        "readout": "standardized_multiclass_ridge",
    }


# The pilot uses the deterministic ridge probe for all readout comparisons.
fit_readout = fit_ridge_readout


def layerwise_similarity(
    reps_a: Sequence[np.ndarray], reps_b: Sequence[np.ndarray]
) -> list[dict]:
    """Compare corresponding layers evaluated on identical ordered examples."""
    if len(reps_a) != len(reps_b):
        raise ValueError("Both models must provide the same number of layers.")
    return [
        {
            "layer": layer,
            "cka": linear_cka(x, y),
            "procrustes_similarity": orthogonal_procrustes_similarity(x, y),
        }
        for layer, (x, y) in enumerate(zip(reps_a, reps_b))
    ]
