"""
metrics.py
Evaluation helpers: accuracy, confusion matrix, per-class report.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Dict, List, Optional


def compute_accuracy(preds: List[int], labels: List[int]) -> float:
    """Top-1 accuracy."""
    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean())


def compute_confusion_matrix(
    preds: List[int],
    labels: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    Returns a (num_classes, num_classes) confusion matrix where
    entry [i, j] is the number of samples whose true label is i
    and predicted label is j.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        matrix[t, p] += 1
    return matrix


def classification_report_dict(
    preds: List[int],
    labels: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-class precision, recall, F1, and support as a dict,
    analogous to sklearn's classification_report but without the dependency.
    """
    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    classes = np.unique(np.concatenate([labels, preds]))
    report: Dict[str, Dict[str, float]] = {}

    for c in classes:
        tp = int(np.sum((preds == c) & (labels == c)))
        fp = int(np.sum((preds == c) & (labels != c)))
        fn = int(np.sum((preds != c) & (labels == c)))
        support = int(np.sum(labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        name = class_names[c] if class_names and c < len(class_names) else str(c)
        report[name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }
    return report


def majority_vote(predictions: List[int]) -> int:
    """Return the most common class index from a list of predictions."""
    if not predictions:
        raise ValueError("predictions list is empty")
    return Counter(predictions).most_common(1)[0][0]
