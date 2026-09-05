from __future__ import annotations

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

from .types import ValidationReport


def compute_validation_metrics(X: np.ndarray, labels: np.ndarray, *, sample_size: int = 2_000) -> ValidationReport:
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique, counts)}

    valid_mask = labels != -1
    effective_labels = labels[valid_mask]
    effective_X = X[valid_mask]
    effective_unique = np.unique(effective_labels)

    metrics: dict[str, float | int | None] = {
        "n_samples": int(X.shape[0]),
        "n_clusters": int(len(effective_unique)),
        "noise_fraction": float(1.0 - np.mean(valid_mask)) if labels.size else 0.0,
        "wss": None,
        "silhouette": None,
        "davies_bouldin": None,
    }

    if effective_X.size == 0 or len(effective_unique) == 0:
        return ValidationReport(metrics=metrics, cluster_sizes=cluster_sizes)

    wss = 0.0
    for cluster_id in effective_unique:
        members = effective_X[effective_labels == cluster_id]
        centroid = members.mean(axis=0)
        wss += float(np.sum((members - centroid) ** 2))
    metrics["wss"] = wss

    if len(effective_unique) >= 2 and effective_X.shape[0] > len(effective_unique):
        if effective_X.shape[0] > sample_size:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(effective_X.shape[0], size=sample_size, replace=False)
            sampled_X = effective_X[sample_idx]
            sampled_labels = effective_labels[sample_idx]
        else:
            sampled_X = effective_X
            sampled_labels = effective_labels

        if len(np.unique(sampled_labels)) >= 2:
            metrics["silhouette"] = float(silhouette_score(sampled_X, sampled_labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(sampled_X, sampled_labels))

    return ValidationReport(metrics=metrics, cluster_sizes=cluster_sizes)
