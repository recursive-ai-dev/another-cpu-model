from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from .config import PipelineConfig
from .types import ClusteringResult
from .utils import compute_centroids, ensure_contiguous_float32


def _minibatch_result(X: np.ndarray, config: PipelineConfig, *, method_name: str) -> ClusteringResult:
    model = MiniBatchKMeans(
        n_clusters=config.n_clusters,
        batch_size=config.batch_size,
        n_init=10,
        random_state=config.random_state,
    )
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_.astype(np.float32)
    return ClusteringResult(
        labels=labels.astype(int),
        model=model,
        centroids=centroids,
        centroid_labels=np.arange(centroids.shape[0], dtype=int),
        method=method_name,
    )


def cluster_embedding(X: np.ndarray, config: PipelineConfig) -> ClusteringResult:
    X, _, _ = ensure_contiguous_float32(X)

    if config.clustering == "minibatch_kmeans":
        return _minibatch_result(X, config, method_name=config.clustering)

    if config.clustering == "agglomerative":
        if X.shape[0] > config.max_hierarchical_n:
            return _minibatch_result(X, config, method_name="minibatch_kmeans(fallback)")

        model = AgglomerativeClustering(n_clusters=config.n_clusters, linkage="average")
        labels = model.fit_predict(X)
        centroids, centroid_labels = compute_centroids(X, labels)
        return ClusteringResult(
            labels=labels.astype(int),
            model=model,
            centroids=centroids,
            centroid_labels=centroid_labels,
            method=config.clustering,
        )

    if config.clustering == "hdbscan":
        try:
            import hdbscan
        except Exception:
            return _minibatch_result(X, config, method_name="minibatch_kmeans(fallback)")

        model = hdbscan.HDBSCAN(min_cluster_size=max(5, min(config.batch_size, 32)), core_dist_n_jobs=-1)
        labels = model.fit_predict(X)
        centroids, centroid_labels = compute_centroids(X, labels)
        return ClusteringResult(
            labels=labels.astype(int),
            model=model,
            centroids=centroids,
            centroid_labels=centroid_labels,
            method=config.clustering,
        )

    raise ValueError(f"Unsupported clustering method: {config.clustering}")
