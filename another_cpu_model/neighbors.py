from __future__ import annotations

import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .types import NeighborSearchResult
from .utils import ensure_contiguous_float32


def build_knn_graph(X: np.ndarray, k: int = 10, use_ann: bool = True, n_jobs: int = -1) -> NeighborSearchResult:
    X, _, _ = ensure_contiguous_float32(X)
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("kNN graph requires at least two samples")

    k = max(1, min(k, n_samples - 1))

    if use_ann:
        hnsw_result = _try_hnswlib(X, k)
        if hnsw_result is not None:
            return hnsw_result

        faiss_result = _try_faiss(X, k)
        if faiss_result is not None:
            return faiss_result

    model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean", n_jobs=n_jobs)
    model.fit(X)
    distances, indices = model.kneighbors(X)
    distances = distances[:, 1:].astype(np.float32)
    indices = indices[:, 1:].astype(np.int32)
    graph = model.kneighbors_graph(X, n_neighbors=k, mode="distance")
    return NeighborSearchResult(graph=graph.tocsr(), indices=indices, distances=distances, backend=f"sklearn(auto, D={n_features})")


def _try_hnswlib(X: np.ndarray, k: int) -> NeighborSearchResult | None:
    try:
        import hnswlib
    except Exception:
        return None

    if X.shape[0] <= 1_000:
        return None

    try:
        index = hnswlib.Index(space="l2", dim=X.shape[1])
        index.init_index(max_elements=X.shape[0], ef_construction=200, M=16)
        index.add_items(X, np.arange(X.shape[0]))
        index.set_ef(max(50, k + 1))
        labels, distances = index.knn_query(X, k=k + 1)
        indices = labels[:, 1:].astype(np.int32)
        distances = distances[:, 1:].astype(np.float32)
        rows = np.repeat(np.arange(X.shape[0]), k)
        graph = csr_matrix((distances.ravel(), (rows, indices.ravel())), shape=(X.shape[0], X.shape[0]))
        return NeighborSearchResult(graph=graph, indices=indices, distances=distances, backend="hnswlib")
    except Exception:
        return None


def _try_faiss(X: np.ndarray, k: int) -> NeighborSearchResult | None:
    try:
        import faiss
    except Exception:
        return None

    if X.shape[0] <= 5_000:
        return None

    try:
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        distances, indices = index.search(X, k + 1)
        indices = indices[:, 1:].astype(np.int32)
        distances = distances[:, 1:].astype(np.float32)
        rows = np.repeat(np.arange(X.shape[0]), k)
        graph = csr_matrix((distances.ravel(), (rows, indices.ravel())), shape=(X.shape[0], X.shape[0]))
        return NeighborSearchResult(graph=graph, indices=indices, distances=distances, backend="faiss")
    except Exception:
        return None
