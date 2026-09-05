from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances


def ensure_contiguous_float32(X: np.ndarray, align: int = 64) -> tuple[np.ndarray, bool, int]:
    array = np.asarray(X)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    address = array.__array_interface__["data"][0]
    return array, address % align == 0, address


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def fingerprint_array(X: np.ndarray, sample_bytes: int = 1_000_000) -> str:
    array = np.ascontiguousarray(np.asarray(X))
    raw = array.view(np.uint8).ravel()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))

    if raw.size <= sample_bytes:
        digest.update(raw.tobytes())
        return digest.hexdigest()

    half = sample_bytes // 2
    digest.update(raw[:half].tobytes())
    digest.update(raw[-half:].tobytes())
    stride = max(1, raw.size // max(1, sample_bytes // 4))
    digest.update(raw[::stride][: sample_bytes // 4].tobytes())
    return digest.hexdigest()


def compute_centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.asarray([label for label in np.unique(labels) if label != -1], dtype=int)
    if unique_labels.size == 0:
        return np.empty((0, X.shape[1]), dtype=np.float32), unique_labels
    centroids = [np.asarray(X[labels == label].mean(axis=0), dtype=np.float32) for label in unique_labels]
    return np.vstack(centroids).astype(np.float32), unique_labels


def assign_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if centroids.size == 0:
        return np.full(X.shape[0], -1, dtype=int)
    distances = pairwise_distances(X, centroids, metric="euclidean")
    return np.argmin(distances, axis=1).astype(int)


def gram_memory_footprint(N: int, dtype: np.dtype = np.float32, symmetric: bool = True, include_diag: bool = True) -> dict[str, int]:
    bytes_per = np.dtype(dtype).itemsize
    if symmetric and include_diag:
        unique_entries = N * (N + 1) // 2
    elif symmetric:
        unique_entries = N * (N - 1) // 2
    else:
        unique_entries = N * N
    return {
        "full_entries": N * N,
        "unique_entries": unique_entries,
        "bytes_per_entry": bytes_per,
        "full_bytes": N * N * bytes_per,
        "unique_bytes": unique_entries * bytes_per,
    }


def blocked_pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    block_size: int = 256,
    metric: str = "euclidean",
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Y = X if Y is None else np.asarray(Y, dtype=np.float32)
    output = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
    for start in range(0, X.shape[0], block_size):
        stop = min(start + block_size, X.shape[0])
        output[start:stop] = pairwise_distances(X[start:stop], Y, metric=metric)
    return output.astype(np.float32)
