from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


@dataclass
class BackendAvailability:
    has_hnswlib: bool
    has_faiss: bool
    has_hdbscan: bool
    blas: str
    threadpools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NeighborSearchResult:
    graph: sparse.csr_matrix
    indices: np.ndarray
    distances: np.ndarray
    backend: str


@dataclass
class EmbeddingResult:
    features: np.ndarray
    model: Any
    method: str


@dataclass
class ClusteringResult:
    labels: np.ndarray
    model: Any
    centroids: np.ndarray
    centroid_labels: np.ndarray
    method: str


@dataclass
class ValidationReport:
    metrics: dict[str, float | int | None]
    cluster_sizes: dict[int, int]


@dataclass
class FitSummary:
    stage_source: dict[str, str]
    stage_durations: dict[str, float]
    data_fingerprint: str
    total_time: float
