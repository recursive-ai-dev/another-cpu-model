from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .backends import detect_backends
from .checkpointing import CheckpointManager
from .clustering import cluster_embedding
from .config import CheckpointConfig, PipelineConfig
from .embedding import fit_embedding
from .neighbors import build_knn_graph
from .types import FitSummary, ValidationReport
from .utils import assign_to_centroids, ensure_contiguous_float32, fingerprint_array
from .validation import compute_validation_metrics


class CPUOptimizedManifoldCluster(BaseEstimator, TransformerMixin, ClusterMixin):
    """Modular CPU-first clustering pipeline with resumable stage checkpoints."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        checkpoint: CheckpointConfig | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config or PipelineConfig()
        self.checkpoint = checkpoint
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CPUOptimizedManifoldCluster":
        del y
        start_total = time.perf_counter()
        X = np.asarray(X)

        self.backends_ = detect_backends()
        self.data_fingerprint_ = fingerprint_array(X)
        self.checkpoint_manager_ = self._build_checkpoint_manager()
        self.fit_summary_ = FitSummary(stage_source={}, stage_durations={}, data_fingerprint=self.data_fingerprint_, total_time=0.0)

        preprocess_payload = self._run_stage("preprocess", lambda: self._fit_preprocess(X))
        self.scaler_ = preprocess_payload["scaler"]
        self.preprocess_info_ = preprocess_payload["info"]
        self.X_scaled_ = preprocess_payload["X_scaled"]

        neighbor_result = self._run_stage(
            "neighbors",
            lambda: build_knn_graph(
                self.X_scaled_,
                k=self.config.n_neighbors,
                use_ann=self.config.use_ann,
                n_jobs=-1,
            ),
        )
        self.knn_graph_ = neighbor_result.graph
        self.knn_indices_ = neighbor_result.indices
        self.knn_distances_ = neighbor_result.distances
        self.neighbor_backend_ = neighbor_result.backend

        embedding_result = self._run_stage("embedding", lambda: fit_embedding(self.X_scaled_, self.config))
        self.embedding_ = embedding_result.features
        self.embedding_model_ = embedding_result.model
        self.embedding_method_ = embedding_result.method

        clustering_result = self._run_stage("clustering", lambda: cluster_embedding(self.embedding_, self.config))
        self.labels_ = clustering_result.labels
        self.clusterer_ = clustering_result.model
        self.centroids_ = clustering_result.centroids
        self.centroid_labels_ = clustering_result.centroid_labels
        self.clustering_method_ = clustering_result.method

        if self.config.validation_enabled:
            validation_report = self._run_stage(
                "validation",
                lambda: compute_validation_metrics(
                    self.embedding_,
                    self.labels_,
                    sample_size=self.config.validation_sample_size,
                ),
            )
        else:
            validation_report = ValidationReport(metrics={}, cluster_sizes={})

        self.metrics_ = validation_report.metrics
        self.cluster_sizes_ = validation_report.cluster_sizes
        self.fit_summary_.total_time = time.perf_counter() - start_total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["scaler_", "embedding_model_"])
        X = np.asarray(X)
        X, _, _ = ensure_contiguous_float32(X)
        X_scaled = self.scaler_.transform(X).astype(np.float32)
        X_scaled, _, _ = ensure_contiguous_float32(X_scaled)
        transformed = self.embedding_model_.transform(X_scaled)
        transformed, _, _ = ensure_contiguous_float32(transformed)
        return transformed

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["clusterer_", "centroids_", "centroid_labels_"])
        X_embedded = self.transform(X)
        if hasattr(self.clusterer_, "predict"):
            return self.clusterer_.predict(X_embedded).astype(int)
        if self.centroids_.size:
            assigned = assign_to_centroids(X_embedded, self.centroids_)
            return self.centroid_labels_[assigned]
        return np.full(X_embedded.shape[0], -1, dtype=int)

    def benchmark(self, X: np.ndarray, *, batch_size: int = 100, n_runs: int = 5) -> dict[str, float]:
        check_is_fitted(self, attributes=["embedding_", "metrics_", "fit_summary_"])
        sample = np.asarray(X)[:batch_size]
        latencies_ms: list[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.transform(sample)
            elapsed = time.perf_counter() - start
            latencies_ms.append(elapsed * 1_000 / max(1, sample.shape[0]))

        return {
            "p50_ms_per_sample": float(np.percentile(latencies_ms, 50)),
            "p95_ms_per_sample": float(np.percentile(latencies_ms, 95)),
            "p99_ms_per_sample": float(np.percentile(latencies_ms, 99)),
            "training_time_s": float(self.fit_summary_.total_time),
            "embedding_memory_mb": float(self.embedding_.nbytes / 1e6),
            "cpu_count": float(os.cpu_count() or 1),
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CPUOptimizedManifoldCluster":
        return joblib.load(path)

    def _build_checkpoint_manager(self) -> CheckpointManager | None:
        if self.checkpoint is None or not self.checkpoint.enabled:
            return None
        return CheckpointManager(self.checkpoint, self.config)

    def _run_stage(self, stage: str, builder: Callable[[], Any]) -> Any:
        stage_start = time.perf_counter()
        source = "computed"

        if self.checkpoint_manager_ is not None:
            payload = self.checkpoint_manager_.load_stage(stage, data_fingerprint=self.data_fingerprint_)
            if payload is not None:
                source = "checkpoint"
                self._record_stage(stage, source, stage_start)
                self._log(stage, source)
                return payload

        payload = builder()
        if self.checkpoint_manager_ is not None:
            self.checkpoint_manager_.save_stage(stage, payload, data_fingerprint=self.data_fingerprint_)
        self._record_stage(stage, source, stage_start)
        self._log(stage, source)
        return payload

    def _record_stage(self, stage: str, source: str, stage_start: float) -> None:
        self.fit_summary_.stage_source[stage] = source
        self.fit_summary_.stage_durations[stage] = time.perf_counter() - stage_start

    def _log(self, stage: str, source: str) -> None:
        if self.verbose:
            print(f"[{stage}] {source}")

    def _fit_preprocess(self, X: np.ndarray) -> dict[str, Any]:
        X, aligned, address = ensure_contiguous_float32(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        X_scaled, _, _ = ensure_contiguous_float32(X_scaled)
        return {
            "scaler": scaler,
            "X_scaled": X_scaled,
            "info": {
                "shape": tuple(int(v) for v in X.shape),
                "aligned": bool(aligned),
                "address": int(address),
                "dtype": str(X.dtype),
            },
        }
