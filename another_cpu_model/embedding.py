from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import Isomap
from sklearn.metrics.pairwise import rbf_kernel

from .config import PipelineConfig
from .types import EmbeddingResult
from .utils import ensure_contiguous_float32


class BaseEmbedder:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def fit(self, X: np.ndarray) -> "BaseEmbedder":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NystromEmbedder(BaseEmbedder):
    def __init__(self, n_components: int, m: int, gamma: float, random_state: int, regularization: float = 1e-6):
        self.n_components = n_components
        self.m = m
        self.gamma = gamma
        self.random_state = random_state
        self.regularization = regularization

    def fit(self, X: np.ndarray) -> "NystromEmbedder":
        X, _, _ = ensure_contiguous_float32(X)
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        m = min(self.m, n_samples)
        self.inducing_indices_ = rng.choice(n_samples, size=m, replace=False)
        self.inducing_points_ = X[self.inducing_indices_]

        K_mm = rbf_kernel(self.inducing_points_, self.inducing_points_, gamma=self.gamma).astype(np.float32)
        K_mm += self.regularization * np.eye(m, dtype=np.float32)

        eigvals, eigvecs = np.linalg.eigh(K_mm)
        order = np.argsort(eigvals)[::-1]
        rank = min(self.n_components, m)
        eigvals = np.maximum(eigvals[order][:rank], 1e-10)
        eigvecs = eigvecs[:, order][:, :rank]
        self.projection_ = (eigvecs / np.sqrt(eigvals)).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X, _, _ = ensure_contiguous_float32(X)
        K_nm = rbf_kernel(X, self.inducing_points_, gamma=self.gamma).astype(np.float32)
        return np.ascontiguousarray(K_nm @ self.projection_, dtype=np.float32)


class RFFEmbedder(BaseEmbedder):
    def __init__(self, n_components: int, gamma: float, random_state: int):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.model_ = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)

    def fit(self, X: np.ndarray) -> "RFFEmbedder":
        self.model_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.model_.transform(X).astype(np.float32), dtype=np.float32)


class IsomapEmbedder(BaseEmbedder):
    def __init__(self, n_neighbors: int, n_components: int):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.model_ = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)

    def fit(self, X: np.ndarray) -> "IsomapEmbedder":
        self.model_.fit(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.model_.fit_transform(X).astype(np.float32), dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.model_.transform(X).astype(np.float32), dtype=np.float32)


class IdentityProjector(BaseEmbedder):
    def __init__(self, n_components: int, random_state: int):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "IdentityProjector":
        X, _, _ = ensure_contiguous_float32(X)
        input_dim = X.shape[1]
        if input_dim <= self.n_components:
            self.projection_ = None
        else:
            rng = np.random.RandomState(self.random_state)
            projection = rng.normal(0.0, 1.0 / np.sqrt(self.n_components), size=(input_dim, self.n_components))
            self.projection_ = projection.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X, _, _ = ensure_contiguous_float32(X)
        if self.projection_ is None:
            return X.astype(np.float32)
        return np.ascontiguousarray(X @ self.projection_, dtype=np.float32)


def build_embedder(config: PipelineConfig) -> BaseEmbedder:
    if config.manifold_method == "nystrom":
        return NystromEmbedder(
            n_components=config.n_components,
            m=config.m_nystrom,
            gamma=config.gamma,
            random_state=config.random_state,
        )
    if config.manifold_method == "rff":
        return RFFEmbedder(
            n_components=config.n_components,
            gamma=config.gamma,
            random_state=config.random_state,
        )
    if config.manifold_method == "isomap":
        return IsomapEmbedder(
            n_neighbors=config.n_neighbors,
            n_components=config.n_components,
        )
    if config.manifold_method == "none":
        return IdentityProjector(
            n_components=config.n_components,
            random_state=config.random_state,
        )
    raise ValueError(f"Unsupported embedding method: {config.manifold_method}")


def fit_embedding(X: np.ndarray, config: PipelineConfig) -> EmbeddingResult:
    embedder = build_embedder(config)
    features = embedder.fit_transform(X)
    features, _, _ = ensure_contiguous_float32(features)
    return EmbeddingResult(features=features, model=embedder, method=config.manifold_method)
