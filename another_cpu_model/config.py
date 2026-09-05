from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

EmbeddingMethod = Literal["nystrom", "rff", "isomap", "none"]
ClusterMethod = Literal["minibatch_kmeans", "agglomerative", "hdbscan"]


@dataclass(frozen=True)
class CheckpointConfig:
    """Controls stage-level persistence during fitting."""

    directory: str = ".checkpoints"
    run_name: str = "default"
    resume: bool = True
    compress: int = 3
    enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the CPU-oriented manifold clustering pipeline."""

    n_neighbors: int = 10
    n_components: int = 10
    manifold_method: EmbeddingMethod = "nystrom"
    m_nystrom: int = 200
    gamma: float = 0.5
    clustering: ClusterMethod = "minibatch_kmeans"
    n_clusters: int = 5
    batch_size: int = 256
    block_size: int = 256
    random_state: int = 42
    max_hierarchical_n: int = 5_000
    use_ann: bool = True
    validation_sample_size: int = 2_000
    validation_enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
