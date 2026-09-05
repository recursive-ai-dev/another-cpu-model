# another-cpu-model

This repository now includes a real Python package around the original notebook so the project can grow beyond a single prototype.

## What changed

The notebook is still here as the research artifact:

- `CPU_Optimized_Model.ipynb`

But the production path is now the package:

- `another_cpu_model/config.py` — immutable pipeline and checkpoint config
- `another_cpu_model/checkpointing.py` — stage-level save/load for resumable training
- `another_cpu_model/neighbors.py` — ANN-aware kNN graph construction with sklearn fallback
- `another_cpu_model/embedding.py` — pluggable embedding backends: Nyström, RFF, Isomap, projection
- `another_cpu_model/clustering.py` — clustering backends and fallbacks
- `another_cpu_model/validation.py` — clustering metrics
- `another_cpu_model/pipeline.py` — end-to-end estimator API

## Checkpointing

Checkpointing is stage-based rather than a single monolithic dump.

Stages:

1. `preprocess`
2. `neighbors`
3. `embedding`
4. `clustering`
5. `validation`

Each stage is saved under:

```text
.checkpoints/<run_name>/<stage>/
```

A stage is reused only when both of these match:

- the pipeline configuration signature
- the input data fingerprint

That makes resume safer than blindly loading old artifacts.

## Quick start

```python
from sklearn.datasets import load_digits

from another_cpu_model import CPUOptimizedManifoldCluster, CheckpointConfig, PipelineConfig

X, _ = load_digits(return_X_y=True)

pipeline = CPUOptimizedManifoldCluster(
    config=PipelineConfig(
        manifold_method="rff",
        n_components=32,
        clustering="minibatch_kmeans",
        n_clusters=10,
        n_neighbors=10,
    ),
    checkpoint=CheckpointConfig(directory=".checkpoints", run_name="digits"),
    verbose=True,
)

pipeline.fit(X)
print(pipeline.metrics_)
print(pipeline.benchmark(X))
```

## Example script

```bash
python examples/train_digits.py
```

## Dev

```bash
python -m pytest
```

## Notes

- Optional backends such as `hnswlib`, `faiss`, and `hdbscan` are detected automatically.
- Nyström now supports proper `transform()` for new samples by persisting inducing points and the learned projection.
- Agglomerative clustering falls back to MiniBatch KMeans when the configured hierarchical size limit would make memory use unsafe.
