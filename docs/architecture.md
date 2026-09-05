# Architecture

## Goals

1. Move the project out of a notebook-only structure.
2. Make training resumable with stage-level checkpoints.
3. Keep CPU-oriented implementation details explicit.
4. Preserve a clean separation between algorithms, orchestration, and persistence.

## Package layout

```text
another_cpu_model/
├── backends.py
├── checkpointing.py
├── clustering.py
├── config.py
├── embedding.py
├── neighbors.py
├── pipeline.py
├── types.py
├── utils.py
└── validation.py
```

## Pipeline stages

```text
raw input
  └─ preprocess
      └─ neighbors
          └─ embedding
              └─ clustering
                  └─ validation
```

### `preprocess`

Responsibilities:

- convert to `float32`
- enforce C-contiguous layout
- standardize with `StandardScaler`
- record low-level memory/alignment metadata

### `neighbors`

Responsibilities:

- build the kNN graph
- prefer ANN backends when available
- degrade safely to sklearn exact search

### `embedding`

Responsibilities:

- choose an embedding backend behind a stable interface
- persist model state needed for `transform()` on new samples

Backends:

- Nyström
- Random Fourier Features
- Isomap
- simple projection/identity path

### `clustering`

Responsibilities:

- cluster the embedded representation
- compute centroids for backends without native `predict()`
- fall back to MiniBatch KMeans when hierarchical clustering would be unsafe

### `validation`

Responsibilities:

- compute WSS
- compute silhouette and Davies-Bouldin when meaningful
- handle noise labels from density-based methods

## Checkpoint design

Checkpointing is stage-oriented instead of whole-model-only.

Each stage stores:

- `artifact.joblib`
- `metadata.json`

A stage is considered valid only when:

- `pipeline_signature` matches the current config
- `data_fingerprint` matches the current dataset

That gives us resumability without silently reusing stale intermediate artifacts.

## Why this is better than the notebook-only design

- individual modules are testable
- stage boundaries are explicit
- new embedders/clusterers can be added without rewriting orchestration
- persistence and computation are separated
- `transform()` for Nyström is now a first-class path rather than a demo-only fallback
