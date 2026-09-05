from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_iris

from another_cpu_model import CPUOptimizedManifoldCluster, CheckpointConfig, PipelineConfig


def test_pipeline_fit_predict_and_save_load(tmp_path: Path) -> None:
    X, _ = load_iris(return_X_y=True)

    pipeline = CPUOptimizedManifoldCluster(
        config=PipelineConfig(
            manifold_method="rff",
            n_components=8,
            clustering="minibatch_kmeans",
            n_clusters=3,
            n_neighbors=5,
            random_state=7,
        ),
        verbose=False,
    )

    pipeline.fit(X)
    predictions = pipeline.predict(X[:12])
    transformed = pipeline.transform(X[:12])

    assert predictions.shape == (12,)
    assert transformed.shape == (12, 8)
    assert pipeline.fit_summary_.stage_source["embedding"] == "computed"

    save_path = tmp_path / "pipeline.joblib"
    pipeline.save(save_path)
    restored = CPUOptimizedManifoldCluster.load(save_path)

    assert restored.predict(X[:12]).shape == (12,)


def test_pipeline_resume_uses_checkpoints(tmp_path: Path) -> None:
    X, _ = load_iris(return_X_y=True)
    checkpoint = CheckpointConfig(directory=str(tmp_path / "checkpoints"), run_name="iris")
    config = PipelineConfig(
        manifold_method="nystrom",
        n_components=6,
        m_nystrom=24,
        clustering="minibatch_kmeans",
        n_clusters=3,
        n_neighbors=5,
        random_state=11,
    )

    first = CPUOptimizedManifoldCluster(config=config, checkpoint=checkpoint, verbose=False)
    first.fit(X)
    assert all(source == "computed" for source in first.fit_summary_.stage_source.values())

    second = CPUOptimizedManifoldCluster(config=config, checkpoint=checkpoint, verbose=False)
    second.fit(X)

    assert second.fit_summary_.stage_source["preprocess"] == "checkpoint"
    assert second.fit_summary_.stage_source["neighbors"] == "checkpoint"
    assert second.fit_summary_.stage_source["embedding"] == "checkpoint"
    assert second.fit_summary_.stage_source["clustering"] == "checkpoint"
