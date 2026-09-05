from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.datasets import load_digits

from another_cpu_model import CPUOptimizedManifoldCluster, CheckpointConfig, PipelineConfig


def main() -> None:
    X, _ = load_digits(return_X_y=True)

    pipeline = CPUOptimizedManifoldCluster(
        config=PipelineConfig(
            manifold_method="rff",
            n_components=32,
            clustering="minibatch_kmeans",
            n_clusters=10,
            n_neighbors=10,
            random_state=42,
        ),
        checkpoint=CheckpointConfig(directory=".checkpoints", run_name="digits-demo"),
        verbose=True,
    )

    pipeline.fit(X)
    print("Metrics:", pipeline.metrics_)
    print("Benchmark:", pipeline.benchmark(X))

    model_path = Path("artifacts") / "digits_pipeline.joblib"
    pipeline.save(model_path)
    print(f"Saved trained pipeline to {model_path}")


if __name__ == "__main__":
    main()
