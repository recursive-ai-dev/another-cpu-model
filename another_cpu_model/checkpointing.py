from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .config import CheckpointConfig, PipelineConfig
from .utils import stable_hash


class CheckpointManager:
    """Stores and reloads stage artifacts for resumable pipeline training."""

    def __init__(self, config: CheckpointConfig, pipeline_config: PipelineConfig):
        self.config = config
        self.pipeline_config = pipeline_config
        self.pipeline_signature = stable_hash(pipeline_config.to_dict())
        self.root = Path(config.directory) / config.run_name
        if self.config.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def _artifact_path(self, stage: str) -> Path:
        return self._stage_dir(stage) / "artifact.joblib"

    def _metadata_path(self, stage: str) -> Path:
        return self._stage_dir(stage) / "metadata.json"

    def load_stage(self, stage: str, *, data_fingerprint: str) -> Any | None:
        if not self.config.enabled or not self.config.resume:
            return None

        artifact_path = self._artifact_path(stage)
        metadata_path = self._metadata_path(stage)
        if not artifact_path.exists() or not metadata_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text())
        if metadata.get("pipeline_signature") != self.pipeline_signature:
            return None
        if metadata.get("data_fingerprint") != data_fingerprint:
            return None

        return joblib.load(artifact_path)

    def save_stage(self, stage: str, payload: Any, *, data_fingerprint: str, extra_metadata: dict[str, Any] | None = None) -> None:
        if not self.config.enabled:
            return

        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = self._artifact_path(stage)
        metadata_path = self._metadata_path(stage)
        joblib.dump(payload, artifact_path, compress=self.config.compress)

        metadata = {
            "stage": stage,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_signature": self.pipeline_signature,
            "pipeline_config": self.pipeline_config.to_dict(),
            "checkpoint_config": self.config.to_dict(),
            "data_fingerprint": data_fingerprint,
        }
        if extra_metadata:
            metadata.update(_serialize_metadata(extra_metadata))
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _serialize_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in payload.items():
        if is_dataclass(value):
            serialized[key] = asdict(value)
        else:
            serialized[key] = value
    return serialized
