from dataclasses import dataclass
from typing import Any


@dataclass
class DataIngestionArtifacts:
    dataset_path: str


@dataclass
class DataTransformationArtifacts:
    train_loader: Any
    val_loader: Any


@dataclass
class ModelTrainerArtifacts:
    model_path: str
    val_accuracy: float
