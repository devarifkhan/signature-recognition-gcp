import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
from src.logger import logging
from src.exception import CustomException
from src.constants import DEVICE
from src.utils.main_utils import save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, artifacts: DataTransformationArtifacts):
        self.config = config
        self.artifacts = artifacts

    def build_model(self):
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model.to(DEVICE)

    def _run_epoch(self, model, loader, optimizer, criterion, train=True):
        model.train() if train else model.eval()
        total_loss, correct = 0.0, 0
        phase = "train" if train else "val"
        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, desc=f"  {phase}", leave=False, ncols=80):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
        n = len(loader.dataset)
        return total_loss / len(loader), correct / n

    def initiate_model_training(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_training")
        try:
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            model = self.build_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

            best_val_acc = 0.0
            for epoch in range(1, self.config.EPOCHS + 1):
                print(f"\nEpoch {epoch}/{self.config.EPOCHS}", flush=True)
                tr_loss, tr_acc = self._run_epoch(model, self.artifacts.train_loader, optimizer, criterion, train=True)
                vl_loss, vl_acc = self._run_epoch(model, self.artifacts.val_loader, optimizer, criterion, train=False)
                msg = (
                    f"  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
                    f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
                )
                print(msg, flush=True)
                logging.info(f"Epoch {epoch}/{self.config.EPOCHS} | " + msg)
                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    save_object(self.config.MODEL_PATH, model)
                    print(f"  ✓ Best model saved (val_acc={vl_acc:.4f})", flush=True)
                    logging.info(f"Best model saved → val_acc={vl_acc:.4f}")

            logging.info(f"Training complete. Best val_acc={best_val_acc:.4f}")
            return ModelTrainerArtifacts(model_path=self.config.MODEL_PATH, val_accuracy=best_val_acc)
        except Exception as e:
            raise CustomException(e, sys) from e
