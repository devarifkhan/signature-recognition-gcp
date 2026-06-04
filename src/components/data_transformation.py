import os
import sys
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class SignatureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []

        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label = 1 if folder.endswith("_forg") else 0
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(folder_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, artifacts: DataIngestionArtifacts):
        self.config = config
        self.artifacts = artifacts

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        logging.info("Entered initiate_data_transformation")
        try:
            dataset = SignatureDataset(self.artifacts.dataset_path, transform=self.get_transforms())
            logging.info(f"Total samples: {len(dataset)}")

            train_size = int(self.config.TRAIN_SPLIT * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

            logging.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
            return DataTransformationArtifacts(train_loader=train_loader, val_loader=val_loader)
        except Exception as e:
            raise CustomException(e, sys) from e
