import os
from datetime import datetime

# Common constants
CONFIG_PATH: str = os.path.join(os.getcwd(), "config", "config.yaml")
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)

try:
    import torch
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError):
    DEVICE = "cpu"

# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
