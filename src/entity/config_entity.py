from src.constants import *
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.BUCKET_NAME: str = self.config['data_ingestion_config']["bucket_name"]
        self.ZIP_FILE_NAME: str = self.config['data_ingestion_config']["zip_file_name"]
        self.DATASET_FOLDER_NAME: str = self.config['data_ingestion_config']["dataset_folder_name"]
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        self.LOCAL_DATA_DIR: str = os.path.join(os.getcwd(), "data", self.DATASET_FOLDER_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.TRAIN_SPLIT: float = self.config['data_transformation_config']["train_split"]
        self.BATCH_SIZE: int = self.config['data_transformation_config']["batch_size"]
        self.IMAGE_SIZE: int = self.config['data_transformation_config']["image_size"]


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.EPOCHS: int = self.config['model_trainer_config']["epochs"]
        self.LEARNING_RATE: float = self.config['model_trainer_config']["learning_rate"]
        self.MODEL_FILE_NAME: str = self.config['model_trainer_config']["model_file_name"]
        self.MODEL_DIR: str = os.path.join(os.getcwd(), "model")
        self.MODEL_PATH: str = os.path.join(self.MODEL_DIR, self.MODEL_FILE_NAME)

