import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts


class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion")
        try:
            ingestion = DataIngestion(self.data_ingestion_config)
            return ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self, ingestion_artifacts: DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("Starting data transformation")
        try:
            transformation = DataTransformation(self.data_transformation_config, ingestion_artifacts)
            return transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_training(self, transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info("Starting model training")
        try:
            trainer = ModelTrainer(self.model_trainer_config, transformation_artifacts)
            return trainer.initiate_model_training()
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> ModelTrainerArtifacts:
        logging.info("===== Training Pipeline Started =====")
        try:
            ingestion_artifacts = self.start_data_ingestion()
            transformation_artifacts = self.start_data_transformation(ingestion_artifacts)
            model_artifacts = self.start_model_training(transformation_artifacts)
            logging.info(f"===== Training Pipeline Complete | val_acc={model_artifacts.val_accuracy:.4f} =====")
            return model_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
