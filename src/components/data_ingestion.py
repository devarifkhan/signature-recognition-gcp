import os
import sys
from zipfile import ZipFile
from src.logger import logging
from src.exception import CustomException
from src.configurations.gcloud_syncer import GCloudSync
from src.constants import ENV_MODE
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data_from_gcloud(self) -> None:

        """
        Method Name :   get_data_from_gcloud
        Description :   This function fetch data from gcloud

        Output      :   Returns data into DataIngestionArtifacts
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the get_data_from_gcloud method of Data ingestion class")

            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.gcloud.sync_file_from_gcloud(self.data_ingestion_config.BUCKET_NAME,
                                              self.data_ingestion_config.ZIP_FILE_NAME,
                                              self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,)

            logging.info("Exited the get_data_from_gcloud method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self) -> None:
        """
        Method Name :   unzip_and_clean
        Description :   This function unzip the dataset

        Output      :   Returns Unzipped Data
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            logging.info("Exited the unzip_and_clean method of Data ingestion class")
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:
            if ENV_MODE == "dev":
                return self._ingest_from_local()
            else:
                return self._ingest_from_gcloud()
        except Exception as e:
            raise CustomException(e, sys) from e

    def _ingest_from_local(self) -> DataIngestionArtifacts:
        local_dir = self.data_ingestion_config.LOCAL_DATA_DIR
        logging.info(f"Dev mode — using local dataset from: {local_dir}")

        if not os.path.exists(local_dir):
            raise FileNotFoundError(
                f"Dev mode: '{local_dir}' folder not found. Create it and place your dataset inside."
            )

        zip_path = os.path.join(local_dir, self.data_ingestion_config.ZIP_FILE_NAME)
        if os.path.exists(zip_path):
            logging.info(f"Found zip file: {zip_path} — extracting...")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_dir)
            os.remove(zip_path)
            logging.info("Extraction complete, zip removed")

        return DataIngestionArtifacts(dataset_path=local_dir)

    def _ingest_from_gcloud(self) -> DataIngestionArtifacts:
        self.get_data_from_gcloud()
        logging.info("Fetched the zipped dataset from GCloud Storage bucket")

        self.unzip_and_clean()
        logging.info("Unzipped the dataset")

        os.remove(os.path.join(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                               self.data_ingestion_config.ZIP_FILE_NAME))

        data_ingestion_artifacts = DataIngestionArtifacts(
            dataset_path=self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)

        logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")
        return data_ingestion_artifacts
