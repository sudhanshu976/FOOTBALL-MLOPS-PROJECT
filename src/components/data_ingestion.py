import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformer

@dataclass
class DataIngestionConfig():
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            logging.info("Reading the dataset")
            df = pd.read_csv('notebook/final_data.csv')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the data in data.csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test Split Initiated")
            # Splitting the data into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion Completed")

            return train_set, test_set  # Return the train_set and test_set

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformer()
    transformed_train_data , transformed_test_data = data_transformation.transform((train_data, test_data))
    data_transformation.save_transformed_data(transformed_train_data, transformed_test_data)
    data_transformation.save_label_encoders()

