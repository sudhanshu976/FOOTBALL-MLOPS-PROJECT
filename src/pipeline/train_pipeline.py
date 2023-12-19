from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer
from src.components.config import Config    

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformer()
    transformed_train_data , transformed_test_data = data_transformation.transform((train_data, test_data))
    data_transformation.save_transformed_data(transformed_train_data, transformed_test_data)
    data_transformation.save_label_encoders()
    trainer = ModelTrainer(Config())
    trainer.train_and_evaluate()