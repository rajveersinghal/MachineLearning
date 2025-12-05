import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def train_pipeline():
    logging.info("===== Training Pipeline Started =====")

    try:
        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train: {train_path}, Test: {test_path}")

        # 2. Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # 3. Model Training
        trainer = ModelTrainer()
        best_score, best_model_name= trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("===== Training Pipeline Completed Successfully =====")
        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"R2 Score: {best_score}")

        return best_model_name, best_score, 

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    model, score, params = train_pipeline()
    print("\nTraining Completed:")
    print("Best Model:", model)
    print("R2 Score:", score)
    print("Best Params:", params)
